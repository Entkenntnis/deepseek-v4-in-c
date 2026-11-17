/* SPDX-License-Identifier: Apache-2.0 */
/* Synchronous 128-token GEMM tiles for FP8, BF16 and FP4 weights. */
#include <string.h>

#include "dsv4_prof.h"
#include "dsv4_prefill_vk_int.h"

static size_t qmat_wbytes(const DSV4QMat *m)
{
    if (!m || m->rows <= 0 || m->cols <= 0) return 0;
    if (m->wdt == DSV4_WBF16) return (size_t)m->rows * (size_t)m->cols * 2u;
    if (m->wdt == DSV4_WFP8)  return (size_t)m->rows * (size_t)m->cols;
    if (m->wdt == DSV4_WFP4)  return (size_t)m->rows * ((size_t)m->cols / 2u);
    return 0;
}

static size_t qmat_sbytes(const DSV4QMat *m)
{
    if (!m || !m->s || m->rows <= 0 || m->cols <= 0) return 0;
    if (m->wdt == DSV4_WFP8)
        return (size_t)((m->rows + 127) / 128)
             * (size_t)((m->cols + 127) / 128);
    if (m->wdt == DSV4_WFP4)
        return (size_t)m->rows * ((size_t)m->cols / 32u);
    return 0;
}

static int run_tile(DSV4PrefillVk *v, int kind, VkPipeline pipe,
                    uint32_t M, uint32_t K, uint32_t N,
                    uint32_t w_off, uint32_t s_off)
{
    double t0 = dsv4_now_s();
    if (vkResetCommandBuffer(v->cmd, 0) != VK_SUCCESS) return -1;
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(v->cmd, &bi) != VK_SUCCESS) return -1;
    vkCmdBindPipeline(v->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(v->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl,
                            0, 1, &v->dset, 0, NULL);
    const DSV4VkPC pc = { M, K, N, w_off, s_off };
    vkCmdPushConstants(v->cmd, v->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof pc, &pc);
    vkCmdDispatch(v->cmd, (M + 63u) / 64u, (N + 15u) / 16u, 1);
    if (vkEndCommandBuffer(v->cmd) != VK_SUCCESS) return -1;
    v->st[kind].record_s += dsv4_now_s() - t0;

    t0 = dsv4_now_s();
    if (vkResetFences(v->dev, 1, &v->fence) != VK_SUCCESS) return -1;
    VkSubmitInfo si;
    memset(&si, 0, sizeof si);
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &v->cmd;
    if (vkQueueSubmit(v->queue, 1, &si, v->fence) != VK_SUCCESS) return -1;
    const int rc = vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE,
                                   UINT64_MAX);
    v->st[kind].submit_s += dsv4_now_s() - t0;
    return rc == VK_SUCCESS ? 0 : -1;
}

int dsv4_prefill_vk_gemm(DSV4PrefillVk *v, float *y, const float *x,
                         const DSV4QMat *m, int n, int kind)
{
    if (!v || v->dead || !m || !x || !y || n <= 0) return -1;
    if (kind < 0 || kind >= DSV4_VK_KIND_COUNT) return -1;
    if (m->rows <= 0 || m->cols <= 0) return -1;
    if ((m->rows & 63) || (m->cols & 127)) return -1;
    if (m->wdt != DSV4_WFP8 && m->wdt != DSV4_WBF16
        && m->wdt != DSV4_WFP4) return -1;
    const double tall0 = dsv4_now_s();
    DSV4PrefillVkStats *st = &v->st[kind];
    st->calls++;

    const size_t wb = qmat_wbytes(m);
    if (!wb) return -1;
    DSV4VkRegion *rw = dsv4_prefill_vk_region_for(v, m->w, wb);
    if (!rw) return -1;
    const VkBuffer wbuf = rw->buf;
    const uint32_t w_off = (uint32_t)((const unsigned char *)m->w - rw->base);

    DSV4VkRegion *rs = rw;
    VkBuffer sbuf = wbuf;
    uint32_t s_off = 0;
    if (m->wdt != DSV4_WBF16) {
        const size_t sb = qmat_sbytes(m);
        if (!m->s || !sb) return -1;
        rs = dsv4_prefill_vk_region_for(v, m->s, sb);
        if (!rs) return -1;
        sbuf = rs->buf;
        s_off = (uint32_t)((const unsigned char *)m->s - rs->base);
    }
    if ((w_off & 3u) || (s_off & 3u)) return -1;

    VkPipeline pipe = m->wdt == DSV4_WFP8 ? v->pipe_fp8
                    : m->wdt == DSV4_WBF16 ? v->pipe_bf16 : v->pipe_fp4;

    VkDescriptorBufferInfo bi[4] = {
        { wbuf, 0, VK_WHOLE_SIZE },
        { sbuf, 0, VK_WHOLE_SIZE },
        { v->xbuf, 0, VK_WHOLE_SIZE },
        { v->ybuf, 0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet wr[4];
    memset(wr, 0, sizeof wr);
    for (int i = 0; i < 4; i++) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = v->dset;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &bi[i];
    }
    const double td = dsv4_now_s();
    vkUpdateDescriptorSets(v->dev, 4, wr, 0, NULL);
    st->desc_s += dsv4_now_s() - td;

    const uint32_t M = (uint32_t)m->rows, K = (uint32_t)m->cols;
    for (int n0 = 0; n0 < n; n0 += (int)DSV4_VK_TILE_N) {
        uint32_t t = (uint32_t)(n - n0);
        if (t > DSV4_VK_TILE_N) t = DSV4_VK_TILE_N;
        const size_t xb = (size_t)t * K * sizeof(float);
        const size_t yb = (size_t)t * M * sizeof(float);
        if (xb > v->xbytes || yb > v->ybytes) return -1;
        const double tx = dsv4_now_s();
        memcpy(v->xmap, x + (size_t)n0 * K, xb);
        st->copy_in_s += dsv4_now_s() - tx;
        st->xbytes += xb;
        if (run_tile(v, kind, pipe, M, K, t, w_off, s_off) != 0) {
            v->dead = 1;
            return -1;
        }
        st->tiles++;
        const double ty = dsv4_now_s();
        memcpy(y + (size_t)n0 * M, v->ymap, yb);
        st->copy_out_s += dsv4_now_s() - ty;
        st->ybytes += yb;
    }
    st->total_s += dsv4_now_s() - tall0;
    return 0;
}

void dsv4_prefill_vk_stats(const DSV4PrefillVk *v,
                           DSV4PrefillVkStats out[DSV4_VK_KIND_COUNT])
{
    for (int i = 0; i < DSV4_VK_KIND_COUNT; i++)
        out[i] = v ? v->st[i] : (DSV4PrefillVkStats){ 0 };
}
