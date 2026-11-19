/* SPDX-License-Identifier: Apache-2.0 */
/* Synchronous 128-token GEMM tiles for FP8, BF16 and FP4 weights. */
#include <string.h>

#include "dsv4_prefill_vk_int.h"

size_t dsv4_vk_qmat_wbytes(const DSV4QMat *m)
{
    if (!m || m->rows <= 0 || m->cols <= 0) return 0;
    if (m->wdt == DSV4_WBF16) return (size_t)m->rows * (size_t)m->cols * 2u;
    if (m->wdt == DSV4_WFP8)  return (size_t)m->rows * (size_t)m->cols;
    if (m->wdt == DSV4_WFP4)  return (size_t)m->rows * ((size_t)m->cols / 2u);
    return 0;
}

size_t dsv4_vk_qmat_sbytes(const DSV4QMat *m)
{
    if (!m || !m->s || m->rows <= 0 || m->cols <= 0) return 0;
    if (m->wdt == DSV4_WFP8)
        return (size_t)((m->rows + 127) / 128)
             * (size_t)((m->cols + 127) / 128);
    if (m->wdt == DSV4_WFP4)
        return (size_t)m->rows * ((size_t)m->cols / 32u);
    return 0;
}

static int submit_tile(DSV4PrefillVk *v, VkPipeline pipe,
                       uint32_t M, uint32_t K, uint32_t N,
                       uint32_t w_off, uint32_t s_off)
{
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

    if (vkResetFences(v->dev, 1, &v->fence) != VK_SUCCESS) return -1;
    VkSubmitInfo si;
    memset(&si, 0, sizeof si);
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &v->cmd;
    if (vkQueueSubmit(v->queue, 1, &si, v->fence) != VK_SUCCESS) return -1;
    return 0;
}

static int wait_tile(DSV4PrefillVk *v)
{
    const VkResult rc = vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE,
                                        UINT64_MAX);
    return rc == VK_SUCCESS ? 0 : -1;
}

static int gemm_impl(DSV4PrefillVk *v, float *y, const float *x,
                     const DSV4QMat *m, int n)
{
    if (!v || v->dead || !m || !x || !y || n <= 0) return -1;
    if (m->rows <= 0 || m->cols <= 0) return -1;
    if ((m->rows & 63) || (m->cols & 127)) return -1;
    if (m->wdt != DSV4_WFP8 && m->wdt != DSV4_WBF16
        && m->wdt != DSV4_WFP4) return -1;

    const size_t wb = dsv4_vk_qmat_wbytes(m);
    if (!wb) return -1;
    DSV4VkRegion *rw = dsv4_prefill_vk_region_for(v, m->w, wb);
    if (!rw) return -1;
    const VkBuffer wbuf = rw->buf;
    const uint32_t w_off = (uint32_t)((const unsigned char *)m->w - rw->base);

    DSV4VkRegion *rs = rw;
    VkBuffer sbuf = wbuf;
    uint32_t s_off = 0;
    if (m->wdt != DSV4_WBF16) {
        const size_t sb = dsv4_vk_qmat_sbytes(m);
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
    vkUpdateDescriptorSets(v->dev, 4, wr, 0, NULL);

    const uint32_t M = (uint32_t)m->rows, K = (uint32_t)m->cols;
    for (int n0 = 0; n0 < n; n0 += (int)DSV4_VK_TILE_N) {
        uint32_t t = (uint32_t)(n - n0);
        if (t > DSV4_VK_TILE_N) t = DSV4_VK_TILE_N;
        const size_t xb = (size_t)t * K * sizeof(float);
        const size_t yb = (size_t)t * M * sizeof(float);
        if (xb > v->xbytes || yb > v->ybytes) return -1;
        memcpy(v->xmap, x + (size_t)n0 * K, xb);
        if (submit_tile(v, pipe, M, K, t, w_off, s_off) != 0) {
            v->dead = 1;
            return -1;
        }
        if (wait_tile(v) != 0) {
            v->dead = 1;
            return -1;
        }
        memcpy(y + (size_t)n0 * M, v->ymap, yb);
    }
    return 0;
}

int dsv4_prefill_vk_gemm(DSV4PrefillVk *v, float *y, const float *x,
                         const DSV4QMat *m, int n)
{
    return gemm_impl(v, y, x, m, n);
}
