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

static int submit_tile(DSV4PrefillVk *v, int kind, VkPipeline pipe,
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
    v->st[kind].submit_s += dsv4_now_s() - t0;
    return 0;
}

static int wait_tile(DSV4PrefillVk *v, int kind)
{
    const double t0 = dsv4_now_s();
    const VkResult rc = vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE,
                                        UINT64_MAX);
    v->st[kind].wait_s += dsv4_now_s() - t0;
    return rc == VK_SUCCESS ? 0 : -1;
}

static int gemm_impl(DSV4PrefillVk *v, float *y, const float *x,
                     const DSV4QMat *m, int n, int kind,
                     DSV4PrefillVkFillFn fill, void *fill_ctx,
                     double *fill_s)
{
    if (fill_s) *fill_s = 0.0;
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
        if (submit_tile(v, kind, pipe, M, K, t, w_off, s_off) != 0) {
            v->dead = 1;
            return -1;
        }
        if (fill) {
            const double tf = dsv4_now_s();
            /* Keep the CPU busy for as long as this tile is still on the
             * device. One CPU bucket per call keeps the granularity small
             * enough that the loop notices the fence promptly; a long bucket
             * can overshoot by at most that bucket. */
            for (;;) {
                const VkResult fs = vkGetFenceStatus(v->dev, v->fence);
                if (fs != VK_NOT_READY) break;
                if (!fill(fill_ctx)) break;
            }
            const double fd = dsv4_now_s() - tf;
            st->fill_s += fd;
            if (fill_s) *fill_s += fd;
        }
        if (wait_tile(v, kind) != 0) {
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

int dsv4_prefill_vk_gemm(DSV4PrefillVk *v, float *y, const float *x,
                         const DSV4QMat *m, int n, int kind)
{
    return gemm_impl(v, y, x, m, n, kind, NULL, NULL, NULL);
}

int dsv4_prefill_vk_gemm_overlap(DSV4PrefillVk *v, float *y, const float *x,
                                 const DSV4QMat *m, int n, int kind,
                                 DSV4PrefillVkFillFn fill, void *fill_ctx,
                                 double *fill_s)
{
    return gemm_impl(v, y, x, m, n, kind, fill, fill_ctx, fill_s);
}

static void compute_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof mb);
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, NULL, 0, NULL);
}

/* Resolve one matrix's W/S region and offsets. */
static int expert_region(DSV4PrefillVk *v, const DSV4QMat *m,
                         VkBuffer *wb, VkBuffer *sb,
                         uint32_t *w_off, uint32_t *s_off)
{
    if (!m || m->wdt != DSV4_WFP4 || m->rows <= 0 || m->cols <= 0 ||
        (m->rows & 63) || (m->cols & 127) || !m->w || !m->s)
        return -1;
    const size_t wb_n = qmat_wbytes(m), sb_n = qmat_sbytes(m);
    DSV4VkRegion *rw = dsv4_prefill_vk_region_for(v, m->w, wb_n);
    DSV4VkRegion *rs = dsv4_prefill_vk_region_for(v, m->s, sb_n);
    if (!rw || !rs) return -1;
    *wb = rw->buf;
    *sb = rs->buf;
    *w_off = (uint32_t)((const unsigned char *)m->w - rw->base);
    *s_off = (uint32_t)((const unsigned char *)m->s - rs->base);
    return ((*w_off & 3u) || (*s_off & 3u)) ? -1 : 0;
}

/* One expert, one command buffer:
 *   w1 * x  -> gate
 *   w3 * x  -> up
 *   swiglu(gate, up) -> mid
 *   w2 * mid -> y
 * All four dispatches share a single submit/fence. The CPU can run routed
 * buckets in fill() while the fence is pending. */
int dsv4_prefill_vk_expert_fused(DSV4PrefillVk *v, float *y, const float *x,
                                 const DSV4QMat *w1, const DSV4QMat *w3,
                                 const DSV4QMat *w2, int n, int inter,
                                 float limit,
                                 DSV4PrefillVkFillFn fill, void *fill_ctx,
                                 double *fill_s)
{
    if (fill_s) *fill_s = 0.0;
    if (!v || v->dead || !y || !x || !w1 || !w3 || !w2 || n <= 0)
        return -1;
    if (inter <= 0 || w1->rows != inter || w3->rows != inter ||
        w2->cols != inter || w1->cols != w2->rows)
        return -1;

    VkBuffer w1b, s1b, w3b, s3b, w2b, s2b;
    uint32_t w1o, s1o, w3o, s3o, w2o, s2o;
    if (expert_region(v, w1, &w1b, &s1b, &w1o, &s1o) ||
        expert_region(v, w3, &w3b, &s3b, &w3o, &s3o) ||
        expert_region(v, w2, &w2b, &s2b, &w2o, &s2o))
        return -1;

    const uint32_t hidden = (uint32_t)w1->cols;
    const uint32_t M1 = (uint32_t)inter, K1 = hidden;
    const uint32_t M2 = hidden, K2 = (uint32_t)inter;
    const double tall0 = dsv4_now_s();
    DSV4PrefillVkStats *st = &v->st[DSV4_VK_KIND_EXPERT];
    st->calls++;

    for (int n0 = 0; n0 < n; n0 += (int)DSV4_VK_TILE_N) {
        uint32_t t = (uint32_t)(n - n0);
        if (t > DSV4_VK_TILE_N) t = DSV4_VK_TILE_N;
        const size_t xb = (size_t)t * K1 * sizeof(float);
        const size_t yb = (size_t)t * M2 * sizeof(float);
        if (xb > v->xbytes || yb > v->ybytes ||
            (size_t)t * (size_t)inter * sizeof(float) > v->expert_mid_bytes)
            return -1;

        const double tx = dsv4_now_s();
        memcpy(v->xmap, x + (size_t)n0 * K1, xb);
        st->copy_in_s += dsv4_now_s() - tx;
        st->xbytes += xb;

        VkDescriptorBufferInfo bi[16];
        VkWriteDescriptorSet wr[16];
        VkBuffer bufs[16] = {
            w1b, s1b, v->xbuf, v->gbuf,
            w3b, s3b, v->xbuf, v->ubuf,
            v->gbuf, v->ubuf, v->mbuf, v->mbuf,
            w2b, s2b, v->mbuf, v->ybuf,
        };
        memset(wr, 0, sizeof wr);
        for (int i = 0; i < 16; i++) {
            bi[i].buffer = bufs[i];
            bi[i].offset = 0;
            bi[i].range = VK_WHOLE_SIZE;
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = v->xdset[i / 4];
            wr[i].dstBinding = (uint32_t)(i & 3);
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &bi[i];
        }
        const double td = dsv4_now_s();
        vkUpdateDescriptorSets(v->dev, 16, wr, 0, NULL);
        st->desc_s += dsv4_now_s() - td;

        const double tr = dsv4_now_s();
        if (vkResetCommandBuffer(v->cmd, 0) != VK_SUCCESS) return -1;
        VkCommandBufferBeginInfo cbi;
        memset(&cbi, 0, sizeof cbi);
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(v->cmd, &cbi) != VK_SUCCESS) return -1;

        const DSV4VkPC pc1 = { M1, K1, t, w1o, s1o };
        vkCmdBindPipeline(v->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pipe_fp4);
        vkCmdBindDescriptorSets(v->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl,
                                0, 1, &v->xdset[0], 0, NULL);
        vkCmdPushConstants(v->cmd, v->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof pc1, &pc1);
        vkCmdDispatch(v->cmd, (M1 + 63u) / 64u, (t + 15u) / 16u, 1);

        const DSV4VkPC pc3 = { M1, K1, t, w3o, s3o };
        vkCmdBindDescriptorSets(v->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl,
                                0, 1, &v->xdset[1], 0, NULL);
        vkCmdPushConstants(v->cmd, v->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof pc3, &pc3);
        vkCmdDispatch(v->cmd, (M1 + 63u) / 64u, (t + 15u) / 16u, 1);

        compute_barrier(v->cmd);

        const DSV4VkSwigluPC sw = { t, (uint32_t)inter, limit };
        vkCmdBindPipeline(v->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pipe_sw);
        vkCmdBindDescriptorSets(v->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                v->pl_sw, 0, 1, &v->xdset[2], 0, NULL);
        vkCmdPushConstants(v->cmd, v->pl_sw, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof sw, &sw);
        vkCmdDispatch(v->cmd, (t * (uint32_t)inter + 255u) / 256u, 1, 1);

        compute_barrier(v->cmd);

        const DSV4VkPC pc2 = { M2, K2, t, w2o, s2o };
        vkCmdBindPipeline(v->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pipe_fp4);
        vkCmdBindDescriptorSets(v->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl,
                                0, 1, &v->xdset[3], 0, NULL);
        vkCmdPushConstants(v->cmd, v->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof pc2, &pc2);
        vkCmdDispatch(v->cmd, (M2 + 63u) / 64u, (t + 15u) / 16u, 1);

        if (vkEndCommandBuffer(v->cmd) != VK_SUCCESS) return -1;
        st->record_s += dsv4_now_s() - tr;

        const double ts = dsv4_now_s();
        if (vkResetFences(v->dev, 1, &v->fence) != VK_SUCCESS) return -1;
        VkSubmitInfo si;
        memset(&si, 0, sizeof si);
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &v->cmd;
        if (vkQueueSubmit(v->queue, 1, &si, v->fence) != VK_SUCCESS) {
            v->dead = 1;
            return -1;
        }
        st->submit_s += dsv4_now_s() - ts;

        if (fill) {
            const double tf = dsv4_now_s();
            for (;;) {
                const VkResult fs = vkGetFenceStatus(v->dev, v->fence);
                if (fs != VK_NOT_READY) break;
                if (!fill(fill_ctx)) break;
            }
            const double fd = dsv4_now_s() - tf;
            st->fill_s += fd;
            if (fill_s) *fill_s += fd;
        }
        const double tw = dsv4_now_s();
        if (vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE, UINT64_MAX)
            != VK_SUCCESS) {
            v->dead = 1;
            return -1;
        }
        st->wait_s += dsv4_now_s() - tw;
        st->tiles++;

        const double ty = dsv4_now_s();
        memcpy(y + (size_t)n0 * M2, v->ymap, yb);
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
