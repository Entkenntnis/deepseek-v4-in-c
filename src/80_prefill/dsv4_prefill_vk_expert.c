/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_vk_expert.c - the fused FP4 expert command buffer. */
#include <string.h>

#include "dsv4_prefill_vk_int.h"

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
    const size_t wb_n = dsv4_vk_qmat_wbytes(m), sb_n = dsv4_vk_qmat_sbytes(m);
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
                                 DSV4PrefillVkFillFn fill, void *fill_ctx)
{
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

    for (int n0 = 0; n0 < n; n0 += (int)DSV4_VK_TILE_N) {
        uint32_t t = (uint32_t)(n - n0);
        if (t > DSV4_VK_TILE_N) t = DSV4_VK_TILE_N;
        const size_t xb = (size_t)t * K1 * sizeof(float);
        const size_t yb = (size_t)t * M2 * sizeof(float);
        if (xb > v->xbytes || yb > v->ybytes ||
            (size_t)t * (size_t)inter * sizeof(float) > v->expert_mid_bytes)
            return -1;

        memcpy(v->xmap, x + (size_t)n0 * K1, xb);

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
        vkUpdateDescriptorSets(v->dev, 16, wr, 0, NULL);

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

        if (fill) {
            for (;;) {
                const VkResult fs = vkGetFenceStatus(v->dev, v->fence);
                if (fs != VK_NOT_READY) break;
                if (!fill(fill_ctx)) break;
            }
        }
        if (vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE, UINT64_MAX)
            != VK_SUCCESS) {
            v->dead = 1;
            return -1;
        }

        memcpy(y + (size_t)n0 * M2, v->ymap, yb);
    }
    return 0;
}
