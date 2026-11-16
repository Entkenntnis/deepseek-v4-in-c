/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_expert_vk.c - one routed FP4 expert's fused pass on the iGPU.
 *
 * The shader is the one from bench/vk_expert: w1 and w3, one barrier, fused
 * SWiGLU, one barrier, w2. The reduction tree still matches the CPU matmul;
 * only the SwiGLU exp differs, which is accepted. Weights come straight from
 * the cache's imported host pages through the pool's VkBuffer.
 *
 * This iteration is deliberately one-by-one: one submit and one fence per
 * expert. It is a correctness/bring-up step, not the performance step.
 */
#define _GNU_SOURCE

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_expert_vk.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "expert_fp4_spv.h"

/* Must match the shader's push-constant block. */
typedef struct {
    uint32_t in_n, out_n, sc, row_bytes;
    uint32_t w_off, s_off, x_off, up_off, y_off, mode;
    float    limit;
} PC;

#define WG_ROWS 64u
#define SWIGLU_WG 256u

struct DSV4ExpertVk {
    DSV4CacheVk *vk;
    VkDevice     dev;
    VkQueue      queue;

    VkDescriptorSetLayout dsl;
    VkPipelineLayout      pl;
    VkDescriptorPool      dpool;
    VkPipeline            pipe;
    VkDescriptorSet      *dset;
    int                   npool;

    VkCommandPool   cpool;
    VkCommandBuffer cmd;
    VkFence         fence;

    DSV4VkMapped scratch;
    int   hidden, inter;
    float limit;
    uint32_t x_off, gate_off, up_off, mid_off, y_off, trio_floats;
};

static int vkchk(VkResult r, const char *what)
{
    if (r == VK_SUCCESS) return 0;
    fprintf(stderr, "dsv4_expert_vk: %s failed (%d)\n", what, (int)r);
    return -1;
}

static void rec_matvec(DSV4ExpertVk *g, VkCommandBuffer cmd,
                       uint32_t w_off, uint32_t s_off,
                       uint32_t in_n, uint32_t out_n, uint32_t x_off,
                       uint32_t y_off)
{
    PC pc;
    memset(&pc, 0, sizeof pc);
    pc.in_n      = in_n;
    pc.out_n     = out_n;
    pc.sc        = in_n / 32u;
    pc.row_bytes = in_n / 2u;
    pc.w_off     = w_off;
    pc.s_off     = s_off;
    pc.x_off     = x_off;
    pc.y_off     = y_off;
    pc.mode      = 0;
    vkCmdPushConstants(cmd, g->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof pc, &pc);
    vkCmdDispatch(cmd, (out_n + WG_ROWS - 1u) / WG_ROWS, 1, 1);
}

static void rec_swiglu(DSV4ExpertVk *g, VkCommandBuffer cmd)
{
    PC pc;
    memset(&pc, 0, sizeof pc);
    pc.in_n   = (uint32_t)g->inter;
    pc.x_off  = g->gate_off;
    pc.up_off = g->up_off;
    pc.y_off  = g->mid_off;
    pc.mode   = 1;
    pc.limit  = g->limit;
    vkCmdPushConstants(cmd, g->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof pc, &pc);
    vkCmdDispatch(cmd, ((uint32_t)g->inter + SWIGLU_WG - 1u) / SWIGLU_WG,
                  1, 1);
}

static void mem_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof mb);
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, NULL, 0, NULL);
}

static uint32_t slot_off(const DSV4ExpertVk *g, const void *p,
                         const DSV4ExpertW *e)
{
    const int64_t d = (const unsigned char *)p - e->vk_base;
    (void)g;
    return (uint32_t)(e->vk_off + d);
}

int dsv4_expert_vk_init(DSV4ExpertVk **out, DSV4CacheVk *cache_vk,
                        const DSV4Cfg *cfg)
{
    if (!out || !cache_vk || !cfg) return -1;
    *out = NULL;

    const int hidden = cfg->hidden, inter = cfg->moe_inter;
    /* The shader has no tail loop: it consumes 128 weights per iteration. */
    if (hidden <= 0 || inter <= 0 || (hidden & 127) || (inter & 127)) {
        fprintf(stderr, "dsv4_expert_vk: geometry %dx%d is not a multiple of "
                        "128\n", hidden, inter);
        return -1;
    }

    DSV4ExpertVk *g = (DSV4ExpertVk *)calloc(1, sizeof *g);
    if (!g) return -1;
    g->vk      = cache_vk;
    g->dev     = dsv4_cache_vk_device(cache_vk);
    g->queue   = dsv4_cache_vk_queue(cache_vk);
    g->hidden  = hidden;
    g->inter   = inter;
    g->limit   = cfg->swiglu_limit;
    g->x_off   = 0;
    g->gate_off = (uint32_t)hidden;
    g->up_off   = g->gate_off + (uint32_t)inter;
    g->mid_off  = g->up_off + (uint32_t)inter;
    g->y_off    = g->mid_off + (uint32_t)inter;
    g->trio_floats = g->y_off + (uint32_t)hidden;
    g->npool   = dsv4_cache_vk_pool_count(cache_vk);
    if (!g->dev || !g->queue || g->npool <= 0) {
        fprintf(stderr, "dsv4_expert_vk: cache has no Vulkan device or pools\n");
        free(g);
        return -1;
    }

    if (dsv4_cache_vk_alloc_mapped(cache_vk,
                                   (uint64_t)g->trio_floats * sizeof(float),
                                   &g->scratch) != 0) {
        fprintf(stderr, "dsv4_expert_vk: could not allocate scratch\n");
        free(g);
        return -1;
    }
    g->dset = (VkDescriptorSet *)calloc((size_t)g->npool, sizeof *g->dset);
    if (!g->dset) { dsv4_expert_vk_free(g); return -1; }

    VkDescriptorSetLayoutBinding b[3];
    memset(b, 0, sizeof b);
    for (int i = 0; i < 3; i++) {
        b[i].binding = (uint32_t)i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli;
    memset(&dli, 0, sizeof dli);
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 3;
    dli.pBindings = b;
    if (vkchk(vkCreateDescriptorSetLayout(g->dev, &dli, NULL, &g->dsl),
              "descriptor set layout")) { dsv4_expert_vk_free(g); return -1; }

    VkPushConstantRange pcr;
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PC);
    VkPipelineLayoutCreateInfo pli;
    memset(&pli, 0, sizeof pli);
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g->dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkchk(vkCreatePipelineLayout(g->dev, &pli, NULL, &g->pl),
              "pipeline layout")) { dsv4_expert_vk_free(g); return -1; }

    VkDescriptorPoolSize psz;
    psz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psz.descriptorCount = (uint32_t)(3 * g->npool);
    VkDescriptorPoolCreateInfo dpi;
    memset(&dpi, 0, sizeof dpi);
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = (uint32_t)g->npool;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &psz;
    if (vkchk(vkCreateDescriptorPool(g->dev, &dpi, NULL, &g->dpool),
              "descriptor pool")) { dsv4_expert_vk_free(g); return -1; }

    VkDescriptorSetAllocateInfo ai;
    memset(&ai, 0, sizeof ai);
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = g->dpool;
    ai.descriptorSetCount = (uint32_t)g->npool;
    ai.pSetLayouts = &g->dsl;
    /* pSetLayouts is an array of npool identical layouts; Vk requires one
     * layout per set. Build it once. */
    VkDescriptorSetLayout *layouts =
        (VkDescriptorSetLayout *)calloc((size_t)g->npool, sizeof *layouts);
    if (!layouts) { dsv4_expert_vk_free(g); return -1; }
    for (int i = 0; i < g->npool; i++) layouts[i] = g->dsl;
    ai.pSetLayouts = layouts;
    VkResult ar = vkAllocateDescriptorSets(g->dev, &ai, g->dset);
    free(layouts);
    if (vkchk(ar, "descriptor sets")) { dsv4_expert_vk_free(g); return -1; }

    for (int p = 0; p < g->npool; p++) {
        VkBuffer pb = dsv4_cache_vk_pool_buffer(cache_vk, p);
        VkDescriptorBufferInfo bi[3] = {
            { pb, 0, VK_WHOLE_SIZE },
            { pb, 0, VK_WHOLE_SIZE },
            { g->scratch.buf, 0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet w[3];
        memset(w, 0, sizeof w);
        for (int i = 0; i < 3; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = g->dset[p];
            w[i].dstBinding = (uint32_t)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(g->dev, 3, w, 0, NULL);
    }

    VkShaderModuleCreateInfo smi;
    memset(&smi, 0, sizeof smi);
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = expert_fp4_spv_len;
    smi.pCode = expert_fp4_spv;
    VkShaderModule mod;
    if (vkchk(vkCreateShaderModule(g->dev, &smi, NULL, &mod),
              "shader module")) { dsv4_expert_vk_free(g); return -1; }

    VkComputePipelineCreateInfo cpi;
    memset(&cpi, 0, sizeof cpi);
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = g->pl;
    VkResult pr = vkCreateComputePipelines(g->dev, VK_NULL_HANDLE, 1, &cpi,
                                           NULL, &g->pipe);
    vkDestroyShaderModule(g->dev, mod, NULL);
    if (vkchk(pr, "compute pipeline")) { dsv4_expert_vk_free(g); return -1; }

    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof pci);
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = dsv4_cache_vk_queue_family(cache_vk);
    if (vkchk(vkCreateCommandPool(g->dev, &pci, NULL, &g->cpool),
              "command pool")) { dsv4_expert_vk_free(g); return -1; }

    VkCommandBufferAllocateInfo cbi;
    memset(&cbi, 0, sizeof cbi);
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = g->cpool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    if (vkchk(vkAllocateCommandBuffers(g->dev, &cbi, &g->cmd),
              "command buffer")) { dsv4_expert_vk_free(g); return -1; }

    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof fci);
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkchk(vkCreateFence(g->dev, &fci, NULL, &g->fence),
              "fence")) { dsv4_expert_vk_free(g); return -1; }

    *out = g;
    return 0;
}

void dsv4_expert_vk_free(DSV4ExpertVk *g)
{
    if (!g) return;
    if (g->dev) vkDeviceWaitIdle(g->dev);
    if (g->fence) vkDestroyFence(g->dev, g->fence, NULL);
    if (g->cpool) vkDestroyCommandPool(g->dev, g->cpool, NULL);
    if (g->pipe) vkDestroyPipeline(g->dev, g->pipe, NULL);
    if (g->dpool) vkDestroyDescriptorPool(g->dev, g->dpool, NULL);
    if (g->pl) vkDestroyPipelineLayout(g->dev, g->pl, NULL);
    if (g->dsl) vkDestroyDescriptorSetLayout(g->dev, g->dsl, NULL);
    if (g->vk && g->scratch.buf)
        dsv4_cache_vk_free_mapped(g->vk, &g->scratch);
    free(g->dset);
    free(g);
}

int dsv4_expert_vk_forward(DSV4ExpertVk *g, const DSV4ExpertW *e,
                           const float *x, float *out)
{
    if (!g || !e || !e->vk_base || !x || !out) return -1;
    if (e->vk_pool < 0 || e->vk_pool >= g->npool) return -1;

    float *f = (float *)g->scratch.cpu;
    memcpy(f + g->x_off, x, (size_t)g->hidden * sizeof(float));

    VkCommandBuffer cmd = g->cmd;
    if (vkchk(vkResetCommandBuffer(cmd, 0), "reset command buffer"))
        return -1;
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkchk(vkBeginCommandBuffer(cmd, &bi), "begin command buffer"))
        return -1;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->pl, 0, 1,
                            &g->dset[e->vk_pool], 0, NULL);

    rec_matvec(g, cmd, slot_off(g, e->w1.w, e), slot_off(g, e->w1.s, e),
               (uint32_t)g->hidden, (uint32_t)g->inter,
               g->x_off, g->gate_off);
    rec_matvec(g, cmd, slot_off(g, e->w3.w, e), slot_off(g, e->w3.s, e),
               (uint32_t)g->hidden, (uint32_t)g->inter,
               g->x_off, g->up_off);
    mem_barrier(cmd);
    rec_swiglu(g, cmd);
    mem_barrier(cmd);
    rec_matvec(g, cmd, slot_off(g, e->w2.w, e), slot_off(g, e->w2.s, e),
               (uint32_t)g->inter, (uint32_t)g->hidden,
               g->mid_off, g->y_off);

    if (vkchk(vkEndCommandBuffer(cmd), "end command buffer")) return -1;

    VkResult fr = vkResetFences(g->dev, 1, &g->fence);
    if (vkchk(fr, "reset fence")) return -1;
    VkSubmitInfo si;
    memset(&si, 0, sizeof si);
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    fr = vkQueueSubmit(g->queue, 1, &si, g->fence);
    if (vkchk(fr, "queue submit")) return -1;
    fr = vkWaitForFences(g->dev, 1, &g->fence, VK_TRUE, UINT64_MAX);
    if (vkchk(fr, "wait for expert fence")) return -1;

    memcpy(out, f + g->y_off, (size_t)g->hidden * sizeof(float));

    /* One-shot diagnostic: compare the first real routed expert against the
     * CPU trio. Expected rel L2 is ~1e-7 (only the SWiGLU exp differs). */
    static int checked, check_ready;
    static const char *check;
    if (!check_ready) {
        check = getenv("DSV4_VK_CHECK");
        check_ready = 1;
    }
    if (check && !checked) {
        checked = 1;
        float *gate = (float *)malloc((size_t)g->inter * sizeof(float));
        float *up   = (float *)malloc((size_t)g->inter * sizeof(float));
        float *mid  = (float *)malloc((size_t)g->inter * sizeof(float));
        float *cpu  = (float *)malloc((size_t)g->hidden * sizeof(float));
        if (gate && up && mid && cpu) {
            dsv4_mmq(gate, x, &e->w1);
            dsv4_mmq(up, x, &e->w3);
            dsv4_swiglu(mid, gate, up, g->inter, g->limit);
            dsv4_mmq(cpu, mid, &e->w2);
            double e2 = 0.0, r2 = 0.0, mx = 0.0;
            for (int i = 0; i < g->hidden; i++) {
                const double d = (double)out[i] - (double)cpu[i];
                e2 += d * d;
                r2 += (double)cpu[i] * (double)cpu[i];
                if (fabs(d) > mx) mx = fabs(d);
            }
            FILE *lf = strcmp(check, "1") ? fopen(check, "a") : stderr;
            if (lf) {
                fprintf(lf, "vk_check: relL2 %.3e max %.3e\n",
                        sqrt(e2 / (r2 > 0.0 ? r2 : 1.0)), mx);
                fprintf(lf, "vk_check: pool %d slot_off %lld\n"
                            "  w1 w %lld s %lld\n"
                            "  w3 w %lld s %lld\n"
                            "  w2 w %lld s %lld\n",
                        e->vk_pool, (long long)e->vk_off,
                        (long long)((const unsigned char *)e->w1.w - e->vk_base),
                        (long long)((const unsigned char *)e->w1.s - e->vk_base),
                        (long long)((const unsigned char *)e->w3.w - e->vk_base),
                        (long long)((const unsigned char *)e->w3.s - e->vk_base),
                        (long long)((const unsigned char *)e->w2.w - e->vk_base),
                        (long long)((const unsigned char *)e->w2.s - e->vk_base));
                if (lf != stderr) fclose(lf);
            }
        }
        free(gate); free(up); free(mid); free(cpu);
    }
    return 0;
}
