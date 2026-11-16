/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_expert_vk.c - routed FP4 experts on the iGPU, submitted in waves.
 *
 * Production runs the fast shader (expert_fp4_fast.comp): shared-LUT FP4
 * decode, no bit-exact CPU reduction tree, gated by tolerance. Set
 * DSV4_VK_EXACT=1 to run the original bit-identical shader instead.
 *
 * One expert is still w1/w3, one barrier, fused SwiGLU, one barrier, w2. What
 * changed is the unit of submission: every expert that is available at the same
 * moment is recorded into one command buffer, so a wave pays one submit and one
 * fence instead of one per expert. The two barriers are therefore per wave, not
 * per expert.
 *
 * Waves do not wait for each other. The caller submits a wave and keeps going;
 * dsv4_expert_vk_wait_wave() is called only when the results are needed. This
 * is what lets the shared expert and the next disk reads overlap with device
 * work.
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
#include "dsv4_prof.h"
#include "expert_fp4_spv.h"
#include "expert_fp4_fast_spv.h"

/* Must match the shader's push-constant block. */
typedef struct {
    uint32_t in_n, out_n, sc, row_bytes;
    uint32_t w_off, s_off, x_off, up_off, y_off, mode;
    float    limit;
} PC;

#define WG_ROWS 64u
#define SWIGLU_WG 256u

/* One slot for the resident/hit wave and one per possible individual miss. */
#define DSV4_VK_MAX_WAVES (DSV4_MAX_TOPK + 1)

/* Timestamps per expert: w1 start/end, w3 end, swiglu start/end, w2 start/end.
 * The "start" stamps matter in a wave: without them the gap to the previous
 * expert's dispatch would be charged to this expert's SwiGLU/w2. */
#define DSV4_VK_TS_JOB 7u

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
    VkCommandBuffer cmd[DSV4_VK_MAX_WAVES];
    VkFence         fence[DSV4_VK_MAX_WAVES];
    int             slot_used[DSV4_VK_MAX_WAVES];

    /* Five queries per expert in a wave: start, w1, w3, swiglu, w2. */
    VkQueryPool qpool;
    int         has_ts;
    uint32_t    ts_valid_bits;
    uint64_t    ts_mask;
    double      ts_period_ns;

    DSV4VkMapped scratch;
    int   hidden, inter, max_jobs;
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

static void rec_swiglu(DSV4ExpertVk *g, VkCommandBuffer cmd,
                       uint32_t gate_off, uint32_t up_off, uint32_t mid_off)
{
    PC pc;
    memset(&pc, 0, sizeof pc);
    pc.in_n   = (uint32_t)g->inter;
    pc.x_off  = gate_off;
    pc.up_off = up_off;
    pc.y_off  = mid_off;
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

static void ts_write(DSV4ExpertVk *g, VkCommandBuffer cmd, uint32_t q)
{
    if (g->has_ts)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            g->qpool, q);
}

static uint32_t slot_off(const DSV4ExpertVk *g, const void *p,
                         const DSV4ExpertW *e)
{
    const int64_t d = (const unsigned char *)p - e->vk_base;
    (void)g;
    return (uint32_t)(e->vk_off + d);
}

static int take_slot(DSV4ExpertVk *g)
{
    for (int i = 0; i < DSV4_VK_MAX_WAVES; i++) {
        if (!g->slot_used[i]) {
            g->slot_used[i] = 1;
            return i;
        }
    }
    return -1;
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
    g->max_jobs = cfg->topk > 0 && cfg->topk <= DSV4_MAX_TOPK
                ? cfg->topk : DSV4_MAX_TOPK;
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
                                   (uint64_t)g->trio_floats * g->max_jobs
                                   * sizeof(float), &g->scratch) != 0) {
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

    /* The fast shader drops the bit-exact CPU reduction tree and decodes FP4
     * through a shared LUT; production uses it. DSV4_VK_EXACT=1 switches back
     * to the exact kernel for validation. Both share the push-constant and
     * descriptor layout, so only the module differs. */
    const char *exact_env = getenv("DSV4_VK_EXACT");
    const int exact = exact_env && *exact_env && strcmp(exact_env, "0") != 0;
    const uint32_t *code = exact ? expert_fp4_spv : expert_fp4_fast_spv;
    const size_t code_len = exact ? expert_fp4_spv_len
                                  : expert_fp4_fast_spv_len;
    fprintf(stderr, "dsv4_expert_vk: expert kernel %s\n",
            exact ? "exact (bit-identical)" : "fast (tolerance gate)");

    VkShaderModuleCreateInfo smi;
    memset(&smi, 0, sizeof smi);
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = code_len;
    smi.pCode = code;
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
    cbi.commandBufferCount = DSV4_VK_MAX_WAVES;
    if (vkchk(vkAllocateCommandBuffers(g->dev, &cbi, g->cmd),
              "command buffers")) { dsv4_expert_vk_free(g); return -1; }

    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof fci);
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    for (int i = 0; i < DSV4_VK_MAX_WAVES; i++)
        if (vkchk(vkCreateFence(g->dev, &fci, NULL, &g->fence[i]),
                  "fence")) { dsv4_expert_vk_free(g); return -1; }

    if (dsv4_cache_vk_has_timestamps(cache_vk)) {
        VkQueryPoolCreateInfo qpi;
        memset(&qpi, 0, sizeof qpi);
        qpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount =
            (uint32_t)(DSV4_VK_MAX_WAVES * g->max_jobs * DSV4_VK_TS_JOB);
        if (vkCreateQueryPool(g->dev, &qpi, NULL, &g->qpool) == VK_SUCCESS) {
            g->has_ts = 1;
            g->ts_period_ns = dsv4_cache_vk_timestamp_period_ns(cache_vk);
            g->ts_valid_bits = dsv4_cache_vk_timestamp_valid_bits(cache_vk);
            /* vkCmdWriteTimestamp may deliver fewer than 64 bits. Mask the
             * unsigned difference so a counter wrap stays a small duration. */
            g->ts_mask = g->ts_valid_bits >= 64
                       ? UINT64_MAX
                       : ((UINT64_C(1) << g->ts_valid_bits) - 1u);
        }
    }

    *out = g;
    return 0;
}

void dsv4_expert_vk_free(DSV4ExpertVk *g)
{
    if (!g) return;
    if (g->dev) vkDeviceWaitIdle(g->dev);
    for (int i = 0; i < DSV4_VK_MAX_WAVES; i++)
        if (g->fence[i]) vkDestroyFence(g->dev, g->fence[i], NULL);
    if (g->qpool) vkDestroyQueryPool(g->dev, g->qpool, NULL);
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

int dsv4_expert_vk_submit_wave(DSV4ExpertVk *g,
                               const DSV4ExpertVkJob *jobs, int n,
                               const float *x, int timing,
                               DSV4ExpertVkWave *w)
{
    if (!g || !jobs || !w || !x || n <= 0 || n > g->max_jobs) return -1;
    for (int j = 0; j < n; j++) {
        if (jobs[j].k < 0 || jobs[j].k >= g->max_jobs ||
            !jobs[j].e || !jobs[j].e->vk_base || !jobs[j].out)
            return -1;
        if (jobs[j].e->vk_pool < 0 || jobs[j].e->vk_pool >= g->npool)
            return -1;
        for (int p = 0; p < j; p++)
            if (jobs[p].k == jobs[j].k) return -1;
    }

    const int slot = take_slot(g);
    if (slot < 0) {
        fprintf(stderr, "dsv4_expert_vk: no free wave slot (max %d)\n",
                DSV4_VK_MAX_WAVES);
        return -1;
    }

    memset(w, 0, sizeof *w);
    w->n = n;
    w->slot = slot;
    w->timing = timing;
    w->valid = 1;
    w->start_at = timing ? dsv4_now_s() : 0.0;
    double ta = w->start_at;

    float *f = (float *)g->scratch.cpu;
    for (int j = 0; j < n; j++) {
        const uint64_t base = (uint64_t)jobs[j].k * g->trio_floats;
        w->ks[j] = jobs[j].k;
        w->out[j] = jobs[j].out;
        memcpy(f + base + g->x_off, x, (size_t)g->hidden * sizeof(float));
    }
    if (timing) {
        const double now = dsv4_now_s();
        w->copy_in_ms = (now - ta) * 1000.0;
        ta = now;
    }

    VkCommandBuffer cmd = g->cmd[slot];
    if (vkchk(vkResetCommandBuffer(cmd, 0), "reset command buffer"))
        { g->slot_used[slot] = 0; return -1; }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkchk(vkBeginCommandBuffer(cmd, &bi), "begin command buffer"))
        { g->slot_used[slot] = 0; return -1; }

    const int gpu_ts = timing && g->has_ts;
    const uint32_t qbase = (uint32_t)slot * (uint32_t)g->max_jobs *
                           DSV4_VK_TS_JOB;
    if (gpu_ts)
        vkCmdResetQueryPool(cmd, g->qpool, qbase,
                            (uint32_t)n * DSV4_VK_TS_JOB);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->pipe);

    /* Phase 1: w1 and w3 for every expert. No barrier between these; they are
     * independent and write different regions of the scratch. */
    for (int j = 0; j < n; j++) {
        const DSV4ExpertW *e = jobs[j].e;
        const uint32_t base = (uint32_t)((uint64_t)jobs[j].k * g->trio_floats);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->pl, 0,
                                1, &g->dset[e->vk_pool], 0, NULL);
        if (gpu_ts) ts_write(g, cmd, qbase + (uint32_t)j * DSV4_VK_TS_JOB + 0u);
        rec_matvec(g, cmd, slot_off(g, e->w1.w, e), slot_off(g, e->w1.s, e),
                   (uint32_t)g->hidden, (uint32_t)g->inter,
                   base + g->x_off, base + g->gate_off);
        if (gpu_ts) ts_write(g, cmd, qbase + (uint32_t)j * DSV4_VK_TS_JOB + 1u);
        rec_matvec(g, cmd, slot_off(g, e->w3.w, e), slot_off(g, e->w3.s, e),
                   (uint32_t)g->hidden, (uint32_t)g->inter,
                   base + g->x_off, base + g->up_off);
        if (gpu_ts) ts_write(g, cmd, qbase + (uint32_t)j * DSV4_VK_TS_JOB + 2u);
    }

    mem_barrier(cmd);

    /* Phase 2: SwiGLU for every expert. */
    for (int j = 0; j < n; j++) {
        const DSV4ExpertW *e = jobs[j].e;
        const uint32_t base = (uint32_t)((uint64_t)jobs[j].k * g->trio_floats);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->pl, 0,
                                1, &g->dset[e->vk_pool], 0, NULL);
        if (gpu_ts) ts_write(g, cmd, qbase + (uint32_t)j * DSV4_VK_TS_JOB + 3u);
        rec_swiglu(g, cmd, base + g->gate_off, base + g->up_off,
                   base + g->mid_off);
        if (gpu_ts) ts_write(g, cmd, qbase + (uint32_t)j * DSV4_VK_TS_JOB + 4u);
    }

    mem_barrier(cmd);

    /* Phase 3: w2 for every expert. */
    for (int j = 0; j < n; j++) {
        const DSV4ExpertW *e = jobs[j].e;
        const uint32_t base = (uint32_t)((uint64_t)jobs[j].k * g->trio_floats);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->pl, 0,
                                1, &g->dset[e->vk_pool], 0, NULL);
        if (gpu_ts) ts_write(g, cmd, qbase + (uint32_t)j * DSV4_VK_TS_JOB + 5u);
        rec_matvec(g, cmd, slot_off(g, e->w2.w, e), slot_off(g, e->w2.s, e),
                   (uint32_t)g->inter, (uint32_t)g->hidden,
                   base + g->mid_off, base + g->y_off);
        if (gpu_ts) ts_write(g, cmd, qbase + (uint32_t)j * DSV4_VK_TS_JOB + 6u);
    }

    if (vkchk(vkEndCommandBuffer(cmd), "end command buffer"))
        { g->slot_used[slot] = 0; return -1; }
    if (timing) {
        const double now = dsv4_now_s();
        w->record_ms = (now - ta) * 1000.0;
        ta = now;
    }

    VkResult fr = vkResetFences(g->dev, 1, &g->fence[slot]);
    if (vkchk(fr, "reset fence")) { g->slot_used[slot] = 0; return -1; }
    VkSubmitInfo si;
    memset(&si, 0, sizeof si);
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    fr = vkQueueSubmit(g->queue, 1, &si, g->fence[slot]);
    if (vkchk(fr, "queue submit")) { g->slot_used[slot] = 0; return -1; }
    if (timing) {
        const double now = dsv4_now_s();
        w->submit_ms = (now - ta) * 1000.0;
        w->submit_at = now;
    }
    return 0;
}

int dsv4_expert_vk_wait_wave(DSV4ExpertVk *g, DSV4ExpertVkWave *w)
{
    if (!g || !w || w->slot < 0 || w->slot >= DSV4_VK_MAX_WAVES ||
        !g->slot_used[w->slot])
        return -1;

    const double tw = w->timing ? dsv4_now_s() : 0.0;
    const VkResult fr =
        vkWaitForFences(g->dev, 1, &g->fence[w->slot], VK_TRUE, UINT64_MAX);
    if (vkchk(fr, "wait for wave fence")) return -1;
    if (w->timing) {
        w->fence_done_at = dsv4_now_s();
        w->wait_ms = (w->fence_done_at - tw) * 1000.0;
    }

    const double tc = w->timing ? dsv4_now_s() : 0.0;
    float *f = (float *)g->scratch.cpu;
    for (int j = 0; j < w->n; j++) {
        const uint64_t base = (uint64_t)w->ks[j] * g->trio_floats;
        memcpy(w->out[j], f + base + g->y_off,
               (size_t)g->hidden * sizeof(float));
    }
    if (w->timing) {
        const double now = dsv4_now_s();
        w->copy_out_ms = (now - tc) * 1000.0;
    }

    if (w->timing && g->has_ts) {
        uint64_t ts[DSV4_MAX_TOPK * DSV4_VK_TS_JOB];
        const uint32_t qbase = (uint32_t)w->slot * (uint32_t)g->max_jobs *
                               DSV4_VK_TS_JOB;
        const uint32_t nq = (uint32_t)w->n * DSV4_VK_TS_JOB;
        const VkResult qr =
            vkGetQueryPoolResults(g->dev, g->qpool, qbase, nq,
                                  sizeof(uint64_t) * nq, ts,
                                  sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT
                                  | VK_QUERY_RESULT_WAIT_BIT);
        if (qr == VK_SUCCESS) {
            const uint64_t m = g->ts_mask;
            const double ms = g->ts_period_ns / 1e6;
            uint64_t t0 = ts[0], t1 = ts[DSV4_VK_TS_JOB - 1];
            for (int j = 1; j < w->n; j++) {
                const uint64_t *q = &ts[j * DSV4_VK_TS_JOB];
                if (q[0] < t0) t0 = q[0];
                if (q[DSV4_VK_TS_JOB - 1] > t1) t1 = q[DSV4_VK_TS_JOB - 1];
            }
            w->gpu_span_ms = (double)((t1 - t0) & m) * ms;
            /* Waves are waited for late, so fence_done_at is not the moment the
             * device finished. Anchor the device span at the submit instead;
             * the relative timestamps inside the wave are exact. */
            w->gpu_anchor_at = w->submit_at;
            w->gpu_ts_valid = 1;
            for (int j = 0; j < w->n; j++) {
                DSV4ExpertVkTiming *e = &w->expert[j];
                const uint64_t *q = &ts[j * DSV4_VK_TS_JOB];
                e->valid = 1;
                e->gpu_ts_valid = 1;
                e->gpu_start_ms = (double)((q[0] - t0) & m) * ms;
                e->gpu_w1_ms = (double)((q[1] - q[0]) & m) * ms;
                e->gpu_w3_ms = (double)((q[2] - q[1]) & m) * ms;
                e->gpu_swiglu_ms = (double)((q[4] - q[3]) & m) * ms;
                e->gpu_w2_ms = (double)((q[6] - q[5]) & m) * ms;
                e->gpu_total_ms = e->gpu_w1_ms + e->gpu_w3_ms +
                                  e->gpu_swiglu_ms + e->gpu_w2_ms;
            }
        }
    }

    if (w->timing) {
        w->total_ms = w->copy_in_ms + w->record_ms + w->submit_ms +
                      (w->gpu_ts_valid ? w->gpu_span_ms : w->wait_ms) +
                      w->copy_out_ms;
    }

    g->slot_used[w->slot] = 0;
    return 0;
}

int dsv4_expert_vk_forward_timed(DSV4ExpertVk *g, const DSV4ExpertW *e,
                                 const float *x, float *out,
                                 DSV4ExpertVkTiming *t)
{
    if (!g || !e || !e->vk_base || !x || !out) return -1;

    const DSV4ExpertVkJob job = { .k = 0, .e = e, .out = out };
    DSV4ExpertVkWave w;
    if (dsv4_expert_vk_submit_wave(g, &job, 1, x, t != NULL, &w) != 0)
        return -1;
    if (dsv4_expert_vk_wait_wave(g, &w) != 0) return -1;

    if (t) {
        memset(t, 0, sizeof *t);
        *t = w.expert[0];
        t->valid = 1;
        t->copy_in_ms = w.copy_in_ms;
        t->record_ms = w.record_ms;
        t->submit_ms = w.submit_ms;
        t->fence_ms = w.wait_ms;
        t->copy_out_ms = w.copy_out_ms;
        t->total_ms = w.total_ms;
        t->submit_at = w.submit_at;
        t->fence_done_at = w.fence_done_at;
        t->gpu_ts_valid = w.gpu_ts_valid;
    }

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

int dsv4_expert_vk_forward(DSV4ExpertVk *g, const DSV4ExpertW *e,
                           const float *x, float *out)
{
    return dsv4_expert_vk_forward_timed(g, e, x, out, NULL);
}
