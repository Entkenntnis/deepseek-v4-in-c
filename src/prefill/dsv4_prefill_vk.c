/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_vk.c - see dsv4_prefill_vk.h. */
#define _GNU_SOURCE

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vulkan/vulkan.h>

#include "dsv4_prefill_vk.h"
#include "prefill_fp8_spv.h"

#define VK_TRY(x)                                                              \
    do {                                                                       \
        VkResult r_ = (x);                                                     \
        if (r_ != VK_SUCCESS) {                                                \
            fprintf(stderr, "prefill-vk: %s failed: %d\n", #x, (int)r_);       \
            return -1;                                                         \
        }                                                                      \
    } while (0)

#define VK_GO(x)                                                               \
    do {                                                                       \
        if ((x) != VK_SUCCESS) {                                               \
            fprintf(stderr, "prefill-vk: %s failed\n", #x);                    \
            goto fail;                                                         \
        }                                                                      \
    } while (0)

typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    void          *map;
    size_t         nbytes;
    const void    *run;
} PVLayer;

struct DSV4PrefillVk {
    VkInstance       instance;
    VkPhysicalDevice phys;
    VkDevice         dev;
    VkQueue          queue;
    uint32_t         qfam;
    VkCommandPool    cmdpool;
    VkCommandBuffer  cmd[DSV4_PV_MAX_JOBS];
    VkFence          fence;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      pl;
    VkDescriptorPool      dpool;
    VkDescriptorSet       dset;
    VkPipeline            pipe;
    VkPhysicalDeviceMemoryProperties mp;

    PVLayer lay[DSV4_PV_MAX_LAYERS];

    VkBuffer       xbuf, ybuf;
    VkDeviceMemory xmem, ymem;
    void          *xmap, *ymap;

    /* Host-visible staging for uploading a whole layer to DEVICE_LOCAL memory.
     * A device-local weight buffer is what keeps the host barrier cheap: the
     * earlier version bound the 130-165 MB layer as host-visible, and the
     * driver's barrier before every dispatch flushed that mapping. */
    VkBuffer       sbuf;
    VkDeviceMemory smem;
    void          *smap;
    size_t         scap;

    uint64_t calls;
    uint64_t bytes;
    double   secs;
};

static double pv_now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint32_t pv_mem_type(const VkPhysicalDeviceMemoryProperties *mp,
                            uint32_t bits)
{
    /* Type 2 on this ANV build: HOST_VISIBLE|HOST_COHERENT|HOST_CACHED|
     * DEVICE_LOCAL. That is the one the standalone bench measured at 74 GB/s. */
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                 VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
        if ((bits & (1u << i)) &&
            (mp->memoryTypes[i].propertyFlags & want) == want)
            return i;
    want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
        if ((bits & (1u << i)) &&
            (mp->memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

static int pv_buffer(DSV4PrefillVk *v, VkDeviceSize size,
                     VkBuffer *buf, VkDeviceMemory *mem, void **map)
{
    VkBufferCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_TRY(vkCreateBuffer(v->dev, &ci, NULL, buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(v->dev, *buf, &req);
    uint32_t ti = pv_mem_type(&v->mp, req.memoryTypeBits);
    if (ti == UINT32_MAX) {
        fprintf(stderr, "prefill-vk: no host-visible memory type for %llu B\n",
                (unsigned long long)size);
        vkDestroyBuffer(v->dev, *buf, NULL);
        return -1;
    }
    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = ti;
    VkResult r = vkAllocateMemory(v->dev, &ai, NULL, mem);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(v->dev, *buf, NULL);
        return -1;
    }
    VK_TRY(vkBindBufferMemory(v->dev, *buf, *mem, 0));
    VK_TRY(vkMapMemory(v->dev, *mem, 0, VK_WHOLE_SIZE, 0, map));
    return 0;
}

/* Weight buffers live in DEVICE_LOCAL memory without HOST_VISIBLE. On this
 * UMA iGPU that is the 101 GB/s path; the host-visible type 2 path is for the
 * small activation scratch only. */
static int pv_buffer_dev(DSV4PrefillVk *v, VkDeviceSize size,
                         VkBuffer *buf, VkDeviceMemory *mem)
{
    VkBufferCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_TRY(vkCreateBuffer(v->dev, &ci, NULL, buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(v->dev, *buf, &req);
    uint32_t ti = UINT32_MAX;
    for (uint32_t i = 0; i < v->mp.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = v->mp.memoryTypes[i].propertyFlags;
        if ((req.memoryTypeBits & (1u << i)) &&
            (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            ti = i; break;
        }
    }
    if (ti == UINT32_MAX)
        ti = pv_mem_type(&v->mp, req.memoryTypeBits);
    if (ti == UINT32_MAX) {
        vkDestroyBuffer(v->dev, *buf, NULL);
        return -1;
    }
    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = ti;
    VkResult r = vkAllocateMemory(v->dev, &ai, NULL, mem);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(v->dev, *buf, NULL);
        return -1;
    }
    VK_TRY(vkBindBufferMemory(v->dev, *buf, *mem, 0));
    return 0;
}

static void pv_layer_free(DSV4PrefillVk *v, PVLayer *l)
{
    if (l->buf) vkDestroyBuffer(v->dev, l->buf, NULL);
    if (l->mem) vkFreeMemory(v->dev, l->mem, NULL);
    memset(l, 0, sizeof *l);
}

DSV4PrefillVk *dsv4_pv_init(void)
{
    DSV4PrefillVk *v = (DSV4PrefillVk *)calloc(1, sizeof *v);
    if (!v) return NULL;

    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dsv4-prefill";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, NULL, &v->instance) != VK_SUCCESS) goto fail;

    uint32_t ndev = 0;
    if (vkEnumeratePhysicalDevices(v->instance, &ndev, NULL) != VK_SUCCESS ||
        ndev == 0)
        goto fail;
    VkPhysicalDevice *devs = (VkPhysicalDevice *)calloc(ndev, sizeof *devs);
    if (!devs) goto fail;
    vkEnumeratePhysicalDevices(v->instance, &ndev, devs);
    for (uint32_t d = 0; d < ndev && !v->phys; d++) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, NULL);
        VkQueueFamilyProperties *qf =
            (VkQueueFamilyProperties *)calloc(nq ? nq : 1, sizeof *qf);
        if (!qf) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, qf);
        for (uint32_t q = 0; q < nq; q++)
            if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                v->phys = devs[d];
                v->qfam = q;
                break;
            }
        free(qf);
    }
    free(devs);
    if (!v->phys) goto fail;

    VkPhysicalDeviceFeatures feats;
    vkGetPhysicalDeviceFeatures(v->phys, &feats);
    if (!feats.shaderFloat64) {
        fprintf(stderr, "prefill-vk: device lacks shaderFloat64\n");
        goto fail;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(v->phys, &props);

    VkPhysicalDeviceFeatures enable = {0};
    enable.shaderFloat64 = VK_TRUE;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = v->qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &enable;
    if (vkCreateDevice(v->phys, &dci, NULL, &v->dev) != VK_SUCCESS) goto fail;
    vkGetDeviceQueue(v->dev, v->qfam, 0, &v->queue);
    vkGetPhysicalDeviceMemoryProperties(v->phys, &v->mp);

    VkCommandPoolCreateInfo pci = {0};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = v->qfam;
    VK_GO(vkCreateCommandPool(v->dev, &pci, NULL, &v->cmdpool));
    VkCommandBufferAllocateInfo cbi = {0};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = v->cmdpool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = DSV4_PV_MAX_JOBS;
    VK_GO(vkAllocateCommandBuffers(v->dev, &cbi, v->cmd));
    VkFenceCreateInfo fci = {0};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_GO(vkCreateFence(v->dev, &fci, NULL, &v->fence));

    VkDescriptorSetLayoutBinding b[4];
    for (uint32_t i = 0; i < 4; i++) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl = {0};
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 4;
    dsl.pBindings = b;
    VK_GO(vkCreateDescriptorSetLayout(v->dev, &dsl, NULL, &v->dsl));

    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 10 * sizeof(uint32_t);
    VkPipelineLayoutCreateInfo plc = {0};
    plc.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plc.setLayoutCount = 1;
    plc.pSetLayouts = &v->dsl;
    plc.pushConstantRangeCount = 1;
    plc.pPushConstantRanges = &pcr;
    VK_GO(vkCreatePipelineLayout(v->dev, &plc, NULL, &v->pl));

    VkDescriptorPoolSize dps = {0};
    dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dps.descriptorCount = 4;
    VkDescriptorPoolCreateInfo dpc = {0};
    dpc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpc.maxSets = 1;
    dpc.poolSizeCount = 1;
    dpc.pPoolSizes = &dps;
    VK_GO(vkCreateDescriptorPool(v->dev, &dpc, NULL, &v->dpool));
    VkDescriptorSetAllocateInfo dsa = {0};
    dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = v->dpool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &v->dsl;
    VK_GO(vkAllocateDescriptorSets(v->dev, &dsa, &v->dset));

    /* pCode must be 4-byte aligned and the generated array is unsigned char. */
    size_t nspv = dsv4_prefill_fp8_spv_len;
    uint32_t *spv = (uint32_t *)malloc(nspv);
    if (!spv) goto fail;
    memcpy(spv, dsv4_prefill_fp8_spv, nspv);
    VkShaderModuleCreateInfo smi = {0};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = nspv;
    smi.pCode = spv;
    VkShaderModule sm;
    VkResult sr = vkCreateShaderModule(v->dev, &smi, NULL, &sm);
    free(spv);
    if (sr != VK_SUCCESS) goto fail;
    VkComputePipelineCreateInfo cp = {0};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.layout = v->pl;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = sm;
    cp.stage.pName = "main";
    VkResult pr = vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1, &cp,
                                           NULL, &v->pipe);
    vkDestroyShaderModule(v->dev, sm, NULL);
    if (pr != VK_SUCCESS) goto fail;

    VkDeviceSize scratch = (VkDeviceSize)DSV4_PV_SCRATCH_FLOATS * 4u;
    if (pv_buffer(v, scratch, &v->xbuf, &v->xmem, &v->xmap) != 0) goto fail;
    if (pv_buffer(v, scratch, &v->ybuf, &v->ymem, &v->ymap) != 0) goto fail;
    memset(v->xmap, 0, (size_t)scratch);
    memset(v->ymap, 0, (size_t)scratch);

    v->scap = 256u << 20;   /* widest released trunk layer is 165.5 MB */
    if (pv_buffer(v, (VkDeviceSize)v->scap, &v->sbuf, &v->smem,
                  &v->smap) != 0)
        goto fail;

    printf("prefill-vk: %s, up to %u x 32768 FP8 GEMM, batch %u\n",
           props.deviceName, DSV4_PV_MAX_LAYERS, DSV4_PV_NTOK);
    return v;

fail:
    fprintf(stderr, "prefill-vk: Vulkan initialisation failed\n");
    dsv4_pv_free(v);
    return NULL;
}

void dsv4_pv_free(DSV4PrefillVk *v)
{
    if (!v) return;
    if (v->dev) vkDeviceWaitIdle(v->dev);
    for (int i = 0; i < DSV4_PV_MAX_LAYERS; i++)
        if (v->dev) pv_layer_free(v, &v->lay[i]);
    if (v->dev) {
        if (v->xbuf) vkDestroyBuffer(v->dev, v->xbuf, NULL);
        if (v->xmem) vkFreeMemory(v->dev, v->xmem, NULL);
        if (v->ybuf) vkDestroyBuffer(v->dev, v->ybuf, NULL);
        if (v->ymem) vkFreeMemory(v->dev, v->ymem, NULL);
        if (v->sbuf) vkDestroyBuffer(v->dev, v->sbuf, NULL);
        if (v->smem) vkFreeMemory(v->dev, v->smem, NULL);
        if (v->pipe) vkDestroyPipeline(v->dev, v->pipe, NULL);
        if (v->pl) vkDestroyPipelineLayout(v->dev, v->pl, NULL);
        if (v->dpool) vkDestroyDescriptorPool(v->dev, v->dpool, NULL);
        if (v->dsl) vkDestroyDescriptorSetLayout(v->dev, v->dsl, NULL);
        if (v->fence) vkDestroyFence(v->dev, v->fence, NULL);
        if (v->cmdpool) vkDestroyCommandPool(v->dev, v->cmdpool, NULL);
        vkDestroyDevice(v->dev, NULL);
    }
    if (v->instance) vkDestroyInstance(v->instance, NULL);
    free(v);
}

int dsv4_pv_prepare_layer(DSV4PrefillVk *v, int layer, const void *run,
                          size_t nbytes)
{
    if (!v || layer < 0 || layer >= DSV4_PV_MAX_LAYERS || !run || !nbytes)
        return -1;
    if (nbytes > v->scap) return -1;
    PVLayer *l = &v->lay[layer];
    if (l->buf) return 0;
    if (pv_buffer_dev(v, (VkDeviceSize)nbytes, &l->buf, &l->mem) != 0)
        return -1;
    l->nbytes = nbytes;
    l->run = run;

    memcpy(v->smap, run, nbytes);

    VkCommandBuffer cb = v->cmd[0];
    if (vkResetCommandBuffer(cb, 0) != VK_SUCCESS) return -1;
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) return -1;
    VkMemoryBarrier hb = {0};
    hb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         1, &hb, 0, NULL, 0, NULL);
    VkBufferCopy bc = {0};
    bc.size = nbytes;
    vkCmdCopyBuffer(cb, v->sbuf, l->buf, 1, &bc);
    VkBufferMemoryBarrier bb = {0};
    bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = l->buf;
    bb.offset = 0;
    bb.size = nbytes;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, NULL, 1, &bb, 0, NULL);
    if (vkEndCommandBuffer(cb) != VK_SUCCESS) return -1;

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    if (vkResetFences(v->dev, 1, &v->fence) != VK_SUCCESS) return -1;
    if (vkQueueSubmit(v->queue, 1, &si, v->fence) != VK_SUCCESS) return -1;
    if (vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE, UINT64_MAX) !=
        VK_SUCCESS)
        return -1;
    return 0;
}

int dsv4_pv_mmq_fp8_multi(DSV4PrefillVk *v, int layer,
                          const DSV4PVJob *jobs, int njobs)
{
    if (!v || !jobs || njobs < 1 || njobs > DSV4_PV_MAX_JOBS) return -1;
    if (layer < 0 || layer >= DSV4_PV_MAX_LAYERS) return -1;
    PVLayer *l = &v->lay[layer];
    if (!l->buf) return -1;

    const char *base = (const char *)l->run;
    uint32_t pc[DSV4_PV_MAX_JOBS][10];
    size_t xoff = 0, yoff = 0;
    uint64_t bytes = 0;

    for (int j = 0; j < njobs; j++) {
        const DSV4QMat *m = jobs[j].m;
        const int n = jobs[j].n;
        if (!m || !m->w || !m->s || !jobs[j].x || !jobs[j].y) return -1;
        if (m->wdt != DSV4_WFP8 || m->blk_c != 128) return -1;
        if (n < 1 || n > DSV4_PV_NTOK) return -1;
        const int in = m->cols, out = m->rows;
        if (in <= 0 || out <= 0 || in > 32768 || out > 32768) return -1;
        const size_t wb = (size_t)out * (size_t)in;
        const int sc = (in + m->blk_c - 1) / m->blk_c;
        const size_t sb = (size_t)((out + m->blk_r - 1) / m->blk_r) * (size_t)sc;
        const uint64_t w_off = (uint64_t)((const char *)m->w - base);
        const uint64_t s_off = (uint64_t)((const char *)m->s - base);
        if (w_off + wb > l->nbytes || s_off + sb > l->nbytes) return -1;
        if ((w_off & 3u) || (s_off & 3u) || (in & 3u)) return -1;

        if (xoff + (size_t)n * (size_t)in > (size_t)DSV4_PV_SCRATCH_FLOATS)
            return -1;
        if (yoff + (size_t)n * (size_t)out > (size_t)DSV4_PV_SCRATCH_FLOATS)
            return -1;

        pc[j][0] = (uint32_t)in;
        pc[j][1] = (uint32_t)out;
        pc[j][2] = (uint32_t)m->blk_r;
        pc[j][3] = (uint32_t)m->blk_c;
        pc[j][4] = (uint32_t)sc;
        pc[j][5] = (uint32_t)w_off;
        pc[j][6] = (uint32_t)s_off;
        pc[j][7] = (uint32_t)in;
        pc[j][8] = (uint32_t)xoff;
        pc[j][9] = (uint32_t)yoff;
        if (getenv("DSV4_PREFILL_CHECK") && j < 8)
            fprintf(stderr, "vk-job %d %dx%d n=%d w_off=%llu s_off=%llu "
                            "x_off=%zu y_off=%zu\n", j, out, in, n,
                    (unsigned long long)w_off, (unsigned long long)s_off,
                    xoff, yoff);
        bytes += (uint64_t)wb + sb;
        xoff += (size_t)n * (size_t)in;
        yoff += (size_t)n * (size_t)out;
    }

    for (int j = 0; j < njobs; j++) {
        const DSV4QMat *m = jobs[j].m;
        const int n = jobs[j].n;
        memcpy((char *)v->xmap + (size_t)pc[j][8] * sizeof(float),
               jobs[j].x, (size_t)n * (size_t)m->cols * sizeof(float));
    }

    VkDescriptorBufferInfo dbi[4] = {
        {l->buf, 0, VK_WHOLE_SIZE},
        {l->buf, 0, VK_WHOLE_SIZE},
        {v->xbuf, 0, VK_WHOLE_SIZE},
        {v->ybuf, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet wr[4];
    for (uint32_t i = 0; i < 4; i++) {
        memset(&wr[i], 0, sizeof wr[i]);
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = v->dset;
        wr[i].dstBinding = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(v->dev, 4, wr, 0, NULL);

    VkCommandBuffer cb = v->cmd[0];
    if (vkResetCommandBuffer(cb, 0) != VK_SUCCESS) return -1;
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) return -1;

    VkMemoryBarrier hb = {0};
    hb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &hb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl,
                            0, 1, &v->dset, 0, NULL);
    for (int j = 0; j < njobs; j++) {
        vkCmdPushConstants(cb, v->pl, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof pc[j], pc[j]);
        vkCmdDispatch(cb, (uint32_t)((jobs[j].m->rows + 7) / 8), 1, 1);
    }
    VkMemoryBarrier fb = {0};
    fb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    fb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    fb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0,
                         1, &fb, 0, NULL, 0, NULL);
    if (vkEndCommandBuffer(cb) != VK_SUCCESS) return -1;

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    double t0 = pv_now();
    VkResult r = vkResetFences(v->dev, 1, &v->fence);
    if (r == VK_SUCCESS) r = vkQueueSubmit(v->queue, 1, &si, v->fence);
    if (r == VK_SUCCESS)
        r = vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "prefill-vk: submit failed: %d\n", (int)r);
        return -1;
    }
    v->secs += pv_now() - t0;
    v->calls += (uint64_t)njobs;
    v->bytes += bytes;

    for (int j = 0; j < njobs; j++) {
        const DSV4QMat *m = jobs[j].m;
        const int n = jobs[j].n;
        memcpy(jobs[j].y,
               (const char *)v->ymap + (size_t)pc[j][9] * sizeof(float),
               (size_t)n * (size_t)m->rows * sizeof(float));
    }
    return 0;
}

int dsv4_pv_mmq_fp8(DSV4PrefillVk *v, int layer, const DSV4QMat *m,
                    const float *x, float *y, int n)
{
    DSV4PVJob job = { m, x, y, n };
    return dsv4_pv_mmq_fp8_multi(v, layer, &job, 1);
}

void dsv4_pv_report(const DSV4PrefillVk *v, const char *label)
{
    if (!v) return;
    printf("prefill-vk %s: %llu GEMMs, %.2f GB weights, %.2f s\n",
           label, (unsigned long long)v->calls,
           (double)v->bytes / 1073741824.0, v->secs);
}
