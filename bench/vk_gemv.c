/* SPDX-License-Identifier: Apache-2.0 */
/* bench/vk_gemv.c - standalone Vulkan GEMV bench against the CPU kernels.
 *
 * WHY THIS EXISTS
 *   Before wiring a Vulkan backend into dsv4_mmq, answer two questions on the
 *   real Flash geometries: how fast is the iGPU, and what does a dispatch cost
 *   end to end?  This program allocates the matrices, imports the host pages
 *   into Vulkan (or copies them, with --copy), dispatches one compute shader,
 *   waits on a fence, and reports both the timestamped kernel time and the
 *   host wall time.  The CPU path is the same dsv4_matmul_fp4 / fp8 kernel the
 *   engine uses, so the comparison is against the shipped code, not a scalar
 *   stand-in.
 *
 * CORRECTNESS
 *   The shader accumulates each 32- or 128-wide block in float and folds the
 *   block scale in double, matching the CPU hybrid.  A GPU cannot reproduce the
 *   CPU's reduction tree bit for bit, so this gates on the same 1e-6 L2 bound
 *   as the CPU FP4/FP8 gates and prints the measured value.
 *
 * USAGE
 *   ./bin/vk_gemv [reps] [--copy] [--spv-dir DIR]
 *   OMP_NUM_THREADS=4 taskset -c 0-3 ./bin/vk_gemv 30
 */
#define _GNU_SOURCE

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vulkan/vulkan.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dsv4.h"

/* The shipped CPU kernels. */
void dsv4_matmul_fp8(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
void dsv4_matmul_fp4(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);

#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult r_ = (x);                                                     \
        if (r_ != VK_SUCCESS) {                                                \
            fprintf(stderr, "vk_gemv: %s failed: %d\n", #x, (int)r_);          \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

typedef struct {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    void          *map;
    VkDeviceSize   size;
    uint32_t       payload_off;  /* byte offset of the matrix in the buffer */
    int            imported;
} GBuf;

typedef struct {
    VkInstance       instance;
    VkPhysicalDevice phys;
    VkDevice         dev;
    VkQueue          queue;
    uint32_t         qfam;
    VkCommandPool    cmdpool;
    VkCommandBuffer  cmd;
    VkFence          fence;
    VkQueryPool      qpool;
    int              has_ts;
    double           ts_period_ns;
    VkPhysicalDeviceMemoryProperties mp;
    int              ext_host;
    uint64_t         host_align;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      pl;
    VkDescriptorPool      dpool;
    VkDescriptorSet       dset;
    VkPipeline            pipe_fp4, pipe_fp8;
} VkCtx;

typedef struct {
    uint32_t in_n, out_n, blk_r, blk_c, sc, w_off, s_off, w_row;
} PC;

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint32_t *read_spv(const char *path, size_t *nbytes)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "vk_gemv: cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n <= 0 || (n & 3) != 0) {
        fprintf(stderr, "vk_gemv: %s is not a SPIR-V word stream\n", path);
        fclose(f);
        return NULL;
    }
    rewind(f);
    uint32_t *p = (uint32_t *)malloc((size_t)n);
    if (!p) { fclose(f); return NULL; }
    if (fread(p, 1, (size_t)n, f) != (size_t)n) {
        free(p); fclose(f); return NULL;
    }
    fclose(f);
    *nbytes = (size_t)n;
    return p;
}

static uint32_t find_mem(const VkCtx *c, uint32_t bits,
                         VkMemoryPropertyFlags want, int prefer_dev)
{
    if (prefer_dev) {
        for (uint32_t i = 0; i < c->mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) &&
                (c->mp.memoryTypes[i].propertyFlags & (want | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) ==
                    (want | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                return i;
    }
    for (uint32_t i = 0; i < c->mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) &&
            (c->mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

static int has_dev_ext(VkPhysicalDevice phys, const char *name)
{
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(phys, NULL, &n, NULL);
    VkExtensionProperties *p = (VkExtensionProperties *)calloc(n ? n : 1, sizeof *p);
    if (!p) return 0;
    vkEnumerateDeviceExtensionProperties(phys, NULL, &n, p);
    int found = 0;
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(p[i].extensionName, name)) { found = 1; break; }
    free(p);
    return found;
}

static void vk_init(VkCtx *c, const char *spv_fp4, const char *spv_fp8)
{
    memset(c, 0, sizeof *c);

    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dsv4-vk-gemv";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, NULL, &c->instance));

    uint32_t ndev = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(c->instance, &ndev, NULL));
    if (ndev == 0) {
        fprintf(stderr, "vk_gemv: no Vulkan physical devices\n");
        exit(1);
    }
    VkPhysicalDevice *devs = (VkPhysicalDevice *)calloc(ndev, sizeof *devs);
    VK_CHECK(vkEnumeratePhysicalDevices(c->instance, &ndev, devs));
    for (uint32_t d = 0; d < ndev; d++) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, NULL);
        VkQueueFamilyProperties *qf = (VkQueueFamilyProperties *)calloc(nq ? nq : 1, sizeof *qf);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, qf);
        uint32_t pick = UINT32_MAX;
        for (uint32_t q = 0; q < nq; q++)
            if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) { pick = q; break; }
        if (pick != UINT32_MAX) {
            c->phys = devs[d];
            c->qfam = pick;
            c->has_ts = qf[pick].timestampValidBits != 0;
            free(qf);
            break;
        }
        free(qf);
    }
    free(devs);
    if (c->phys == VK_NULL_HANDLE) {
        fprintf(stderr, "vk_gemv: no Vulkan device with a compute queue\n");
        exit(1);
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(c->phys, &props);
    VkPhysicalDeviceFeatures feats;
    vkGetPhysicalDeviceFeatures(c->phys, &feats);
    if (!feats.shaderFloat64) {
        fprintf(stderr, "vk_gemv: device does not support shaderFloat64\n");
        exit(1);
    }

    c->ext_host = has_dev_ext(c->phys, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    if (c->ext_host) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = {0};
        hp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 p2 = {0};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &hp;
        vkGetPhysicalDeviceProperties2(c->phys, &p2);
        c->host_align = hp.minImportedHostPointerAlignment;
        if (c->host_align == 0) c->host_align = 4096;
    }

    VkPhysicalDeviceFeatures enable = {0};
    enable.shaderFloat64 = VK_TRUE;
    const char *dext[1];
    uint32_t nde = 0;
    if (c->ext_host) dext[nde++] = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = c->qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nde;
    dci.ppEnabledExtensionNames = dext;
    dci.pEnabledFeatures = &enable;
    VK_CHECK(vkCreateDevice(c->phys, &dci, NULL, &c->dev));
    vkGetDeviceQueue(c->dev, c->qfam, 0, &c->queue);
    vkGetPhysicalDeviceMemoryProperties(c->phys, &c->mp);

    VkCommandPoolCreateInfo pci = {0};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = c->qfam;
    VK_CHECK(vkCreateCommandPool(c->dev, &pci, NULL, &c->cmdpool));
    VkCommandBufferAllocateInfo cbi = {0};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = c->cmdpool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(c->dev, &cbi, &c->cmd));
    VkFenceCreateInfo fci = {0};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(c->dev, &fci, NULL, &c->fence));

    if (c->has_ts) {
        VkQueryPoolCreateInfo qp = {0};
        qp.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(c->dev, &qp, NULL, &c->qpool));
        c->ts_period_ns = (double)props.limits.timestampPeriod;
    }

    VkDescriptorSetLayoutBinding b[4];
    for (uint32_t i = 0; i < 4; i++) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo dsl = {0};
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 4;
    dsl.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(c->dev, &dsl, NULL, &c->dsl));

    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PC);
    VkPipelineLayoutCreateInfo plc = {0};
    plc.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plc.setLayoutCount = 1;
    plc.pSetLayouts = &c->dsl;
    plc.pushConstantRangeCount = 1;
    plc.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(c->dev, &plc, NULL, &c->pl));

    VkDescriptorPoolSize dps = {0};
    dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dps.descriptorCount = 4;
    VkDescriptorPoolCreateInfo dpc = {0};
    dpc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpc.maxSets = 1;
    dpc.poolSizeCount = 1;
    dpc.pPoolSizes = &dps;
    VK_CHECK(vkCreateDescriptorPool(c->dev, &dpc, NULL, &c->dpool));
    VkDescriptorSetAllocateInfo dsa = {0};
    dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = c->dpool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &c->dsl;
    VK_CHECK(vkAllocateDescriptorSets(c->dev, &dsa, &c->dset));

    size_t n4 = 0, n8 = 0;
    uint32_t *spv4 = read_spv(spv_fp4, &n4);
    uint32_t *spv8 = read_spv(spv_fp8, &n8);
    if (!spv4 || !spv8) {
        fprintf(stderr, "vk_gemv: shaders missing (compile with make bin/vk_gemv)\n");
        exit(1);
    }
    VkShaderModuleCreateInfo smi = {0};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = n4;
    smi.pCode = spv4;
    VkShaderModule m4;
    VK_CHECK(vkCreateShaderModule(c->dev, &smi, NULL, &m4));
    smi.codeSize = n8;
    smi.pCode = spv8;
    VkShaderModule m8;
    VK_CHECK(vkCreateShaderModule(c->dev, &smi, NULL, &m8));

    VkComputePipelineCreateInfo cp = {0};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.layout = c->pl;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.pName = "main";
    cp.stage.module = m4;
    VK_CHECK(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cp, NULL, &c->pipe_fp4));
    cp.stage.module = m8;
    VK_CHECK(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cp, NULL, &c->pipe_fp8));
    free(spv4);
    free(spv8);

    printf("device        : %s\n", props.deviceName);
    printf("api           : %u.%u.%u\n",
           VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion));
    printf("import host   : %s", c->ext_host ? "yes" : "no");
    if (c->ext_host) printf(" (alignment %llu)", (unsigned long long)c->host_align);
    printf("\n");
    printf("timestamps    : %s\n", c->has_ts ? "yes" : "no");
}

static int make_plain(VkCtx *c, VkDeviceSize size, GBuf *b)
{
    memset(b, 0, sizeof *b);
    VkBufferCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(c->dev, &ci, NULL, &b->buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->dev, b->buffer, &req);
    uint32_t ti = find_mem(c, req.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                           VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 1);
    if (ti == UINT32_MAX)
        ti = find_mem(c, req.memoryTypeBits,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);
    if (ti == UINT32_MAX)
        ti = find_mem(c, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 1);
    if (ti == UINT32_MAX) {
        fprintf(stderr, "vk_gemv: no host-visible memory type\n");
        return -1;
    }
    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = ti;
    VK_CHECK(vkAllocateMemory(c->dev, &ai, NULL, &b->memory));
    VK_CHECK(vkBindBufferMemory(c->dev, b->buffer, b->memory, 0));
    VK_CHECK(vkMapMemory(c->dev, b->memory, 0, VK_WHOLE_SIZE, 0, &b->map));
    b->size = size;
    b->payload_off = 0;
    b->imported = 0;
    return 0;
}

/* Import host pages with VK_EXT_external_memory_host.  `host_size` is the full
 * allocation, so the alignment round-down is checked against a real range. */
static int make_import(VkCtx *c, void *ptr, VkDeviceSize size,
                       VkDeviceSize host_size, GBuf *b)
{
    memset(b, 0, sizeof *b);
    if (!c->ext_host) return -1;
    uint64_t align = c->host_align ? c->host_align : 4096;
    uintptr_t base = (uintptr_t)ptr & ~(uintptr_t)(align - 1);
    uint64_t off = (uint64_t)((uintptr_t)ptr - base);

    VkPhysicalDeviceExternalBufferInfo ebi = {0};
    ebi.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    ebi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    ebi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkExternalBufferProperties ebp = {0};
    ebp.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(c->phys, &ebi, &ebp);
    if (!(ebp.externalMemoryProperties.externalMemoryFeatures &
          VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT))
        return -1;

    VkExternalMemoryBufferCreateInfo ext = {0};
    ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.pNext = &ext;
    ci.size = size;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(c->dev, &ci, NULL, &b->buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->dev, b->buffer, &req);
    if (off + req.size > host_size) {
        vkDestroyBuffer(c->dev, b->buffer, NULL);
        b->buffer = VK_NULL_HANDLE;
        return -1;
    }
    uint32_t ti = find_mem(c, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
    if (ti == UINT32_MAX) {
        vkDestroyBuffer(c->dev, b->buffer, NULL);
        b->buffer = VK_NULL_HANDLE;
        return -1;
    }
    VkImportMemoryHostPointerInfoEXT imp = {0};
    imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = (void *)base;
    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &imp;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = ti;
    VkResult r = vkAllocateMemory(c->dev, &ai, NULL, &b->memory);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(c->dev, b->buffer, NULL);
        b->buffer = VK_NULL_HANDLE;
        return -1;
    }
    VK_CHECK(vkBindBufferMemory(c->dev, b->buffer, b->memory, 0));
    b->size = size;
    b->payload_off = (uint32_t)off;
    b->imported = 1;
    return 0;
}

static void free_gbuf(VkCtx *c, GBuf *b)
{
    if (b->buffer) vkDestroyBuffer(c->dev, b->buffer, NULL);
    if (b->memory) vkFreeMemory(c->dev, b->memory, NULL);
    memset(b, 0, sizeof *b);
}

static void record(VkCtx *c, VkPipeline pipe, const PC *pc, uint32_t groups,
                   uint32_t batch, uint32_t w_stride, uint32_t s_stride)
{
    VK_CHECK(vkResetCommandBuffer(c->cmd, 0));
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(c->cmd, &bi));
    if (c->has_ts) vkCmdResetQueryPool(c->cmd, c->qpool, 0, 2);

    VkMemoryBarrier hb = {0};
    hb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(c->cmd, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &hb, 0, NULL, 0, NULL);

    if (c->has_ts)
        vkCmdWriteTimestamp(c->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            c->qpool, 0);

    vkCmdBindPipeline(c->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(c->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c->pl,
                            0, 1, &c->dset, 0, NULL);
    for (uint32_t r = 0; r < batch; r++) {
        PC one = *pc;
        one.w_off += r * w_stride;
        one.s_off += r * s_stride;
        vkCmdPushConstants(c->cmd, c->pl, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof one, &one);
        vkCmdDispatch(c->cmd, groups, 1, 1);
    }

    if (c->has_ts)
        vkCmdWriteTimestamp(c->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            c->qpool, 1);

    VkMemoryBarrier fb = {0};
    fb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    fb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    fb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(c->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0,
                         1, &fb, 0, NULL, 0, NULL);
    VK_CHECK(vkEndCommandBuffer(c->cmd));
}

typedef struct { double host, kern; } Times;

static Times run_gpu(VkCtx *c, int reps, uint32_t batch)
{
    for (int i = 0; i < 5; i++) {
        VK_CHECK(vkResetFences(c->dev, 1, &c->fence));
        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c->cmd;
        VK_CHECK(vkQueueSubmit(c->queue, 1, &si, c->fence));
        VK_CHECK(vkWaitForFences(c->dev, 1, &c->fence, VK_TRUE, UINT64_MAX));
    }

    Times best = {1e30, 1e30};
    for (int i = 0; i < reps; i++) {
        VK_CHECK(vkResetFences(c->dev, 1, &c->fence));
        double t0 = now_s();
        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c->cmd;
        VK_CHECK(vkQueueSubmit(c->queue, 1, &si, c->fence));
        VK_CHECK(vkWaitForFences(c->dev, 1, &c->fence, VK_TRUE, UINT64_MAX));
        double t1 = now_s();
        double dt = t1 - t0;
        if (dt < best.host) best.host = dt;

        if (c->has_ts) {
            uint64_t ts[2] = {0, 0};
            VkResult r = vkGetQueryPoolResults(c->dev, c->qpool, 0, 2,
                                                sizeof ts, ts, sizeof ts[0],
                                                VK_QUERY_RESULT_64_BIT |
                                                VK_QUERY_RESULT_WAIT_BIT);
            if (r == VK_SUCCESS) {
                double k = (double)(ts[1] - ts[0]) * c->ts_period_ns * 1e-9;
                k /= (double)batch;
                if (k > 0.0 && k < best.kern) best.kern = k;
            }
        }
    }
    best.host /= (double)batch;
    if (best.kern > 1e29) best.kern = best.host;
    return best;
}

static double run_cpu(int reps, int fp4, float *y, const float *x,
                      const uint8_t *w, const uint8_t *s,
                      int in, int out, int blk_r, int blk_c)
{
    double best = 1e30;
    for (int i = 0; i < reps; i++) {
        double t0 = now_s();
        if (fp4) dsv4_matmul_fp4(y, x, w, s, in, out, blk_r, blk_c);
        else     dsv4_matmul_fp8(y, x, w, s, in, out, blk_r, blk_c);
        double t1 = now_s();
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

typedef struct {
    const char *name;
    int in, out, blk_r, blk_c, fp4;
} Job;

static unsigned rng = 987654321u;
static unsigned nextr(void) { rng = rng * 1103515245u + 12345u; return rng; }

static int bench_job(VkCtx *c, const Job *j, int reps, int try_import,
                     int normals, uint32_t batch, const char *spv_dir)
{
    (void)spv_dir;
    const int in = j->in, out = j->out;
    const size_t wbytes = j->fp4 ? (size_t)out * in / 2 : (size_t)out * in;
    const int sc = (in + j->blk_c - 1) / j->blk_c;
    const size_t sbytes = (size_t)((out + j->blk_r - 1) / j->blk_r) * (size_t)sc;
    const size_t align = 65536;
    const size_t slop = 4 * align;

    float *x = NULL, *ycpu = NULL;
    uint8_t *w = NULL, *s = NULL;
    if (posix_memalign((void **)&x, align, (size_t)in * sizeof *x) != 0) return -1;
    if (posix_memalign((void **)&ycpu, align, (size_t)out * sizeof *ycpu) != 0) return -1;
    if (posix_memalign((void **)&w, align, wbytes + slop) != 0) return -1;
    if (posix_memalign((void **)&s, align, sbytes + slop) != 0) return -1;

    for (int i = 0; i < in; i++)
        x[i] = (float)((int)(nextr() >> 16 & 0xffff) - 32768) * 1e-3f;
    for (size_t i = 0; i < wbytes; i++) {
        uint8_t b = (uint8_t)(nextr() >> 16);
        if (!j->fp4) {
            if ((b & 0x7Fu) == 0x7Fu) b &= 0xFEu;          /* no E4M3 NaNs */
            if (normals && (b & 0x78u) == 0u) b |= 0x08u;  /* no subnormals */
        }
        w[i] = b;
    }
    for (size_t i = 0; i < sbytes; i++)
        s[i] = (uint8_t)(120u + (nextr() & 15u));          /* 2^-7 .. 2^8 */

    /* CPU reference and its timing. */
    double cpu = run_cpu(reps, j->fp4, ycpu, x, w, s, in, out, j->blk_r, j->blk_c);

    GBuf bw = {0}, bs = {0}, bx = {0}, by = {0};
    int imported = 0;
    /* VK_EXT_external_memory_host imports 4 MB happily on this ANV build but a
     * 33.5 MB import has produced VK_ERROR_DEVICE_LOST on the first submit.
     * Keep import for the expert-sized matrices (where it is the shape the
     * engine would use) and copy the 33 MB dense ones until that is understood.
     * The copy happens once, outside the timed reps. */
    int big = wbytes > (4u << 20);
    if (try_import && batch == 1 && !big &&
        make_import(c, w, wbytes, wbytes + slop, &bw) == 0 &&
        make_import(c, s, sbytes, sbytes + slop, &bs) == 0) {
        imported = 1;
    } else {
        free_gbuf(c, &bw);
        free_gbuf(c, &bs);
        if (make_plain(c, (VkDeviceSize)wbytes * batch, &bw) != 0) return -1;
        if (make_plain(c, (VkDeviceSize)sbytes * batch, &bs) != 0) return -1;
        for (uint32_t r = 0; r < batch; r++) {
            memcpy((char *)bw.map + (size_t)r * wbytes, w, wbytes);
            memcpy((char *)bs.map + (size_t)r * sbytes, s, sbytes);
        }
        bw.payload_off = 0;
        bs.payload_off = 0;
    }
    if (make_plain(c, (VkDeviceSize)in * sizeof(float), &bx) != 0) return -1;
    if (make_plain(c, (VkDeviceSize)out * sizeof(float), &by) != 0) return -1;
    memcpy(bx.map, x, (size_t)in * sizeof *x);
    memset(by.map, 0, (size_t)out * sizeof(float));

    VkDescriptorBufferInfo dbi[4] = {
        {bw.buffer, 0, VK_WHOLE_SIZE},
        {bs.buffer, 0, VK_WHOLE_SIZE},
        {bx.buffer, 0, VK_WHOLE_SIZE},
        {by.buffer, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet wr[4];
    for (uint32_t i = 0; i < 4; i++) {
        memset(&wr[i], 0, sizeof wr[i]);
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = c->dset;
        wr[i].dstBinding = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(c->dev, 4, wr, 0, NULL);

    PC pc = {0};
    pc.in_n = (uint32_t)in;
    pc.out_n = (uint32_t)out;
    pc.blk_r = (uint32_t)j->blk_r;
    pc.blk_c = (uint32_t)j->blk_c;
    pc.sc = (uint32_t)sc;
    pc.w_off = bw.payload_off;
    pc.s_off = bs.payload_off;
    pc.w_row = (uint32_t)(j->fp4 ? in / 2 : in);

    if ((pc.w_off & 3u) || (pc.s_off & 3u) || (pc.w_row & 3u)) {
        fprintf(stderr, "vk_gemv: %s: offsets not 4-byte aligned (%u %u %u)\n",
                j->name, pc.w_off, pc.s_off, pc.w_row);
        return -1;
    }

    /* The shader packs 32 lanes into rows; when a row has fewer than 32
     * 128-wide blocks (in = 1024 -> 8), put several rows in one lane group
     * so the idle lanes do not cost 4x.  Must match the shader's rule. */
    const int nblk = (in + j->blk_c - 1) / j->blk_c;
    const int lpr  = nblk >= 32 ? 32 : (nblk >= 16 ? 16 : 8);
    const int rpg  = 32 / lpr;
    const int rows_wg = 8 * rpg;
    const uint32_t groups = (uint32_t)((out + rows_wg - 1) / rows_wg);
    record(c, j->fp4 ? c->pipe_fp4 : c->pipe_fp8, &pc, groups, batch,
           (uint32_t)wbytes, (uint32_t)sbytes);
    Times t = run_gpu(c, reps, batch);

    const float *yg = (const float *)by.map;
    double err2 = 0.0, ref2 = 0.0, maxabs = 0.0;
    for (int i = 0; i < out; i++) {
        double d = (double)yg[i] - (double)ycpu[i];
        if (fabs(d) > maxabs) maxabs = fabs(d);
        err2 += d * d;
        ref2 += (double)ycpu[i] * (double)ycpu[i];
    }
    double rel = sqrt(err2 / (ref2 > 0.0 ? ref2 : 1.0));
    int bad = !(rel <= 1e-6);

    const double bytes = (double)wbytes + (double)sbytes +
                         (double)in * 4.0 + (double)out * 4.0;
    const double flop = 2.0 * (double)in * (double)out;
    printf("%-24s %8.3f %8.3f %8.3f %7.2fx %7.2fx %7.2f %7.1f %8.1e%s%s\n",
           j->name,
           cpu * 1e3, t.kern * 1e3, t.host * 1e3,
           cpu / (t.host > 0 ? t.host : 1e-30),
           cpu / (t.kern > 0 ? t.kern : 1e-30),
           bytes / (t.kern > 0 ? t.kern : 1e-30) * 1e-9,
           flop / (t.kern > 0 ? t.kern : 1e-30) * 1e-9,
           rel,
           bad ? "  FAIL" : "",
           imported ? "" : " [copy]");
    fflush(stdout);

    free_gbuf(c, &bw);
    free_gbuf(c, &bs);
    free_gbuf(c, &bx);
    free_gbuf(c, &by);
    free(x); free(ycpu); free(w); free(s);
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    int reps = 20;
    int try_import = 0;
    int normals = 0;
    int only = -1;
    uint32_t batch = 1;
    const char *spv_dir = "build";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--copy")) try_import = 0;
        else if (!strcmp(argv[i], "--import")) try_import = 1;
        else if (!strcmp(argv[i], "--normals")) normals = 1;
        else if (!strcmp(argv[i], "--spv-dir") && i + 1 < argc) spv_dir = argv[++i];
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = (uint32_t)atoi(argv[++i]);
        else if (argv[i][0] != '-') reps = atoi(argv[i]);
        else {
            fprintf(stderr, "usage: %s [reps] [--copy|--import] [--normals] "
                            "[--only N] [--batch N] [--spv-dir DIR]\n", argv[0]);
            return 1;
        }
    }
    if (reps < 1) reps = 1;
    if (batch < 1) batch = 1;

#ifdef _OPENMP
    if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(4);
    printf("CPU threads   : %d\n", omp_get_max_threads());
#else
    printf("CPU threads   : 1 (no OpenMP)\n");
#endif

    char p4[1024], p8[1024];
    snprintf(p4, sizeof p4, "%s/vk_gemv_fp4.spv", spv_dir);
    snprintf(p8, sizeof p8, "%s/vk_gemv_fp8.spv", spv_dir);

    VkCtx c;
    vk_init(&c, p4, p8);

    static const Job jobs[] = {
        { "fp4 expert w1 2048x4096", 4096, 2048, 1, 32, 1 },
        { "fp4 expert w2 4096x2048", 2048, 4096, 1, 32, 1 },
        { "fp8 attn wo_b 4096x8192", 8192, 4096, 128, 128, 0 },
        { "fp8 attn wq_b 32768x1024", 1024, 32768, 128, 128, 0 },
        { "fp8 shared w1 2048x4096", 4096, 2048, 128, 128, 0 },
    };

    printf("reps          : %d  (batch %u, %s%s)\n", reps, batch,
           try_import ? "import if possible" : "copy",
           normals ? ", normal-only FP8" : "");
    printf("%-24s %8s %8s %8s %7s %7s %7s %7s %8s\n",
           "kernel", "CPU ms", "GPU ms", "wall ms", "wallx", "kernx",
           "GB/s", "GF/s", "L2");
    int failed = 0;
    for (unsigned i = 0; i < sizeof jobs / sizeof jobs[0]; i++) {
        if (only >= 0 && (unsigned)only != i) continue;
        if (bench_job(&c, &jobs[i], reps, try_import, normals, batch, spv_dir) != 0)
            failed = 1;
    }

    vkDeviceWaitIdle(c.dev);
    vkDestroyQueryPool(c.dev, c.qpool, NULL);
    vkDestroyFence(c.dev, c.fence, NULL);
    vkDestroyCommandPool(c.dev, c.cmdpool, NULL);
    vkDestroyPipeline(c.dev, c.pipe_fp4, NULL);
    vkDestroyPipeline(c.dev, c.pipe_fp8, NULL);
    vkDestroyPipelineLayout(c.dev, c.pl, NULL);
    vkDestroyDescriptorPool(c.dev, c.dpool, NULL);
    vkDestroyDescriptorSetLayout(c.dev, c.dsl, NULL);
    vkDestroyDevice(c.dev, NULL);
    vkDestroyInstance(c.instance, NULL);
    return failed ? 1 : 0;
}
