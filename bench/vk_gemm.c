/* SPDX-License-Identifier: Apache-2.0 */
/* bench/vk_gemm.c - iGPU microbenchmark for the DSV4 FP8 and BF16 GEMMs.
 *
 *   Y[N, M] = X[N, K] * W[M, K]^T
 *
 * FP8 weights are E4M3 with E8M0 scales on 128x128 blocks, exactly the cache
 * layout. BF16 weights are row-major. The shaders are deliberately simple
 * tiled kernels (64x16 output tile, BK=64, four rows per thread); they measure
 * what the iGPU can do on the prefill shapes, not the final production tiling.
 *
 * The CPU reference is dsv4_mmq_n on the same bytes, so the reported GF/s is
 * directly comparable to ./bin/bench's wq_a_nt54 / wo_b_nt54 rows.
 */
#define _GNU_SOURCE

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vulkan/vulkan.h>

#include "bench.h"
#include "dsv4.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"

#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult r_ = (x);                                                     \
        if (r_ != VK_SUCCESS) {                                                \
            fprintf(stderr, "vk_gemm: %s failed: %d\n", #x, (int)r_);          \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    void          *map;
    VkDeviceSize   size;
    void          *raw;     /* posix_memalign pages for the import path */
} GBuf;

typedef struct {
    VkInstance       inst;
    VkPhysicalDevice phys;
    VkDevice         dev;
    VkQueue          queue;
    uint32_t         qfam;
    VkPhysicalDeviceMemoryProperties mp;
    VkCommandPool    pool;
    VkCommandBuffer  cmd;
    VkFence          fence;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      pl;
    VkDescriptorPool      dpool;
    VkPipeline            pipe_fp8, pipe_bf16, pipe_fp4;
    PFN_vkGetMemoryHostPointerPropertiesEXT get_host_props;
    VkDeviceSize          import_align;
    char                  devname[256];
} Vk;

static int g_import;

static double now_s(void) { return bench_now_s(); }

static uint32_t pick_mem_type(Vk *v, uint32_t bits)
{
    const VkMemoryPropertyFlags want =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
      | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    for (int pass = 0; pass < 2; pass++) {
        const VkMemoryPropertyFlags w =
            pass == 0
            ? (want | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            : want;
        for (uint32_t i = 0; i < v->mp.memoryTypeCount; i++) {
            if (!(bits & (1u << i))) continue;
            if ((v->mp.memoryTypes[i].propertyFlags & w) == w) return i;
        }
    }
    fprintf(stderr, "vk_gemm: no host-visible memory type\n");
    exit(1);
}

static void make_buf(Vk *v, GBuf *b, VkDeviceSize size)
{
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size ? size : 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(v->dev, &bi, NULL, &b->buf));

    VkMemoryRequirements rq;
    vkGetBufferMemoryRequirements(v->dev, b->buf, &rq);

    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof ai);
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = rq.size;
    ai.memoryTypeIndex = pick_mem_type(v, rq.memoryTypeBits);
    VK_CHECK(vkAllocateMemory(v->dev, &ai, NULL, &b->mem));
    VK_CHECK(vkBindBufferMemory(v->dev, b->buf, b->mem, 0));
    VK_CHECK(vkMapMemory(v->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map));
    b->size = size ? size : 4;
}

static void make_buf_import(Vk *v, GBuf *b, VkDeviceSize size)
{
    if (!v->get_host_props) {
        fprintf(stderr, "vk_gemm: host import unavailable\n");
        exit(1);
    }
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size ? size : 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(v->dev, &bi, NULL, &b->buf));

    VkMemoryRequirements rq;
    vkGetBufferMemoryRequirements(v->dev, b->buf, &rq);

    VkDeviceSize align = v->import_align ? v->import_align : 4096u;
    if (align < 4096u) align = 4096u;
    void *raw = NULL;
    if (posix_memalign(&raw, (size_t)align,
                       (size_t)rq.size + (size_t)align) != 0)
        bench_die("host import allocation failed");
    unsigned char *base = (unsigned char *)
        (((uintptr_t)raw + (uintptr_t)align - 1u) & ~((uintptr_t)align - 1u));

    VkMemoryHostPointerPropertiesEXT hp;
    memset(&hp, 0, sizeof hp);
    hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    VK_CHECK(v->get_host_props(
        v->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        base, &hp));

    const uint32_t allowed = rq.memoryTypeBits & hp.memoryTypeBits;
    const uint32_t mt = pick_mem_type(v, allowed);
    if (mt == UINT32_MAX) {
        fprintf(stderr, "vk_gemm: no importable memory type\n");
        exit(1);
    }

    VkImportMemoryHostPointerInfoEXT imp;
    memset(&imp, 0, sizeof imp);
    imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = base;

    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof ai);
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &imp;
    ai.allocationSize = rq.size;
    ai.memoryTypeIndex = mt;
    VK_CHECK(vkAllocateMemory(v->dev, &ai, NULL, &b->mem));
    VK_CHECK(vkBindBufferMemory(v->dev, b->buf, b->mem, 0));
    b->map = base;
    b->raw = raw;
    b->size = size ? size : 4;
}

static void free_buf(Vk *v, GBuf *b)
{
    if (!b->buf) return;
    vkUnmapMemory(v->dev, b->mem);
    vkDestroyBuffer(v->dev, b->buf, NULL);
    vkFreeMemory(v->dev, b->mem, NULL);
    free(b->raw);
    memset(b, 0, sizeof *b);
}

static uint32_t *read_spv(const char *path, size_t *nbytes)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "vk_gemm: cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    if (n <= 0 || (n & 3) != 0) {
        fprintf(stderr, "vk_gemm: %s is not SPIR-V\n", path);
        exit(1);
    }
    uint32_t *p = (uint32_t *)malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) exit(1);
    fclose(f);
    *nbytes = (size_t)n;
    return p;
}

static void make_pipeline(Vk *v, const char *spv, VkPipeline *out)
{
    size_t sn = 0;
    uint32_t *code = read_spv(spv, &sn);
    VkShaderModuleCreateInfo smi;
    memset(&smi, 0, sizeof smi);
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = sn;
    smi.pCode = code;
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(v->dev, &smi, NULL, &mod));
    free(code);

    VkComputePipelineCreateInfo cpi;
    memset(&cpi, 0, sizeof cpi);
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = v->pl;
    VK_CHECK(vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1, &cpi,
                                      NULL, out));
    vkDestroyShaderModule(v->dev, mod, NULL);
}

static int dev_ext_supported(VkPhysicalDevice phys, const char *name)
{
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, NULL, &n, NULL)
        != VK_SUCCESS) return 0;
    VkExtensionProperties *e = (VkExtensionProperties *)calloc(n, sizeof *e);
    if (!e) return 0;
    int found = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, NULL, &n, e) == VK_SUCCESS)
        for (uint32_t i = 0; i < n; i++)
            if (strcmp(e[i].extensionName, name) == 0) { found = 1; break; }
    free(e);
    return found;
}

static void vk_init(Vk *v)
{
    memset(v, 0, sizeof *v);

    VkApplicationInfo app;
    memset(&app, 0, sizeof app);
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dsv4 vk_gemm";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici;
    memset(&ici, 0, sizeof ici);
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, NULL, &v->inst));

    uint32_t np = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(v->inst, &np, NULL));
    if (!np) { fprintf(stderr, "vk_gemm: no device\n"); exit(1); }
    VkPhysicalDevice *pd = (VkPhysicalDevice *)calloc(np, sizeof *pd);
    VK_CHECK(vkEnumeratePhysicalDevices(v->inst, &np, pd));
    v->phys = pd[0];
    free(pd);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(v->phys, &props);
    snprintf(v->devname, sizeof v->devname, "%s", props.deviceName);
    vkGetPhysicalDeviceMemoryProperties(v->phys, &v->mp);

    if (g_import) {
        if (!dev_ext_supported(v->phys,
                               VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
            fprintf(stderr, "vk_gemm: device lacks host import\n");
            exit(1);
        }
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp;
        memset(&hp, 0, sizeof hp);
        hp.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 p2;
        memset(&p2, 0, sizeof p2);
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &hp;
        vkGetPhysicalDeviceProperties2(v->phys, &p2);
        v->import_align = hp.minImportedHostPointerAlignment;
        if (!v->import_align) {
            fprintf(stderr, "vk_gemm: zero import alignment\n");
            exit(1);
        }
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(v->phys, &nq, NULL);
    VkQueueFamilyProperties *qf =
        (VkQueueFamilyProperties *)calloc(nq, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(v->phys, &nq, qf);
    v->qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            v->qfam = i;
            break;
        }
    free(qf);
    if (v->qfam == UINT32_MAX) { fprintf(stderr, "vk_gemm: no queue\n"); exit(1); }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci;
    memset(&qci, 0, sizeof qci);
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = v->qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci;
    memset(&dci, 0, sizeof dci);
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    const char *dev_exts[1];
    if (g_import) {
        dev_exts[0] = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = dev_exts;
    }
    VK_CHECK(vkCreateDevice(v->phys, &dci, NULL, &v->dev));
    vkGetDeviceQueue(v->dev, v->qfam, 0, &v->queue);

    if (g_import) {
        v->get_host_props = (PFN_vkGetMemoryHostPointerPropertiesEXT)
            vkGetDeviceProcAddr(v->dev, "vkGetMemoryHostPointerPropertiesEXT");
        if (!v->get_host_props) {
            fprintf(stderr, "vk_gemm: missing host-pointer proc\n");
            exit(1);
        }
    }

    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof pci);
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = v->qfam;
    VK_CHECK(vkCreateCommandPool(v->dev, &pci, NULL, &v->pool));

    VkCommandBufferAllocateInfo cbi;
    memset(&cbi, 0, sizeof cbi);
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = v->pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(v->dev, &cbi, &v->cmd));

    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof fci);
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(v->dev, &fci, NULL, &v->fence));

    VkDescriptorSetLayoutBinding b[4];
    memset(b, 0, sizeof b);
    for (int i = 0; i < 4; i++) {
        b[i].binding = (uint32_t)i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli;
    memset(&dli, 0, sizeof dli);
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 4;
    dli.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(v->dev, &dli, NULL, &v->dsl));

    VkPushConstantRange pcr;
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 3 * sizeof(uint32_t);
    VkPipelineLayoutCreateInfo pli;
    memset(&pli, 0, sizeof pli);
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &v->dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(v->dev, &pli, NULL, &v->pl));

    VkDescriptorPoolSize psz;
    psz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psz.descriptorCount = 4u * 64u;
    VkDescriptorPoolCreateInfo dpi;
    memset(&dpi, 0, sizeof dpi);
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 64;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &psz;
    VK_CHECK(vkCreateDescriptorPool(v->dev, &dpi, NULL, &v->dpool));

    make_pipeline(v, "build/gemm_fp8.spv", &v->pipe_fp8);
    make_pipeline(v, "build/gemm_bf16.spv", &v->pipe_bf16);
    make_pipeline(v, "build/gemm_fp4.spv", &v->pipe_fp4);
}

static VkDescriptorSet bind_case(Vk *v, GBuf *w, GBuf *s, GBuf *x, GBuf *y)
{
    VkDescriptorSetAllocateInfo ai;
    memset(&ai, 0, sizeof ai);
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = v->dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &v->dsl;
    VkDescriptorSet d;
    VK_CHECK(vkAllocateDescriptorSets(v->dev, &ai, &d));

    VkDescriptorBufferInfo bi[4] = {
        { w->buf, 0, VK_WHOLE_SIZE },
        { s->buf, 0, VK_WHOLE_SIZE },
        { x->buf, 0, VK_WHOLE_SIZE },
        { y->buf, 0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet wr[4];
    memset(wr, 0, sizeof wr);
    for (int i = 0; i < 4; i++) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = d;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(v->dev, 4, wr, 0, NULL);
    return d;
}

static void dispatch_once(Vk *v, VkPipeline pipe, VkDescriptorSet d,
                          uint32_t M, uint32_t K, uint32_t N)
{
    VK_CHECK(vkResetCommandBuffer(v->cmd, 0));
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(v->cmd, &bi));
    vkCmdBindPipeline(v->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(v->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl,
                            0, 1, &d, 0, NULL);
    const uint32_t pc[3] = { M, K, N };
    vkCmdPushConstants(v->cmd, v->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof pc, pc);
    vkCmdDispatch(v->cmd, (M + 63u) / 64u, (N + 15u) / 16u, 1);
    VK_CHECK(vkEndCommandBuffer(v->cmd));

    VK_CHECK(vkResetFences(v->dev, 1, &v->fence));
    VkSubmitInfo si;
    memset(&si, 0, sizeof si);
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &v->cmd;
    VK_CHECK(vkQueueSubmit(v->queue, 1, &si, v->fence));
    VK_CHECK(vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE, UINT64_MAX));
}

static double rel_l2(const float *a, const float *b, size_t n)
{
    double e = 0.0, r = 0.0, mx = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        e += d * d;
        r += (double)b[i] * (double)b[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    fprintf(stderr, "  rel L2 %.3e max |d| %.3e\n",
            sqrt(e / (r > 0.0 ? r : 1.0)), mx);
    return sqrt(e / (r > 0.0 ? r : 1.0));
}

static void check_format(Vk *v, int fmt)
{
    const int M = 128, K = 512, N = 8;
    const int fp8 = fmt == 1, fp4 = fmt == 2;
    const int wdt = fp8 ? DSV4_WFP8 : fp4 ? DSV4_WFP4 : DSV4_WBF16;
    const VkDeviceSize wb = fp4 ? (VkDeviceSize)M * K / 2
                                : (VkDeviceSize)M * K * (fp8 ? 1 : 2);
    const VkDeviceSize sb = fp8
        ? (VkDeviceSize)((M + 127) / 128) * ((K + 127) / 128)
        : fp4 ? (VkDeviceSize)M * (K / 32) : 4;
    QFix q;
    qfix_init(&q, wdt, M, K);

    GBuf w, s, x, y;
    memset(&w, 0, sizeof w); memset(&s, 0, sizeof s);
    memset(&x, 0, sizeof x); memset(&y, 0, sizeof y);
    make_buf(v, &w, wb);
    make_buf(v, &s, sb);
    make_buf(v, &x, (VkDeviceSize)N * K * sizeof(float));
    make_buf(v, &y, (VkDeviceSize)N * M * sizeof(float));
    memcpy(w.map, q.m.w, (size_t)wb);
    if (fp8 || fp4) memcpy(s.map, q.m.s, (size_t)sb);
    else memset(s.map, 0, 4);
    float *xh = (float *)x.map;
    bench_fill_f32(xh, (size_t)N * K, 1.0f);
    memset(y.map, 0, (size_t)N * M * sizeof(float));

    VkDescriptorSet d = bind_case(v, &w, &s, &x, &y);

    if (fp8 && getenv("VK_GEMM_DEBUG_FP8")) {
        memset(w.map, 0x38, (size_t)wb);           /* E4M3 1.0 */
        memset(s.map, 127, (size_t)sb);
        for (int i = 0; i < N * K; i++) xh[i] = 1.0f;
        dispatch_once(v, v->pipe_fp8, d, M, K, N);
        fprintf(stderr, "uniform fp8: y[0]=%g y[1]=%g (expect %d)\n",
                ((float *)y.map)[0], ((float *)y.map)[1], K);
        free_buf(v, &w); free_buf(v, &s); free_buf(v, &x); free_buf(v, &y);
        qfix_free(&q);
        return;
    }

    dispatch_once(v, fp8 ? v->pipe_fp8 : fp4 ? v->pipe_fp4 : v->pipe_bf16,
                  d, M, K, N);

    float *ref = (float *)bench_xalloc((size_t)N * M * sizeof(float));
    dsv4_mmq_n(ref, xh, &q.m, N);
    fprintf(stderr, "correctness %s: ", fp8 ? "fp8" : fp4 ? "fp4" : "bf16");
    rel_l2((float *)y.map, ref, (size_t)N * M);
    fprintf(stderr, "  y[0]=%g ref[0]=%g  y[1]=%g ref[1]=%g\n",
            ((float *)y.map)[0], ref[0], ((float *)y.map)[1], ref[1]);

    free(ref);
    free_buf(v, &w); free_buf(v, &s); free_buf(v, &x); free_buf(v, &y);
    qfix_free(&q);
}

typedef struct {
    int fmt;              /* 0 = bf16, 1 = fp8, 2 = fp4 */
    const char *name;
    int M, K, N;
} Case;

static const Case cases[] = {
    { 1, "wq_a fp8",          1024, 4096,  54 },
    { 1, "wq_a fp8 nt16",     1024, 4096,  16 },
    { 1, "wq_a fp8 nt204",    1024, 4096, 204 },
    { 1, "wq_b fp8",         32768, 1024,  54 },
    { 1, "wo_a fp8 (1/8)",    1024, 4096,  54 },
    { 1, "wo_a fp8 nt16",     1024, 4096,  16 },
    { 1, "wo_a fp8 nt204",    1024, 4096, 204 },
    { 1, "wo_b fp8",          4096, 8192,  54 },
    { 1, "wo_b fp8 nt204",    4096, 8192, 204 },
    { 1, "shared_gate fp8",   2048, 4096,  54 },
    { 1, "shared_gate fp8 nt204", 2048, 4096, 204 },
    { 1, "shared_down fp8",   4096, 2048,  54 },
    { 1, "shared_down fp8 nt204", 4096, 2048, 204 },
    { 1, "indexer_wq_b fp8",  8192, 1024,  54 },
    { 2, "expert_w1 fp4 nt5",  2048, 4096,   5 },
    { 2, "expert_w1 fp4 nt16", 2048, 4096,  16 },
    { 2, "expert_w1 fp4 nt54", 2048, 4096,  54 },
    { 2, "expert_w1 fp4 nt204",2048, 4096, 204 },
    { 2, "expert_w2 fp4 nt5",  4096, 2048,   5 },
    { 2, "expert_w2 fp4 nt16", 4096, 2048,  16 },
    { 2, "expert_w2 fp4 nt54", 4096, 2048,  54 },
    { 2, "expert_w2 fp4 nt204",4096, 2048, 204 },
    { 0, "gate bf16",          256, 4096,  54 },
    { 0, "gate bf16 nt1",      256, 4096,   1 },
    { 0, "weights_proj bf16",   64, 4096,  54 },
};

int main(int argc, char **argv)
{
    const char *filter = NULL;
    double target_ms = 200.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--filter") && i + 1 < argc) filter = argv[++i];
        else if (!strcmp(argv[i], "--target-ms") && i + 1 < argc)
            target_ms = atof(argv[++i]);
        else if (!strcmp(argv[i], "--import")) g_import = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            fprintf(stderr, "usage: %s [--filter SUBSTR] [--target-ms MS] "
                            "[--import]\n", argv[0]);
            return 0;
        }
    }

    Vk vk;
    vk_init(&vk);
    printf("device: %s\n", vk.devname);
    printf("tile  : 64x16 output, BK=64, 256 threads\n");
    printf("weights: %s\n", g_import ? "VK_EXT_external_memory_host import"
                                    : "native VkDeviceMemory");
    printf("correctness gate (CPU dsv4_mmq_n):\n");
    check_format(&vk, 1);
    check_format(&vk, 0);
    check_format(&vk, 2);
    printf("\n%-24s %8s %7s %9s %9s %9s\n",
           "case", "M x K", "N", "ms/call", "GF/s", "GB/s");

    for (size_t ci = 0; ci < sizeof cases / sizeof cases[0]; ci++) {
        const Case *c = &cases[ci];
        if (filter && !strstr(c->name, filter)) continue;

        QFix q;
        const int fp8 = c->fmt == 1, fp4 = c->fmt == 2;
        const int wdt = fp8 ? DSV4_WFP8 : fp4 ? DSV4_WFP4 : DSV4_WBF16;
        qfix_init(&q, wdt, c->M, c->K);
        GBuf w, s, x, y;
        memset(&w, 0, sizeof w); memset(&s, 0, sizeof s);
        memset(&x, 0, sizeof x); memset(&y, 0, sizeof y);
        const VkDeviceSize wb = fp4
            ? (VkDeviceSize)c->M * (VkDeviceSize)c->K / 2
            : (VkDeviceSize)c->M * (VkDeviceSize)c->K * (fp8 ? 1 : 2);
        const VkDeviceSize sb = fp8
            ? (VkDeviceSize)((c->M + 127) / 128) * ((c->K + 127) / 128)
            : fp4 ? (VkDeviceSize)c->M * (c->K / 32) : 4;
        if (g_import) make_buf_import(&vk, &w, wb);
        else          make_buf(&vk, &w, wb);
        make_buf(&vk, &s, sb);
        make_buf(&vk, &x, (VkDeviceSize)c->N * c->K * sizeof(float));
        make_buf(&vk, &y, (VkDeviceSize)c->N * c->M * sizeof(float));
        memcpy(w.map, q.m.w, (size_t)wb);
        if (fp8 || fp4) memcpy(s.map, q.m.s, (size_t)sb);
        else memset(s.map, 0, 4);
        bench_fill_f32((float *)x.map, (size_t)c->N * c->K, 1.0f);
        memset(y.map, 0, (size_t)c->N * c->M * sizeof(float));

        VkDescriptorSet d = bind_case(&vk, &w, &s, &x, &y);
        VkPipeline pipe = fp8 ? vk.pipe_fp8
                        : fp4 ? vk.pipe_fp4 : vk.pipe_bf16;

        for (int i = 0; i < 5; i++)
            dispatch_once(&vk, pipe, d, (uint32_t)c->M, (uint32_t)c->K,
                          (uint32_t)c->N);

        int reps = 0;
        const double t0 = now_s();
        double el;
        do {
            dispatch_once(&vk, pipe, d, (uint32_t)c->M, (uint32_t)c->K,
                          (uint32_t)c->N);
            reps++;
            el = now_s() - t0;
        } while (el * 1000.0 < target_ms && reps < 20000);

        const double ms = el * 1e3 / (double)reps;
        const double gf = 2.0 * (double)c->M * (double)c->K * (double)c->N;
        printf("%-24s %5d x %-5d %7d %9.4f %9.1f %9.2f\n",
               c->name, c->M, c->K, c->N, ms,
               gf * (double)reps / el / 1e9,
               (double)wb * (double)reps / el / 1e9);

        free_buf(&vk, &w); free_buf(&vk, &s);
        free_buf(&vk, &x); free_buf(&vk, &y);
        qfix_free(&q);
    }
    return 0;
}
