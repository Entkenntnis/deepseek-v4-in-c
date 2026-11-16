/* SPDX-License-Identifier: Apache-2.0 */
/* bench/vk_expert.c - one routed expert (w1, w3, SWiGLU, w2) on the iGPU,
 * measured against the shipped CPU FP4 kernel on the same bytes.
 *
 * WHY A SEPARATE BINARY
 *   This is a decision tool, not a backend. It answers three questions before
 *   anyone touches src/10_compute: what does a routed expert cost on the iGPU,
 *   what does one dispatch cost end to end, and can the GPU reproduce the CPU's
 *   reduction tree bit for bit. Nothing here is reachable from the engine.
 *
 * THE SLOT IS A VULKAN ALLOCATION
 *   On this iGPU the whole device heap is host memory, and memory type 2 is
 *   DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED. So an expert
 *   slot can simply be allocated through Vulkan, mapped once, and read by the
 *   GPU with no copy and no import extension -- the same 13,369,344 bytes the
 *   CPU kernel reads, in the same layout dsv4_cache_expert_geometry() produces.
 *   The CPU baseline here therefore runs from the very same buffer, which is
 *   the only way the two numbers are comparable.
 *
 * CORRECTNESS IS A GATE
 *   The shader reproduces dsv4_matmul_fp4.c's tree exactly (see the .comp), so
 *   `exact` is checked with memcmp, not a tolerance. The only deliberately
 *   inexact step is SWiGLU, whose GLSL exp() is not libm expf(); the `fused`
 *   timing variant keeps it on the GPU and reports the error, while `exact`
 *   runs it on the CPU through dsv4_swiglu and stays bit-identical.
 *
 * USAGE
 *   ./bin/vk_expert [reps] [--slots N] [--spv FILE] [--threads N]
 *                   [--pin off|all|p|e] [--cpu-only] [--gpu-only] [--no-energy]
 *   OMP_NUM_THREADS=4 taskset -c 0-3 ./bin/vk_expert
 */
#define _GNU_SOURCE

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include <vulkan/vulkan.h>

#include "bench.h"
#include "dsv4.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_threads.h"

/* Flash geometry, the same literals bench/cases_qmat.c uses. Pro would be
 * 7168/3072 and is not what this machine runs. */
#define HID 4096
#define INT 2048
#define NEXP 6

/* Scratch layout, in floats, one region per expert of a batch. */
#define X_OFF    0u
#define GATE_OFF (X_OFF + HID)
#define UP_OFF   (GATE_OFF + INT)
#define MID_OFF  (UP_OFF + INT)
#define Y_OFF    (MID_OFF + INT)
#define TRIO_FLOATS (Y_OFF + HID)          /* 14336 floats = 56 KiB */

#define MAX_SLOTS 8

#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult r_ = (x);                                                     \
        if (r_ != VK_SUCCESS) {                                                \
            fprintf(stderr, "vk_expert: %s failed: %d\n", #x, (int)r_);        \
            exit(1);                                                          \
        }                                                                      \
    } while (0)

/* Slot layout, byte offsets. Index order is w1, w2, w3 -- the disk order and
 * the order dsv4_cache_bind_slot() uses. w1/w3 are [INT, HID], w2 [HID, INT]. */
static int64_t g_off_s[3], g_len_s[3], g_off_w[3], g_len_w[3], g_expert_bytes;

static void geom_init(void)
{
    const int64_t wb[3] = { INT * (HID / 2), HID * (INT / 2), INT * (HID / 2) };
    const int64_t sb[3] = { INT * (HID / 32), HID * (INT / 32), INT * (HID / 32) };
    int64_t off = 0;
    for (int i = 0; i < 3; i++) { g_off_s[i] = off; g_len_s[i] = sb[i]; off += sb[i]; }
    for (int i = 0; i < 3; i++) { g_off_w[i] = off; g_len_w[i] = wb[i]; off += wb[i]; }
    g_expert_bytes = off;
}

typedef struct {
    uint32_t in_n, out_n, sc, row_bytes;
    uint32_t w_off, s_off, x_off, up_off, y_off, mode;
    float    limit;
} PC;

typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    void          *map;
    VkDeviceSize   size;
} GBuf;

typedef struct {
    VkInstance       inst;
    VkPhysicalDevice phys;
    VkDevice         dev;
    VkQueue          queue;
    uint32_t         qfam;
    VkCommandPool    pool;
    VkCommandBuffer  cmd[2];
    VkFence          fence;
    VkQueryPool      qpool;
    int              has_ts;
    double           ts_ns;
    VkPhysicalDeviceMemoryProperties mp;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      pl;
    VkDescriptorPool      dpool;
    VkPipeline            pipe;
    uint32_t              mem_type;
    PFN_vkGetMemoryHostPointerPropertiesEXT get_host_props;
    VkDeviceSize          import_align;
    char                  devname[256];
} Vk;

static GBuf  g_slot[MAX_SLOTS];
static int   g_nslot_alloc;
static GBuf  g_scratch;
static VkDescriptorSet g_dset[MAX_SLOTS];
static int   g_nslot = 4;
static int   g_import;                   /* slots via VK_EXT_external_memory_host */
static int   g_huge;                     /* align/madvise imported slots to 2 MiB */
static int   g_unique;                   /* fill each slot with unique bytes      */
static double g_kern_s;                  /* last run's timestamped kernel time */
static int   g_use_gpu = 1, g_use_cpu = 1;
static unsigned g_rot;
static double g_submit_s, g_wait_s;
static Vk   *g_vk;
static float *g_xsrc;                    /* HID floats, the layer's activation */
static float *g_cpu_y;                   /* HID floats, the CPU reference     */
static double g_cpu_ms;
/* When non-zero, measure() sleeps this long before every timed call, so a
 * steady-state case can be repeated in the engine's cold-clock condition. */
static double g_measure_gap_s;

#define MAX_GAPS 32
static double g_gap_ms[MAX_GAPS] = { 0.0, 0.25, 0.5, 1.0, 2.0, 4.0,
                                     8.0, 16.0, 32.0, 64.0 };
static int    g_ngap = 10;
static double g_warmup_s = 5.0;
static double g_case_s = 3.0;
static int    g_legacy;

static double now_s(void) { return bench_now_s(); }

/* ------------------------------------------------------------- energy ----- */
static const char *g_rapl = "/sys/class/powercap/intel-rapl:0/energy_uj";

static double rapl_j(void)
{
    FILE *f = fopen(g_rapl, "r");
    if (!f) return -1.0;
    unsigned long long u = 0;
    int ok = fscanf(f, "%llu", &u) == 1;
    fclose(f);
    return ok ? (double)u * 1e-6 : -1.0;
}

/* --------------------------------------------------------------- vulkan --- */
static uint32_t *read_spv(const char *path, size_t *nbytes)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "vk_expert: cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); exit(1); }
    long n = ftell(f);
    if (n <= 0 || (n & 3) != 0) {
        fprintf(stderr, "vk_expert: %s is not a SPIR-V word stream\n", path);
        fclose(f);
        exit(1);
    }
    rewind(f);
    uint32_t *p = (uint32_t *)malloc((size_t)n);
    if (!p) { fclose(f); exit(1); }
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); exit(1); }
    fclose(f);
    *nbytes = (size_t)n;
    return p;
}

/* Type 2 here: DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT|HOST_CACHED. HOST_CACHED
 * matters -- a write-combined mapping would make the CPU baseline path crawl. */
static uint32_t pick_mem_type(Vk *v, uint32_t bits)
{
    for (int pass = 0; pass < 2; pass++) {
        const VkMemoryPropertyFlags want =
            pass == 0
            ? (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
             | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
             | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
             | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
            : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
             | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        for (uint32_t i = 0; i < v->mp.memoryTypeCount; i++) {
            if (!(bits & (1u << i))) continue;
            if ((v->mp.memoryTypes[i].propertyFlags & want) == want) return i;
        }
    }
    fprintf(stderr, "vk_expert: no host-visible memory type\n");
    exit(1);
}

static void make_buf(Vk *v, GBuf *b, VkDeviceSize size)
{
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
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
    v->mem_type = ai.memoryTypeIndex;
    VK_CHECK(vkAllocateMemory(v->dev, &ai, NULL, &b->mem));
    VK_CHECK(vkBindBufferMemory(v->dev, b->buf, b->mem, 0));
    VK_CHECK(vkMapMemory(v->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map));
    b->size = size;
}

/* Sum AnonHugePages in the smaps entry that covers [addr, addr+len). */
static unsigned long long smaps_anon_huge_kb(const void *addr, size_t len)
{
    FILE *f = fopen("/proc/self/smaps", "r");
    if (!f) return 0;
    const unsigned long long a0 = (unsigned long long)(uintptr_t)addr;
    const unsigned long long a1 = a0 + (unsigned long long)len;
    unsigned long long sum = 0, start, end, v;
    int inside = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%llx-%llx", &start, &end) == 2) {
            inside = start < a1 && end > a0;
            continue;
        }
        if (inside && sscanf(line, "AnonHugePages: %llu kB", &v) == 1)
            sum += v;
    }
    fclose(f);
    return sum;
}

/* Same bytes, same VkBuffer, but backed by posix_memalign pages imported with
 * VK_EXT_external_memory_host -- the path the engine cache uses so O_DIRECT can
 * land directly in the slot. */
static void make_buf_import(Vk *v, GBuf *b, VkDeviceSize size)
{
    if (!v->get_host_props) {
        fprintf(stderr, "vk_expert: import requested but extension unavailable\n");
        exit(1);
    }

    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
             | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(v->dev, &bi, NULL, &b->buf));

    VkMemoryRequirements rq;
    vkGetBufferMemoryRequirements(v->dev, b->buf, &rq);

    VkDeviceSize align = v->import_align ? v->import_align : 4096u;
    if (align < 4096u) align = 4096u;
    if (g_huge && align < (2u << 20)) align = 2u << 20;

    void *raw = NULL;
    if (posix_memalign(&raw, (size_t)align,
                       (size_t)rq.size + (size_t)align) != 0)
        bench_die("host import allocation failed");
    unsigned char *base = (unsigned char *)
        (((uintptr_t)raw + (uintptr_t)align - 1u) & ~((uintptr_t)align - 1u));

    if (g_huge) {
        if (madvise(base, (size_t)rq.size, MADV_HUGEPAGE) != 0)
            fprintf(stderr, "vk_expert: madvise(MADV_HUGEPAGE): %s\n",
                    strerror(errno));
        memset(base, 0, (size_t)rq.size);   /* fault in before the import */
        const unsigned long long kb = smaps_anon_huge_kb(base, (size_t)rq.size);
        fprintf(stderr, "vk_expert: imported %p..%p, AnonHugePages %llu KiB"
                        " (align %llu)\n", (void *)base, (void *)(base + rq.size),
                kb, (unsigned long long)align);
    }

    VkMemoryHostPointerPropertiesEXT hp;
    memset(&hp, 0, sizeof hp);
    hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    VK_CHECK(v->get_host_props(
        v->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        base, &hp));

    const uint32_t allowed = rq.memoryTypeBits & hp.memoryTypeBits;
    const uint32_t mt = pick_mem_type(v, allowed);
    if (mt == UINT32_MAX) {
        fprintf(stderr, "vk_expert: no importable memory type\n");
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
    v->mem_type = mt;
    VK_CHECK(vkAllocateMemory(v->dev, &ai, NULL, &b->mem));
    VK_CHECK(vkBindBufferMemory(v->dev, b->buf, b->mem, 0));
    b->map = base;
    b->size = size;
}

static int dev_ext_supported(VkPhysicalDevice phys, const char *name)
{
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, NULL, &n, NULL)
        != VK_SUCCESS)
        return 0;
    VkExtensionProperties *e =
        (VkExtensionProperties *)calloc(n, sizeof *e);
    if (!e) return 0;
    int found = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, NULL, &n, e) == VK_SUCCESS)
        for (uint32_t i = 0; i < n; i++)
            if (strcmp(e[i].extensionName, name) == 0) { found = 1; break; }
    free(e);
    return found;
}

static void vk_init(Vk *v, const char *spv)
{
    memset(v, 0, sizeof *v);

    VkApplicationInfo app;
    memset(&app, 0, sizeof app);
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dsv4 vk_expert";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici;
    memset(&ici, 0, sizeof ici);
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, NULL, &v->inst));

    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(v->inst, &n, NULL));
    if (n == 0) { fprintf(stderr, "vk_expert: no Vulkan device\n"); exit(1); }
    VkPhysicalDevice *pd = (VkPhysicalDevice *)calloc(n, sizeof *pd);
    VK_CHECK(vkEnumeratePhysicalDevices(v->inst, &n, pd));
    v->phys = pd[0];
    free(pd);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(v->phys, &props);
    snprintf(v->devname, sizeof v->devname, "%s", props.deviceName);
    vkGetPhysicalDeviceMemoryProperties(v->phys, &v->mp);

    if (g_import) {
        if (!dev_ext_supported(v->phys,
                               VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
            fprintf(stderr, "vk_expert: device lacks %s\n",
                    VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
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
        if (v->import_align == 0) {
            fprintf(stderr, "vk_expert: host import alignment is zero\n");
            exit(1);
        }
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(v->phys, &nq, NULL);
    VkQueueFamilyProperties *qf =
        (VkQueueFamilyProperties *)calloc(nq, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(v->phys, &nq, qf);
    int found = 0;
    for (uint32_t i = 0; i < nq; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            v->qfam = i;
            if (qf[i].timestampValidBits > 0) v->has_ts = 1;
            found = 1;
            break;
        }
    }
    free(qf);
    if (!found) { fprintf(stderr, "vk_expert: no compute queue\n"); exit(1); }

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
            fprintf(stderr, "vk_expert: vkGetMemoryHostPointerPropertiesEXT "
                            "missing\n");
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
    cbi.commandBufferCount = 2;
    VK_CHECK(vkAllocateCommandBuffers(v->dev, &cbi, v->cmd));

    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof fci);
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(v->dev, &fci, NULL, &v->fence));

    if (v->has_ts) {
        VkQueryPoolCreateInfo qpi;
        memset(&qpi, 0, sizeof qpi);
        qpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 8;
        VK_CHECK(vkCreateQueryPool(v->dev, &qpi, NULL, &v->qpool));
        v->ts_ns = (double)props.limits.timestampPeriod;
    }

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
    VK_CHECK(vkCreateDescriptorSetLayout(v->dev, &dli, NULL, &v->dsl));

    VkPushConstantRange pcr;
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PC);
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
    psz.descriptorCount = 3u * MAX_SLOTS;
    VkDescriptorPoolCreateInfo dpi;
    memset(&dpi, 0, sizeof dpi);
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = MAX_SLOTS;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &psz;
    VK_CHECK(vkCreateDescriptorPool(v->dev, &dpi, NULL, &v->dpool));

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
    VK_CHECK(vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1, &cpi, NULL,
                                      &v->pipe));
    vkDestroyShaderModule(v->dev, mod, NULL);
}

static void bind_descriptors(Vk *v, int slot)
{
    VkDescriptorSetAllocateInfo ai;
    memset(&ai, 0, sizeof ai);
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = v->dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &v->dsl;
    VK_CHECK(vkAllocateDescriptorSets(v->dev, &ai, &g_dset[slot]));

    VkDescriptorBufferInfo bi[3] = {
        { g_slot[slot].buf, 0, VK_WHOLE_SIZE },   /* weights, uvec4 view */
        { g_slot[slot].buf, 0, VK_WHOLE_SIZE },   /* scales, uint view   */
        { g_scratch.buf,    0, VK_WHOLE_SIZE },   /* activations         */
    };
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < 3; i++) {
        memset(&w[i], 0, sizeof w[i]);
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = g_dset[slot];
        w[i].dstBinding = (uint32_t)i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(v->dev, 3, w, 0, NULL);
}

static void submit(Vk *v, VkCommandBuffer cmd)
{
    VK_CHECK(vkResetFences(v->dev, 1, &v->fence));
    VkSubmitInfo si;
    memset(&si, 0, sizeof si);
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    double a = now_s();
    VK_CHECK(vkQueueSubmit(v->queue, 1, &si, v->fence));
    g_submit_s += now_s() - a;
    a = now_s();
    VK_CHECK(vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE, UINT64_MAX));
    g_wait_s += now_s() - a;
}

static void ts_begin(Vk *v, VkCommandBuffer cmd, uint32_t q)
{
    if (v->has_ts) vkCmdResetQueryPool(cmd, v->qpool, q, 2);
}

static void ts_write(Vk *v, VkCommandBuffer cmd, uint32_t q)
{
    if (v->has_ts)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, v->qpool, q);
}

/* Sum of (end - start) across one or two submits. */
static double ts_read(Vk *v, uint32_t q0, uint32_t nq)
{
    if (!v->has_ts) return -1.0;
    uint64_t t[8];
    VK_CHECK(vkGetQueryPoolResults(v->dev, v->qpool, q0, nq,
                                   sizeof(uint64_t) * nq, t,
                                   sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT
                                   | VK_QUERY_RESULT_WAIT_BIT));
    double s = 0.0;
    for (uint32_t i = 0; i + 1 < nq; i += 2) {
        double d = (double)(t[i + 1] - t[i]) * v->ts_ns;
        if (d > 0.0) s += d;
    }
    return s * 1e-9;
}

/* Must match ROWS_PER_WG in bench/shaders/expert_fp4.comp. */
#define WG_ROWS 64u

/* ------------------------------------------------------------- recording -- */
static void rec_matvec(Vk *v, VkCommandBuffer cmd, uint32_t w_off,
                       uint32_t s_off, uint32_t in_n, uint32_t out_n,
                       uint32_t sc, uint32_t row_bytes,
                       uint32_t x_off, uint32_t y_off)
{
    PC pc;
    memset(&pc, 0, sizeof pc);
    pc.in_n = in_n;
    pc.out_n = out_n;
    pc.sc = sc;
    pc.row_bytes = row_bytes;
    pc.w_off = w_off;
    pc.s_off = s_off;
    pc.x_off = x_off;
    pc.y_off = y_off;
    pc.mode = 0;
    vkCmdPushConstants(cmd, v->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof pc, &pc);
    vkCmdDispatch(cmd, (out_n + WG_ROWS - 1u) / WG_ROWS, 1, 1);
}

static void rec_swiglu(Vk *v, VkCommandBuffer cmd, uint32_t gate, uint32_t up,
                       uint32_t mid)
{
    PC pc;
    memset(&pc, 0, sizeof pc);
    pc.in_n = INT;
    pc.x_off = gate;
    pc.up_off = up;
    pc.y_off = mid;
    pc.mode = 1;
    pc.limit = 10.0f;
    vkCmdPushConstants(cmd, v->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof pc, &pc);
    vkCmdDispatch(cmd, (INT + 255u) / 256u, 1, 1);
}

static void mem_barrier(Vk *v, VkCommandBuffer cmd)
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

static VkCommandBuffer begin_rec(Vk *v, int which)
{
    VkCommandBuffer cmd = v->cmd[which];
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pipe);
    return cmd;
}

/* ---------------------------------------------------------------- cases --- */
static DSV4ExpertW bind_expert(const uint8_t *base)
{
    DSV4ExpertW e;
    memset(&e, 0, sizeof e);
    e.w1.wdt = DSV4_WFP4; e.w1.rows = INT; e.w1.cols = HID;
    e.w1.blk_r = 1; e.w1.blk_c = 32;
    e.w1.w = base + g_off_w[0]; e.w1.s = base + g_off_s[0];
    e.w2.wdt = DSV4_WFP4; e.w2.rows = HID; e.w2.cols = INT;
    e.w2.blk_r = 1; e.w2.blk_c = 32;
    e.w2.w = base + g_off_w[1]; e.w2.s = base + g_off_s[1];
    e.w3.wdt = DSV4_WFP4; e.w3.rows = INT; e.w3.cols = HID;
    e.w3.blk_r = 1; e.w3.blk_c = 32;
    e.w3.w = base + g_off_w[2]; e.w3.s = base + g_off_s[2];
    return e;
}

static float *scratch_f(int slot, uint32_t off)
{
    return (float *)g_scratch.map + (size_t)slot * TRIO_FLOATS + off;
}

/* The cache's on-disk layout: scales w1,w2,w3 then weights w1,w2,w3. */
static void fill_slot(void *base_)
{
    uint8_t *base = (uint8_t *)base_;
    QFix q[3];
    qfix_init(&q[0], DSV4_WFP4, INT, HID);   /* w1 */
    qfix_init(&q[1], DSV4_WFP4, HID, INT);   /* w2 */
    qfix_init(&q[2], DSV4_WFP4, INT, HID);   /* w3 */
    for (int t = 0; t < 3; t++) {
        memcpy(base + g_off_s[t], q[t].s, (size_t)g_len_s[t]);
        memcpy(base + g_off_w[t], q[t].w, (size_t)g_len_w[t]);
        qfix_free(&q[t]);
    }
}

/* No 64 KiB pattern: every byte of every slot is independent. Each slot gets
 * a different image, so neither L2 nor any memory compression can exploit
 * repetition. Scale bytes stay in the same exponent band the fixture uses. */
static void fill_slot_unique(void *base_)
{
    uint8_t *base = (uint8_t *)base_;
    for (int t = 0; t < 3; t++) {
        uint8_t *s = base + g_off_s[t];
        for (int64_t i = 0; i < g_len_s[t]; i++)
            s[i] = (uint8_t)(118u + (bench_rnd32() % 7u));
        uint8_t *w = base + g_off_w[t];
        for (int64_t i = 0; i < g_len_w[t]; i++)
            w[i] = (uint8_t)(bench_rnd32() >> 24);
    }
}

static int next_slot(void) { return (int)(g_rot++ % (unsigned)g_nslot); }

static void run_cpu(void *p)
{
    (void)p;
    const int slot = next_slot();
    memcpy(scratch_f(slot, X_OFF), g_xsrc, HID * sizeof(float));
    const DSV4ExpertW e = bind_expert((const uint8_t *)g_slot[slot].map);
    dsv4_mmq(scratch_f(slot, GATE_OFF), scratch_f(slot, X_OFF), &e.w1);
    dsv4_mmq(scratch_f(slot, UP_OFF),   scratch_f(slot, X_OFF), &e.w3);
    dsv4_swiglu(scratch_f(slot, MID_OFF), scratch_f(slot, GATE_OFF),
                scratch_f(slot, UP_OFF), INT, 10.0f);
    dsv4_mmq(scratch_f(slot, Y_OFF), scratch_f(slot, MID_OFF), &e.w2);
}

/* Submit w1+w3, let the CPU do SWiGLU, submit w2. Bit-identical to run_cpu. */
static void gpu_exact(Vk *v, int slot)
{
    const uint32_t base = (uint32_t)slot * TRIO_FLOATS;
    VkCommandBuffer cmd = begin_rec(v, 0);
    ts_begin(v, cmd, 0);
    ts_write(v, cmd, 0);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl, 0, 1,
                            &g_dset[slot], 0, NULL);
    rec_matvec(v, cmd, (uint32_t)g_off_w[0], (uint32_t)g_off_s[0],
               HID, INT, HID / 32, HID / 2, base + X_OFF, base + GATE_OFF);
    rec_matvec(v, cmd, (uint32_t)g_off_w[2], (uint32_t)g_off_s[2],
               HID, INT, HID / 32, HID / 2, base + X_OFF, base + UP_OFF);
    ts_write(v, cmd, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));
    submit(v, cmd);

    dsv4_swiglu(scratch_f(slot, MID_OFF), scratch_f(slot, GATE_OFF),
                scratch_f(slot, UP_OFF), INT, 10.0f);

    cmd = begin_rec(v, 1);
    ts_begin(v, cmd, 2);
    ts_write(v, cmd, 2);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl, 0, 1,
                            &g_dset[slot], 0, NULL);
    rec_matvec(v, cmd, (uint32_t)g_off_w[1], (uint32_t)g_off_s[1],
               INT, HID, INT / 32, INT / 2, base + MID_OFF, base + Y_OFF);
    ts_write(v, cmd, 3);
    VK_CHECK(vkEndCommandBuffer(cmd));
    submit(v, cmd);

    const double k = ts_read(v, 0, 4);
    if (k >= 0.0) g_kern_s = k;
}

static void run_gpu_exact(void *p)
{
    (void)p;
    const int slot = next_slot();
    memcpy(scratch_f(slot, X_OFF), g_xsrc, HID * sizeof(float));
    gpu_exact(g_vk, slot);
}

/* One submit: w1, w3, SWiGLU (GPU), w2. */
static void run_gpu_fused(void *p)
{
    (void)p;
    Vk *v = g_vk;
    const int slot = next_slot();
    const uint32_t base = (uint32_t)slot * TRIO_FLOATS;
    memcpy(scratch_f(slot, X_OFF), g_xsrc, HID * sizeof(float));

    VkCommandBuffer cmd = begin_rec(v, 0);
    ts_begin(v, cmd, 0);
    ts_write(v, cmd, 0);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl, 0, 1,
                            &g_dset[slot], 0, NULL);
    rec_matvec(v, cmd, (uint32_t)g_off_w[0], (uint32_t)g_off_s[0],
               HID, INT, HID / 32, HID / 2, base + X_OFF, base + GATE_OFF);
    rec_matvec(v, cmd, (uint32_t)g_off_w[2], (uint32_t)g_off_s[2],
               HID, INT, HID / 32, HID / 2, base + X_OFF, base + UP_OFF);
    mem_barrier(v, cmd);
    rec_swiglu(v, cmd, base + GATE_OFF, base + UP_OFF, base + MID_OFF);
    mem_barrier(v, cmd);
    rec_matvec(v, cmd, (uint32_t)g_off_w[1], (uint32_t)g_off_s[1],
               INT, HID, INT / 32, INT / 2, base + MID_OFF, base + Y_OFF);
    ts_write(v, cmd, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));
    submit(v, cmd);

    const double k = ts_read(v, 0, 2);
    if (k >= 0.0) g_kern_s = k;
}

/* Six experts, twelve matvecs plus six SWiGLUs, one submit: the per-layer
 * shape the decoder would actually use. Each expert has its own slot and its
 * own scratch region; barriers only between the three phases. */
static void run_gpu_batch6(void *p)
{
    (void)p;
    Vk *v = g_vk;
    for (int k = 0; k < NEXP; k++)
        memcpy(scratch_f(k, X_OFF), g_xsrc, HID * sizeof(float));

    VkCommandBuffer cmd = begin_rec(v, 0);
    ts_begin(v, cmd, 0);
    ts_write(v, cmd, 0);

    for (int k = 0; k < NEXP; k++) {
        const uint32_t base = (uint32_t)k * TRIO_FLOATS;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl, 0,
                                1, &g_dset[k], 0, NULL);
        rec_matvec(v, cmd, (uint32_t)g_off_w[0], (uint32_t)g_off_s[0],
                   HID, INT, HID / 32, HID / 2, base + X_OFF, base + GATE_OFF);
        rec_matvec(v, cmd, (uint32_t)g_off_w[2], (uint32_t)g_off_s[2],
                   HID, INT, HID / 32, HID / 2, base + X_OFF, base + UP_OFF);
    }
    mem_barrier(v, cmd);
    for (int k = 0; k < NEXP; k++) {
        const uint32_t base = (uint32_t)k * TRIO_FLOATS;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl, 0,
                                1, &g_dset[k], 0, NULL);
        rec_swiglu(v, cmd, base + GATE_OFF, base + UP_OFF, base + MID_OFF);
    }
    mem_barrier(v, cmd);
    for (int k = 0; k < NEXP; k++) {
        const uint32_t base = (uint32_t)k * TRIO_FLOATS;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl, 0,
                                1, &g_dset[k], 0, NULL);
        rec_matvec(v, cmd, (uint32_t)g_off_w[1], (uint32_t)g_off_s[1],
                   INT, HID, INT / 32, INT / 2, base + MID_OFF, base + Y_OFF);
    }
    ts_write(v, cmd, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));
    submit(v, cmd);

    const double k = ts_read(v, 0, 2);
    if (k >= 0.0) g_kern_s = k;
}

/* The pessimistic shape: one submit per expert, fenced, as the decoder would
 * have to do it if it starts each expert the moment its read lands. */
static void run_gpu_stream6(void *p)
{
    (void)p;
    Vk *v = g_vk;
    for (int k = 0; k < NEXP; k++) {
        const uint32_t base = (uint32_t)k * TRIO_FLOATS;
        memcpy(scratch_f(k, X_OFF), g_xsrc, HID * sizeof(float));
        VkCommandBuffer cmd = begin_rec(v, 0);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl, 0,
                                1, &g_dset[k], 0, NULL);
        rec_matvec(v, cmd, (uint32_t)g_off_w[0], (uint32_t)g_off_s[0],
                   HID, INT, HID / 32, HID / 2, base + X_OFF, base + GATE_OFF);
        rec_matvec(v, cmd, (uint32_t)g_off_w[2], (uint32_t)g_off_s[2],
                   HID, INT, HID / 32, HID / 2, base + X_OFF, base + UP_OFF);
        mem_barrier(v, cmd);
        rec_swiglu(v, cmd, base + GATE_OFF, base + UP_OFF, base + MID_OFF);
        mem_barrier(v, cmd);
        rec_matvec(v, cmd, (uint32_t)g_off_w[1], (uint32_t)g_off_s[1],
                   INT, HID, INT / 32, INT / 2, base + MID_OFF, base + Y_OFF);
        VK_CHECK(vkEndCommandBuffer(cmd));
        submit(v, cmd);
    }
}

/* --------------------------------------------------------------- report --- */
static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Linear-interpolated percentile. a must be sorted. */
static double pct_sorted(const double *a, int n, double q)
{
    if (n <= 0) return 0.0;
    if (n == 1) return a[0];
    const double x = q * (double)(n - 1);
    const int i = (int)x;
    const double f = x - (double)i;
    if (i >= n - 1) return a[n - 1];
    return a[i] * (1.0 - f) + a[i + 1] * f;
}

static void sleep_s(double s)
{
    if (s <= 0.0) return;
    struct timespec ts;
    ts.tv_sec  = (time_t)s;
    ts.tv_nsec = (long)((s - (double)ts.tv_sec) * 1e9);
    if (ts.tv_nsec < 0) ts.tv_nsec = 0;
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) { }
}

static const double TRIO_FLOPS = 2.0 * 2048.0 * 4096.0 * 3.0;

static double trio_bytes(void)
{
    return (double)g_expert_bytes
         + (double)(HID + 3 * INT + HID) * 4.0;
}

static void warm_gpu(double seconds);

/* A gap point must start from the same sustained clock state as the steady
 * case, otherwise a large gap from the previous point leaks into this one.
 * 3 s of back-to-back fused trios is enough to reach the settled clock. */
#define GAP_REWARM_S 3.0

static void output_header(void)
{
    printf("%-13s %7s %8s %8s %8s %8s %7s %7s %7s %6s %6s %10s  %s\n",
           "case", "submits", "p10 ms", "p50 ms", "p90 ms", "kern ms",
           "sub ms", "wait ms", "GF/s", "GB/s", "xCPU", "J/expert", "gate");
}

static double measure(const char *name, void (*fn)(void *), int reps,
                      int submits, int per_expert, int exact)
{
    for (int i = 0; i < 8; i++) fn(NULL);          /* warm up / clock ramp */

    double *t = (double *)calloc((size_t)reps, sizeof *t);
    double *k = (double *)calloc((size_t)reps, sizeof *k);
    if (!t || !k) bench_die("oom");
    g_submit_s = g_wait_s = 0.0;
    const double j0 = rapl_j();
    int nk = 0;
    for (int i = 0; i < reps; i++) {
        if (g_measure_gap_s > 0.0) sleep_s(g_measure_gap_s);
        g_kern_s = -1.0;
        const double a = now_s();
        fn(NULL);
        t[i] = now_s() - a;
        if (g_kern_s >= 0.0) k[nk++] = g_kern_s;
    }
    const double j1 = rapl_j();

    qsort(t, (size_t)reps, sizeof *t, cmp_d);
    if (nk) qsort(k, (size_t)nk, sizeof *k, cmp_d);
    const double div = (double)per_expert;
    const double p10 = pct_sorted(t, reps, 0.10) * 1e3 / div;
    const double p50 = pct_sorted(t, reps, 0.50) * 1e3 / div;
    const double p90 = pct_sorted(t, reps, 0.90) * 1e3 / div;
    const double kern = nk ? pct_sorted(k, nk, 0.50) * 1e3 / div : -1.0;

    printf("%-13s %7d %8.4f %8.4f %8.4f", name, submits, p10, p50, p90);
    if (kern >= 0.0) printf(" %8.4f", kern);
    else             printf(" %8s", "-");

    printf(" %7.4f %7.4f %7.1f %6.2f",
           g_submit_s / reps * 1e3 / div, g_wait_s / reps * 1e3 / div,
           TRIO_FLOPS / (p50 * 1e-3) / 1e9,
           trio_bytes() / (p50 * 1e-3) / 1e9);
    if (g_cpu_ms > 0.0) printf(" %5.2fx", g_cpu_ms / p50);
    else                printf(" %6s", "-");

    if (j0 >= 0.0 && j1 >= 0.0) printf(" %10.4f", (j1 - j0) / reps / div);
    else                        printf(" %10s", "-");

    printf("  %s\n", exact ? "memcmp" : "exp() differs");
    free(t);
    free(k);
    return p50;
}

/* One timed call. g_kern_s is set by the fused path when timestamps exist. */
typedef struct { double wall, kern; } Sample;

static void timed_fused(Sample *s)
{
    g_kern_s = -1.0;
    const double a = now_s();
    run_gpu_fused(NULL);
    s->wall = now_s() - a;
    s->kern = g_kern_s;
}

/* Fill a distribution for one call position, sorted ascending. */
static void sort_samples(double *v, int n)
{
    if (n > 0) qsort(v, (size_t)n, sizeof *v, cmp_d);
}

/* GAP SWEEP
 *
 * Each iteration sleeps for `gap_s` with the GPU idle, then times two fused
 * trios back to back: `cold` is the first expert after the gap (it pays for
 * whatever clock the GPU dropped to), `warm` is the second (steady state).
 * That pair is what a live decoder sees: one first expert per layer after IO,
 * then the rest of the layer's experts.
 *
 * A pre-roll of the same gap/run pattern keeps each gap point from inheriting
 * the clock state of the previous one. The sleep is outside the timed region,
 * so wall is the trio only.
 */
static double measure_gap(double gap_s, int reps)
{
    warm_gpu(GAP_REWARM_S);
    for (int i = 0; i < 8; i++) {
        sleep_s(gap_s);
        run_gpu_fused(NULL);
    }

    Sample *cold = (Sample *)calloc((size_t)reps, sizeof *cold);
    Sample *warm = (Sample *)calloc((size_t)reps, sizeof *warm);
    double *cw = (double *)calloc((size_t)reps, sizeof *cw);
    double *ck = (double *)calloc((size_t)reps, sizeof *ck);
    double *ww = (double *)calloc((size_t)reps, sizeof *ww);
    double *wk = (double *)calloc((size_t)reps, sizeof *wk);
    if (!cold || !warm || !cw || !ck || !ww || !wk) bench_die("oom");

    int nk = 0;
    for (int i = 0; i < reps; i++) {
        sleep_s(gap_s);
        timed_fused(&cold[i]);
        timed_fused(&warm[i]);
        cw[i] = cold[i].wall;
        ww[i] = warm[i].wall;
        if (cold[i].kern >= 0.0) ck[nk++] = cold[i].kern;
        if (warm[i].kern >= 0.0) wk[i] = warm[i].kern;
    }

    sort_samples(cw, reps);
    sort_samples(ww, reps);
    sort_samples(ck, nk);
    if (nk) sort_samples(wk, nk);

    const double c10 = pct_sorted(cw, reps, 0.10) * 1e3;
    const double c50 = pct_sorted(cw, reps, 0.50) * 1e3;
    const double c90 = pct_sorted(cw, reps, 0.90) * 1e3;
    const double w50 = pct_sorted(ww, reps, 0.50) * 1e3;
    const double ck50 = nk ? pct_sorted(ck, nk, 0.50) * 1e3 : -1.0;
    const double wk50 = nk ? pct_sorted(wk, nk, 0.50) * 1e3 : -1.0;

    printf("%7.2f %8.4f %8.4f %8.4f", gap_s * 1e3, c10, c50, c90);
    if (ck50 >= 0.0) printf(" %9.4f", ck50);
    else             printf(" %9s", "-");
    printf(" %9.4f", w50);
    if (wk50 >= 0.0) printf(" %9.4f", wk50);
    else             printf(" %9s", "-");
    printf(" %8.1f %7.2f %7.2f\n",
           TRIO_FLOPS / (c50 * 1e-3) / 1e9,
           trio_bytes() / (c50 * 1e-3) / 1e9,
           g_cpu_ms > 0.0 ? g_cpu_ms / c50 : 0.0);

    free(cold); free(warm); free(cw); free(ck); free(ww); free(wk);
    return c50;
}

static void gap_header(void)
{
    printf("%7s %8s %8s %8s %9s %9s %9s %8s %7s %7s\n",
           "gap ms", "cold p10", "cold p50", "cold p90", "cold kern",
           "warm p50", "warm kern", "coldGF/s", "GB/s", "xCPU");
}

/* POST-GAP RECOVERY
 *
 * Six fused trios in a row after an idle gap, each one its own submit. This is
 * the per-expert stream a layer would issue if nothing were batched: it shows
 * whether the GPU ramps back up inside a layer, or stays at the idle clock for
 * all six. Columns are the p50 of each position in the six-call burst.
 */
#define BURST_N 6
static void measure_burst(double gap_s, int reps)
{
    warm_gpu(GAP_REWARM_S);
    for (int i = 0; i < 8; i++) {
        sleep_s(gap_s);
        for (int j = 0; j < BURST_N; j++) run_gpu_fused(NULL);
    }

    double *w = (double *)calloc((size_t)reps * BURST_N, sizeof *w);
    double *k = (double *)calloc((size_t)reps * BURST_N, sizeof *k);
    double *col = (double *)calloc((size_t)reps, sizeof *col);
    if (!w || !k || !col) bench_die("oom");

    for (int i = 0; i < reps; i++) {
        sleep_s(gap_s);
        for (int j = 0; j < BURST_N; j++) {
            Sample s;
            timed_fused(&s);
            w[i * BURST_N + j] = s.wall;
            k[i * BURST_N + j] = s.kern;
        }
    }

    printf("%7.2f", gap_s * 1e3);
    for (int j = 0; j < BURST_N; j++) {
        for (int i = 0; i < reps; i++) col[i] = w[i * BURST_N + j];
        sort_samples(col, reps);
        printf(" %7.4f", pct_sorted(col, reps, 0.50) * 1e3);
    }
    printf("   |");
    for (int j = 0; j < BURST_N; j++) {
        int n = 0;
        for (int i = 0; i < reps; i++)
            if (k[i * BURST_N + j] >= 0.0) col[n++] = k[i * BURST_N + j];
        sort_samples(col, n);
        if (n) printf(" %7.4f", pct_sorted(col, n, 0.50) * 1e3);
        else   printf(" %7s", "-");
    }
    printf("\n");

    free(w); free(k); free(col);
}

/* Run one fused trio for `seconds` to put the GPU in its sustained clock
 * state before any measurement. Round 1 showed a 30% swing between the first
 * and second case, so this is not optional. */
static void warm_gpu(double seconds)
{
    if (!g_use_gpu || seconds <= 0.0) return;
    const double t0 = now_s();
    do {
        run_gpu_fused(NULL);
    } while (now_s() - t0 < seconds);
}

/* ----------------------------------------------------------------- main --- */
static void usage(const char *me)
{
    fprintf(stderr,
            "usage: %s [reps] [--slots N] [--spv FILE] [--threads N]\n"
            "       [--pin off|all|p|e] [--cpu-only] [--gpu-only] [--no-energy]\n"
            "       [--warmup S] [--case-sec S] [--gaps MS,MS,...]\n"
            "       [--no-gaps] [--legacy]\n"
            "       [--import] [--huge] [--unique]\n"
            "\n"
            "  --warmup S      seconds of fused GPU load before timing (5)\n"
            "  --case-sec S    target seconds per steady-state case (3)\n"
            "  --cold-batch MS time gpu-batch6 with an MS idle gap before each call\n"
            "  --gaps LIST     comma-separated idle gaps in ms:\n"
            "                  0,0.25,0.5,1,2,4,8,16,32,64\n"
            "  --no-gaps       skip the idle-gap sweep\n"
            "  --legacy        also time gpu-exact, gpu-batch6, gpu-stream6\n"
            "  --import        back slots with VK_EXT_external_memory_host pages,\n"
            "                  like the engine cache, instead of native memory\n"
            "  --huge          with --import: 2 MiB alignment + MADV_HUGEPAGE\n"
            "  --unique        fill every slot with unique random bytes instead\n"
            "                  of the repeated 64 KiB fixture pattern\n",
            me);
}

/* Parse "0,0.5,4,16" as milliseconds into g_gap_ms. */
static int parse_gaps(const char *s)
{
    int n = 0;
    const char *p = s;
    while (*p && n < MAX_GAPS) {
        char *end = NULL;
        const double v = strtod(p, &end);
        if (end == p || v < 0.0) return -1;
        g_gap_ms[n++] = v;
        p = end;
        if (*p == ',') p++;
        else if (*p != '\0') return -1;
    }
    if (n == 0) return -1;
    g_ngap = n;
    return 0;
}

int main(int argc, char **argv)
{
    int reps = 0;                 /* 0 = auto */
    const char *spv = "build/expert_fp4.spv";
    int pin = DSV4_PIN_P, threads = 4;
    int energy = 1, gaps = 1;
    double cold_gap_ms = 0.0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--slots") && i + 1 < argc) {
            g_nslot = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--spv") && i + 1 < argc) {
            spv = argv[++i];
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--pin") && i + 1 < argc) {
            pin = dsv4_threads_parse_pin(argv[++i]);
            if (pin < 0) { usage(argv[0]); return 1; }
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            g_warmup_s = strtod(argv[++i], NULL);
            if (g_warmup_s < 0.0) { usage(argv[0]); return 1; }
        } else if (!strcmp(argv[i], "--case-sec") && i + 1 < argc) {
            g_case_s = strtod(argv[++i], NULL);
            if (g_case_s <= 0.0) { usage(argv[0]); return 1; }
        } else if (!strcmp(argv[i], "--gaps") && i + 1 < argc) {
            if (parse_gaps(argv[++i]) != 0) { usage(argv[0]); return 1; }
        } else if (!strcmp(argv[i], "--no-gaps")) {
            gaps = 0;
        } else if (!strcmp(argv[i], "--cold-batch") && i + 1 < argc) {
            cold_gap_ms = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--legacy")) {
            g_legacy = 1;
        } else if (!strcmp(argv[i], "--cpu-only")) {
            g_use_gpu = 0;
        } else if (!strcmp(argv[i], "--gpu-only")) {
            g_use_cpu = 0;
        } else if (!strcmp(argv[i], "--no-energy")) {
            energy = 0;
        } else if (!strcmp(argv[i], "--import")) {
            g_import = 1;
        } else if (!strcmp(argv[i], "--huge")) {
            g_import = 1;
            g_huge = 1;
        } else if (!strcmp(argv[i], "--unique")) {
            g_unique = 1;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            reps = atoi(argv[i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (g_nslot < 1) g_nslot = 1;
    if (g_nslot > MAX_SLOTS) g_nslot = MAX_SLOTS;
    g_nslot_alloc = g_nslot > NEXP ? g_nslot : NEXP;
    if (!energy) g_rapl = "/nonexistent";

    const DSV4ThreadOpts to = { pin, threads, 0, DSV4_PIN_OFF };
    dsv4_threads_apply(&to);

    geom_init();
    if (g_expert_bytes != 13369344) {
        fprintf(stderr, "vk_expert: geometry is %lld B, expected 13369344\n",
                (long long)g_expert_bytes);
        return 1;
    }

    g_xsrc  = (float *)bench_xalloc((size_t)HID * sizeof(float));
    g_cpu_y = (float *)bench_xalloc((size_t)HID * sizeof(float));
    bench_fill_f32(g_xsrc, HID, 1.0f);

    Vk vk;
    memset(&vk, 0, sizeof vk);
    const VkDeviceSize scratch_bytes =
        (VkDeviceSize)g_nslot_alloc * TRIO_FLOATS * sizeof(float);

    if (g_use_gpu) {
        vk_init(&vk, spv);
        g_vk = &vk;
        for (int i = 0; i < g_nslot_alloc; i++)
            if (g_import)
                make_buf_import(&vk, &g_slot[i],
                                (VkDeviceSize)g_expert_bytes);
            else
                make_buf(&vk, &g_slot[i], (VkDeviceSize)g_expert_bytes);
        make_buf(&vk, &g_scratch, scratch_bytes);
    } else {
        /* Host fallback: the same bytes, plain malloc, so --cpu-only is a
         * baseline for the shipped kernel and nothing else. */
        for (int i = 0; i < g_nslot_alloc; i++) {
            if (posix_memalign(&g_slot[i].map, 4096,
                               (size_t)g_expert_bytes) != 0)
                bench_die("slot allocation failed");
            g_slot[i].size = (VkDeviceSize)g_expert_bytes;
        }
        if (posix_memalign(&g_scratch.map, 4096, (size_t)scratch_bytes) != 0)
            bench_die("scratch allocation failed");
        g_scratch.size = scratch_bytes;
    }

    /* Fill every slot with the same fixture the CPU bench uses, in the cache's
     * own layout. One-time, outside every timed region. */
    for (int i = 0; i < g_nslot_alloc; i++) {
        if (g_unique) fill_slot_unique(g_slot[i].map);
        else          fill_slot(g_slot[i].map);
        if (g_use_gpu) bind_descriptors(&vk, i);
    }
    memset(g_scratch.map, 0, (size_t)scratch_bytes);

    if (g_use_gpu) {
        printf("device        : %s (%s)\n", vk.devname,
               vk.has_ts ? "timestamps" : "no timestamps");
        printf("slot          : %lld B, memory type %u, %d slots\n",
               (long long)g_expert_bytes, vk.mem_type, g_nslot_alloc);
        printf("slot alloc    : %s%s\n",
               g_import ? "VK_EXT_external_memory_host import"
                        : "native VkDeviceMemory",
               g_import && g_huge ? " + 2 MiB alignment/MADV_HUGEPAGE" : "");
        printf("fixture       : %s\n",
               g_unique ? "unique random bytes per slot"
                        : "repeated 64 KiB pattern per slot");
    } else {
        printf("slot          : %lld B, host malloc, %d slots\n",
               (long long)g_expert_bytes, g_nslot_alloc);
    }

    /* ---- correctness: CPU reference, then GPU, on slot 0 ---- */
    g_rot = 0;
    {
        memcpy(scratch_f(0, X_OFF), g_xsrc, HID * sizeof(float));
        const DSV4ExpertW e = bind_expert((const uint8_t *)g_slot[0].map);
        dsv4_mmq(scratch_f(0, GATE_OFF), scratch_f(0, X_OFF), &e.w1);
        dsv4_mmq(scratch_f(0, UP_OFF),   scratch_f(0, X_OFF), &e.w3);
        dsv4_swiglu(scratch_f(0, MID_OFF), scratch_f(0, GATE_OFF),
                    scratch_f(0, UP_OFF), INT, 10.0f);
        dsv4_mmq(scratch_f(0, Y_OFF), scratch_f(0, MID_OFF), &e.w2);
        memcpy(g_cpu_y, scratch_f(0, Y_OFF), HID * sizeof(float));
    }

    if (g_use_gpu) {
        g_rot = 0;
        gpu_exact(g_vk, 0);
        int ndiff = 0;
        double maxd = 0.0;
        for (int i = 0; i < HID; i++) {
            const float g = scratch_f(0, Y_OFF)[i];
            if (memcmp(&g, &g_cpu_y[i], sizeof(float)) != 0) ndiff++;
            const double d = fabs((double)g - (double)g_cpu_y[i]);
            if (d > maxd) maxd = d;
        }
        printf("exact gate    : %s (%d of %d floats differ, max |d| %.3e)\n",
               ndiff ? "MISMATCH" : "memcmp clean", ndiff, HID, maxd);

        g_rot = 0;
        run_gpu_fused(NULL);
        double e2 = 0.0, r2 = 0.0;
        maxd = 0.0;
        for (int i = 0; i < HID; i++) {
            const double d = (double)scratch_f(0, Y_OFF)[i] - (double)g_cpu_y[i];
            e2 += d * d;
            r2 += (double)g_cpu_y[i] * (double)g_cpu_y[i];
            if (fabs(d) > maxd) maxd = fabs(d);
        }
        printf("fused gate    : rel L2 %.3e, max |d| %.3e (GPU exp vs expf)\n",
               sqrt(e2 / (r2 > 0.0 ? r2 : 1.0)), maxd);
    }
    printf("\n");

    /* Calibrate the CPU baseline before anything is printed: it sizes the rep
     * count and it is the denominator of the xCPU column. */
    if (g_use_cpu) {
        g_rot = 0;
        for (int i = 0; i < 10; i++) run_cpu(NULL);
        const double a = now_s();
        for (int i = 0; i < 50; i++) run_cpu(NULL);
        g_cpu_ms = (now_s() - a) / 50.0 * 1e3;
    }

    /* ---- auto reps: target seconds per steady-state case ---- */
    if (reps <= 0) {
        double unit = g_cpu_ms > 0.0 ? g_cpu_ms : 0.50;  /* ms per trio */
        if (g_use_gpu && unit > 0.50) unit = 0.50;
        reps = (int)(g_case_s * 1e3 / unit);
        if (reps < 40) reps = 40;
        if (reps > 6000) reps = 6000;
    }

    printf("reps          : %d per case, %d slots, warmup %.1f s\n\n",
           reps, g_nslot, g_warmup_s);
    output_header();

    if (g_use_cpu) {
        /* The ratio column must use the sustained CPU distribution, not the
         * short calibration above. Clear it so the CPU row prints "-" and the
         * GPU rows divide by the p50 of this very run. */
        g_cpu_ms = 0.0;
        g_rot = 0;
        const double cpu_p50 = measure("cpu", run_cpu, reps, 1, 1, 1);
        if (cpu_p50 > 0.0) g_cpu_ms = cpu_p50;
    }

    if (g_use_gpu) {
        /* One long sustained load, then the fused single-expert case. The
         * exact/batch/stream shapes stay behind --legacy: the port under
         * consideration is fused, one expert per submit. */
        warm_gpu(g_warmup_s);
        g_rot = 0;
        measure("gpu-fused", run_gpu_fused, reps, 1, 1, 0);

        if (g_legacy) {
            g_rot = 0;
            measure("gpu-exact", run_gpu_exact, reps, 2, 1, 1);
            g_rot = 0;
            measure("gpu-batch6", run_gpu_batch6, reps, 1, NEXP, 0);
            g_rot = 0;
            measure("gpu-stream6", run_gpu_stream6, reps, 6, NEXP, 0);
        }

        if (cold_gap_ms > 0.0) {
            g_measure_gap_s = cold_gap_ms / 1000.0;
            g_rot = 0;
            measure("gpu-batch6-cold", run_gpu_batch6, reps, 1, NEXP, 0);
            g_measure_gap_s = 0.0;
        }

        if (gaps && g_ngap > 0) {
            printf("\nidle-gap sweep (cold = first fused trio after the gap, "
                   "warm = the one right after it)\n");
            gap_header();
            double base = -1.0, at10 = -1.0, atcpu = -1.0;
            for (int i = 0; i < g_ngap; i++) {
                const double gap_s = g_gap_ms[i] / 1e3;
                int gr = (int)(1.5 / (gap_s + 0.00045));
                if (gr < 48) gr = 48;
                if (gr > 800) gr = 800;
                const double c50 = measure_gap(gap_s, gr);
                if (i == 0) base = c50;
                if (at10 < 0.0 && base > 0.0 && c50 > base * 1.10) at10 = gap_s * 1e3;
                if (atcpu < 0.0 && g_cpu_ms > 0.0 && c50 > g_cpu_ms) atcpu = gap_s * 1e3;
            }
            if (base > 0.0) {
                const double rel = (g_cpu_ms > 0.0)
                                 ? 100.0 * (1.0 - base / g_cpu_ms) : 0.0;
                printf("\ngap summary  : back-to-back fused p50 %.4f ms", base);
                if (g_cpu_ms > 0.0)
                    printf(" (%.1f%% %s than CPU %.4f ms)",
                           fabs(rel), rel >= 0.0 ? "faster" : "slower",
                           g_cpu_ms);
                printf("\n");
                if (at10 >= 0.0)
                    printf("               cold p50 exceeds +10%% at a >= %.2f ms "
                           "idle gap\n", at10);
                else
                    printf("               cold p50 never exceeds +10%% in this "
                           "sweep\n");
                if (atcpu >= 0.0)
                    printf("               cold p50 exceeds CPU at a >= %.2f ms "
                           "idle gap\n", atcpu);
                else
                    printf("               cold p50 stays below CPU in this "
                           "sweep\n");
            }

            printf("\npost-gap recovery: %d sequential fused trios per burst, "
                   "p50 ms by position\n", BURST_N);
            printf("%7s %7s %7s %7s %7s %7s %7s   |"
                   "%7s %7s %7s %7s %7s %7s\n",
                   "gap ms", "pos0", "pos1", "pos2", "pos3", "pos4", "pos5",
                   "kern0", "kern1", "kern2", "kern3", "kern4", "kern5");
            for (int i = 0; i < g_ngap; i++) {
                const double gap_s = g_gap_ms[i] / 1e3;
                const int br = gap_s >= 0.008 ? 32 : 64;
                measure_burst(gap_s, br);
            }
        }
    }

    printf("\nnotes: p50/p90 are over individual calls; xCPU is the CPU trio on\n"
           "       the same slot bytes. GB/s counts the expert's packed weights\n"
           "       plus its activations. J/expert needs root for RAPL and prints\n"
           "       - otherwise.\n");
    return 0;
}
