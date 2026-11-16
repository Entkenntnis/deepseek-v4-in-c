/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_vk.c - allocate the expert slots as Vulkan-owned memory.
 *
 * WHY HOST-POINTER IMPORT
 *   The obvious implementation is vkAllocateMemory() from a host-visible
 *   DEVICE_LOCAL heap and vkMapMemory() for the CPU. That works for a GPU
 *   kernel, but O_DIRECT pread() into the resulting mapping fails with EFAULT:
 *   the driver maps that allocation with VM_PFNMAP/VM_IO pages, and the kernel
 *   refuses to pin them for direct I/O. The cache depends on O_DIRECT landing
 *   straight in the slot.
 *
 *   So the pages come from posix_memalign() and the VkDeviceMemory is created
 *   with VK_EXT_external_memory_host. Vulkan owns the device memory object and
 *   the VkBuffer; the CPU owns the pages that back it. pread() writes directly
 *   into those pages, the CPU FP4 kernel reads the same bytes, and the GPU
 *   binds the same VkBuffer. No staging and no copy.
 *
 * MEMORY TYPE
 *   Prefer DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, then
 *   HOST_VISIBLE | HOST_COHERENT | HOST_CACHED. HOST_CACHED is not optional:
 *   the CPU FP4 kernel reads these bytes. HOST_COHERENT lets pread writes
 *   become GPU-visible without an explicit flush. Vulkan is required; there is
 *   no host-memory fallback.
 *
 * POOLS
 *   One VkBuffer/VkDeviceMemory per slot would spend the driver's allocation
 *   table. Instead this module imports a small number of large chunks (1 GiB
 *   target, halved automatically if the driver refuses) and slices each one
 *   into 4096-aligned slots. LRU reuse writes into the same slice.
 */
#define _GNU_SOURCE

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_cache_vk.h"

#define DSV4_SLOT_ALIGN  4096u
#define DSV4_POOL_TARGET ((size_t)1 << 30)   /* 1 GiB, halved on failure */

typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    void          *raw;       /* posix_memalign allocation, owns the pages    */
    unsigned char *base;      /* raw rounded up to the import alignment       */
    size_t         bytes;     /* usable aligned bytes                         */
} DSV4VkPool;

struct DSV4CacheVk {
    VkInstance       inst;
    VkPhysicalDevice phys;
    VkDevice         dev;
    VkQueue          queue;
    uint32_t         qfam;
    uint32_t         mem_type;
    VkDeviceSize     import_align;
    PFN_vkGetMemoryHostPointerPropertiesEXT get_host_ptr_props;
    VkPhysicalDeviceMemoryProperties mp;
    uint32_t         ts_valid_bits;   /* timestamp bits of the compute queue  */
    double           ts_period_ns;    /* nanoseconds per timestamp tick       */

    DSV4VkPool *pool;
    int         npool;
    int         pool_cap;
    char        devname[256];
};

static uint32_t pick_mem_type(const VkPhysicalDeviceMemoryProperties *mp,
                              uint32_t allowed)
{
    static const VkMemoryPropertyFlags first =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
      | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
      | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    static const VkMemoryPropertyFlags second =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
      | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    for (int pass = 0; pass < 2; pass++) {
        const VkMemoryPropertyFlags want = pass == 0 ? first : second;
        for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
            if (!(allowed & (1u << i))) continue;
            if ((mp->memoryTypes[i].propertyFlags & want) == want) return i;
        }
    }
    return UINT32_MAX;
}

static int device_ext_supported(VkPhysicalDevice phys, const char *name)
{
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, NULL, &n, NULL)
        != VK_SUCCESS)
        return 0;
    VkExtensionProperties *e =
        (VkExtensionProperties *)calloc(n, sizeof *e);
    if (!e) return 0;
    vkEnumerateDeviceExtensionProperties(phys, NULL, &n, e);
    int found = 0;
    for (uint32_t i = 0; i < n; i++)
        if (strcmp(e[i].extensionName, name) == 0) { found = 1; break; }
    free(e);
    return found;
}

static void pool_destroy(DSV4CacheVk *v, DSV4VkPool *p)
{
    if (!v || !p) return;
    /* The buffer must be unbound before the imported memory is freed. */
    if (p->buf) vkDestroyBuffer(v->dev, p->buf, NULL);
    if (p->mem) vkFreeMemory(v->dev, p->mem, NULL);
    free(p->raw);
    memset(p, 0, sizeof *p);
}

static int pool_alloc(DSV4CacheVk *v, size_t want, DSV4VkPool *out)
{
    DSV4VkPool p;
    memset(&p, 0, sizeof p);

    VkDeviceSize align = v->import_align;
    if (align < DSV4_SLOT_ALIGN) align = DSV4_SLOT_ALIGN;
    const size_t want_round =
        ((size_t)want + (size_t)align - 1u) & ~((size_t)align - 1u);

    void *raw = NULL;
    if (posix_memalign(&raw, (size_t)align,
                       want_round + (size_t)align) != 0)
        return -1;
    unsigned char *base = (unsigned char *)
        (((uintptr_t)raw + (uintptr_t)align - 1u)
         & ~((uintptr_t)align - 1u));
    if ((size_t)(base - (unsigned char *)raw) + want_round
        > want_round + (size_t)align) {
        free(raw);
        return -1;
    }

    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = (VkDeviceSize)want_round;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
             | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(v->dev, &bi, NULL, &p.buf) != VK_SUCCESS) {
        free(raw);
        return -1;
    }

    VkMemoryRequirements rq;
    vkGetBufferMemoryRequirements(v->dev, p.buf, &rq);

    VkMemoryHostPointerPropertiesEXT hp;
    memset(&hp, 0, sizeof hp);
    hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    if (v->get_host_ptr_props(
            v->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
            base, &hp) != VK_SUCCESS) {
        vkDestroyBuffer(v->dev, p.buf, NULL);
        free(raw);
        return -1;
    }

    const uint32_t allowed = rq.memoryTypeBits & hp.memoryTypeBits;
    const uint32_t mt = pick_mem_type(&v->mp, allowed);
    if (mt == UINT32_MAX) {
        vkDestroyBuffer(v->dev, p.buf, NULL);
        free(raw);
        return -1;
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
    ai.allocationSize = (VkDeviceSize)want_round;
    ai.memoryTypeIndex = mt;
    if (vkAllocateMemory(v->dev, &ai, NULL, &p.mem) != VK_SUCCESS) {
        vkDestroyBuffer(v->dev, p.buf, NULL);
        free(raw);
        return -1;
    }
    if (vkBindBufferMemory(v->dev, p.buf, p.mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(v->dev, p.buf, NULL);
        vkFreeMemory(v->dev, p.mem, NULL);
        free(raw);
        return -1;
    }

    p.raw = raw;
    p.base = base;
    p.bytes = want;
    v->mem_type = mt;
    *out = p;
    return 0;
}

static int pool_push(DSV4CacheVk *v, const DSV4VkPool *p)
{
    if (v->npool == v->pool_cap) {
        int cap = v->pool_cap ? v->pool_cap * 2 : 8;
        DSV4VkPool *n = (DSV4VkPool *)realloc(v->pool,
                                              (size_t)cap * sizeof *n);
        if (!n) return -1;
        v->pool = n;
        v->pool_cap = cap;
    }
    v->pool[v->npool++] = *p;
    return 0;
}

DSV4CacheVk *dsv4_cache_vk_open(void)
{
    DSV4CacheVk *v = (DSV4CacheVk *)calloc(1, sizeof *v);
    if (!v) return NULL;
    v->mem_type = UINT32_MAX;

    VkApplicationInfo app;
    memset(&app, 0, sizeof app);
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dsv4 expert cache";
    app.applicationVersion = 1;
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici;
    memset(&ici, 0, sizeof ici);
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, NULL, &v->inst) != VK_SUCCESS) {
        free(v);
        return NULL;
    }

    uint32_t np = 0;
    if (vkEnumeratePhysicalDevices(v->inst, &np, NULL) != VK_SUCCESS || np == 0) {
        vkDestroyInstance(v->inst, NULL);
        free(v);
        return NULL;
    }
    VkPhysicalDevice *pd = (VkPhysicalDevice *)calloc(np, sizeof *pd);
    if (!pd) {
        vkDestroyInstance(v->inst, NULL);
        free(v);
        return NULL;
    }
    if (vkEnumeratePhysicalDevices(v->inst, &np, pd) != VK_SUCCESS) np = 0;

    int found = 0;
    for (uint32_t i = 0; i < np && !found; i++) {
        if (!device_ext_supported(pd[i],
                                  VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME))
            continue;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd[i], &props);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(pd[i], &mp);
        if (pick_mem_type(&mp, UINT32_MAX) == UINT32_MAX) continue;

        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp;
        memset(&hp, 0, sizeof hp);
        hp.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 p2;
        memset(&p2, 0, sizeof p2);
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &hp;
        vkGetPhysicalDeviceProperties2(pd[i], &p2);
        if (hp.minImportedHostPointerAlignment == 0) continue;

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, NULL);
        VkQueueFamilyProperties *qf =
            (VkQueueFamilyProperties *)calloc(nq, sizeof *qf);
        if (!qf) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, qf);
        uint32_t qfam = UINT32_MAX;
        uint32_t ts_bits = 0;
        for (uint32_t q = 0; q < nq; q++)
            if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                qfam = q;
                ts_bits = qf[q].timestampValidBits;
                break;
            }
        free(qf);
        if (qfam == UINT32_MAX) continue;

        const char *exts[] = { VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME };
        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci;
        memset(&qci, 0, sizeof qci);
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        VkDeviceCreateInfo dci;
        memset(&dci, 0, sizeof dci);
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = exts;
        if (vkCreateDevice(pd[i], &dci, NULL, &v->dev) != VK_SUCCESS) continue;

        PFN_vkGetMemoryHostPointerPropertiesEXT fn =
            (PFN_vkGetMemoryHostPointerPropertiesEXT)
            vkGetDeviceProcAddr(v->dev, "vkGetMemoryHostPointerPropertiesEXT");
        if (!fn) {
            vkDestroyDevice(v->dev, NULL);
            v->dev = VK_NULL_HANDLE;
            continue;
        }

        v->phys = pd[i];
        v->qfam = qfam;
        v->mp = mp;
        v->ts_valid_bits = ts_bits;
        v->ts_period_ns = (double)props.limits.timestampPeriod;
        v->import_align = hp.minImportedHostPointerAlignment;
        v->get_host_ptr_props = fn;
        snprintf(v->devname, sizeof v->devname, "%s", props.deviceName);
        vkGetDeviceQueue(v->dev, qfam, 0, &v->queue);
        found = 1;
    }
    free(pd);

    if (!found) {
        if (v->dev) vkDestroyDevice(v->dev, NULL);
        vkDestroyInstance(v->inst, NULL);
        free(v);
        return NULL;
    }
    return v;
}

void dsv4_cache_vk_close(DSV4CacheVk *v)
{
    if (!v) return;
    if (v->dev) {
        vkDeviceWaitIdle(v->dev);
        for (int i = 0; i < v->npool; i++) pool_destroy(v, &v->pool[i]);
        vkDestroyDevice(v->dev, NULL);
    }
    free(v->pool);
    if (v->inst) vkDestroyInstance(v->inst, NULL);
    free(v);
}

int dsv4_cache_vk_alloc_slots(DSV4CacheVk *v, int64_t slot_bytes, int nslot,
                              unsigned char **out, int *pool_idx,
                              int64_t *pool_off)
{
    if (!v || !out || !pool_idx || !pool_off || nslot <= 0) return -1;
    if (slot_bytes <= 0 || (slot_bytes & (DSV4_SLOT_ALIGN - 1u)) != 0) {
        fprintf(stderr, "dsv4_cache_vk: slot_bytes %" PRId64
                        " is not a multiple of %u\n",
                slot_bytes, DSV4_SLOT_ALIGN);
        return -1;
    }

    size_t per_cap = DSV4_POOL_TARGET / (size_t)slot_bytes;
    if (per_cap < 1) per_cap = 1;

    int made = 0;
    while (made < nslot) {
        int per = (int)per_cap;
        if (per > nslot - made) per = nslot - made;
        if (per > INT32_MAX) per = INT32_MAX;

        DSV4VkPool p;
        size_t want = 0;
        while (per > 0) {
            want = (size_t)per * (size_t)slot_bytes;
            if (want / (size_t)slot_bytes == (size_t)per
                && pool_alloc(v, want, &p) == 0)
                break;
            per >>= 1;
        }
        if (per <= 0) return -1;

        if (pool_push(v, &p) != 0) {
            pool_destroy(v, &p);
            return -1;
        }
        const int pidx = v->npool - 1;
        for (int i = 0; i < per; i++) {
            out[made + i] = p.base + (size_t)i * (size_t)slot_bytes;
            pool_idx[made + i] = pidx;
            pool_off[made + i] = (int64_t)i * slot_bytes;
        }
        made += per;
        per_cap = (size_t)per;
    }

    fprintf(stderr, "dsv4_cache: %d expert slots in Vulkan %s memory "
                    "(%d pool%s, type %u, host import %" PRIu64 " B)\n",
            nslot, v->devname, v->npool, v->npool == 1 ? "" : "s",
            v->mem_type, (uint64_t)v->import_align);
    return 0;
}

int dsv4_cache_vk_alloc_mapped(DSV4CacheVk *v, uint64_t bytes,
                               DSV4VkMapped *out)
{
    if (!v || !out || bytes == 0 || bytes > SIZE_MAX) return -1;
    DSV4VkPool p;
    if (pool_alloc(v, (size_t)bytes, &p) != 0) return -1;
    memset(out, 0, sizeof *out);
    out->buf = p.buf;
    out->mem = p.mem;
    out->cpu = p.base;
    out->bytes = (uint64_t)p.bytes;
    out->raw = p.raw;
    return 0;
}

void dsv4_cache_vk_free_mapped(DSV4CacheVk *v, DSV4VkMapped *m)
{
    if (!v || !m) return;
    DSV4VkPool p;
    memset(&p, 0, sizeof p);
    p.buf = m->buf;
    p.mem = m->mem;
    p.raw = m->raw;
    pool_destroy(v, &p);
    memset(m, 0, sizeof *m);
}

const char *dsv4_cache_vk_device_name(const DSV4CacheVk *v)
{
    return v ? v->devname : "";
}

uint32_t dsv4_cache_vk_memory_type(const DSV4CacheVk *v)
{
    return v ? v->mem_type : UINT32_MAX;
}

VkDevice dsv4_cache_vk_device(const DSV4CacheVk *v)
{
    return v ? v->dev : VK_NULL_HANDLE;
}

VkQueue dsv4_cache_vk_queue(const DSV4CacheVk *v)
{
    return v ? v->queue : VK_NULL_HANDLE;
}

uint32_t dsv4_cache_vk_queue_family(const DSV4CacheVk *v)
{
    return v ? v->qfam : UINT32_MAX;
}

int dsv4_cache_vk_has_timestamps(const DSV4CacheVk *v)
{
    return v && v->ts_valid_bits > 0 && v->ts_period_ns > 0.0;
}

uint32_t dsv4_cache_vk_timestamp_valid_bits(const DSV4CacheVk *v)
{
    return v ? v->ts_valid_bits : 0;
}

double dsv4_cache_vk_timestamp_period_ns(const DSV4CacheVk *v)
{
    return v ? v->ts_period_ns : 0.0;
}

int dsv4_cache_vk_pool_count(const DSV4CacheVk *v)
{
    return v ? v->npool : 0;
}

VkBuffer dsv4_cache_vk_pool_buffer(const DSV4CacheVk *v, int pool)
{
    if (!v || pool < 0 || pool >= v->npool) return VK_NULL_HANDLE;
    return v->pool[pool].buf;
}

VkDeviceSize dsv4_cache_vk_pool_size(const DSV4CacheVk *v, int pool)
{
    if (!v || pool < 0 || pool >= v->npool) return 0;
    return (VkDeviceSize)v->pool[pool].bytes;
}
