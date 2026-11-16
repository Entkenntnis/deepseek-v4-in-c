/* SPDX-License-Identifier: Apache-2.0 */
/* Host-memory import: trunk runs, pinned layers and the expert-slot arena. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_prefill_vk_int.h"

static int region_push(DSV4PrefillVk *v, const DSV4VkRegion *r)
{
    if (v->nreg == v->regcap) {
        int cap = v->regcap ? v->regcap * 2 : 16;
        DSV4VkRegion *n = (DSV4VkRegion *)realloc(
            v->regions, (size_t)cap * sizeof *n);
        if (!n) return -1;
        v->regions = n;
        v->regcap = cap;
    }
    v->regions[v->nreg++] = *r;
    return 0;
}

/* Import one existing host allocation. `avail` is the allocation size; the
 * imported range is rounded down to the driver's alignment. */
int dsv4_prefill_vk_region_import(DSV4PrefillVk *v, unsigned char *base,
                                  size_t avail)
{
    if (!v || !base || avail < DSV4_VK_SLOT_ALIGN || v->import_align == 0)
        return -1;
    if ((uintptr_t)base % (uintptr_t)v->import_align) return -1;
    const size_t bytes = avail & ~((size_t)v->import_align - 1u);
    if (bytes < DSV4_VK_SLOT_ALIGN) return -1;
    for (int i = 0; i < v->nreg; i++)
        if (v->regions[i].base == base) return 0;

    DSV4VkRegion r;
    memset(&r, 0, sizeof r);
    r.base = base;
    r.bytes = bytes;

    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = (VkDeviceSize)bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
             | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(v->dev, &bi, NULL, &r.buf) != VK_SUCCESS) return -1;

    VkMemoryRequirements rq;
    vkGetBufferMemoryRequirements(v->dev, r.buf, &rq);

    VkMemoryHostPointerPropertiesEXT hp;
    memset(&hp, 0, sizeof hp);
    hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    if (v->get_host_props(
            v->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
            base, &hp) != VK_SUCCESS) {
        vkDestroyBuffer(v->dev, r.buf, NULL);
        return -1;
    }
    const uint32_t allowed = rq.memoryTypeBits & hp.memoryTypeBits;
    const uint32_t mt = dsv4_prefill_vk_pick_mem_type(&v->mp, allowed);
    if (mt == UINT32_MAX) {
        vkDestroyBuffer(v->dev, r.buf, NULL);
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
    ai.allocationSize = (VkDeviceSize)bytes;
    ai.memoryTypeIndex = mt;
    if (vkAllocateMemory(v->dev, &ai, NULL, &r.mem) != VK_SUCCESS) {
        vkDestroyBuffer(v->dev, r.buf, NULL);
        return -1;
    }
    if (vkBindBufferMemory(v->dev, r.buf, r.mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(v->dev, r.buf, NULL);
        vkFreeMemory(v->dev, r.mem, NULL);
        return -1;
    }
    v->mem_type = mt;
    if (region_push(v, &r) != 0) {
        vkDestroyBuffer(v->dev, r.buf, NULL);
        vkFreeMemory(v->dev, r.mem, NULL);
        return -1;
    }
    return 0;
}

DSV4VkRegion *dsv4_prefill_vk_region_for(DSV4PrefillVk *v, const void *ptr,
                                         size_t need)
{
    if (!v || !ptr) return NULL;
    const unsigned char *p = (const unsigned char *)ptr;
    for (int i = 0; i < v->nreg; i++) {
        DSV4VkRegion *r = &v->regions[i];
        if (p >= r->base && p + need <= r->base + r->bytes) return r;
    }
    if (v->tr) {
        if (v->tr->ring && v->tr->nslot > 0) {
            const size_t b = (size_t)v->tr->nslot * (size_t)v->tr->slot_bytes;
            if (p >= v->tr->ring && p + need <= v->tr->ring + b) {
                if (dsv4_prefill_vk_region_import(v, v->tr->ring, b) == 0)
                    return &v->regions[v->nreg - 1];
                return NULL;
            }
        }
        for (int L = 0; L < v->tr->npin; L++) {
            unsigned char *b = v->tr->pin[L];
            if (!b) continue;
            const size_t sz = (size_t)v->tr->lay[L].nbytes
                            + 2u * DSV4_VK_SLOT_ALIGN;
            if (p >= b && p + need <= b + sz) {
                if (dsv4_prefill_vk_region_import(v, b, sz) == 0)
                    return &v->regions[v->nreg - 1];
                return NULL;
            }
        }
    }
    if (v->cache) {
        if (v->cache->slot_arena) {
            const size_t total = (size_t)v->cache->nslot
                               * (size_t)v->cache->slot_bytes;
            const size_t chunk = (size_t)1 << 30;
            for (size_t off = 0; off < total; off += chunk) {
                size_t n = total - off < chunk ? total - off : chunk;
                n &= ~((size_t)v->import_align - 1u);
                unsigned char *b = v->cache->slot_arena + off;
                if (p >= b && p + need <= b + n) {
                    if (dsv4_prefill_vk_region_import(v, b, n) == 0)
                        return &v->regions[v->nreg - 1];
                    return NULL;
                }
            }
        }
        for (int i = 0; i < v->cache->nslot; i++) {
            unsigned char *b = v->cache->slot[i].bytes;
            if (!b) continue;
            if (p >= b && p + need <= b + (size_t)v->cache->slot_bytes) {
                if (dsv4_prefill_vk_region_import(
                        v, b, (size_t)v->cache->slot_bytes) == 0)
                    return &v->regions[v->nreg - 1];
                return NULL;
            }
        }
    }
    return NULL;
}
