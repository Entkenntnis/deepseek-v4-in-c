/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_vk.h - expert-slot memory owned by Vulkan.
 *
 * The cache asks this module for one CPU pointer per slot. The pointers are
 * slices of host pages imported into Vulkan with VK_EXT_external_memory_host:
 * O_DIRECT can pread straight into them, the CPU FP4 kernel can read them, and
 * the compute pass binds the same VkBuffer without a copy. The driver's own
 * device mapping cannot be used for O_DIRECT (get_user_pages rejects it), which
 * is why the pages are host-allocated and imported rather than vkMapMemory'd.
 *
 * The module owns the instance/device and the underlying pool allocations. The
 * cache only stores the opaque handle and the per-slot pointers;
 * dsv4_cache_free() releases everything through dsv4_cache_vk_close().
 */
#ifndef DSV4_CACHE_VK_H
#define DSV4_CACHE_VK_H

#include <stdint.h>

#include <vulkan/vulkan.h>

typedef struct DSV4CacheVk DSV4CacheVk;

/* Vulkan is required: returns NULL only on a real init failure. */
DSV4CacheVk *dsv4_cache_vk_open(void);

/* Release every pool, then the device and instance. NULL is a no-op. */
void dsv4_cache_vk_close(DSV4CacheVk *v);

/* Fill out[0..nslot-1] with 4096-aligned CPU pointers into Vulkan memory.
 * pool_idx[i] and pool_off[i] identify the same slice as a (VkBuffer, offset)
 * pair for the compute pass. slot_bytes must be a multiple of 4096. Returns 0
 * on success. On failure the caller still owns v and must close it. */
int dsv4_cache_vk_alloc_slots(DSV4CacheVk *v, int64_t slot_bytes, int nslot,
                              unsigned char **out, int *pool_idx,
                              int64_t *pool_off);

const char *dsv4_cache_vk_device_name(const DSV4CacheVk *v);
uint32_t    dsv4_cache_vk_memory_type(const DSV4CacheVk *v);

/* Handles for the later compute pass. A slot is described by
 * (dsv4_cache_vk_pool_buffer(v, slot->vk_pool), slot->vk_off, slot_bytes). */
VkDevice   dsv4_cache_vk_device(const DSV4CacheVk *v);
VkQueue    dsv4_cache_vk_queue(const DSV4CacheVk *v);
uint32_t   dsv4_cache_vk_queue_family(const DSV4CacheVk *v);
int        dsv4_cache_vk_pool_count(const DSV4CacheVk *v);
VkBuffer   dsv4_cache_vk_pool_buffer(const DSV4CacheVk *v, int pool);
VkDeviceSize dsv4_cache_vk_pool_size(const DSV4CacheVk *v, int pool);

#endif /* DSV4_CACHE_VK_H */
