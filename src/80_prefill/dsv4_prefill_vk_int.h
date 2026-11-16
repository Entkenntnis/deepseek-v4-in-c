/* SPDX-License-Identifier: Apache-2.0 */
/* Private state shared by the prefill Vulkan backend files. */
#ifndef DSV4_PREFILL_VK_INT_H
#define DSV4_PREFILL_VK_INT_H

#include <stddef.h>
#include <stdint.h>

#include <vulkan/vulkan.h>

#include "dsv4_cache_int.h"
#include "dsv4_prefill_vk.h"

#define DSV4_VK_TILE_N    128u
#define DSV4_VK_SLOT_ALIGN 4096u

typedef struct {
    unsigned char *base;
    size_t         bytes;
    VkBuffer       buf;
    VkDeviceMemory mem;
} DSV4VkRegion;

struct DSV4PrefillVk {
    VkInstance       inst;
    VkPhysicalDevice phys;
    VkDevice         dev;
    VkQueue          queue;
    uint32_t         qfam;
    uint32_t         mem_type;
    VkDeviceSize     import_align;
    PFN_vkGetMemoryHostPointerPropertiesEXT get_host_props;
    VkPhysicalDeviceMemoryProperties mp;

    VkCommandPool   cpool;
    VkCommandBuffer cmd;
    VkFence         fence;

    VkDescriptorSetLayout dsl;
    VkDescriptorPool      dpool;
    VkDescriptorSet       dset;
    VkPipelineLayout      pl;
    VkPipeline            pipe_fp8, pipe_bf16, pipe_fp4;

    VkBuffer       xbuf, ybuf;
    VkDeviceMemory xmem, ymem;
    void          *xmap, *ymap;
    size_t         xbytes, ybytes;

    DSV4VkRegion *regions;
    int           nreg, regcap;

    const DSV4Cfg *cfg;
    DSV4Trunk     *tr;
    DSV4Cache     *cache;
    int            dead;
    char           devname[256];
};

typedef struct {
    uint32_t M, K, N, w_off, s_off;
} DSV4VkPC;

uint32_t dsv4_prefill_vk_pick_mem_type(
    const VkPhysicalDeviceMemoryProperties *mp, uint32_t allowed);
int  dsv4_prefill_vk_open(DSV4PrefillVk *v);

int  dsv4_prefill_vk_region_import(DSV4PrefillVk *v, unsigned char *base,
                                   size_t avail);
DSV4VkRegion *dsv4_prefill_vk_region_for(DSV4PrefillVk *v, const void *ptr,
                                         size_t need);

int  dsv4_prefill_vk_make_mapped(DSV4PrefillVk *v, VkDeviceSize bytes,
                                 VkBuffer *buf, VkDeviceMemory *mem,
                                 void **map);
int  dsv4_prefill_vk_make_pipeline(DSV4PrefillVk *v, const uint32_t *code,
                                   size_t len, VkPipeline *out);

#endif /* DSV4_PREFILL_VK_INT_H */
