/* SPDX-License-Identifier: Apache-2.0 */
/* One-time Vulkan setup for prefill: descriptors, pipelines and X/Y tiles. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_prefill_vk_int.h"

#include "gemm_fp8_spv.h"
#include "gemm_bf16_spv.h"
#include "gemm_fp4_spv.h"
#include "swiglu_spv.h"

static int vk_make_sync(DSV4PrefillVk *v)
{
    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof pci);
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = v->qfam;
    if (vkCreateCommandPool(v->dev, &pci, NULL, &v->cpool) != VK_SUCCESS)
        return -1;
    VkCommandBufferAllocateInfo cbi;
    memset(&cbi, 0, sizeof cbi);
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = v->cpool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(v->dev, &cbi, &v->cmd) != VK_SUCCESS)
        return -1;
    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof fci);
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(v->dev, &fci, NULL, &v->fence) != VK_SUCCESS)
        return -1;
    return 0;
}

static int vk_make_descriptors(DSV4PrefillVk *v)
{
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
    if (vkCreateDescriptorSetLayout(v->dev, &dli, NULL, &v->dsl)
        != VK_SUCCESS) return -1;

    VkPushConstantRange pcr;
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(DSV4VkPC);
    VkPipelineLayoutCreateInfo pli;
    memset(&pli, 0, sizeof pli);
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &v->dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(v->dev, &pli, NULL, &v->pl) != VK_SUCCESS)
        return -1;

    VkPushConstantRange pcr_sw;
    pcr_sw.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr_sw.offset = 0;
    pcr_sw.size = sizeof(DSV4VkSwigluPC);
    VkPipelineLayoutCreateInfo pli_sw;
    memset(&pli_sw, 0, sizeof pli_sw);
    pli_sw.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli_sw.setLayoutCount = 1;
    pli_sw.pSetLayouts = &v->dsl;
    pli_sw.pushConstantRangeCount = 1;
    pli_sw.pPushConstantRanges = &pcr_sw;
    if (vkCreatePipelineLayout(v->dev, &pli_sw, NULL, &v->pl_sw)
        != VK_SUCCESS) return -1;

    VkDescriptorPoolSize psz;
    psz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psz.descriptorCount = 20;      /* 5 sets x 4 bindings */
    VkDescriptorPoolCreateInfo dpi;
    memset(&dpi, 0, sizeof dpi);
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 5;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &psz;
    if (vkCreateDescriptorPool(v->dev, &dpi, NULL, &v->dpool)
        != VK_SUCCESS) return -1;
    VkDescriptorSetAllocateInfo ai;
    memset(&ai, 0, sizeof ai);
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = v->dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &v->dsl;
    if (vkAllocateDescriptorSets(v->dev, &ai, &v->dset) != VK_SUCCESS)
        return -1;
    VkDescriptorSetLayout xl[4] = { v->dsl, v->dsl, v->dsl, v->dsl };
    VkDescriptorSetAllocateInfo xai;
    memset(&xai, 0, sizeof xai);
    xai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    xai.descriptorPool = v->dpool;
    xai.descriptorSetCount = 4;
    xai.pSetLayouts = xl;
    if (vkAllocateDescriptorSets(v->dev, &xai, v->xdset) != VK_SUCCESS)
        return -1;
    return 0;
}

static int vk_make_buffers(DSV4PrefillVk *v, const DSV4Cfg *cfg)
{
    uint32_t max_m = (uint32_t)cfg->n_heads * (uint32_t)cfg->head_dim;
    uint32_t max_k = (uint32_t)cfg->hidden;
    if ((uint32_t)cfg->o_groups * (uint32_t)cfg->o_lora > max_k)
        max_k = (uint32_t)cfg->o_groups * (uint32_t)cfg->o_lora;
    if ((uint32_t)cfg->moe_inter * (uint32_t)cfg->n_shared > max_k)
        max_k = (uint32_t)cfg->moe_inter * (uint32_t)cfg->n_shared;
    if ((uint32_t)cfg->moe_inter > max_m)
        max_m = (uint32_t)cfg->moe_inter;
    if ((uint32_t)cfg->index_n_heads * (uint32_t)cfg->index_head_dim > max_m)
        max_m = (uint32_t)cfg->index_n_heads * (uint32_t)cfg->index_head_dim;
    if (max_m < (uint32_t)cfg->hidden) max_m = (uint32_t)cfg->hidden;

    v->xbytes = (size_t)DSV4_VK_TILE_N * max_k * sizeof(float);
    v->ybytes = (size_t)DSV4_VK_TILE_N * max_m * sizeof(float);
    if (dsv4_prefill_vk_make_mapped(v, (VkDeviceSize)v->xbytes, &v->xbuf,
                                    &v->xmem, &v->xmap) != 0) return -1;
    if (dsv4_prefill_vk_make_mapped(v, (VkDeviceSize)v->ybytes, &v->ybuf,
                                    &v->ymem, &v->ymap) != 0) return -1;
    v->expert_mid_bytes = (size_t)DSV4_VK_TILE_N
                        * (size_t)cfg->moe_inter * sizeof(float);
    if (dsv4_prefill_vk_make_mapped(v, (VkDeviceSize)v->expert_mid_bytes,
                                    &v->gbuf, &v->gmem, &v->gmap) != 0 ||
        dsv4_prefill_vk_make_mapped(v, (VkDeviceSize)v->expert_mid_bytes,
                                    &v->ubuf, &v->umem, &v->umap) != 0 ||
        dsv4_prefill_vk_make_mapped(v, (VkDeviceSize)v->expert_mid_bytes,
                                    &v->mbuf, &v->mmem, &v->mmap) != 0)
        return -1;
    return 0;
}

int dsv4_prefill_vk_init(DSV4PrefillVk **out, const DSV4Cfg *cfg,
                         DSV4Trunk *tr, DSV4Cache *cache)
{
    if (!out || !cfg) return -1;
    *out = NULL;
    DSV4PrefillVk *v = (DSV4PrefillVk *)calloc(1, sizeof *v);
    if (!v) return -1;
    v->cfg = cfg;
    v->tr = tr;
    v->cache = cache;

    if (dsv4_prefill_vk_open(v) != 0) goto fail;
    if (vk_make_sync(v) != 0) goto fail;
    if (vk_make_descriptors(v) != 0) goto fail;
    if (dsv4_prefill_vk_make_pipeline(v, gemm_fp8_spv, gemm_fp8_spv_len,
                                      &v->pipe_fp8) ||
        dsv4_prefill_vk_make_pipeline(v, gemm_bf16_spv, gemm_bf16_spv_len,
                                      &v->pipe_bf16) ||
        dsv4_prefill_vk_make_pipeline(v, gemm_fp4_spv, gemm_fp4_spv_len,
                                      &v->pipe_fp4) ||
        dsv4_prefill_vk_make_pipeline_layout(v, swiglu_spv, swiglu_spv_len,
                                             v->pl_sw, &v->pipe_sw))
        goto fail;
    if (vk_make_buffers(v, cfg) != 0) goto fail;

    fprintf(stderr, "prefill: Vulkan GEMM on %s (tile %u tokens, import %llu)\n",
            v->devname, DSV4_VK_TILE_N,
            (unsigned long long)v->import_align);
    *out = v;
    return 0;

fail:
    fprintf(stderr, "prefill: Vulkan init failed, staying on CPU\n");
    dsv4_prefill_vk_free(v);
    return -1;
}
