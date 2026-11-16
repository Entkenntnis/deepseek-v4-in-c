/* SPDX-License-Identifier: Apache-2.0 */
/* Small Vulkan resource helpers: mapped host buffers and compute pipelines. */
#include <string.h>

#include "dsv4_prefill_vk_int.h"

int dsv4_prefill_vk_make_mapped(DSV4PrefillVk *v, VkDeviceSize bytes,
                                VkBuffer *buf, VkDeviceMemory *mem, void **map)
{
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(v->dev, &bi, NULL, buf) != VK_SUCCESS) return -1;
    VkMemoryRequirements rq;
    vkGetBufferMemoryRequirements(v->dev, *buf, &rq);
    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof ai);
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = rq.size;
    ai.memoryTypeIndex = dsv4_prefill_vk_pick_mem_type(&v->mp,
                                                       rq.memoryTypeBits);
    if (ai.memoryTypeIndex == UINT32_MAX) return -1;
    if (vkAllocateMemory(v->dev, &ai, NULL, mem) != VK_SUCCESS) return -1;
    if (vkBindBufferMemory(v->dev, *buf, *mem, 0) != VK_SUCCESS) return -1;
    return vkMapMemory(v->dev, *mem, 0, VK_WHOLE_SIZE, 0, map) == VK_SUCCESS
         ? 0 : -1;
}

int dsv4_prefill_vk_make_pipeline(DSV4PrefillVk *v, const uint32_t *code,
                                  size_t len, VkPipeline *out)
{
    VkShaderModuleCreateInfo smi;
    memset(&smi, 0, sizeof smi);
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = len;
    smi.pCode = code;
    VkShaderModule mod;
    if (vkCreateShaderModule(v->dev, &smi, NULL, &mod) != VK_SUCCESS) return -1;
    VkComputePipelineCreateInfo cpi;
    memset(&cpi, 0, sizeof cpi);
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = v->pl;
    const VkResult r = vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1,
                                                &cpi, NULL, out);
    vkDestroyShaderModule(v->dev, mod, NULL);
    return r == VK_SUCCESS ? 0 : -1;
}
