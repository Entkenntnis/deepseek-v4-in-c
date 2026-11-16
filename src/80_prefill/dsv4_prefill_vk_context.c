/* SPDX-License-Identifier: Apache-2.0 */
/* Device selection and teardown for the prefill Vulkan backend. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_prefill_vk_int.h"

uint32_t dsv4_prefill_vk_pick_mem_type(
    const VkPhysicalDeviceMemoryProperties *mp, uint32_t allowed)
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

static int ext_supported(VkPhysicalDevice phys, const char *name)
{
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, NULL, &n, NULL)
        != VK_SUCCESS) return 0;
    VkExtensionProperties *e = (VkExtensionProperties *)calloc(n, sizeof *e);
    if (!e) return 0;
    int found = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, NULL, &n, e) == VK_SUCCESS)
        for (uint32_t i = 0; i < n; i++)
            if (!strcmp(e[i].extensionName, name)) { found = 1; break; }
    free(e);
    return found;
}

int dsv4_prefill_vk_open(DSV4PrefillVk *v)
{
    VkApplicationInfo app;
    memset(&app, 0, sizeof app);
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dsv4 prefill";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici;
    memset(&ici, 0, sizeof ici);
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, NULL, &v->inst) != VK_SUCCESS) return -1;

    uint32_t np = 0;
    if (vkEnumeratePhysicalDevices(v->inst, &np, NULL) != VK_SUCCESS || !np)
        return -1;
    VkPhysicalDevice *pd = (VkPhysicalDevice *)calloc(np, sizeof *pd);
    if (!pd) return -1;
    if (vkEnumeratePhysicalDevices(v->inst, &np, pd) != VK_SUCCESS) np = 0;

    int found = 0;
    for (uint32_t i = 0; i < np && !found; i++) {
        if (!ext_supported(pd[i], VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME))
            continue;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd[i], &props);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(pd[i], &mp);
        if (dsv4_prefill_vk_pick_mem_type(&mp, UINT32_MAX) == UINT32_MAX)
            continue;

        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp;
        memset(&hp, 0, sizeof hp);
        hp.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 p2;
        memset(&p2, 0, sizeof p2);
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &hp;
        vkGetPhysicalDeviceProperties2(pd[i], &p2);
        if (!hp.minImportedHostPointerAlignment) continue;

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, NULL);
        VkQueueFamilyProperties *qf =
            (VkQueueFamilyProperties *)calloc(nq, sizeof *qf);
        if (!qf) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, qf);
        uint32_t qfam = UINT32_MAX;
        for (uint32_t q = 0; q < nq; q++)
            if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = q; break; }
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

        v->get_host_props = (PFN_vkGetMemoryHostPointerPropertiesEXT)
            vkGetDeviceProcAddr(v->dev, "vkGetMemoryHostPointerPropertiesEXT");
        if (!v->get_host_props) {
            vkDestroyDevice(v->dev, NULL);
            v->dev = VK_NULL_HANDLE;
            continue;
        }
        v->phys = pd[i];
        v->qfam = qfam;
        v->mp = mp;
        v->import_align = hp.minImportedHostPointerAlignment;
        v->mem_type = UINT32_MAX;
        snprintf(v->devname, sizeof v->devname, "%s", props.deviceName);
        vkGetDeviceQueue(v->dev, qfam, 0, &v->queue);
        found = 1;
    }
    free(pd);
    return found ? 0 : -1;
}

void dsv4_prefill_vk_free(DSV4PrefillVk *v)
{
    if (!v) return;
    if (v->dev) vkDeviceWaitIdle(v->dev);
    if (v->xbuf) vkDestroyBuffer(v->dev, v->xbuf, NULL);
    if (v->xmem) vkFreeMemory(v->dev, v->xmem, NULL);
    if (v->ybuf) vkDestroyBuffer(v->dev, v->ybuf, NULL);
    if (v->ymem) vkFreeMemory(v->dev, v->ymem, NULL);
    for (int i = 0; i < v->nreg; i++) {
        vkDestroyBuffer(v->dev, v->regions[i].buf, NULL);
        vkFreeMemory(v->dev, v->regions[i].mem, NULL);
    }
    free(v->regions);
    if (v->pipe_fp8) vkDestroyPipeline(v->dev, v->pipe_fp8, NULL);
    if (v->pipe_bf16) vkDestroyPipeline(v->dev, v->pipe_bf16, NULL);
    if (v->pipe_fp4) vkDestroyPipeline(v->dev, v->pipe_fp4, NULL);
    if (v->dpool) vkDestroyDescriptorPool(v->dev, v->dpool, NULL);
    if (v->pl) vkDestroyPipelineLayout(v->dev, v->pl, NULL);
    if (v->dsl) vkDestroyDescriptorSetLayout(v->dev, v->dsl, NULL);
    if (v->cpool) vkDestroyCommandPool(v->dev, v->cpool, NULL);
    if (v->fence) vkDestroyFence(v->dev, v->fence, NULL);
    if (v->dev) vkDestroyDevice(v->dev, NULL);
    if (v->inst) vkDestroyInstance(v->inst, NULL);
    free(v);
}
