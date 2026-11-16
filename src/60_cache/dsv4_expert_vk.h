/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_expert_vk.h - one routed FP4 expert's fused gate/up/SwiGLU/down pass.
 *
 * This is the first GPU iteration: one expert, one command buffer, one submit,
 * one fence. It reuses the cache's Vulkan device and the imported slot pools;
 * no batching and no cross-expert queueing yet.
 */
#ifndef DSV4_EXPERT_VK_H
#define DSV4_EXPERT_VK_H

#include "dsv4.h"
#include "dsv4_cache_vk.h"

typedef struct DSV4ExpertVk DSV4ExpertVk;

/* cache_vk must outlive the returned context. Returns 0 on success. */
int  dsv4_expert_vk_init(DSV4ExpertVk **out, DSV4CacheVk *cache_vk,
                         const DSV4Cfg *cfg);

void dsv4_expert_vk_free(DSV4ExpertVk *g);

/* x and out are ordinary CPU buffers. The expert's weights are read from its
 * Vulkan slot slice; out receives the final w2 result. Returns 0 on success. */
int dsv4_expert_vk_forward(DSV4ExpertVk *g, const DSV4ExpertW *e,
                           const float *x, float *out);

#endif /* DSV4_EXPERT_VK_H */
