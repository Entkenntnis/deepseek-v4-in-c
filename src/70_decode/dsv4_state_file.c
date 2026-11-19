/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_state_file.c - the prompt-prefix KV cache file.
 *
 * FORMAT (all integers and floats little-endian):
 *
 *   char     magic[8]        "DSV4KVC\0"
 *   uint32   version         1
 *   uint32   reserved        0
 *   uint64   cfg_hash        dsv4_cfg_hash()
 *   uint64   payload_hash    FNV-1a over the payload that follows
 *   uint64   payload_bytes
 *   int32    n_layers, hidden, head_dim, sliding_window, qk_rope, vocab
 *   int32    saved_max_pos, n_ids
 *   char     model_tag[128]
 *   int32    compress_ratios[n_layers]
 *   int32    n_compressed[n_layers]
 *   int32    ids[n_ids]
 *   per layer:
 *     float window[sliding_window * head_dim]
 *     float compressed[n_compressed * head_dim]      (ratio != 0)
 *     float comp_kv[cn]                              (ratio != 0)
 *     float comp_score[cn]                           (ratio != 0)
 *
 * WHY THE WHOLE WINDOW RING IS SAVED
 *   The sliding window is a ring indexed by absolute position. Saving only the
 *   "last N tokens" and rebuilding the ring would have to reproduce the exact
 *   modulo placement the model used. Copying the whole ring is both exact and
 *   simpler; it is ~11 MB for Flash.
 *
 * WHY COMPRESSED ROWS AND comp_* ARE SAVED
 *   The compressed rows are append-only and live after the ring in kv_cache.
 *   comp_kv/comp_score are the in-progress pooling window, including the
 *   -INFINITY score slots. Dropping either would make the first new token
 *   attend to a plausible but different past.
 */
#include "dsv4_state_file.h"
#include "dsv4_state_file_int.h"

static uint64_t fnv_u64(uint64_t h, uint64_t v)
{
    for (int i = 0; i < 8; i++) {
        h ^= (v >> (8 * i)) & 0xffu;
        h *= KV_FNV_PRIME;
    }
    return h;
}

static uint64_t fnv_i32(uint64_t h, int32_t v)
{
    return fnv_u64(h, (uint64_t)(uint32_t)v);
}

static uint64_t fnv_f32(uint64_t h, float f)
{
    union { float f; uint32_t u; } v;
    v.f = f;
    return fnv_u64(h, (uint64_t)v.u);
}

uint64_t dsv4_cfg_hash(const DSV4Cfg *c)
{
    uint64_t h = KV_FNV_OFFSET;
    if (!c) return h;

    h = fnv_i32(h, c->hidden);
    h = fnv_i32(h, c->n_layers);
    h = fnv_i32(h, c->vocab);
    h = fnv_f32(h, c->rms_eps);
    h = fnv_i32(h, c->max_pos);

    h = fnv_i32(h, c->n_heads);
    h = fnv_i32(h, c->n_kv_heads);
    h = fnv_i32(h, c->head_dim);
    h = fnv_i32(h, c->q_lora);
    h = fnv_i32(h, c->o_lora);
    h = fnv_i32(h, c->o_groups);
    h = fnv_i32(h, c->qk_rope);
    h = fnv_i32(h, c->sliding_window);

    h = fnv_i32(h, c->index_n_heads);
    h = fnv_i32(h, c->index_head_dim);
    h = fnv_i32(h, c->index_topk);

    h = fnv_i32(h, c->n_compress);
    for (int i = 0; i < c->n_compress && c->compress_ratios; i++)
        h = fnv_i32(h, c->compress_ratios[i]);
    h = fnv_f32(h, c->compress_rope_theta);

    h = fnv_i32(h, c->hc_mult);
    h = fnv_i32(h, c->hc_sinkhorn_iters);
    h = fnv_f32(h, c->hc_eps);

    h = fnv_i32(h, c->n_experts);
    h = fnv_i32(h, c->topk);
    h = fnv_i32(h, c->n_shared);
    h = fnv_i32(h, c->moe_inter);
    h = fnv_f32(h, c->routed_scale);
    h = fnv_i32(h, c->norm_topk);
    h = fnv_i32(h, c->scoring_func);
    h = fnv_i32(h, c->topk_method);

    h = fnv_f32(h, c->swiglu_limit);
    h = fnv_f32(h, c->rope_theta);
    h = fnv_f32(h, c->yarn_factor);
    h = fnv_f32(h, c->yarn_beta_fast);
    h = fnv_f32(h, c->yarn_beta_slow);
    h = fnv_i32(h, c->yarn_orig_ctx);
    h = fnv_i32(h, c->wblock_m);
    h = fnv_i32(h, c->wblock_n);
    h = fnv_i32(h, c->num_hash_layers);
    h = fnv_i32(h, c->num_nextn_predict);
    return h;
}
