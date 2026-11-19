/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cfg.c - the DeepSeek-V4-Flash geometry, as constants.
 *
 * These are the values from the released Flash config.json. They are compiled
 * in because this engine implements exactly this architecture; see the header
 * for why a runtime reader was removed rather than simplified.
 */
#include <string.h>

#include "dsv4.h"
#include "dsv4_cfg.h"

/* 43 decoder layers + 3 trailing MTP entries. Only the first 43 are decoder
 * layers; the zeros after them are load-bearing in the file format and are
 * kept here so the table has the same shape as the checkpoint. */
#define DSV4_FLASH_N_LAYERS 43

static const int flash_compress_ratios[] = {
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 0, 0, 0,
};

void dsv4_cfg_init(DSV4Cfg *c)
{
    memset(c, 0, sizeof *c);

    c->hidden     = 4096;
    c->n_layers   = DSV4_FLASH_N_LAYERS;
    c->vocab      = 129280;
    c->rms_eps    = 1e-6f;
    c->max_pos    = 1048576;

    /* Attention: MLA-shaped, one kv head, q and o both low-rank. */
    c->n_heads        = 64;
    c->n_kv_heads     = 1;
    c->head_dim       = 512;
    c->q_lora         = 1024;
    c->o_lora         = 1024;
    c->o_groups       = 8;
    c->qk_rope        = 64;
    c->sliding_window = 128;

    /* CSA indexer. */
    c->index_n_heads  = 64;
    c->index_head_dim = 128;
    c->index_topk     = 512;

    /* HCA compressor. */
    c->n_compress         = DSV4_FLASH_N_LAYERS;
    c->compress_ratios    = flash_compress_ratios;
    c->compress_rope_theta = 160000.0f;

    /* mHC. */
    c->hc_mult           = 4;
    c->hc_sinkhorn_iters = 20;
    c->hc_eps            = 1e-6f;

    /* MoE. */
    c->n_experts    = 256;
    c->topk         = 6;
    c->n_shared     = 1;
    c->moe_inter    = 2048;
    c->routed_scale = 1.5f;
    c->norm_topk    = 1;
    c->scoring_func = DSV4_SCORE_SQRTSOFTPLUS;
    c->topk_method  = DSV4_TOPK_NOAUX_TC;

    c->swiglu_limit = 10.0f;

    /* RoPE with YaRN. */
    c->rope_theta     = 10000.0f;
    c->yarn_factor    = 16.0f;
    c->yarn_beta_fast = 32.0f;
    c->yarn_beta_slow = 1.0f;
    c->yarn_orig_ctx  = 65536;

    /* FP8 weight blocks. FP4 experts use their own 1x32 blocks and do not read
     * these fields. */
    c->wblock_m = 128;
    c->wblock_n = 128;

    c->num_hash_layers   = 3;
    c->num_nextn_predict = 1;
}
