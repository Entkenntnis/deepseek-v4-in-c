/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_weights.h - the checkpoint layouts the kernels read.
 *
 * Included by dsv4.h so consumers still include one header for the model types.
 */
#ifndef DSV4_WEIGHTS_H
#define DSV4_WEIGHTS_H

#include <stdint.h>
#include <stddef.h>

/* ---------------------------------------------------------------- weights ---
 * Every shape below was read out of the released DeepSeek-V4-Flash checkpoint
 * headers and is annotated with both the literal Flash value and the config
 * expression it comes from, so Pro is the same code with different numbers.
 *
 * mHC geometry, from model.py:663-667 -- NOT guessed:
 *     mix_hc = (2 + hc_mult) * hc_mult      Flash: (2+4)*4 = 24
 *     hc_dim = hc_mult * hidden             Flash: 4*4096  = 16384
 * so hc_attn_fn / hc_ffn_fn are [24, 16384] and the head's is [hc_mult, hc_dim].
 */
/* A matrix that carries its own block scales.
 *
 * THE TWO BLOCK GEOMETRIES ARE NOT INTERCHANGEABLE and this struct exists to
 * stop them being confused:
 *   DSV4_WFP8  F8_E4M3 weights, F8_E8M0 scales, 128 x 128 blocks. `rows`/`cols`
 *              are element counts and equal the stored byte counts.
 *   DSV4_WFP4  I8 storage holding TWO 4-bit values per byte, F8_E8M0 scales,
 *              1 x 32 blocks. `cols` is the LOGICAL element count, so the
 *              stored row is cols/2 bytes wide. Getting this backwards reads
 *              half a row and still produces finite numbers.
 * A plain BF16/F32 matrix sets s = NULL and leaves blk_* at 0.
 */
typedef struct {
    const void *w;          /* packed weight bytes, never widened            */
    const void *s;          /* F8_E8M0 block scales, or NULL                 */
    int         wdt;        /* DSV4_WF32 | DSV4_WBF16 | DSV4_WFP8 | DSV4_WFP4 */
    int         rows, cols; /* LOGICAL element dims                          */
    int         blk_r, blk_c;
} DSV4QMat;

/* HCA compressor. Present only where compress_ratio != 0 (model.py:466).
 * coff = 2 when the ratio is 4 (overlapped), else 1; it widens the projection,
 * which is why layer 2's wkv is [1024, 4096] and layer 3's is [512, 4096]. */
typedef struct {
    const float *ape;       /* F32 [ratio, coff*head_dim]                    */
    const float *norm;      /* widened from BF16 [head_dim]                  */
    DSV4QMat     wkv;       /* BF16 [coff*head_dim, hidden]                  */
    DSV4QMat     wgate;     /* BF16 [coff*head_dim, hidden]                  */
} DSV4CompressorW;

/* CSA indexer. Present ONLY where compress_ratio == 4 -- invariant 2, and
 * confirmed in the checkpoint: layer 2 has these, layer 3 does not. */
typedef struct {
    DSV4QMat        wq_b;         /* FP8 [index_n_heads*index_head_dim, q_lora] = [8192,1024] */
    DSV4QMat        weights_proj; /* BF16 [index_n_heads, hidden] = [64, 4096]  */
    DSV4CompressorW comp;         /* its own compressor, narrower than the main one */
} DSV4IndexerW;

typedef struct {
    const float *sink;      /* F32 [n_heads], attention sinks                */
    const float *q_norm;    /* widened from BF16 [q_lora]                    */
    const float *kv_norm;   /* widened from BF16 [head_dim]                  */
    DSV4QMat     wq_a;      /* FP8 [q_lora, hidden]            = [1024, 4096] */
    DSV4QMat     wq_b;      /* FP8 [n_heads*head_dim, q_lora]  = [32768,1024] */
    DSV4QMat     wkv;       /* FP8 [head_dim, hidden]          = [512,  4096] */
    DSV4QMat     wo_a;      /* FP8 [o_lora*o_groups, n_heads*head_dim/o_groups] = [8192,4096] */
    DSV4QMat     wo_b;      /* FP8 [hidden, o_lora*o_groups]   = [4096, 8192] */
    DSV4CompressorW comp;   /* valid when has_comp                            */
    DSV4IndexerW    idx;    /* valid when has_idx                             */
} DSV4AttnW;

/* The wo_a matrix is stored as o_groups stacked slices. Decode runs one GEMV
 * per slice and prefill runs one GEMM per slice, so the offset arithmetic is
 * shared here rather than repeated at both call sites. */
static inline DSV4QMat dsv4_wo_a_group(const DSV4AttnW *a, int group,
                                       int o_groups, int o_lora,
                                       int heads_dim)
{
    const int gw = heads_dim / o_groups;
    const int gr = o_lora;
    DSV4QMat s = a->wo_a;
    s.rows = gr;
    s.cols = gw;

    const size_t esz = (s.wdt == DSV4_WBF16) ? 2u : 1u;
    const size_t row_bytes = (s.wdt == DSV4_WFP4) ? (size_t)gw / 2u
                                                  : (size_t)gw * esz;
    s.w = (const unsigned char *)a->wo_a.w + (size_t)group * (size_t)gr
        * row_bytes;
    if (s.s) {
        const size_t sc_row = (size_t)((gw + s.blk_c - 1) / s.blk_c);
        s.s = (const unsigned char *)a->wo_a.s
            + (size_t)group * (size_t)(gr / (s.blk_r ? s.blk_r : 1))
            * sc_row;
    }
    return s;
}

/* One routed expert, as it sits in the cache: still packed FP4, never widened.
 * Measured: 13,369,344 B on disk per expert for Flash. Widening to f32 would be
 * 8x that, and a token touches 6 per layer across 43 layers. */
typedef struct {
    DSV4QMat w1, w3;        /* FP4 [moe_inter, hidden]                       */
    DSV4QMat w2;            /* FP4 [hidden, moe_inter]                       */
} DSV4ExpertW;

typedef struct {
    DSV4QMat       gate;    /* BF16 [n_experts, hidden]; scores ALWAYS computed */
    const float   *bias;    /* F32 [n_experts] on scored layers, else NULL   */
    const int64_t *tid2eid; /* I64 [vocab, topk] on hash layers, else NULL   */
    DSV4ExpertW    shared;  /* the one shared expert, FP8 not FP4, resident  */
    /* Routed experts are streamed per token and are deliberately absent here. */
} DSV4MoeW;

/* mHC mixing parameters. Two sets per block, one for attention, one for FFN. */
typedef struct {
    const float *fn;        /* F32 [mix_hc, hc_dim]  = [24, 16384]           */
    const float *base;      /* F32 [mix_hc]          = [24]                  */
    const float *scale;     /* F32 [3] on a block, [1] on the head           */
} DSV4HcW;

typedef struct {
    const float *attn_norm; /* widened from BF16 [hidden]                    */
    const float *ffn_norm;  /* widened from BF16 [hidden]                    */
    DSV4AttnW    attn;
    DSV4MoeW     moe;
    DSV4HcW      hc_attn, hc_ffn;
    int          layer;
    int          compress_ratio;
    int          has_comp;      /* compress_ratio != 0                       */
    int          has_idx;       /* compress_ratio == 4                       */
} DSV4LayerW;

/* Model-level weights. All six live outside layers.* and mtp.*:
 * embed.weight (shard 1) and norm/head/hc_head_* (shard 45). Both embed and
 * head are BF16 [vocab, hidden] = 0.986 GB each; tie_word_embeddings is false,
 * so they are genuinely two tensors. */
typedef struct {
    const void  *embed;     /* BF16 [vocab, hidden]                          */
    const void  *head;      /* BF16 [vocab, hidden]                          */
    const float *norm;      /* widened from BF16 [hidden]                    */
    DSV4HcW      hc_head;   /* fn is [hc_mult, hc_dim], scale is [1]         */
    int          wdt;
} DSV4ModelW;

/* Gather one embedding row, widening if the table is BF16. The table is INDEXED,
 * not multiplied, so it cannot go through a matmul dispatch: a memcpy with a
 * float stride would read half a row of the wrong values. */
static inline void dsv4_embed_row(float *dst, const void *table, int wdt,
                                  int64_t row, int hidden)
{
    if (wdt == DSV4_WBF16) {
        const uint16_t *p = (const uint16_t *)table + row * hidden;
        for (int i = 0; i < hidden; i++) {
            union { uint32_t u; float f; } v;
            v.u = (uint32_t)p[i] << 16;
            dst[i] = v.f;
        }
    } else {
        const float *p = (const float *)table + row * hidden;
        for (int i = 0; i < hidden; i++) dst[i] = p[i];
    }
}


#endif /* DSV4_WEIGHTS_H */
