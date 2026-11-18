/* SPDX-License-Identifier: Apache-2.0 */
/* test_kv.c - round-trip and rejection gates for the prompt-prefix cache. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4.h"
#include "dsv4_layer.h"
#include "dsv4_state_file.h"

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

#define NL   2
#define HID  8
#define HDIM 4
#define WIN  4
#define NIDS 8

static int g_cr[NL] = { 0, 4 };

static void make_cfg(DSV4Cfg *c, int hidden)
{
    memset(c, 0, sizeof *c);
    c->hidden = hidden;
    c->n_layers = NL;
    c->vocab = 16;
    c->rms_eps = 1e-6f;
    c->max_pos = 64;
    c->n_heads = 1;
    c->n_kv_heads = 1;
    c->head_dim = HDIM;
    c->q_lora = 4;
    c->o_lora = 4;
    c->o_groups = 1;
    c->qk_rope = 2;
    c->sliding_window = WIN;
    c->index_n_heads = 1;
    c->index_head_dim = 2;
    c->index_topk = 2;
    c->n_compress = NL;
    c->compress_ratios = g_cr;
    c->compress_rope_theta = 160000.0f;
    c->hc_mult = 1;
    c->hc_sinkhorn_iters = 1;
    c->hc_eps = 1e-6f;
    c->n_experts = 2;
    c->topk = 1;
    c->n_shared = 1;
    c->moe_inter = 4;
    c->routed_scale = 1.0f;
    c->norm_topk = 1;
    c->scoring_func = DSV4_SCORE_SQRTSOFTPLUS;
    c->topk_method = DSV4_TOPK_NOAUX_TC;
    c->swiglu_limit = 10.0f;
    c->rope_theta = 10000.0f;
    c->yarn_factor = 16.0f;
    c->yarn_beta_fast = 32.0f;
    c->yarn_beta_slow = 1.0f;
    c->yarn_orig_ctx = 65536;
    c->wblock_m = 128;
    c->wblock_n = 128;
    c->num_hash_layers = 0;
    c->num_nextn_predict = 1;
}

static size_t cn_for(const DSV4Cfg *c, int layer)
{
    const int ratio = dsv4_compress_ratio(c, layer);
    if (ratio <= 0) return 0;
    const int coff = ratio == 4 ? 2 : 1;
    return (size_t)coff * (size_t)ratio * (size_t)coff * (size_t)c->head_dim;
}

static void fill_state(const DSV4Cfg *c, DSV4LayerState *lst)
{
    const size_t wn = (size_t)c->sliding_window * (size_t)c->head_dim;
    for (int L = 0; L < NL; L++) {
        for (size_t i = 0; i < wn; i++)
            lst[L].kv_cache[i] = (float)(100 * L) + (float)i + 0.25f;
        const int ratio = dsv4_compress_ratio(c, L);
        if (ratio <= 0) continue;
        lst[L].n_compressed = NIDS / ratio;
        const size_t rows = (size_t)lst[L].n_compressed * (size_t)c->head_dim;
        for (size_t i = 0; i < rows; i++)
            lst[L].kv_cache[wn + i] = (float)(1000 * L) + (float)i + 0.5f;
        const size_t cn = cn_for(c, L);
        for (size_t i = 0; i < cn; i++) {
            lst[L].comp_kv[i] = (float)(2000 * L) + (float)i + 0.75f;
            lst[L].comp_score[i] = (float)(3000 * L) + (float)i - 8.0f;
        }
    }
}

static int same_state(const DSV4Cfg *c, const DSV4LayerState *a,
                      const DSV4LayerState *b)
{
    const size_t wn = (size_t)c->sliding_window * (size_t)c->head_dim;
    for (int L = 0; L < NL; L++) {
        if (a[L].n_compressed != b[L].n_compressed) return 0;
        if (memcmp(a[L].kv_cache, b[L].kv_cache, wn * sizeof(float)) != 0)
            return 0;
        const int ratio = dsv4_compress_ratio(c, L);
        if (ratio > 0) {
            const size_t rows = (size_t)a[L].n_compressed *
                                (size_t)c->head_dim;
            const size_t cn = cn_for(c, L);
            if (memcmp(a[L].kv_cache + wn, b[L].kv_cache + wn,
                       rows * sizeof(float)) != 0) return 0;
            if (memcmp(a[L].comp_kv, b[L].comp_kv,
                       cn * sizeof(float)) != 0) return 0;
            if (memcmp(a[L].comp_score, b[L].comp_score,
                       cn * sizeof(float)) != 0) return 0;
        }
    }
    return 1;
}

static int init_all(const DSV4Cfg *c, DSV4LayerState *lst)
{
    for (int L = 0; L < NL; L++)
        if (dsv4_state_init(&lst[L], c, L, c->max_pos) != 0) return -1;
    return 0;
}

static void free_all(DSV4LayerState *lst)
{
    for (int L = 0; L < NL; L++) dsv4_state_free(&lst[L]);
}

int main(void)
{
    printf("DeepSeek-V4 prompt-prefix KV cache gate\n");

    DSV4Cfg c;
    make_cfg(&c, HID);
    DSV4LayerState src[NL], dst[NL];
    memset(src, 0, sizeof src);
    memset(dst, 0, sizeof dst);
    CHECK(init_all(&c, src) == 0, "source state allocation failed");
    CHECK(init_all(&c, dst) == 0, "destination state allocation failed");

    int32_t ids[NIDS], bad_ids[NIDS];
    for (int i = 0; i < NIDS; i++) { ids[i] = i + 1; bad_ids[i] = i + 2; }
    fill_state(&c, src);

    char path[] = "/tmp/dsv4_kv_test_XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp failed");
    if (fd >= 0) close(fd);

    printf("\n-- GATE 1  save/load preserves every persistent float --\n");
    CHECK(dsv4_state_file_save(path, "tiny-model", &c, src, ids, NIDS) == 0,
          "save failed");
    int n = 0;
    CHECK(dsv4_state_file_load(path, "tiny-model", &c, dst, c.max_pos,
                               ids, NIDS, &n, ids, NIDS) == 0, "load failed");
    CHECK(n == NIDS, "loaded %d ids, expected %d", n, NIDS);
    CHECK(same_state(&c, src, dst), "state changed across the round trip");
    printf("  ok    window, compressed rows, compressor state and ids match\n");

    printf("\n-- GATE 2  a different prefix is rejected atomically --\n");
    {
        const float sentinel = 1234.5f;
        dst[0].kv_cache[0] = sentinel;
        int got = 0;
        CHECK(dsv4_state_file_load(path, "tiny-model", &c, dst, c.max_pos,
                                   ids, NIDS, &got, bad_ids, NIDS) != 0,
              "mismatched prefix was accepted");
        CHECK(dst[0].kv_cache[0] == sentinel,
              "rejected load modified the live state");
        printf("  ok    mismatch rejected, live state untouched\n");
    }

    printf("\n-- GATE 3  a different model geometry is rejected --\n");
    {
        DSV4Cfg other;
        make_cfg(&other, HID + 1);
        const float sentinel = 2345.5f;
        dst[0].kv_cache[0] = sentinel;
        int got = 0;
        CHECK(dsv4_state_file_load(path, "tiny-model", &other, dst,
                                   other.max_pos, ids, NIDS, &got,
                                   ids, NIDS) != 0,
              "cache built for another geometry was accepted");
        CHECK(dst[0].kv_cache[0] == sentinel,
              "geometry rejection modified the live state");
        printf("  ok    geometry mismatch rejected\n");
    }

    printf("\n-- GATE 4  a missing file is a clean miss --\n");
    {
        int got = 0;
        CHECK(dsv4_state_file_load("/tmp/dsv4_kv_does_not_exist.bin",
                                   "tiny-model", &c, dst, c.max_pos, ids,
                                   NIDS, &got, ids, NIDS) != 0,
              "missing file was accepted");
        printf("  ok    missing file rejected\n");
    }

    unlink(path);
    free_all(src);
    free_all(dst);
    printf("\n%s\n", fails ? "FAILED" : "all KV cache gates passed");
    return fails ? 1 : 0;
}
