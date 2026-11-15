/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_bind_layer.c - the per-layer plan. This is the single declaration of
 * what a layer needs; both the resident and the streamed binder walk it. */
#include <stdio.h>
#include <string.h>

#include "dsv4_bind.h"
#include "dsv4_bind_int.h"

/* The compressor's projections widen by coff: 2 on the overlapped ratio-4 path
 * and 1 otherwise (model.py: self.overlap = ratio == 4). */
static int comp_coff(int ratio) { return ratio == 4 ? 2 : 1; }

static void plan_compressor(DSV4BindPlan *p, const DSV4Cfg *c,
                            DSV4CompressorW *w, int ratio, int head_dim,
                            const char *base)
{
    const int coff = comp_coff(ratio);
    w->ratio = ratio;
    w->coff  = coff;
    dsv4_bind_reqw(p, &w->ape,  (int64_t)ratio * coff * head_dim, "%s.ape", base);
    dsv4_bind_reqw(p, &w->norm, head_dim, "%s.norm.weight", base);
    dsv4_bind_reqq(p, &w->wkv, DSV4_WBF16, (int64_t)coff * head_dim, c->hidden,
                   "%s.wkv", base);
    dsv4_bind_reqq(p, &w->wgate, DSV4_WBF16, (int64_t)coff * head_dim, c->hidden,
                   "%s.wgate", base);
}

/* Everything one decoder layer needs, and NOTHING it must not have. See
 * README.md for the five layer shapes this branches on. */
static void plan_layer(DSV4BindPlan *p, const DSV4Cfg *c, int L, DSV4LayerW *w)
{
    const int ratio = dsv4_compress_ratio(c, L);
    const int hash  = dsv4_is_hash_routed(c, L);

    memset(w, 0, sizeof *w);
    w->layer = L;
    w->compress_ratio = ratio;
    w->hash_routed = hash;
    w->has_comp = (ratio != 0);
    w->has_idx  = dsv4_has_indexer(c, L);

    dsv4_bind_reqw(p, &w->attn_norm, c->hidden, "layers.%d.attn_norm.weight", L);
    dsv4_bind_reqw(p, &w->ffn_norm,  c->hidden, "layers.%d.ffn_norm.weight",  L);

    /* ---- attention ---- */
    DSV4AttnW *a = &w->attn;
    a->has_comp = w->has_comp;
    a->has_idx  = w->has_idx;
    dsv4_bind_reqw(p, &a->sink,    c->n_heads,  "layers.%d.attn.attn_sink",      L);
    dsv4_bind_reqw(p, &a->q_norm,  c->q_lora,   "layers.%d.attn.q_norm.weight",  L);
    dsv4_bind_reqw(p, &a->kv_norm, c->head_dim, "layers.%d.attn.kv_norm.weight", L);

    dsv4_bind_reqq(p, &a->wq_a, DSV4_WFP8, c->q_lora, c->hidden,
                   "layers.%d.attn.wq_a", L);
    dsv4_bind_reqq(p, &a->wq_b, DSV4_WFP8, (int64_t)c->n_heads * c->head_dim,
                   c->q_lora, "layers.%d.attn.wq_b", L);
    dsv4_bind_reqq(p, &a->wkv,  DSV4_WFP8, c->head_dim, c->hidden,
                   "layers.%d.attn.wkv", L);
    /* The output projection is low-rank AND grouped. The column count is gw,
     * NOT hidden: Flash makes them equal, and writing c->hidden here encoded a
     * coincidence as a rule and rejected a config where they differ. */
    const int64_t gw = (int64_t)c->n_heads * c->head_dim / c->o_groups;
    dsv4_bind_reqq(p, &a->wo_a, DSV4_WFP8, (int64_t)c->o_lora * c->o_groups, gw,
                   "layers.%d.attn.wo_a", L);
    dsv4_bind_reqq(p, &a->wo_b, DSV4_WFP8, c->hidden,
                   (int64_t)c->o_lora * c->o_groups, "layers.%d.attn.wo_b", L);

    /* HCA. Absent on ratio-0 layers: model.py constructs no Compressor there. */
    if (w->has_comp) {
        char nm[192];
        snprintf(nm, sizeof nm, "layers.%d.attn.compressor", L);
        plan_compressor(p, c, &a->comp, ratio, c->head_dim, nm);
    }

    /* CSA. Present ONLY where the ratio is 4. Its compressor is narrower: it
     * works in index_head_dim, not head_dim. */
    if (w->has_idx) {
        char nm[192];
        dsv4_bind_reqq(p, &a->idx.wq_b, DSV4_WFP8,
                       (int64_t)c->index_n_heads * c->index_head_dim, c->q_lora,
                       "layers.%d.attn.indexer.wq_b", L);
        dsv4_bind_reqq(p, &a->idx.weights_proj, DSV4_WBF16, c->index_n_heads,
                       c->hidden, "layers.%d.attn.indexer.weights_proj", L);
        snprintf(nm, sizeof nm, "layers.%d.attn.indexer.compressor", L);
        plan_compressor(p, c, &a->idx.comp, ratio, c->index_head_dim, nm);
    }

    /* ---- mHC. mix_hc rows per block, 3 scales (pre, post, comb). ---- */
    {
        char nm[192];
        snprintf(nm, sizeof nm, "layers.%d.hc_attn", L);
        dsv4_bind_plan_hc(p, c, &w->hc_attn, dsv4_mix_hc(c), 3, nm);
        snprintf(nm, sizeof nm, "layers.%d.hc_ffn", L);
        dsv4_bind_plan_hc(p, c, &w->hc_ffn,  dsv4_mix_hc(c), 3, nm);
    }

    /* ---- MoE ---- */
    DSV4MoeW *mo = &w->moe;
    dsv4_bind_reqq(p, &mo->gate, DSV4_WBF16, c->n_experts, c->hidden,
                   "layers.%d.ffn.gate", L);
    /* Exactly one of these two exists. Requesting both fails on every layer;
     * requesting neither loses the routing. */
    if (hash)
        dsv4_bind_reqn(p, (const void **)&mo->tid2eid,
                       (int64_t)c->vocab * c->topk, "layers.%d.ffn.gate.tid2eid", L);
    else
        dsv4_bind_reqw(p, &mo->bias, c->n_experts, "layers.%d.ffn.gate.bias", L);

    /* The shared expert is FP8 and resident. Routed experts are FP4 and
     * streamed, so they never appear here. */
    for (int i = 1; i <= 3; i++) {
        DSV4QMat *q = (i == 1) ? &mo->shared.w1
                    : (i == 3) ? &mo->shared.w3 : &mo->shared.w2;
        const int64_t inter = (int64_t)c->moe_inter * c->n_shared;
        const char *f = "layers.%d.ffn.shared_experts.w%d";
        if (i == 2) dsv4_bind_reqq(p, q, DSV4_WFP8, c->hidden, inter, f, L, i);
        else        dsv4_bind_reqq(p, q, DSV4_WFP8, inter, c->hidden, f, L, i);
    }
}

int dsv4_bind_layer_plan(const DSV4Cfg *c, int L,
                         int (*emit)(void *ctx, int i, const char *name),
                         void *ctx)
{
    DSV4BindPlan p; memset(&p, 0, sizeof p);
    DSV4LayerW tmp;
    plan_layer(&p, c, L, &tmp);
    if (p.bad) return -1;
    for (int i = 0; i < p.n; i++)
        if (emit && emit(ctx, i, p.r[i].name) != 0) return -1;
    return p.n;
}

void dsv4_bind_plan_layer(DSV4BindPlan *p, const DSV4Cfg *c, int L,
                          DSV4LayerW *w)
{
    plan_layer(p, c, L, w);
}
