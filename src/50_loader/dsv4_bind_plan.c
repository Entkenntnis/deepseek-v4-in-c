/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_bind_plan.c - the request builders and the model-level plan. */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "dsv4_bind.h"
#include "dsv4_bind_int.h"

static void req_(DSV4BindPlan *p, const void **dest, int narrow, int64_t want,
                    const char *fmt, va_list ap)
{
    if (p->n >= DSV4_BIND_MAXB) {
        fprintf(stderr, "dsv4_bind: too many tensors\n");
        p->bad++;
        return;
    }
    DSV4BindReq *q = &p->r[p->n];
    vsnprintf(q->name, sizeof q->name, fmt, ap);
    q->dest = dest; q->want = want; q->narrow = narrow;
    q->t = NULL; q->off = 0;
    p->n++;
}

/* WIDE: widened to fp32. Only for vectors read elementwise. */
void dsv4_bind_reqw(DSV4BindPlan *p, const float **dest, int64_t want,
                    const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    req_(p, (const void **)dest, 0, want, fmt, ap);
    va_end(ap);
}

/* NARROW: kept as the checkpoint's own bytes, widened inside the matmul. */
void dsv4_bind_reqn(DSV4BindPlan *p, const void **dest, int64_t want,
                    const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    req_(p, dest, 1, want, fmt, ap);
    va_end(ap);
}

/* QUANTISED PAIR. rows/cols are LOGICAL dimensions; stored width differs:
 *   FP8   [rows, cols] bytes,     scale [rows/128, cols/128]
 *   FP4   [rows, cols/2] bytes,   scale [rows, cols/32]  (1x32 blocks)
 *   BF16  [rows, cols],           no scale tensor
 * All three request functions take format strings; an earlier version let only
 * reqq take a plain string, and callers passing "%s.wkv" got a literal name. */
void dsv4_bind_reqq(DSV4BindPlan *p, DSV4QMat *m, int wdt, int64_t rows,
                    int64_t cols, const char *fmt, ...)
{
    char base[192];
    va_list ap; va_start(ap, fmt);
    vsnprintf(base, sizeof base, fmt, ap);
    va_end(ap);

    memset(m, 0, sizeof *m);
    m->wdt = wdt; m->rows = (int)rows; m->cols = (int)cols;

    int64_t stored;
    switch (wdt) {
    case DSV4_WFP8:
        m->blk_r = 128; m->blk_c = 128;
        stored = rows * cols;
        break;
    case DSV4_WFP4:
        /* 1x32 blocks, NOT quantization_config.weight_block_size: that field
         * describes the FP8 tensors, and applying it here misreads every scale
         * while the byte counts still add up. */
        m->blk_r = 1; m->blk_c = 32;
        stored = rows * (cols / 2);
        break;
    default:
        m->blk_r = 0; m->blk_c = 0;
        stored = rows * cols;
        break;
    }

    dsv4_bind_reqn(p, &m->w, stored, "%s.weight", base);
    if (m->blk_c) {
        const int64_t ns = ((rows + m->blk_r - 1) / m->blk_r)
                         * ((cols + m->blk_c - 1) / m->blk_c);
        dsv4_bind_reqn(p, &m->s, ns, "%s.scale", base);
    }
}

/* mHC: rows x hc_dim parameters, plus nscale bias/scale vectors. */
void dsv4_bind_plan_hc(DSV4BindPlan *p, const DSV4Cfg *c, DSV4HcW *w, int rows,
                       int nscale, const char *base)
{
    dsv4_bind_reqw(p, &w->fn,    (int64_t)rows * dsv4_hc_dim(c), "%s_fn",    base);
    dsv4_bind_reqw(p, &w->base,  rows,                           "%s_base",  base);
    dsv4_bind_reqw(p, &w->scale, nscale,                         "%s_scale", base);
}

void dsv4_bind_plan_model(DSV4BindPlan *p, const DSV4Cfg *c, int want_head,
                          DSV4ModelW *w)
{
    memset(w, 0, sizeof *w);
    w->wdt = DSV4_WBF16;
    dsv4_bind_reqn(p, &w->embed, (int64_t)c->vocab * c->hidden, "embed.weight");
    dsv4_bind_reqw(p, &w->norm,  c->hidden,                     "norm.weight");
    if (want_head)
        dsv4_bind_reqn(p, &w->head, (int64_t)c->vocab * c->hidden, "head.weight");
    /* The head's mHC has hc_mult rows and ONE scale, against a block's mix_hc
     * rows and three. */
    dsv4_bind_plan_hc(p, c, &w->hc_head, c->hc_mult, 1, "hc_head");
}
