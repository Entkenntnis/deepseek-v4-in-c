/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_bind_int.h - the plan shared by both binding paths. Not public. */
#ifndef DSV4_BIND_INT_H
#define DSV4_BIND_INT_H

#include <stdarg.h>
#include <stdint.h>

#include "dsv4.h"
#include "dsv4_st.h"

/* The widest layer, a ratio-4 indexed one, needs 46 requests. */
#define DSV4_BIND_MAXB 96

typedef struct {
    char              name[224];
    const DSV4Tensor *t;
    int64_t           want;    /* STORED elements expected; -1 accepts anything */
    int               narrow;  /* 1 = keep bytes, 0 = widen to fp32 */
    const void      **dest;
    size_t            off;     /* byte offset into the blob */
} DSV4BindReq;

typedef struct {
    DSV4BindReq r[DSV4_BIND_MAXB];
    int         n;
    int         bad;
} DSV4BindPlan;

static inline size_t dsv4_bind_align8(size_t x)
{
    return (x + 7u) & ~(size_t)7u;
}

/* The three request classes. All take format strings, deliberately. */
void dsv4_bind_req_(DSV4BindPlan *p, const void **dest, int narrow, int64_t want,
                    const char *fmt, va_list ap);
void dsv4_bind_reqw(DSV4BindPlan *p, const float **dest, int64_t want,
                    const char *fmt, ...);
void dsv4_bind_reqn(DSV4BindPlan *p, const void **dest, int64_t want,
                    const char *fmt, ...);
void dsv4_bind_reqq(DSV4BindPlan *p, DSV4QMat *m, int wdt, int64_t rows,
                    int64_t cols, const char *fmt, ...);

void dsv4_bind_plan_hc(DSV4BindPlan *p, const DSV4Cfg *c, DSV4HcW *w, int rows,
                       int nscale, const char *base);

/* Plan one layer / the model-level tensors. */
void dsv4_bind_plan_layer(DSV4BindPlan *p, const DSV4Cfg *c, int L,
                          DSV4LayerW *w);
void dsv4_bind_plan_model(DSV4BindPlan *p, const DSV4Cfg *c, int want_head,
                          DSV4ModelW *w);

/* Validate against the config, lay out the blob, then read into it. */
int64_t dsv4_bind_plan_resolve(DSV4BindPlan *p, const DSV4St *s);
int     dsv4_bind_plan_load(DSV4BindPlan *p, const DSV4St *s,
                            unsigned char *blob);

#endif /* DSV4_BIND_INT_H */
