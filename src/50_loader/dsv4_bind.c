/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_bind.c - resolve, validate and read a planned layer/model.
 *
 * THE CHECK THAT EARNS ITS KEEP is plan_resolve: every tensor's element count is
 * compared against what the CONFIG implies before a single byte is read. The
 * expectations are expressions of config fields, never literals. A checkpoint
 * that disagrees with its own config fails at load, loudly, instead of feeding
 * every kernel downstream the wrong strides while producing plausible output.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_bind.h"
#include "dsv4_bind_int.h"

static int64_t plan_resolve(DSV4BindPlan *p, const DSV4St *s)
{
    size_t off = 0;
    for (int i = 0; i < p->n; i++) {
        DSV4BindReq *q = &p->r[i];
        q->t = dsv4_st_find(s, q->name);
        if (!q->t) {
            fprintf(stderr, "dsv4_bind: missing tensor %s\n", q->name);
            p->bad++;
            continue;
        }
        const int64_t have = dsv4_st_numel(q->t);
        if (q->want >= 0 && have != q->want) {
            fprintf(stderr, "dsv4_bind: %s has %lld elements, config implies %lld\n",
                    q->name, (long long)have, (long long)q->want);
            p->bad++;
            continue;
        }
        /* Asking to widen an FP8/FP4/I64 tensor is a planning bug: those have
         * no meaning without their scale, or are not floats at all. */
        if (!q->narrow && dsv4_st_elemsize(q->t->dtype) > 0 &&
            (q->t->dtype == DSV4_DT_F8_E4M3 || q->t->dtype == DSV4_DT_F8_E8M0 ||
             q->t->dtype == DSV4_DT_I8 || q->t->dtype == DSV4_DT_I64)) {
            fprintf(stderr, "dsv4_bind: %s was planned WIDE but is a quantised or "
                            "index dtype; it must be requested narrow\n", q->name);
            p->bad++;
            continue;
        }
        const size_t bytes = q->narrow
            ? (size_t)q->t->nbytes
            : (size_t)have * sizeof(float);
        q->off = off;
        off = dsv4_bind_align8(off + bytes);
    }
    return p->bad ? -1 : (int64_t)off;
}

static int plan_load(DSV4BindPlan *p, const DSV4St *s, unsigned char *blob)
{
    for (int i = 0; i < p->n; i++) {
        DSV4BindReq *q = &p->r[i];
        unsigned char *dst = blob + q->off;
        if (q->narrow) {
            if (dsv4_st_read(s, q->t, dst) != q->t->nbytes) {
                fprintf(stderr, "dsv4_bind: short read on %s\n", q->name);
                return -1;
            }
        } else {
            const int64_t n = dsv4_st_numel(q->t);
            if (dsv4_st_read_f32(s, q->t, (float *)dst) != n) {
                fprintf(stderr, "dsv4_bind: short widen on %s\n", q->name);
                return -1;
            }
        }
        *q->dest = dst;
    }
    return 0;
}

int64_t dsv4_bind_model_bytes(const DSV4St *s, const DSV4Cfg *c, int want_head)
{
    DSV4BindPlan p; memset(&p, 0, sizeof p);
    DSV4ModelW tmp;
    dsv4_bind_plan_model(&p, c, want_head, &tmp);
    return plan_resolve(&p, s);
}

int dsv4_bind_model(const DSV4St *s, const DSV4Cfg *c, int want_head,
                    DSV4ModelBind *m)
{
    memset(m, 0, sizeof *m);

    DSV4BindPlan p; memset(&p, 0, sizeof p);
    dsv4_bind_plan_model(&p, c, want_head, &m->w);
    const int64_t need = plan_resolve(&p, s);
    if (need < 0) {
        fprintf(stderr, "dsv4_bind: model could not be planned\n");
        return -1;
    }

    m->blob = malloc((size_t)need);
    if (!m->blob) {
        fprintf(stderr, "dsv4_bind: OOM allocating %lld bytes for the model\n",
                (long long)need);
        return -1;
    }

    if (plan_load(&p, s, (unsigned char *)m->blob) != 0) {
        free(m->blob); m->blob = NULL;
        return -1;
    }
    return 0;
}

void dsv4_bind_model_free(DSV4ModelBind *m)
{
    if (!m) return;
    free(m->blob);
    memset(m, 0, sizeof *m);
}
