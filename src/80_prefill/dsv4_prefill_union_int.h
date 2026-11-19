/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_union_int.h - scheduler state shared by the union files. */
#ifndef DSV4_PREFILL_UNION_INT_H
#define DSV4_PREFILL_UNION_INT_H

#include "dsv4_prefill_int.h"

typedef struct {
    DSV4Prefill *p;
    const DSV4ExpertW **ex;
    const int *order;
    int ng;
    int *done;
    int *remaining;
    int *q, *qh, *qt;
    int io_done;
} DSV4PFillCtx;

int dsv4_pf_union_run_one_ready(DSV4Prefill *p, DSV4PFillCtx *f);

#endif /* DSV4_PREFILL_UNION_INT_H */
