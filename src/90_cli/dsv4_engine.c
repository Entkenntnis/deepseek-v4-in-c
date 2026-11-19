/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_engine.c - teardown, context reset and sequence building. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_serve_int.h"

void dsv4_engine_free(Engine *e)
{
    dsv4_prefill_free(e->pf);
    dsv4_state_snapshot_free(e->snap);
    for (int L = 0; L < e->c.n_layers && e->lst; L++) dsv4_state_free(&e->lst[L]);
    free(e->lst);
    free(e->h); free(e->xh); free(e->logits); free(e->ids);
    free(e->cs_d); free(e->sn_d); free(e->cs_c); free(e->sn_c);
    free(e->in_ids);
    dsv4_scratch_free(&e->scratch);
    dsv4_bind_model_free(&e->mb);
    dsv4_cache_free(&e->cache);
    dsv4_trunk_close(&e->tr);
    dsv4_st_close(&e->st);
    memset(e, 0, sizeof *e);
}

int dsv4_engine_reset_context(Engine *e)
{
    dsv4_prefill_release(e->pf);
    for (int L = 0; L < e->c.n_layers; L++) {
        dsv4_state_free(&e->lst[L]);
        if (dsv4_state_init(&e->lst[L], &e->c, L, e->maxpos) != 0) return -1;
    }
    e->n_ids = 0;
    e->pending_forward = 0;
    e->kv_prefix = 0;
    memset(&e->prof_base, 0, sizeof e->prof_base);
    memset(&dsv4_prof, 0, sizeof dsv4_prof);
    return 0;
}

/* Append token ids to the sequence. Range-check first so a malformed command
 * cannot index an embedding row out of bounds and a context overflow cannot
 * leave a half-appended turn behind. */
int dsv4_append_ids(Engine *e, const int32_t *ids, int n)
{
    if (n < 0) return -1;
    for (int i = 0; i < n; i++)
        if (ids[i] < 0 || ids[i] >= e->c.vocab) return -1;
    if (n > e->cap_ids - e->n_ids) return -2;
    memcpy(e->ids + e->n_ids, ids, (size_t)n * sizeof(int32_t));
    e->n_ids += n;
    return n;
}
