/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_engine.c - teardown, context reset and sequence building. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_expert_vk.h"
#include "dsv4_serve_int.h"

void dsv4_engine_free(Engine *e)
{
    dsv4_prefill_free(e->pf);
    for (int L = 0; L < e->c.n_layers && e->lst; L++) dsv4_state_free(&e->lst[L]);
    free(e->lst);
    free(e->h); free(e->xh); free(e->logits); free(e->ids);
    free(e->cs_d); free(e->sn_d); free(e->cs_c); free(e->sn_c);
    free(e->enc_tmp); free(e->tok_bytes); free(e->hexbuf);
    dsv4_scratch_free(&e->scratch);
    dsv4_bind_model_free(&e->mb);
    dsv4_expert_vk_free(e->expert_vk);
    e->expert_vk = NULL;
    dsv4_cache_free(&e->cache);
    dsv4_trunk_close(&e->tr);
    dsv4_tok_free(&e->tok);
    dsv4_st_close(&e->st);
    free(e->system_prompt);
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
    memset(&e->prof_base, 0, sizeof e->prof_base);
    memset(&dsv4_prof, 0, sizeof dsv4_prof);
    return 0;
}

/* Append `text` to the sequence, encoding into a temporary first so a context
 * overflow cannot leave a half-appended turn behind. */
int dsv4_append_text(Engine *e, const char *text)
{
    const int len = (int)strlen(text);
    const int need = len + 64;
    if (need > e->enc_cap) {
        int32_t *p = realloc(e->enc_tmp, (size_t)need * sizeof(int32_t));
        if (!p) return -1;
        e->enc_tmp = p;
        e->enc_cap = need;
    }
    int n = dsv4_tok_encode(&e->tok, text, len, e->enc_tmp, e->enc_cap);
    if (n < 0) return -1;
    if (n > e->cap_ids - e->n_ids) return -2;
    memcpy(e->ids + e->n_ids, e->enc_tmp, (size_t)n * sizeof(int32_t));
    e->n_ids += n;
    return n;
}
