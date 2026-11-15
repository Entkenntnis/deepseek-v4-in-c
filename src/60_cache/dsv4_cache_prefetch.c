/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_prefetch.c - queue known-ahead experts into the main cache. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_cache.h"
#include "dsv4_cache_int.h"

int dsv4_cache_enable_prefetch(DSV4Cache *c)
{
    if (!c || c->pf_enabled) return 0;
    if (c->nrth <= 0) return -1;      /* without readers a pending slot hangs */
    if (!c->pf_batch) {
        c->pf_batch = (DSV4CacheBatch *)calloc(1, sizeof *c->pf_batch);
        if (!c->pf_batch) return -1;
        c->pf_batch->is_prefetch = 1;
    }
    c->pf_enabled = 1;
    return 0;
}

/* True when the main cache can hold `n` prefetched experts AND still leave one
 * layer's top-k worth of slots for normal misses. */
static int pf_has_room(const DSV4Cache *c, int n)
{
    return c->pf_enabled && c->nrth > 0 && n > 0 &&
           c->nslot > n + c->cfg->topk;
}

static int pf_queue(DSV4Cache *c, const int *layers, const int *experts,
                    int n)
{
    if (!c || n <= 0 || n > DSV4_MAX_PREFETCH) return 0;
    if (!pf_has_room(c, n)) return 0;

    pthread_mutex_lock(&c->rmx);
    /* The previous token's prefetch must have landed before its batch is
     * reused. Its slots are normal residents by now. */
    while (c->pf_batch->ntodo > c->pf_batch->rdone)
        pthread_cond_wait(&c->rdone, &c->rmx);
    c->pf_batch->ntodo = 0;
    c->pf_batch->rdone = 0;
    c->pf_batch->is_prefetch = 1;

    int queued = 0;
    for (int i = 0; i < n; i++) {
        const int layer = layers[i], expert = experts[i];

        int resident = -1;
        for (int j = 0; j < c->nslot; j++)
            if (c->slot[j].layer == layer && c->slot[j].expert == expert) {
                resident = j; break;
            }
        if (resident >= 0) {
            if (c->slot[resident].pending) continue; /* already in flight */
            c->slot[resident].stamp = ++c->clock;
            c->pf_hits++;
            continue;
        }

        const DSV4Tensor *t[6];
        if (dsv4_cache_resolve(c, layer, expert, t) != 0) {
            c->pf_fail++;
            continue;
        }
        const int slot = dsv4_cache_claim_slot(c, layer, expert, NULL, 0);
        if (slot < 0) break;              /* cannot happen with pf_has_room */
        DSV4CacheBatch *b = c->pf_batch;
        const int j = b->ntodo++;
        b->slot_of[j]   = slot;
        b->k_of[j]      = i;
        b->layer_of[j]  = layer;
        b->expert_of[j] = expert;
        memcpy(b->t_of[j], t, sizeof t);
        b->failed[j] = 0;
        b->got[j] = 0;
        if (dsv4_cache_rq_push(c, b, j) != 0) {
            c->slot[slot].layer = -1;
            c->slot[slot].pending = 0;
            b->ntodo--;
            c->pf_fail++;
            break;
        }
        queued++;
    }
    c->pf_requests += (uint64_t)queued;
    if (queued > 0) pthread_cond_broadcast(&c->rwork);
    pthread_mutex_unlock(&c->rmx);
    return 0;
}

int dsv4_cache_prefetch_hash_layers(DSV4Cache *c, int token_id)
{
    if (!c || !c->pf_enabled) return 0;
    const DSV4Cfg *cfg = c->cfg;
    if (!cfg || token_id < 0 || token_id >= cfg->vocab) return -1;
    if (cfg->topk <= 0 || cfg->topk > DSV4_MAX_TOPK) return -1;
    const int n_total = cfg->num_hash_layers * cfg->topk;
    if (n_total > DSV4_MAX_PREFETCH) return 0;
    if (!pf_has_room(c, n_total)) return 0;

    int layers[DSV4_MAX_PREFETCH], experts[DSV4_MAX_PREFETCH], n = 0;
    for (int L = 0; L < cfg->num_hash_layers && L < cfg->n_layers; L++) {
        char name[224];
        snprintf(name, sizeof name, "layers.%d.ffn.gate.tid2eid", L);
        const DSV4Tensor *t = dsv4_st_find(c->st, name);
        if (!t || t->dtype != DSV4_DT_I64) continue;
        const int64_t row_bytes = (int64_t)cfg->topk * (int64_t)sizeof(int64_t);
        if ((int64_t)(token_id + 1) * row_bytes > t->nbytes) continue;
        DSV4Tensor row = *t;
        row.off += (int64_t)token_id * row_bytes;
        row.nbytes = row_bytes;
        int64_t vals[DSV4_MAX_TOPK];
        if (dsv4_st_read(c->st, &row, vals) != row_bytes) continue;
        for (int k = 0; k < cfg->topk && n < DSV4_MAX_PREFETCH; k++) {
            layers[n] = L;
            experts[n] = (int)vals[k];
            n++;
        }
    }
    if (n == 0) return 0;
    return pf_queue(c, layers, experts, n);
}
