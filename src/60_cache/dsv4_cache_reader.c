/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_reader.c - the pread() pool. See README.md for why it is pthreads
 * and why the width is topk. */
#include <stdio.h>
#include <stdlib.h>

#include "dsv4_cache.h"
#include "dsv4_cache_int.h"
#include "dsv4_threads.h"

/* Caller holds rmx. */
int dsv4_cache_rq_push(DSV4Cache *c, DSV4CacheBatch *b, int j)
{
    if (c->rq_n >= c->rq_cap) return -1;
    c->rq[c->rq_tail].b = b;
    c->rq[c->rq_tail].j = j;
    c->rq_tail = (c->rq_tail + 1) % c->rq_cap;
    c->rq_n++;
    return 0;
}

static void *reader_main(void *arg)
{
    ReaderArg *ra = (ReaderArg *)arg;
    DSV4Cache *c = ra->c;

    /* These threads inherited the COMPUTE affinity mask. Put them where the
     * thread policy asked for them -- on this machine the E-cores, where they
     * spend their lives blocked in pread() instead of competing with FP4. */
    dsv4_threads_pin_current_reader();

    pthread_mutex_lock(&c->rmx);
    for (;;) {
        while (!c->rshutdown && c->rq_n == 0)
            pthread_cond_wait(&c->rwork, &c->rmx);
        if (c->rshutdown) break;

        const struct DSV4CacheJob job = c->rq[c->rq_head];
        c->rq_head = (c->rq_head + 1) % c->rq_cap;
        c->rq_n--;
        pthread_mutex_unlock(&c->rmx);

        DSV4CacheBatch *b = job.b;
        const int j = job.j;
        uint64_t got = 0;
        const int bad = dsv4_cache_load(c, &c->slot[b->slot_of[j]], b->t_of[j],
                                        c->bounce[ra->bi], &got) != 0;

        pthread_mutex_lock(&c->rmx);
        b->failed[j] = bad;
        b->got[j]    = got;
        if (b->is_prefetch) {
            /* Publish immediately: a layer's begin() may be waiting on this
             * exact slot, and there is no end() call for a prefetch batch. */
            DSV4Slot *s = &c->slot[b->slot_of[j]];
            s->pending = 0;
            if (bad) {
                s->layer = s->expert = -1;
                c->pf_fail++;
            } else {
                s->layer  = b->layer_of[j];
                s->expert = b->expert_of[j];
                c->pf_loads++;
                c->pf_hits++;
                c->pf_bytes += got;
            }
        } else if (b->streaming) {
            /* Publish this miss immediately, so the caller can run its matmul
             * while the rest of the batch is still on the wire. All eviction
             * and slot decisions were made in begin(). */
            DSV4Slot *s = &c->slot[b->slot_of[j]];
            s->pending = 0;
            if (bad) {
                s->layer = s->expert = -1;     /* no half-read slot goes live */
                b->rc = -1;
                b->out[b->k_of[j]] = NULL;
            } else {
                s->layer  = b->layer_of[j];
                s->expert = b->expert_of[j];
                c->misses++;
                c->bytes_read += got;
                b->out[b->k_of[j]] = &s->w;
            }
            b->published[j] = 1;

            /* Keep the read pipeline full: queue the next job as this one
             * retires. With cap >= ntodo this queues everything at once. */
            if (b->next_j < b->ntodo) {
                const int nj = b->next_j++;
                if (dsv4_cache_rq_push(c, b, nj) != 0) {
                    /* Cannot happen with the FIFO sized at construction, but
                     * never leave end() waiting on jobs nobody will run. */
                    b->rc = -1;
                    for (int t = nj; t < b->ntodo; t++) {
                        DSV4Slot *st = &c->slot[b->slot_of[t]];
                        st->pending = 0;
                        st->layer = st->expert = -1;
                        b->failed[t] = 1;
                        b->published[t] = 1;
                        b->out[b->k_of[t]] = NULL;
                        b->rdone++;
                    }
                    b->next_j = b->ntodo;
                }
            }
        }
        b->rdone++;
        pthread_cond_broadcast(&c->rdone);
    }
    pthread_mutex_unlock(&c->rmx);
    return NULL;
}

/* Failure here is NOT fatal: nrth stays 0 and every batch falls back to reading
 * on the calling thread. An engine that runs slower beats one that refuses. */
void dsv4_cache_pool_start(DSV4Cache *c)
{
    int want = dsv4_threads_readers();
    if (want < 1) want = 1;
    if (want > DSV4_MAX_TOPK) want = DSV4_MAX_TOPK;
    /* bounce[0] is reserved for synchronous get(); readers start at 1. */
    if (want > c->nbounce - 1) want = c->nbounce - 1;
    if (want < 1) return;

    if (pthread_mutex_init(&c->rmx, NULL) != 0) return;
    if (pthread_cond_init(&c->rwork, NULL) != 0) {
        pthread_mutex_destroy(&c->rmx);
        return;
    }
    if (pthread_cond_init(&c->rdone, NULL) != 0) {
        pthread_cond_destroy(&c->rwork);
        pthread_mutex_destroy(&c->rmx);
        return;
    }

    c->rth = (pthread_t *)calloc((size_t)want, sizeof *c->rth);
    ReaderArg *args = (ReaderArg *)calloc((size_t)want, sizeof *args);
    if (!c->rth || !args) { free(c->rth); free(args); c->rth = NULL; return; }
    c->rargs = args;
    for (int i = 0; i < want; i++) {
        args[i].c = c;
        args[i].bi = i + 1;          /* 0 is the synchronous get() buffer */
        if (pthread_create(&c->rth[i], NULL, reader_main, &args[i]) != 0) break;
        c->nrth++;
    }
    if (c->nrth == 0) {
        free(c->rth); c->rth = NULL;
        free(args); c->rargs = NULL;
    }
}

void dsv4_cache_pool_stop(DSV4Cache *c)
{
    if (!c->rth) return;
    pthread_mutex_lock(&c->rmx);
    c->rshutdown = 1;
    pthread_cond_broadcast(&c->rwork);
    pthread_mutex_unlock(&c->rmx);
    for (int i = 0; i < c->nrth; i++) pthread_join(c->rth[i], NULL);
    free(c->rth);
    free(c->rargs);
    c->rth  = NULL;
    c->rargs = NULL;
    c->nrth = 0;
    pthread_cond_destroy(&c->rwork);
    pthread_cond_destroy(&c->rdone);
    pthread_mutex_destroy(&c->rmx);
}
