/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul_fp8.c - FP8 batch body and the reference entry point. */
#include <stdint.h>

#include "dsv4_matmul.h"
#include "dsv4_matmul_fp8_int.h"

/* Decode one 128-byte block once, into exact f32 values. */
static inline void fp8_decode_block_128(const uint8_t *wb, float *wf)
{
    if (fp8_block_has_special(wb)) {
        for (int i = 0; i < 128; i++)
            wf[i] = dsv4_e4m3_to_f32(wb[i]);
        return;
    }
    for (int i = 0; i < 128; i += 32) {
        _mm256_storeu_ps(wf + i,      fp8_8_normal(wb, i));
        _mm256_storeu_ps(wf + i + 8,  fp8_8_normal(wb, i + 8));
        _mm256_storeu_ps(wf + i + 16, fp8_8_normal(wb, i + 16));
        _mm256_storeu_ps(wf + i + 24, fp8_8_normal(wb, i + 24));
    }
}

/* Keep the rare gather body OUT of the hot function. When it was `static
 * inline`, GCC hoisted its vpmovzxbd loads and spilled eight YMM registers to
 * the stack *before* the special/not-special branch, so the common fast path
 * paid for the rare slow path on every block. A profile showed those spills
 * costing cycles; one call on a special block is nothing next to that. */
static __attribute__((noinline)) __m256
fp8_block_sum_128_gather(const uint8_t *wb, const float *xb)
{
    __m256 s;
#define FP8_LOAD8(k) load8_fp8(wb, (k))
    FP8_BLOCK_SUM(FP8_LOAD8);
#undef FP8_LOAD8
    return s;
}

static inline __m256 fp8_block_sum_128(const uint8_t *wb, const float *xb)
{
    return fp8_block_has_special(wb) ? fp8_block_sum_128_gather(wb, xb)
                                     : fp8_block_sum_128_normal(wb, xb);
}

/* FP8. Both schedules accumulate four consecutive block sums and four scales
 * as one float vector, into four partial accumulators, with the same final
 * tree and tail order, so the CPU batch has the same per-row arithmetic as nt
 * GEMVs; the gate in `make test` checks it in L2. */
void fp8_n(float *y, const float *x, const uint8_t *W,
           const uint8_t *S, int in, int out, int blk_r, int blk_c, int nt)
{
    const int nblk = in / 128;
    const int nrb  = (out + 127) / 128;

    if (blk_c == 128 && (in & 127) == 0 && blk_r == 128 && nblk <= 256) {
        if (nt == 1) {
            _Pragma("omp parallel for schedule(static) if (out > 64)")
            for (int rb = 0; rb < nrb; rb++) {
                const uint8_t *srow = S + (size_t)rb * nblk;
                float sf[256];
                int si = 0;
                for (; si + 3 < nblk; si += 4)
                    _mm_storeu_ps(sf + si, dsv4_e8m0x4_to_ps(srow + si));
                for (; si < nblk; si++) sf[si] = dsv4_e8m0_to_f32(srow[si]);
                const int o0 = rb * 128;
                int o1 = o0 + 128;
                if (o1 > out) o1 = out;

                for (int o = o0; o < o1; o++) {
                    const uint8_t *rw = W + (size_t)o * in;
                    __m128 a0 = _mm_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                    float tail = 0.0f;
                    int bi = 0;

                    for (; bi + 3 < nblk; bi += 4) {
                        const __m256 s0 = fp8_block_sum_128(rw + (size_t)bi * 128, x + (size_t)bi * 128);
                        const __m256 s1 = fp8_block_sum_128(rw + (size_t)(bi + 1) * 128, x + (size_t)(bi + 1) * 128);
                        const __m256 s2 = fp8_block_sum_128(rw + (size_t)(bi + 2) * 128, x + (size_t)(bi + 2) * 128);
                        const __m256 s3 = fp8_block_sum_128(rw + (size_t)(bi + 3) * 128, x + (size_t)(bi + 3) * 128);
                        const __m128 b  = dsv4_fold4_ps(s0, s1, s2, s3);
                        const __m128 sc = _mm_loadu_ps(sf + bi);
                        switch ((bi >> 2) & 3) {
                        case 0: a0 = _mm_fmadd_ps(b, sc, a0); break;
                        case 1: a1 = _mm_fmadd_ps(b, sc, a1); break;
                        case 2: a2 = _mm_fmadd_ps(b, sc, a2); break;
                        default: a3 = _mm_fmadd_ps(b, sc, a3); break;
                        }
                    }
                    for (; bi < nblk; bi++)
                        tail += dsv4_hsum256(fp8_block_sum_128(
                                    rw + (size_t)bi * 128,
                                    x + (size_t)bi * 128)) * sf[bi];

                    const __m128 t = _mm_add_ps(_mm_add_ps(a0, a1),
                                                _mm_add_ps(a2, a3));
                    y[o] = dsv4_hsum128(t) + tail;
                }
            }
            return;
        }

        _Pragma("omp parallel for schedule(static) if (nrb > 1)")
        for (int rb = 0; rb < nrb; rb++) {
            const uint8_t *srow = S + (size_t)rb * nblk;
            float sf[256];
            int si = 0;
            for (; si + 3 < nblk; si += 4)
                _mm_storeu_ps(sf + si, dsv4_e8m0x4_to_ps(srow + si));
            for (; si < nblk; si++) sf[si] = dsv4_e8m0_to_f32(srow[si]);
            const int o0 = rb * 128;
            int o1 = o0 + 128;
            if (o1 > out) o1 = out;

            for (int o = o0; o < o1; o++) {
                const uint8_t *rw = W + (size_t)o * in;
                __m128 acc[4][DSV4_MAX_BATCH];
                float tail[DSV4_MAX_BATCH];
                float wf[512];
                for (int k = 0; k < 4; k++)
                    for (int t = 0; t < nt; t++)
                        acc[k][t] = _mm_setzero_ps();
                for (int t = 0; t < nt; t++) tail[t] = 0.0f;

                int bi = 0;
                for (; bi + 3 < nblk; bi += 4) {
                    for (int k = 0; k < 4; k++)
                        fp8_decode_block_128(rw + (size_t)(bi + k) * 128,
                                             wf + (size_t)k * 128);
                    const __m128 sc = _mm_loadu_ps(sf + bi);
                    const int slot = (bi >> 2) & 3;
                    for (int t = 0; t < nt; t++) {
                        const float *xb = x + (size_t)t * in + (size_t)bi * 128;
                        const __m256 s0 = fp8_block_sum_wf(wf,       xb);
                        const __m256 s1 = fp8_block_sum_wf(wf + 128, xb + 128);
                        const __m256 s2 = fp8_block_sum_wf(wf + 256, xb + 256);
                        const __m256 s3 = fp8_block_sum_wf(wf + 384, xb + 384);
                        const __m128 b = dsv4_fold4_ps(s0, s1, s2, s3);
                        acc[slot][t] = _mm_fmadd_ps(b, sc, acc[slot][t]);
                    }
                }
                for (; bi < nblk; bi++) {
                    fp8_decode_block_128(rw + (size_t)bi * 128, wf);
                    for (int t = 0; t < nt; t++) {
                        const float *xb = x + (size_t)t * in + (size_t)bi * 128;
                        tail[t] += dsv4_hsum256(fp8_block_sum_wf(wf, xb)) * sf[bi];
                    }
                }
                for (int t = 0; t < nt; t++) {
                    const __m128 v = _mm_add_ps(_mm_add_ps(acc[0][t], acc[1][t]),
                                                _mm_add_ps(acc[2][t], acc[3][t]));
                    y[(size_t)t * out + o] = dsv4_hsum128(v) + tail[t];
                }
            }
        }
        return;
    }

    const int scg = (in + blk_c - 1) / blk_c;
    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int o = 0; o < out; o++) {
        const uint8_t *rw = W + (size_t)o * in;
        const uint8_t *srow = S + (size_t)(o / blk_r) * scg;
        float total[DSV4_MAX_BATCH];
        for (int t = 0; t < nt; t++) total[t] = 0.0f;

        int bi = 0;
        for (int c0 = 0; c0 < in; c0 += blk_c, bi++) {
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;
            const float s = dsv4_e8m0_to_f32(srow[bi]);
            for (int t = 0; t < nt; t++)
                total[t] += (float)fp8_block_dot_f32(rw + c0,
                                                     x + (size_t)t * in + c0, n) * s;
        }
        for (int t = 0; t < nt; t++)
            y[(size_t)t * out + o] = total[t];
    }
}
