/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul_fp4.c - FP4 batch body and its nt == 1 entry point. */
#include <stdint.h>

#include "dsv4_matmul.h"
#include "dsv4_matmul_int.h"
#include "dsv4_qmat_dec.h"

/* Float inner block, float cross-block fold. Bounded by the matmul gate in
 * L2, not bitwise equal to the scalar double kernel. */
static inline double fp4_block_dot_f32(const uint8_t *row, int64_t base,
                                       const float *xb, int n)
{
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        a0 = _mm256_fmadd_ps(fp4_8(row, base + i),
                             _mm256_loadu_ps(xb + i), a0);
        a1 = _mm256_fmadd_ps(fp4_8(row, base + i + 8),
                             _mm256_loadu_ps(xb + i + 8), a1);
        a2 = _mm256_fmadd_ps(fp4_8(row, base + i + 16),
                             _mm256_loadu_ps(xb + i + 16), a2);
        a3 = _mm256_fmadd_ps(fp4_8(row, base + i + 24),
                             _mm256_loadu_ps(xb + i + 24), a3);
    }

    __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1),
                             _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    double acc = (double)_mm_cvtss_f32(t);

    for (; i < n; i++)
        acc += (double)dsv4_fp4_at(row, base + i) * (double)xb[i];
    return acc;
}

/* FP4. One token decodes four 32-wide blocks at a time; more than one token
 * decodes each 128-weight group once into wf[128]. Both accumulate four block
 * sums and four scales as one float vector, with the same four partial
 * accumulators and the same final tree, so batched output remains bitwise
 * equal to nt one-token calls. */
void fp4_n(float *y, const float *x, const uint8_t *W,
           const uint8_t *S, int in, int out, int blk_r, int blk_c, int nt)
{
    const int sc = (in + blk_c - 1) / blk_c;

    if (blk_r > 0 && blk_c == 32 && (in & 31) == 0 && sc <= FP4_SC_MAX) {
        if (nt == 1) {
            _Pragma("omp parallel for schedule(static) if (out > 64)")
            for (int o = 0; o < out; o++) {
                const uint8_t *row = W + (size_t)o * ((size_t)in / 2);
                const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
                __m128 a0 = _mm_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                float tail = 0.0f;
                int bi = 0, c0 = 0;

                for (; c0 + 127 < in; c0 += 128, bi += 4) {
                    const __m256 s0 = fp4_block_sum32(row, c0, x + c0);
                    const __m256 s1 = fp4_block_sum32(row, c0 + 32, x + c0 + 32);
                    const __m256 s2 = fp4_block_sum32(row, c0 + 64, x + c0 + 64);
                    const __m256 s3 = fp4_block_sum32(row, c0 + 96, x + c0 + 96);
                    const __m128 b  = dsv4_fold4_ps(s0, s1, s2, s3);
                    const __m128 sc4 = dsv4_e8m0x4_to_ps(srow + bi);
                    switch ((bi >> 2) & 3) {
                    case 0: a0 = _mm_fmadd_ps(b, sc4, a0); break;
                    case 1: a1 = _mm_fmadd_ps(b, sc4, a1); break;
                    case 2: a2 = _mm_fmadd_ps(b, sc4, a2); break;
                    default: a3 = _mm_fmadd_ps(b, sc4, a3); break;
                    }
                }
                for (; c0 < in; c0 += 32, bi++) {
                    const __m256 s = fp4_block_sum32(row, c0, x + c0);
                    tail += dsv4_hsum256(s) * dsv4_e8m0_to_f32(srow[bi]);
                }
                const __m128 t = _mm_add_ps(_mm_add_ps(a0, a1),
                                            _mm_add_ps(a2, a3));
                y[o] = dsv4_hsum128(t) + tail;
            }
            return;
        }

        _Pragma("omp parallel for schedule(static) if (out > 64)")
        for (int o = 0; o < out; o++) {
            const uint8_t *row = W + (size_t)o * ((size_t)in / 2);
            const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
            __m128 acc[4][DSV4_MAX_BATCH];
            float tail[DSV4_MAX_BATCH];
            float wf[128];
            for (int k = 0; k < 4; k++)
                for (int t = 0; t < nt; t++)
                    acc[k][t] = _mm_setzero_ps();
            for (int t = 0; t < nt; t++) tail[t] = 0.0f;

            int bi = 0, c0 = 0;
            for (; c0 + 127 < in; c0 += 128, bi += 4) {
                for (int i = 0; i < 128; i += 8)
                    _mm256_storeu_ps(wf + i, fp4_8(row, (int64_t)c0 + i));
                const __m128 sc4 = dsv4_e8m0x4_to_ps(srow + bi);
                const int slot = (bi >> 2) & 3;
                for (int t = 0; t < nt; t++) {
                    const float *xb = x + (size_t)t * in + c0;
                    const __m256 s0 = fp4_block_sum32_w(wf,      xb);
                    const __m256 s1 = fp4_block_sum32_w(wf + 32, xb + 32);
                    const __m256 s2 = fp4_block_sum32_w(wf + 64, xb + 64);
                    const __m256 s3 = fp4_block_sum32_w(wf + 96, xb + 96);
                    const __m128 b = dsv4_fold4_ps(s0, s1, s2, s3);
                    acc[slot][t] = _mm_fmadd_ps(b, sc4, acc[slot][t]);
                }
            }
            for (; c0 < in; c0 += 32, bi++) {
                for (int i = 0; i < 32; i += 8)
                    _mm256_storeu_ps(wf + i, fp4_8(row, (int64_t)c0 + i));
                const float s = dsv4_e8m0_to_f32(srow[bi]);
                for (int t = 0; t < nt; t++) {
                    const __m256 v = fp4_block_sum32_w(
                        wf, x + (size_t)t * in + c0);
                    tail[t] += dsv4_hsum256(v) * s;
                }
            }
            for (int t = 0; t < nt; t++) {
                const __m128 v = _mm_add_ps(_mm_add_ps(acc[0][t], acc[1][t]),
                                            _mm_add_ps(acc[2][t], acc[3][t]));
                y[(size_t)t * out + o] = dsv4_hsum128(v) + tail[t];
            }
        }
        return;
    }

    /* Generic block geometry, one block at a time; only fixtures reach it. */
    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int o = 0; o < out; o++) {
        const uint8_t *row = W + (size_t)o * ((size_t)in / 2);
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
        float total[DSV4_MAX_BATCH];
        for (int t = 0; t < nt; t++) total[t] = 0.0f;

        int bi = 0;
        for (int c0 = 0; c0 < in; c0 += blk_c, bi++) {
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;
            const float s = dsv4_e8m0_to_f32(srow[bi]);
            for (int t = 0; t < nt; t++)
                total[t] += (float)fp4_block_dot_f32(row, (int64_t)c0,
                                                     x + (size_t)t * in + c0, n) * s;
        }
        for (int t = 0; t < nt; t++)
            y[(size_t)t * out + o] = total[t];
    }
}
