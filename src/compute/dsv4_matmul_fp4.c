/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul_fp4.c - FP4 batch body and its nt == 1 entry point. */
#include <stdint.h>

#include "dsv4_matmul.h"
#include "dsv4_matmul_int.h"
#include "dsv4_qmat_dec.h"
/* Float inner block, double fold: 32 products in four independent float
 * accumulators, then the block dot and scale in double. Bounded by the matmul
 * gate in L2, not bitwise equal to the scalar double kernel. */
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

    /* Only hand-built fixtures reach this tail; the real FP4 block is 32. */
    for (; i < n; i++)
        acc += (double)dsv4_fp4_at(row, base + i) * (double)xb[i];
    return acc;
}

static inline double fp4_block_dot_f32_32(const uint8_t *row, int64_t base,
                                          const float *xb)
{
    return (double)dsv4_hsum256(fp4_block_sum32(row, base, xb));
}

/* FP4. One token keeps the four-blocks-at-a-time register schedule. More than
 * one token decodes each 128-weight group ONCE into wf[128], then runs the
 * same four block sums and the same four-lane double fold per token. Both
 * paths fold the block sums in the same order, so (row, token) is bitwise
 * equal to nt one-token calls. */
void fp4_n(float *y, const float *x, const uint8_t *W,
                  const uint8_t *S, int in, int out, int blk_r, int blk_c,
                  int nt)
{
    const int sc = (in + blk_c - 1) / blk_c;

    if (nt == 1 && blk_r > 0 && blk_c == 32 &&
        (in & 31) == 0 && sc <= FP4_SC_MAX) {
        _Pragma("omp parallel for schedule(static) if (out > 64)")
        for (int o = 0; o < out; o++) {
            const uint8_t *row = W + (size_t)o * ((size_t)in / 2);
            const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
            double sdec[FP4_SC_MAX];
            for (int i = 0; i < sc; i++)
                sdec[i] = (double)dsv4_e8m0_to_f32(srow[i]);

            double total = 0.0;
            __m256d acc4 = _mm256_setzero_pd();
            int bi = 0;
            int c0 = 0;
            for (; c0 + 127 < in; c0 += 128, bi += 4) {
                const __m256 s0 = fp4_block_sum32(row, c0, x + c0);
                const __m256 s1 = fp4_block_sum32(row, c0 + 32, x + c0 + 32);
                const __m256 s2 = fp4_block_sum32(row, c0 + 64, x + c0 + 64);
                const __m256 s3 = fp4_block_sum32(row, c0 + 96, x + c0 + 96);

                const __m256 h01 = _mm256_hadd_ps(s0, s1);
                const __m256 h23 = _mm256_hadd_ps(s2, s3);
                const __m256 h   = _mm256_hadd_ps(h01, h23);
                const __m128 lo  = _mm256_castps256_ps128(h);
                const __m128 hi  = _mm256_extractf128_ps(h, 1);
                const __m128 b4  = _mm_add_ps(lo, hi);

                acc4 = _mm256_add_pd(
                    acc4,
                    _mm256_mul_pd(_mm256_cvtps_pd(b4),
                                  _mm256_loadu_pd(sdec + bi)));
            }
            for (; c0 < in; c0 += 32, bi++)
                total += fp4_block_dot_f32_32(row, (int64_t)c0, x + c0) *
                         sdec[bi];

            const __m128d dlo = _mm256_castpd256_pd128(acc4);
            const __m128d dhi = _mm256_extractf128_pd(acc4, 1);
            const __m128d d2  = _mm_add_pd(dlo, dhi);
            const __m128d d1  = _mm_unpackhi_pd(d2, d2);
            total += _mm_cvtsd_f64(_mm_add_sd(d2, d1));
            y[o] = (float)total;
        }
        return;
    }

    if (blk_r > 0 && blk_c == 32 && (in & 31) == 0 && sc <= FP4_SC_MAX) {
        _Pragma("omp parallel for schedule(static) if (out > 64)")
        for (int o = 0; o < out; o++) {
            const uint8_t *row = W + (size_t)o * ((size_t)in / 2);
            const uint8_t *srow = S + (size_t)(o / blk_r) * sc;

            double sdec[FP4_SC_MAX];
            for (int i = 0; i < sc; i++)
                sdec[i] = (double)dsv4_e8m0_to_f32(srow[i]);

            __m256d acc4[DSV4_MAX_BATCH];
            double total[DSV4_MAX_BATCH];
            for (int t = 0; t < nt; t++) {
                acc4[t] = _mm256_setzero_pd();
                total[t] = 0.0;
            }

            int bi = 0;
            int c0 = 0;
            for (; c0 + 127 < in; c0 += 128, bi += 4) {
                float wf[128];
                for (int i = 0; i < 128; i += 8)
                    _mm256_storeu_ps(wf + i, fp4_8(row, (int64_t)c0 + i));

                const __m256d sl = _mm256_loadu_pd(sdec + bi);
                for (int t = 0; t < nt; t++) {
                    const float *xb = x + (size_t)t * in + c0;
                    const __m256 s0 = fp4_block_sum32_w(wf,      xb);
                    const __m256 s1 = fp4_block_sum32_w(wf + 32, xb + 32);
                    const __m256 s2 = fp4_block_sum32_w(wf + 64, xb + 64);
                    const __m256 s3 = fp4_block_sum32_w(wf + 96, xb + 96);

                    const __m256 h01 = _mm256_hadd_ps(s0, s1);
                    const __m256 h23 = _mm256_hadd_ps(s2, s3);
                    const __m256 h   = _mm256_hadd_ps(h01, h23);
                    const __m128 lo  = _mm256_castps256_ps128(h);
                    const __m128 hi  = _mm256_extractf128_ps(h, 1);
                    const __m128 b4  = _mm_add_ps(lo, hi);

                    acc4[t] = _mm256_add_pd(
                        acc4[t],
                        _mm256_mul_pd(_mm256_cvtps_pd(b4), sl));
                }
            }
            for (; c0 < in; c0 += 32, bi++) {
                float wf[32];
                for (int i = 0; i < 32; i += 8)
                    _mm256_storeu_ps(wf + i, fp4_8(row, (int64_t)c0 + i));
                for (int t = 0; t < nt; t++) {
                    const __m256 s = fp4_block_sum32_w(
                        wf, x + (size_t)t * in + c0);
                    total[t] += dsv4_hsum256(s) * sdec[bi];
                }
            }

            for (int t = 0; t < nt; t++) {
                const __m128d dlo = _mm256_castpd256_pd128(acc4[t]);
                const __m128d dhi = _mm256_extractf128_pd(acc4[t], 1);
                const __m128d d2  = _mm_add_pd(dlo, dhi);
                const __m128d d1  = _mm_unpackhi_pd(d2, d2);
                total[t] += _mm_cvtsd_f64(_mm_add_sd(d2, d1));
                y[(size_t)t * out + o] = (float)total[t];
            }
        }
        return;
    }

    /* Generic block geometry. Same scale fold as the fast path, one block at
     * a time; only hand-built fixtures reach it. */
    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int o = 0; o < out; o++) {
        const uint8_t *row = W + (size_t)o * ((size_t)in / 2);
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
        double total[DSV4_MAX_BATCH];
        for (int t = 0; t < nt; t++) total[t] = 0.0;

        int bi = 0;
        for (int c0 = 0; c0 < in; c0 += blk_c, bi++) {
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;
            const double s = (double)dsv4_e8m0_to_f32(srow[bi]);
            for (int t = 0; t < nt; t++)
                total[t] += fp4_block_dot_f32(row, (int64_t)c0,
                                              x + (size_t)t * in + c0, n) * s;
        }
        for (int t = 0; t < nt; t++)
            y[(size_t)t * out + o] = (float)total[t];
    }
}

void dsv4_matmul_fp4(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    fp4_n(y, x, W, S, in, out, blk_r, blk_c, 1);
}
