/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_gemm.c - batched dense matmuls for prompt prefill.
 *
 * THE POINT IS THE WEIGHT, NOT THE ACTIVATION.
 *   A GEMV reads one weight matrix and performs one multiply-add per weight, so
 *   it is bound by how fast those weights arrive. Feeding `nt` token rows
 *   through the same loaded weight block turns that into nt multiply-adds per
 *   weight and moves the kernel toward its compute limit. The kernels below
 *   therefore keep the token loop inside the block loop and decode the weight
 *   block once.
 *
 * WHY THIS IS NOT IN src/core/dsv4_matmul.c
 *   Decode is one token at a time and must stay that way. Keeping the batch
 *   kernels in the prefill module keeps the core file as the single-token
 *   reference and prevents a decode change from silently acquiring a second
 *   reduction path. The price is that the FP8/BF16 decoders below are repeated
 *   from the core; the matmul gate compares every batch result against nt
 *   separate GEMVs, so a divergence is caught rather than trusted.
 *
 * BIT-EXACTNESS
 *   Per (output row, token) every dot product uses the same columns, in the
 *   same order, with the same accumulator tree as the GEMV it replaces.
 *   Batching changes only which independent dot products are in flight. BF16
 *   is bitwise; FP8 reuses the core rowscale body's exact block arithmetic
 *   (normal vs. scalar-special chosen by the same test), so it is bitwise
 *   against dsv4_matmul_fp8 rows as well. FP4 reproduces the production
 *   four-block fold per (row, token), and the matmul gate checks it bitwise.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>

#include "dsv4_prefill_gemm.h"
#include "dsv4_qmat_dec.h"
#include "dsv4_quant.h"

/* ---------------------------------------------------------------- FP8 ----- */
/* Decode one 128-byte block once, into exact f32 values. The normal path uses
 * the same closed form as the core kernel; the rare subnormal/NaN block falls
 * back to the scalar definition, which is what the core table was built from. */
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

/* --------------------------------------------------------- batch kernels -- */
static void bf16_n(float *y, const float *x, const uint16_t *W,
                   int in, int out, int nt)
{
    double *xd = (double *)malloc((size_t)nt * (size_t)in * sizeof *xd);
    if (!xd) {
        fprintf(stderr, "dsv4_prefill_gemm: out of memory\n");
        return;
    }
    for (int t = 0; t < nt; t++)
        for (int i = 0; i < in; i++)
            xd[(size_t)t * in + i] = (double)x[(size_t)t * in + i];

    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int o = 0; o < out; o++) {
        const uint16_t *row = W + (size_t)o * in;
        const int n = in;
        for (int t = 0; t < nt; t++) {
            const float  *xb  = x  + (size_t)t * in;
            const double *xdb = xd + (size_t)t * in; (void)xb;
            double acc;
            BLOCK_DOT_V(LOAD1_BF16, LOAD4_BF16);
            y[(size_t)t * out + o] = (float)acc;
        }
    }
    free(xd);
}

static void fp8_n(float *y, const float *x, const uint8_t *W,
                  const uint8_t *S, int in, int out, int nt)
{
    const int sc = in / 128;
    const int nrb = (out + 127) / 128;

    _Pragma("omp parallel for schedule(static) if (nrb > 1)")
    for (int rb = 0; rb < nrb; rb++) {
        const uint8_t *srow = S + (size_t)rb * sc;
        double sdec[256];
        for (int i = 0; i < sc; i++)
            sdec[i] = (double)dsv4_e8m0_to_f32(srow[i]);

        const int o0 = rb * 128;
        int o1 = o0 + 128;
        if (o1 > out) o1 = out;

        for (int o = o0; o < o1; o++) {
            const uint8_t *rw = W + (size_t)o * in;
            double total[DSV4_MAX_BATCH];
            for (int t = 0; t < nt; t++) total[t] = 0.0;

            int bi = 0;
            for (int c0 = 0; c0 < in; c0 += 128, bi++) {
                const uint8_t *wb = rw + c0;
                float wf[128];
                fp8_decode_block_128(wb, wf);

                const double s = sdec[bi];
                for (int t = 0; t < nt; t++) {
                    const float *xb = x + (size_t)t * in + c0;
                    __m256 a0 = _mm256_setzero_ps();
                    __m256 a1 = _mm256_setzero_ps();
                    __m256 a2 = _mm256_setzero_ps();
                    __m256 a3 = _mm256_setzero_ps();
                    for (int i = 0; i < 128; i += 32) {
                        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(wf + i),
                                             _mm256_loadu_ps(xb + i), a0);
                        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(wf + i + 8),
                                             _mm256_loadu_ps(xb + i + 8), a1);
                        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(wf + i + 16),
                                             _mm256_loadu_ps(xb + i + 16), a2);
                        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(wf + i + 24),
                                             _mm256_loadu_ps(xb + i + 24), a3);
                    }
                    const double acc = dsv4_hsum256(_mm256_add_ps(
                        _mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
                    total[t] += acc * s;
                }
            }
            for (int t = 0; t < nt; t++)
                y[(size_t)t * out + o] = (float)total[t];
        }
    }
}

/* Batched FP4 with the production kernel's arithmetic, not a new reduction
 * order. Each 128-weight group is decoded ONCE into wf[128], then every token
 * runs the same four block sums and the same four-lane double fold. The four
 * block sums are folded exactly as dsv4_matmul_fp4_avx2 folds them, so the
 * result for (row, token) is bitwise equal to nt separate GEMVs. */
static void fp4_n(float *y, const float *x, const uint8_t *W,
                  const uint8_t *S, int in, int out, int blk_r, int blk_c,
                  int nt)
{
    const int sc = (in + blk_c - 1) / blk_c;

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
}

/* One kernel tile. nt is 1..DSV4_MAX_BATCH, which bounds the fixed-size
 * accumulator arrays above and keeps the activation working set small enough
 * for the last-level cache. dsv4_mmq_n() below is the public entry point and
 * walks as many tiles as the caller's token block needs. */
static void mmq_n_tile(float *y, const float *x, const DSV4QMat *m, int nt)
{
    if (nt <= 1) {
        dsv4_mmq(y, x, m);
        return;
    }

    switch (m->wdt) {
    case DSV4_WBF16:
        bf16_n(y, x, (const uint16_t *)m->w, m->cols, m->rows, nt);
        break;
    case DSV4_WFP8:
        /* The fast body assumes the released 128x128 geometry and the fixed
         * scale buffer. Anything else stays on the per-token GEMV. */
        if (m->blk_c == 128 && (m->cols & 127) == 0 &&
            m->blk_r == 128 && m->cols / 128 <= 256) {
            fp8_n(y, x, (const uint8_t *)m->w, (const uint8_t *)m->s,
                  m->cols, m->rows, nt);
        } else {
            for (int t = 0; t < nt; t++)
                dsv4_mmq(y + (size_t)t * m->rows,
                         x + (size_t)t * m->cols, m);
        }
        break;
    case DSV4_WFP4:
        if (m->blk_r > 0 && m->blk_c == 32 && (m->cols & 31) == 0 &&
            m->cols / 32 <= FP4_SC_MAX) {
            fp4_n(y, x, (const uint8_t *)m->w, (const uint8_t *)m->s,
                  m->cols, m->rows, m->blk_r, m->blk_c, nt);
        } else {
            for (int t = 0; t < nt; t++)
                dsv4_mmq(y + (size_t)t * m->rows,
                         x + (size_t)t * m->cols, m);
        }
        break;
    default:
        fprintf(stderr, "dsv4_mmq_n: matrix has unimplemented wdt %d "
                        "(%dx%d)\n", m->wdt, m->rows, m->cols);
        abort();
    }
}

void dsv4_mmq_n(float *y, const float *x, const DSV4QMat *m, int nt)
{
    if (nt <= 0) return;

    /* The prefill block is the whole prompt, but no kernel has to hold the
     * whole block: every token row is independent. Walk the block in
     * DSV4_MAX_BATCH-row tiles. This is also what keeps the activation
     * footprint inside LLC; one giant nt turns the dense GEMMs into a DRAM
     * stream because the output-row loop re-walks x for every row. */
    for (int off = 0; off < nt; off += DSV4_MAX_BATCH) {
        int t = nt - off;
        if (t > DSV4_MAX_BATCH) t = DSV4_MAX_BATCH;
        mmq_n_tile(y + (size_t)off * (size_t)m->rows,
                   x + (size_t)off * (size_t)m->cols, m, t);
    }
}
