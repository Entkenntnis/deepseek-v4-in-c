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
 *   against dsv4_matmul_fp8 rows as well.
 *
 * FP4 is a per-token fallback for now: the dense prefill path never calls it,
 * and a real FP4 batch kernel belongs with routed-expert work.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>

#include "dsv4_prefill_gemm.h"
#include "dsv4_quant.h"

/* --------------------------------------------------------------- BF16 ----- */
static inline double bf16d(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return (double)v.f;
}

static inline __m256d load4_bf16(const uint16_t *row, int i)
{
    const __m128i h = _mm_loadl_epi64((const __m128i *)(row + i));
    const __m128i w = _mm_slli_epi32(_mm_cvtepu16_epi32(h), 16);
    return _mm256_cvtps_pd(_mm_castsi128_ps(w));
}

#define XB4(i) _mm256_loadu_pd(xdb + (i))
#define BLOCK_DOT_V(LOAD1, LOAD4)                                              \
    do {                                                                       \
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();            \
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();            \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16) {                                          \
            v0 = _mm256_fmadd_pd(LOAD4(i),      XB4(i),      v0);              \
            v1 = _mm256_fmadd_pd(LOAD4(i + 4),  XB4(i + 4),  v1);              \
            v2 = _mm256_fmadd_pd(LOAD4(i + 8),  XB4(i + 8),  v2);              \
            v3 = _mm256_fmadd_pd(LOAD4(i + 12), XB4(i + 12), v3);              \
        }                                                                      \
        double bb[4];                                                          \
        _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(v0, v1),              \
                                           _mm256_add_pd(v2, v3)));            \
        acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);                               \
        for (; i < n; i++) acc = fma(LOAD1(i), (double)xb[i], acc);            \
    } while (0)

#define LOAD1_BF16(k) bf16d(row[k])
#define LOAD4_BF16(k) load4_bf16(row, (k))

/* ---------------------------------------------------------------- FP8 ----- */
/* Exact f32 bit pattern for one E4M3 byte. This is the scalar decoder from
 * dsv4_quant.h; the fast vector form below must produce the same bits. */
static inline __m256 fp8_8_normal(const uint8_t *wb, int i)
{
    const __m128i b8 = _mm_loadl_epi64((const __m128i *)(wb + i));
    const __m256i b = _mm256_cvtepu8_epi32(b8);
    const __m256i bits = _mm256_add_epi32(
        _mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x7F)), 20),
        _mm256_set1_epi32(0x3C000000));
    const __m256i sign = _mm256_and_si256(_mm256_slli_epi32(b, 24),
                                          _mm256_set1_epi32((int)0x80000000u));
    return _mm256_castsi256_ps(_mm256_or_si256(bits, sign));
}

/* Any byte with exp==0 (subnormal or zero) or abs==0x7F (NaN). */
static inline int fp8_block_has_special(const uint8_t *wb)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i emask = _mm256_set1_epi8(0x78);
    const __m256i amask = _mm256_set1_epi8(0x7F);
    const __m256i nan = _mm256_set1_epi8(0x7F);
    __m256i any = zero;
    for (int i = 0; i < 128; i += 32) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(wb + i));
        const __m256i abs = _mm256_and_si256(v, amask);
        any = _mm256_or_si256(any, _mm256_cmpeq_epi8(
            _mm256_and_si256(abs, emask), zero));
        any = _mm256_or_si256(any, _mm256_cmpeq_epi8(abs, nan));
    }
    return !_mm256_testz_si256(any, any);
}

static inline double hsum_old(__m256 s)
{
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return (double)_mm_cvtss_f32(t);
}

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
                    const double acc = hsum_old(_mm256_add_ps(
                        _mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
                    total[t] += acc * s;
                }
            }
            for (int t = 0; t < nt; t++)
                y[(size_t)t * out + o] = (float)total[t];
        }
    }
}

void dsv4_mmq_n(float *y, const float *x, const DSV4QMat *m, int nt)
{
    if (nt <= 1) {
        dsv4_mmq(y, x, m);
        return;
    }
    if (nt > DSV4_MAX_BATCH) {
        fprintf(stderr, "dsv4_mmq_n: batch %d exceeds DSV4_MAX_BATCH %d "
                        "for %dx%d\n", nt, DSV4_MAX_BATCH, m->rows, m->cols);
        abort();
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
        for (int t = 0; t < nt; t++)
            dsv4_mmq(y + (size_t)t * m->rows, x + (size_t)t * m->cols, m);
        break;
    default:
        fprintf(stderr, "dsv4_mmq_n: matrix has unimplemented wdt %d "
                        "(%dx%d)\n", m->wdt, m->rows, m->cols);
        abort();
    }
}
