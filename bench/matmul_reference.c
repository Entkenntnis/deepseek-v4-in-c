/* SPDX-License-Identifier: Apache-2.0 */
/* bench/matmul_reference.c - the slow reference kernels.
 *
 * These are the scalar double-precision bodies, plus the double-accumulator
 * AVX2 FP8 body, that the gates and microbenchmarks compare the production
 * kernels against. They are not part of the engine: src/core/dsv4_matmul.c
 * contains only the paths dsv4 actually dispatches to.
 *
 * The copies are deliberately standalone. A reference must be independent of
 * the production code it checks, and test_matmul/matmul_variants compare the
 * two bitwise (or against the L2 bound) before timing or accepting a result.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "dsv4.h"
#include "dsv4_quant.h"

#define BLOCK_DOT_S(LOAD)                                                      \
    do {                                                                       \
        double a[16] = {0};                                                    \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16)                                            \
            for (int l = 0; l < 16; l++)                                       \
                a[l] = fma(LOAD(i + l), (double)xb[i + l], a[l]);              \
        double b0 = (a[0] + a[4]) + (a[8]  + a[12]);                           \
        double b1 = (a[1] + a[5]) + (a[9]  + a[13]);                           \
        double b2 = (a[2] + a[6]) + (a[10] + a[14]);                           \
        double b3 = (a[3] + a[7]) + (a[11] + a[15]);                           \
        acc = (b0 + b1) + (b2 + b3);                                           \
        for (; i < n; i++) acc = fma(LOAD(i), (double)xb[i], acc);             \
    } while (0)

static inline double bf16d(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return (double)v.f;
}

#define LOAD1_BF16(k) bf16d(row[k])
#define LOAD1_FP8(k)  ((double)dsv4_e4m3_to_f32(wb[k]))
#define LOAD1_FP4(k)  ((double)dsv4_fp4_at(row, (int64_t)c0 + (k)))

static const uint32_t dsv4_e4m3_bits[256] = {
    0x00000000u, 0x3B000000u, 0x3B800000u, 0x3BC00000u, 0x3C000000u, 0x3C200000u, 0x3C400000u, 0x3C600000u,
    0x3C800000u, 0x3C900000u, 0x3CA00000u, 0x3CB00000u, 0x3CC00000u, 0x3CD00000u, 0x3CE00000u, 0x3CF00000u,
    0x3D000000u, 0x3D100000u, 0x3D200000u, 0x3D300000u, 0x3D400000u, 0x3D500000u, 0x3D600000u, 0x3D700000u,
    0x3D800000u, 0x3D900000u, 0x3DA00000u, 0x3DB00000u, 0x3DC00000u, 0x3DD00000u, 0x3DE00000u, 0x3DF00000u,
    0x3E000000u, 0x3E100000u, 0x3E200000u, 0x3E300000u, 0x3E400000u, 0x3E500000u, 0x3E600000u, 0x3E700000u,
    0x3E800000u, 0x3E900000u, 0x3EA00000u, 0x3EB00000u, 0x3EC00000u, 0x3ED00000u, 0x3EE00000u, 0x3EF00000u,
    0x3F000000u, 0x3F100000u, 0x3F200000u, 0x3F300000u, 0x3F400000u, 0x3F500000u, 0x3F600000u, 0x3F700000u,
    0x3F800000u, 0x3F900000u, 0x3FA00000u, 0x3FB00000u, 0x3FC00000u, 0x3FD00000u, 0x3FE00000u, 0x3FF00000u,
    0x40000000u, 0x40100000u, 0x40200000u, 0x40300000u, 0x40400000u, 0x40500000u, 0x40600000u, 0x40700000u,
    0x40800000u, 0x40900000u, 0x40A00000u, 0x40B00000u, 0x40C00000u, 0x40D00000u, 0x40E00000u, 0x40F00000u,
    0x41000000u, 0x41100000u, 0x41200000u, 0x41300000u, 0x41400000u, 0x41500000u, 0x41600000u, 0x41700000u,
    0x41800000u, 0x41900000u, 0x41A00000u, 0x41B00000u, 0x41C00000u, 0x41D00000u, 0x41E00000u, 0x41F00000u,
    0x42000000u, 0x42100000u, 0x42200000u, 0x42300000u, 0x42400000u, 0x42500000u, 0x42600000u, 0x42700000u,
    0x42800000u, 0x42900000u, 0x42A00000u, 0x42B00000u, 0x42C00000u, 0x42D00000u, 0x42E00000u, 0x42F00000u,
    0x43000000u, 0x43100000u, 0x43200000u, 0x43300000u, 0x43400000u, 0x43500000u, 0x43600000u, 0x43700000u,
    0x43800000u, 0x43900000u, 0x43A00000u, 0x43B00000u, 0x43C00000u, 0x43D00000u, 0x43E00000u, 0x7FC00000u,
    0x80000000u, 0xBB000000u, 0xBB800000u, 0xBBC00000u, 0xBC000000u, 0xBC200000u, 0xBC400000u, 0xBC600000u,
    0xBC800000u, 0xBC900000u, 0xBCA00000u, 0xBCB00000u, 0xBCC00000u, 0xBCD00000u, 0xBCE00000u, 0xBCF00000u,
    0xBD000000u, 0xBD100000u, 0xBD200000u, 0xBD300000u, 0xBD400000u, 0xBD500000u, 0xBD600000u, 0xBD700000u,
    0xBD800000u, 0xBD900000u, 0xBDA00000u, 0xBDB00000u, 0xBDC00000u, 0xBDD00000u, 0xBDE00000u, 0xBDF00000u,
    0xBE000000u, 0xBE100000u, 0xBE200000u, 0xBE300000u, 0xBE400000u, 0xBE500000u, 0xBE600000u, 0xBE700000u,
    0xBE800000u, 0xBE900000u, 0xBEA00000u, 0xBEB00000u, 0xBEC00000u, 0xBED00000u, 0xBEE00000u, 0xBEF00000u,
    0xBF000000u, 0xBF100000u, 0xBF200000u, 0xBF300000u, 0xBF400000u, 0xBF500000u, 0xBF600000u, 0xBF700000u,
    0xBF800000u, 0xBF900000u, 0xBFA00000u, 0xBFB00000u, 0xBFC00000u, 0xBFD00000u, 0xBFE00000u, 0xBFF00000u,
    0xC0000000u, 0xC0100000u, 0xC0200000u, 0xC0300000u, 0xC0400000u, 0xC0500000u, 0xC0600000u, 0xC0700000u,
    0xC0800000u, 0xC0900000u, 0xC0A00000u, 0xC0B00000u, 0xC0C00000u, 0xC0D00000u, 0xC0E00000u, 0xC0F00000u,
    0xC1000000u, 0xC1100000u, 0xC1200000u, 0xC1300000u, 0xC1400000u, 0xC1500000u, 0xC1600000u, 0xC1700000u,
    0xC1800000u, 0xC1900000u, 0xC1A00000u, 0xC1B00000u, 0xC1C00000u, 0xC1D00000u, 0xC1E00000u, 0xC1F00000u,
    0xC2000000u, 0xC2100000u, 0xC2200000u, 0xC2300000u, 0xC2400000u, 0xC2500000u, 0xC2600000u, 0xC2700000u,
    0xC2800000u, 0xC2900000u, 0xC2A00000u, 0xC2B00000u, 0xC2C00000u, 0xC2D00000u, 0xC2E00000u, 0xC2F00000u,
    0xC3000000u, 0xC3100000u, 0xC3200000u, 0xC3300000u, 0xC3400000u, 0xC3500000u, 0xC3600000u, 0xC3700000u,
    0xC3800000u, 0xC3900000u, 0xC3A00000u, 0xC3B00000u, 0xC3C00000u, 0xC3D00000u, 0xC3E00000u, 0xFFC00000u,
};

static inline __m256 load8_fp8(const uint8_t *wb, int i)
{
    const __m128i b8 = _mm_loadl_epi64((const __m128i *)(wb + i));
    const __m256i idx = _mm256_cvtepu8_epi32(b8);
    const __m256i bits = _mm256_i32gather_epi32((const int *)dsv4_e4m3_bits,
                                                idx, 4);
    return _mm256_castsi256_ps(bits);
}

#define PD_LO(f) _mm256_cvtps_pd(_mm256_castps256_ps128(f))
#define PD_HI(f) _mm256_cvtps_pd(_mm256_extractf128_ps((f), 1))

#define XB4(i) _mm256_loadu_pd(xdb + (i))

#define BLOCK_DOT_V_8(LOAD1, LOAD8)                                            \
    do {                                                                       \
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();            \
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();            \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16) {                                          \
            const __m256 f0 = LOAD8(i);                                        \
            const __m256 f1 = LOAD8(i + 8);                                    \
            v0 = _mm256_fmadd_pd(PD_LO(f0), XB4(i),      v0);                  \
            v1 = _mm256_fmadd_pd(PD_HI(f0), XB4(i + 4),  v1);                  \
            v2 = _mm256_fmadd_pd(PD_LO(f1), XB4(i + 8),  v2);                  \
            v3 = _mm256_fmadd_pd(PD_HI(f1), XB4(i + 12), v3);                  \
        }                                                                      \
        double bb[4];                                                          \
        _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(v0, v1),              \
                                           _mm256_add_pd(v2, v3)));            \
        acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);                               \
        for (; i < n; i++) acc = fma(LOAD1(i), (double)xb[i], acc);            \
    } while (0)

#define LOAD8_FP8(i) load8_fp8(wb, (i))
#define BLOCK_DOT_FP8(LOAD1) BLOCK_DOT_V_8(LOAD1, LOAD8_FP8)

static double *widen_x(const float *x, int in)
{
    double *d = (double *)malloc((size_t)in * sizeof *d);
    if (!d) {
        fprintf(stderr, "dsv4_matmul: could not widen %d activations\n", in);
        return NULL;
    }
    for (int i = 0; i < in; i++) d[i] = (double)x[i];
    return d;
}

#define BLOCK_DOT_S2(LOAD1, ...) BLOCK_DOT_S(LOAD1)

#define GEN_FP8_BODY(NAME, DOT)                                                \
void NAME(float *y, const float *x, const uint8_t *W,                          \
                          const uint8_t *S, int in, int out,                   \
                          int blk_r, int blk_c)                                \
{                                                                              \
    const int sc = (in + blk_c - 1) / blk_c;                                   \
    double *xd = widen_x(x, in);                                               \
    if (!xd) return;                                                           \
    _Pragma("omp parallel for schedule(static) if (out > 64)")                  \
    for (int o = 0; o < out; o++) {                                            \
        const uint8_t *rw = W + (size_t)o * in;                                \
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;                    \
        double total = 0.0;                                                    \
        int bi = 0;                                                            \
        for (int c0 = 0; c0 < in; c0 += blk_c) {                               \
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;               \
            const uint8_t *wb  = rw + c0;                                      \
            const float   *xb  = x + c0;                                       \
            const double  *xdb = xd + c0; (void)xdb;                           \
            double acc;                                                        \
            DOT(LOAD1_FP8);                                                    \
            /* One multiply per block, as kernel.py does -- not per element. */\
            total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);               \
        }                                                                      \
        y[o] = (float)total;                                                   \
    }                                                                          \
    free(xd);                                                                  \
}

#define GEN_BF16(SUF, DOT)                                                     \
void dsv4_matmul_bf16##SUF(float *y, const float *x, const uint16_t *W,        \
                           int in, int out)                                    \
{                                                                              \
    double *xd = widen_x(x, in);                                               \
    if (!xd) return;                                                           \
    _Pragma("omp parallel for schedule(static) if (out > 64)")                  \
    for (int o = 0; o < out; o++) {                                            \
        const uint16_t *row = W + (size_t)o * in;                              \
        const float  *xb  = x;                                                 \
        const double *xdb = xd; (void)xdb;                                     \
        const int n = in;                                                      \
        double acc;                                                            \
        DOT(LOAD1_BF16, LOAD4_BF16);                                           \
        y[o] = (float)acc;                                                     \
    }                                                                          \
    free(xd);                                                                  \
}

#define GEN_FP4_SCALAR(SUF)                                                    \
void dsv4_matmul_fp4##SUF(float *y, const float *x, const uint8_t *W,          \
                          const uint8_t *S, int in, int out,                   \
                          int blk_r, int blk_c)                                \
{                                                                              \
    const int sc = (in + blk_c - 1) / blk_c;                                   \
    double *xd = widen_x(x, in);                                               \
    if (!xd) return;                                                           \
    _Pragma("omp parallel for schedule(static) if (out > 64)")                  \
    for (int o = 0; o < out; o++) {                                            \
        /* Stored row is in/2 bytes wide. Getting this stride wrong reads half  \
         * a row of the wrong values and still produces finite numbers. */     \
        const uint8_t *row = W + (size_t)o * ((size_t)in / 2);                 \
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;                    \
        double total = 0.0;                                                    \
        int bi = 0;                                                            \
        for (int c0 = 0; c0 < in; c0 += blk_c) {                               \
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;               \
            const float  *xb  = x + c0;                                        \
            const double *xdb = xd + c0; (void)xdb;                            \
            double acc;                                                        \
            BLOCK_DOT_S2(LOAD1_FP4);                                           \
            total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);               \
        }                                                                      \
        y[o] = (float)total;                                                   \
    }                                                                          \
    free(xd);                                                                  \
}

/* The scalar bf16 body ignores the four-wide loader, so it can be a no-op. */
#undef LOAD4_BF16
#define LOAD4_BF16(k) 0

GEN_BF16(_scalar, BLOCK_DOT_S2)
GEN_FP4_SCALAR(_scalar)
GEN_FP8_BODY(dsv4_matmul_fp8_scalar, BLOCK_DOT_S2)
GEN_FP8_BODY(dsv4_matmul_fp8_avx2, BLOCK_DOT_FP8)
