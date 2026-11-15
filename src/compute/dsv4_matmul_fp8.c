/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul_fp8.c - FP8 batch body and the reference entry point. */
#include <stdint.h>

#include "dsv4_matmul.h"
#include "dsv4_matmul_int.h"
#include "dsv4_matmul_fp8_int.h"

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


/* FP8. One token keeps the register schedule: the block is decoded straight
 * into accumulators, with the exact fallback for subnormal/NaN blocks. More
 * than one token decodes the block once into wf[128] and runs the same tree
 * over it per token. Both apply the scale to the block dot product in double,
 * and both use FP8_BLOCK_SUM, so only the schedule differs. */
void fp8_n(float *y, const float *x, const uint8_t *W,
                  const uint8_t *S, int in, int out, int blk_r, int blk_c,
                  int nt)
{
    const int sc = in / 128;
    const int nrb = (out + 127) / 128;

    if (nt == 1 && blk_c == 128 && (in & 127) == 0 &&
        blk_r == 128 && sc <= 256) {
        _Pragma("omp parallel for schedule(static) if (out > 64)")
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
                double total = 0.0;
                int bi = 0;
                for (int c0 = 0; c0 < in; c0 += 128) {
                    const uint8_t *wb = rw + c0;
                    const float *xb = x + c0;
                    const double acc = fp8_block_has_special(wb)
                                     ? fp8_block_dot_f32_128_gather(wb, xb)
                                     : fp8_block_dot_f32_128_normal(wb, xb);
                    total += acc * sdec[bi++];
                }
                y[o] = (float)total;
            }
        }
        return;
    }

    if (blk_c == 128 && (in & 127) == 0 && blk_r == 128 && sc <= 256) {
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
                        total[t] += (double)dsv4_hsum256(
                                        fp8_block_sum_wf(wf, xb)) * s;
                    }
                }
                for (int t = 0; t < nt; t++)
                    y[(size_t)t * out + o] = (float)total[t];
            }
        }
        return;
    }

    /* Generic block geometry, for hand-built fixtures and any future shape.
     * Same per-block scale fold and same order as the row reference. */
    const int scg = (in + blk_c - 1) / blk_c;
    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int o = 0; o < out; o++) {
        const uint8_t *rw = W + (size_t)o * in;
        const uint8_t *srow = S + (size_t)(o / blk_r) * scg;
        double total[DSV4_MAX_BATCH];
        for (int t = 0; t < nt; t++) total[t] = 0.0;

        int bi = 0;
        for (int c0 = 0; c0 < in; c0 += blk_c, bi++) {
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;
            const double s = (double)dsv4_e8m0_to_f32(srow[bi]);
            for (int t = 0; t < nt; t++)
                total[t] += fp8_block_dot_f32(rw + c0,
                                              x + (size_t)t * in + c0, n) * s;
        }
        for (int t = 0; t < nt; t++)
            y[(size_t)t * out + o] = (float)total[t];
    }
}


void dsv4_matmul_fp8(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    fp8_n(y, x, W, S, in, out, blk_r, blk_c, 1);
}
