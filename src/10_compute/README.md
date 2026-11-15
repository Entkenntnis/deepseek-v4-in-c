# src/10_compute — the kernel layer

This directory contains the compute-heavy code. A function here may assume that
its inputs are already resident in memory; it must not know about safetensors,
the trunk, the expert cache, the reader pool, tokenization or the CLI.

## Contract

- **No file I/O and no cache/trunk types.** The only model-shaped types used
  here are the weight descriptors in `include/dsv4/dsv4.h` (`DSV4QMat`,
  `DSV4HcW`) and scalar geometry passed as arguments.
- **Explicit inputs and outputs.** A kernel receives packed weights, scales and
  activation buffers and writes its result. It does not decide where a weight
  comes from or whether it is resident.
- **Fixed reduction order.** The decode GEMV and the batched GEMM share the
  decoders and reduction trees in `dsv4_qmat_dec.h`; a scheduling change may
  not change the arithmetic. FP4/FP8 accumulate block sums in float with four
  partials and a pairwise final tree; the scalar references remain double.
- **No orchestration.** Layer sequencing, routing decisions across layers,
  prefetching and prefill scheduling live in `src/70_decode`, `src/80_prefill`,
  `src/60_cache`, `src/50_loader` and `src/40_io`. This layer only executes the requested math.

## Files

| file | contents |
|---|---|
| `dsv4_matmul.c/.h` | packed GEMV/GEMM dispatch and token tiling |
| `dsv4_matmul_bf16.c` | BF16 body |
| `dsv4_matmul_fp8.c` | FP8 batch body and `dsv4_matmul_fp8()` |
| `dsv4_matmul_fp8_ref.c` | exact-table FP8 reference for the gate |
| `dsv4_matmul_fp4.c` | FP4 body and `dsv4_matmul_fp4()` |
| `dsv4_matmul_fp8_int.h` | FP8 table and block helpers |
| `dsv4_matmul_int.h` | private declarations between dispatch and bodies |
| `dsv4_quant.h` | E8M0/E4M3/E2M1 decoders |
| `dsv4_qmat_dec.h` | aggregator for the shared block decoders |
| `dsv4_dec_common.h` | horizontal sum |
| `dsv4_dec_bf16.h` | BF16 widening |
| `dsv4_dec_reduce.h` | four-lane double reduction tree (BF16) |
| `dsv4_dec_fp8.h` | E4M3 fast decoder and 128-wide sum |
| `dsv4_dec_fp4.h` | E2M1 vector decode and 32-wide sum |
| `dsv4_dec_scale.h` | float E8M0 scale vectors and 4-block fold |
| `dsv4_dot.h` | f32-input / double-accumulate dot product |
| `dsv4_ops.c` | RMSNorm, clamped SwiGLU, router |
| `dsv4_rope.c` | YaRN table and interleaved rotation |
| `dsv4_attn.c` | sparse attention with sinks, window index list |
| `dsv4_compress.c` | HCA pooling step |
| `dsv4_indexer.c` | CSA score pass and index offsets |
| `dsv4_topk.c` | CSA top-k heap |
| `dsv4_hc.c` | mHC Sinkhorn split, reduce and expand |
| `dsv4_kernels.h` | one declaration of every non-matmul entry point |

The thread-affinity and OpenMP policy is deliberately in `src/20_runtime`; it is
not a kernel and the compute layer does not depend on it.
