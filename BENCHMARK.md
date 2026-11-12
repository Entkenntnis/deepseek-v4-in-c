# Benchmark: decode rate

This file records the baseline this machine produces, so a change can be judged
against something. It is intentionally narrow: **decode only, one fixed
invocation, one machine.** Kernel microbenchmarks live in `bench/matmul_bw.c`,
`bench/cache_bw.c` and `bench/fp8_decode.c`; throughput
claims for those belong in the README, not here.

## The metric

`bin/dsv4` prints

    N tokens in T s, T/N s/token

but `T` starts before prefill and `N` includes the one token the prefill head
picks. On a short prompt that folds 9–12 s of prefill into the rate and makes it
look worse than decode is. This benchmark subtracts TTFT and divides by the
tokens actually decoded:

    decode_s_per_token = (T - TTFT) / (N - 1)

Prefill is deliberately **not** measured. It is a different question (disk
layout, prompt length) and mixing the two hides changes in both.

**Current baseline (2026-11-12, after the thread-policy and gather8 changes):
0.965 s/token decode, 2.6% spread over 3 runs.** Same build with the previous
FP8 decoder (four scalar loads) at the same thread policy measured 1.035; with
the old 8-thread/no-affinity default it was 1.18–1.24. The measurements are in
[Thread policy](#thread-policy-4-p-cores-beat-all-8--2026-11-12) and
[FP8 gather8](#p-core-fp8-decode-gather8-is-now-the-default--2026-11-12).
Earlier numbers are kept in the change history so the deltas can be checked.

## How to run

```sh
make bin/dsv4
python3 bench/bench_decode.py --reps 3
```

Defaults: budget **16 GB**, prompt `"Das Leben ist"` (3 tokens), **20 generated
tokens**, 3 repetitions, model/trunk/tokenizer from `~/colibri-models/`. The
generated ids are checked across repetitions; they must be identical.

Equivalent one-shot command (the script only adds the repetition, parsing and
the TTFT subtraction):

```sh
./bin/dsv4 ~/colibri-models/deepseek-v4-flash \
           --trunk ~/colibri-models/dsv4-trunk-flash \
           --tok   ~/colibri-models/dsv4-tok-flash.bin \
           --prompt "Das Leben ist" --gen 20 --budget 16
```

Useful options:

| option | meaning |
|---|---|
| `--reps N` | repetitions; median is reported (default 3) |
| `--gen N` | generated tokens (default 20) |
| `--budget G` | expert-cache budget in GB (default 16) |
| `--pin pcores` | legacy alias for `--cpu-pin p --threads 4 --readers 4` |
| `--cpu-pin X` | engine affinity: `off`, `p`, `e` or `all` |
| `--threads N` | compute threads passed to the engine |
| `--readers N` | expert-reader threads (independent of compute) |
| `--reader-pin X` | affinity for the reader threads |
| `--keep DIR` | keep raw logs |
| `--json FILE` | write every parsed repetition |

## Machine

| | |
|---|---|
| CPU | Intel Core Ultra 7 258V, 8 CPUs: 4 P-cores (Lion Cove) + 4 E-cores (Skymont), no SMT |
| RAM | 30 GB, 1 GB swap (full) |
| Disk | NVMe, ~4.8 GB/s measured for the trunk's O_DIRECT reads |
| OS / governor | Linux, `powersave` |
| Model layout | trunk pinned (6.27 GB), expert cache 767 slots x 12.75 MB |

## Baseline before scheduling change — 2026-11-12

`--budget 16 --gen 20`, default threads (8), no affinity.

| rep | wall | TTFT | decode s/token | reported s/token |
|---|---|---|---|---|
| 1 | 58.6 s | 11.6 s | 2.350 | 2.79 |
| 2 | 57.2 s | 11.1 s | 2.305 | 2.72 |
| 3 | 57.7 s | 11.4 s | 2.315 | 2.75 |
| **median** | **57.7 s** | **11.4 s** | **2.315** | **2.75** |

Spread over the three runs: **1.9%** (min 2.305, max 2.350).

Fixed per-run counters (identical every repetition, as they should be):

| | |
|---|---|
| generated ids | identical across all 3 runs |
| peak RSS | 17.84 GB |
| trunk | 6.27 GB read in 1.4 s, 4.8 GB/s O_DIRECT |
| expert cache | 767 slots, 5776 requests, 3059 hits (53.0%), 2717 misses, 1950 evictions |
| expert reads | 33.83 GB from disk |

### Where the time went (median run)

| component | seconds | share | ms/pass |
|---|---|---|---|
| attention (FP8 projections + sparse attn) | 27.5 | 47.8% | 1311.3 |
| routed experts (matmul) | 12.9 | 22.5% | 613.4 |
| routed experts (disk) | 6.8 | 11.6% | 322.6 |
| shared expert (FP8) | 7.0 | 12.0% | 333.6 |
| mHC + norms | 1.1 | 1.9% | 51.0 |
| router | 0.1 | 0.3% | 6.7 |
| **accounted** | **55.4** | **96%** | |

The "attention" row is mostly the FP8 GEMVs (`wq_a`, `wq_b`, `wo_a`, `wo_b`,
`wkv`), not the attention kernel: `perf` attributes ~65% of all samples to
`dsv4_matmul_fp8_avx2` and <2% to everything that is not a matmul.

## Change history

### 2026-11-12 — dynamic scheduling in the matmul loops

**Change:** `schedule(static)` → `schedule(dynamic, 4)` in the six
`omp parallel for` loops in `src/core/dsv4_matmul.c`.

**Why:** the rows are independent and equal in cost, but the CPU is 4 P-cores +
4 E-cores. Static hands every thread the same number of rows, so the E-cores
finish last and the P-cores idle at the barrier. Dynamic lets a fast core take
the next chunk. It cannot change a result: rows do not interact, and the
reduction inside a row is untouched.

**Measured** by interleaving the two binaries, 3 runs each, alternating order
(budget 16, gen 20). The static arm had one 2.49 outlier (thermal); even against
its best run the change is ~8%:

| arm | runs (decode s/token) | median |
|---|---|---|
| static | 2.285 / 2.490 / 2.290 | 2.290 |
| dynamic, chunk 8 | 2.100 / 2.110 / 2.105 | **2.105** |

Chunk sweep, one run each in the same session: chunk 4 = 2.100, chunk 8 =
2.115, chunk 16 = 2.120, chunk 32 = 2.185, `guided` = 2.180. Chunk 4 was kept.

Per-component effect (median runs): attention 27.3 → 25.0 s, routed-expert
matmul 12.8 → 9.8 s, shared expert 6.8 → 6.5 s, exposed expert disk 6.4 →
7.4 s (faster compute exposes more of the I/O), mHC 1.0 → 1.3 s.

A later 3-rep run of the shipped build gave 2.090 / 2.185 / 2.140, median
**2.140** (spread 4.4%; the machine was warmer, TTFT 10.3 s against 10.2 s).
That was the state before the FP8 table change below; the current figure is
**1.21**.

**Verified:** generated ids identical to the pre-change baseline in all runs;
`bin/test_matmul` passes including the 1-vs-8-thread bit-identity gate.

### 2026-11-12 — static scheduling after P-core pinning

**Change:** `schedule(dynamic, 4)` → `schedule(static)` in the matmul loops.

**Why:** the dynamic schedule existed for a 4P+4E compute team, where equal row
counts left the P-cores waiting on the E-cores. The default is now four compute
threads pinned to the four identical P-cores, so there is nothing to balance,
and dynamic only adds a dispatch fetch per four rows.

**Measured** with two binaries interleaved over two 3-rep rounds in opposite
order, budget 16, gen 20, identical ids:

| arm | runs (decode s/token) | median |
|---|---|---:|
| dynamic, chunk 4 | 0.955 / 0.995 / 0.950 / 0.940 / 0.945 / 0.955 | 0.9525 |
| **static** | 0.945 / 0.950 / 0.930 / 0.935 / 0.945 / 0.945 | **0.945** |

That is −0.8%, inside the run-to-run noise. The honest conclusion is that
removing dynamic is **neutral** in the pinned configuration, not a speedup; it
is kept for the simplification and because the reason dynamic existed is gone.

**Verified:** generated ids identical across both arms and all runs;
`bin/test_matmul` passes.

### 2026-11-12 — FP8 decode from a 256-entry table

**Change:** the then-current four-load AVX2 FP8 decoder in
`src/core/dsv4_matmul.c` now decodes E4M3 through a
1 KB table of exact f32 bit patterns instead of inlining the branchy
`dsv4_e4m3_to_f32()` bit-twiddle. The table was generated from that function;
the scalar path still uses the bit-twiddle, so `test_matmul`'s bitwise
AVX2-vs-scalar FP8 gate proves the table cannot drift.

**Why not a gather.** `bench/fp8_decode.c` also prototypes an 8-wide
`vpgatherdd` form. It wins on the P-cores (1.15–1.3× over the four loads) but is
about 2× slower on the E-cores — Skymont's gather is microcoded — and the
matmul loops now run on both core types. Four L1 loads win on the mix.

**Kernel microbenchmark** (`make bin/fp8_decode`, best of 5):

| arm | 8 threads | P-cores only | E-cores only |
|---|---|---|---|
| `bit` (old) | 2.9–7.1 GF/s | 2.7–4.0 GF/s | 3.2–3.3 GF/s |
| `lut4` | **19.6–60.9 GF/s** | 30.8–36.4 GF/s | 23.7–30.4 GF/s |
| `gather8` | 20.5–55.5 GF/s | 36.2–44.3 GF/s | 15.0–15.9 GF/s |

At the large shapes the table path runs at ~23–30 GB/s of weight traffic, i.e.
it is finally near the memory stream rather than the instruction limit. All
three arms produced bit-identical output to the scalar reference on every
shape, and the all-256 decode check passes.

**Full-run measurement** (3 reps, no core pinning, budget 16, gen 20, after both
changes):

| rep | wall | TTFT | decode s/token |
|---|---|---|---|
| 1 | 31.9 s | 7.5 s | 1.220 |
| 2 | 31.7 s | 7.5 s | 1.210 |
| 3 | 31.5 s | 7.4 s | 1.205 |
| **median** | **31.7 s** | **7.5 s** | **1.210** |

Spread 1.2%, generated ids identical across all reps. That is **43% less time
per token than the 2.140 dynamic-scheduling baseline** (1.77×) and 1.91×
against the original 2.315.

Per-component, median runs (dynamic only → dynamic + table): attention
25.1 → 7.6 s, shared expert 6.5 → 1.8 s, routed-expert matmul 9.8 → 11.6 s,
exposed expert disk 8.0 → 6.7 s, mHC 1.3 → 1.6 s. The two FP8 components are
3.3–3.6× faster. The routed FP4 matmul is now the largest single component; its
reported seconds rose slightly because the run is shorter and it gets a larger
share of the same thermal budget, not because the FP4 path changed.

Peak RSS and cache counters are unchanged: 17.84 GB, 767 slots, 53.0% hits,
33.8 GB read from disk.

**Repeatability.** The same binary (sha256 `fb70dd6…`) was run again
immediately afterwards: 1.220 / 1.235 / 1.240, median **1.235** (spread 1.6%).
The two medians differ by 2.1%, with identical ids and counters, so quote
**~1.22** as the consensus and treat differences under ~3% as machine state.

**Verified:** `bin/test_matmul` passes, including bf16/FP8 bitwise equality,
the FP4 L2 bound, and the 1-vs-8-thread gate.

## Thread policy: 4 P-cores beat all 8 — 2026-11-12

This machine is 4 P-cores (CPUs 0–3) + 4 LP E-cores (4–7), no SMT. The engine
was measuring a **confound**: `--pin pcores` in the bench meant `taskset -c 0-3`
**and** `OMP_NUM_THREADS=4`, and `OMP_NUM_THREADS` also sized the expert-reader
pool (`nbounce = omp_get_max_threads()`), so it changed core selection, compute
width and queue depth at once. The reader width is now independent
(`max(compute threads, DSV4_READERS)`, default 6), and the engine pins itself.

Matrix, `--budget 16`, `--gen 20`, one rep per cell, two rounds in opposite
order. All cells produced **identical token ids**.

| config | compute | affinity | readers | decode s/token |
|---|---|---|---:|---|
| base | 8 | off | 6 | 1.180 / 1.235 |
| t4all | 4 | off | 6 | 1.040 / 1.040 |
| p4r4 (old `--pin pcores`) | 4 | P 0–3 | 4 | 1.025 / 1.080 |
| p8r6 | 8 | P 0–3 | 6 | 1.030 / 1.030 |
| e4r4 | 4 | E 4–7 | 4 | 1.500 / 1.495 |
| **p4r6** | **4** | **P 0–3** | **6** | **1.015 / 1.035** |
| p4r6e (readers on E) | 4 | P 0–3 | 6 | 1.030 / 1.025 |

Finalists, three interleaved reps:

| config | runs | median |
|---|---|---:|
| **p4r6** | 1.025 / 1.015 / 1.025 | **1.025** |
| p4r6e | 1.075 / 1.015 / 1.020 | 1.020 (one outlier) |
| t4all | 1.040 / 1.050 / 1.035 | 1.040 |
| p8r6 | 1.080 / 1.045 / 1.045 | 1.045 |

Components explain it. Going from the 8-thread default to p4r6: routed-expert
matmul 11.1–11.7 → 8.4–8.5 s (−25%), mHC 1.6 → 0.9 s (−44%), shared expert
1.8 → 1.5 s, attention 7.5–7.8 → 7.4–7.5 s (flat), trunk I/O unchanged. E-cores
are 2.4x slower at the FP4 kernel and worse at the FP64 mHC dots; FP8 attention
barely notices losing them. Pinning the readers to the E-cores (p4r6e) buys
nothing measurable — the drive, not the reader CPU, is the limit.

Isolating the reader pool matters: p4r6 (6 readers) is consistently at or below
p4r4 (4 readers, the old alias), and the old 1.080 outlier in p4r4 is exactly
the kind of variance the confound introduced.

**Default is now `--threads 4 --pin p --readers 6`.** The P-core group is
hardcoded for the Core Ultra 7 258V (`DSV4_P_LO..DSV4_E_HI` in
`src/core/dsv4_threads.c`, commented); on other CPUs pass `--pin off` or
`--pin all`. `--threads 4 --pin off` recovers most of the win (1.040) if
affinity is undesirable.

## P-core FP8 decode: gather8 is now the default — 2026-11-12

The 8-wide `vpgatherdd` decoder had been rejected for a good reason: Skymont's
gather is microcoded, so it was ~2x slower than four scalar loads there. That
reason stopped applying when the compute team moved to the four P-cores. On
Lion Cove the same path is faster on every shape this model uses.

`bench/fp8_decode.c` now interleaves the variants per repetition instead of
measuring all of one and then all of the next -- the old order made the last
variant the warmest one. Pinned to CPU 0–3, `OMP_NUM_THREADS=4`, 20 reps, two
runs, median GF/s:

| FP8 shape | lut4 | gather8 | ratio |
|---|---:|---:|---:|
| attn wq_a 1024×4096 | 30.9 | 39.6 | **1.28x** |
| shared w1 2048×4096 | 34.8 | 42.8 | **1.23x** |
| attn wo_b 4096×8192 | 31.1 | 39.2 | **1.26x** |
| attn wq_b 32768×1024 | 29.4 | 35.4 | **1.20x** |

Every variant is bitwise identical to the scalar reference (`test_matmul`
checks the gather path against it too). Whole-model A/B on the p4r6 thread
policy, three interleaved pairs, identical token ids:

| arm | runs | median |
|---|---|---:|
| four scalar loads | 1.035 / 1.025 / 1.055 | 1.035 |
| **gather8** | 0.940 / 0.950 / 0.960 | **0.950** |

Components: attention 7.5–7.7 → 6.3–6.5 s, shared expert 1.5–1.6 → 1.3 s;
routed FP4 is untouched, as it should be. That is **−8.2%** wall clock on top of
the thread-policy change. A final default 3-rep run measured 0.945 / 0.965 /
0.970 (median 0.965).

Gather8 is now the only FP8 decoder; the four-load path and its environment
override have been removed.

## FP32 inner accumulation is now the FP4 default — 2026-11-12

`bench/fp4_f32.c` asks the two questions separately: what the CPU's raw FMA
ceiling is, and how much of the FP4 expert kernel is actually FMA rather than
dequant. `dsv4_matmul_fp4_avx2` is now the shipped kernel: one 32-weight scale
block accumulates with 8-lane float FMAs, then the block dot is widened and
folded in double. The scalar double kernel remains as the reference and as the
non-AVX2 path. The AVX2 kernel is deliberately **not** bitwise-equal to that
reference; `test_matmul` bounds the difference in L2 norm (1e-6).

**Raw FMA ceiling**, eight independent chains, register-only, `bin/fp4_f32`:

| threads | FP32 | FP64 | ratio |
|---|---:|---:|---:|
| 1, pinned CPU 0 | 146.9 GF/s | 73.6 GF/s | **2.00x** |
| 4, pinned CPU 0–3 | 501.2 GF/s | 214.6 GF/s | **2.34x** |
| 8 | 796.4 GF/s | 396.4 GF/s | **2.01x** |

The one-thread ratio is exactly the vector width (8 floats vs 4 doubles), so
FP64 FMA instruction throughput is not reduced on Lion Cove — FP32 simply does
twice the FLOPs per FMA.

**Kernel result**, real FP4 expert shapes, best of 15:

| shape | double | float inner | speedup |
|---|---:|---:|---:|
| w1 2048x4096 | 44.9 GF/s | 61.3 GF/s | **1.36x** |
| w2 4096x2048 | 44.9 GF/s | 60.6 GF/s | **1.35x** |

The later 32-wide block specialization lifts the shipped kernel to 63.9/63.8
GF/s; see [Loop-variant experiments](#loop-variant-experiments-after-fp32--2026-11-12).

Accuracy against the double reference: L2 relative error `7.1–7.4e-8`; max
absolute `7.6e-5–9.2e-5`; the worst per-element relative error is `4e-3`, but
only on outputs near zero. That is the expected float-block/double-fold shape.

**Full decode**, two binaries interleaved, 3 reps each, `--budget 16 --gen 20`:

| arm | runs (s/token) | median |
|---|---|---:|
| double | 0.960 / 0.980 / 0.970 | 0.970 |
| **float inner** | 0.995 / 0.915 / 0.910 | **0.915** |

That is **−5.7%**, with identical generated ids across both arms. The component
that moved is the routed-expert matmul: `417.5 → 322.1 ms/pass`. Exposed disk
rose `279.2 → 317.4 ms/pass`, because faster compute exposes more of the I/O;
attention and the shared expert were flat.

A reversed-order round on the same binaries repeated the result on a warmer
machine: double `1.080 / 1.080 / 1.105` (median 1.080) against float
`0.925 / 1.010 / 1.025` (median 1.010), **−6.5%**, identical ids. The routed
matmul again fell about 23% (`478.5 → 366.8 ms/pass`), so the direction and size
survive the thermal swing even when the absolute numbers do not.

**Adopted.** `dsv4_matmul_fp4` now always uses this kernel on AVX2+FMA; there
is no switch or environment override. `test_matmul` keeps bf16 and FP8 bitwise
and bounds FP4 at 1e-6 L2 relative error, about an order of magnitude above the
measured 7e-8. The first float rep above (0.995) is the reason to judge the
change by the interleaved medians, not a single run. `make bin/fp4_f32`
reproduces the kernel numbers and the error bound.

## Loop-variant experiments after FP32 — 2026-11-12

`bench/matmul_variants.c` compares experimental loop bodies against the
shipped kernel. It is built with `-DDSV4_MATMUL_EXPERIMENTS`, so the variants
exist only in `bin/matmul_variants`, never in the engine. It prints one progress
line per repetition with every variant's running best, and it checks every
variant against the shipped kernel with `memcmp` before timing.

Pinned to CPU 0–3, `OMP_NUM_THREADS=4`, 10 reps, best of run:

| variant | FP4 w1 | FP4 w2 | verdict |
|---|---:|---:|---|
| original f32 baseline | 60.2 GF/s | 60.1 GF/s | — |
| block-index counter | 62.5 GF/s | 61.9 GF/s | **+3–4%** |
| 32-wide block specialization | **65.3 GF/s** | **64.7 GF/s** | **+8–9%** |
| 32-wide + row-block loop | 65.1 GF/s | 64.3 GF/s | neutral |
| + FP4 LUT gather | 43.3 GF/s | 43.2 GF/s | **−28%** |

The per-block `c0 / blk_c` integer division was real but smaller than expected.
The bigger FP4 win is removing the generic block loop and tail entirely for the
real 32-wide block. The 16-entry LUT gather is a clear loss on this core: the
sign/magnitude ALU decode is cheaper than eight gathers.

For FP8:

| variant | wo_b | wq_b | verdict |
|---|---:|---:|---|
| baseline | 42.7 GF/s | 44.1 GF/s | — |
| block-index counter | 43.1 GF/s | 44.2 GF/s | **+1%** |
| unroll 4 | 40.2 GF/s | 40.2 GF/s | **−6–12%** |
| double-table gather | 37.8 GF/s | 39.1 GF/s | **−12–15%** |

FP8 is already close to the memory stream, so only the counter change is kept.
The double table removes the `vcvtps2pd` instructions but doubles the gather
count and loses more than it saves.

**Hard-coded now:** FP4 uses the 32-wide specialized kernel (with the generic
counter path as a non-32 fallback); FP8 uses the counter-based scale indexing.
All variants are bitwise-equal to the kernel they replaced.

Full-model A/B, old f32 kernel versus the new one, interleaved 3 reps each:

| round | old f32 median | new f32 median | delta |
|---|---:|---:|---:|
| old first | 0.885 | 0.880 | **−0.6%** |
| new first | 0.885 | 0.870 | **−1.7%** |
| combined 6+6 runs | 0.885 | 0.875 | **−1.1%** |

All runs produced identical ids. The routed-expert matmul fell
`315.5–317.8 → 300.2–300.5 ms/pass`; exposed disk and attention were flat, so
most of the kernel gain was absorbed by the other components. The change is
kept because it is strictly faster and bitwise-neutral, not because it moves
the full model much.

## The shared expert runs during the expert reads — 2026-11-12

**Change:** `moe()` now computes the shared expert between `begin()` and
`end()`, into a new `DSV4Scratch.shared_acc`, instead of after the routed sum.
The final `out[i] += shared[i]` still runs immediately after the routed k-order
sum, so the arithmetic into `out` is unchanged.

**Why:** the shared expert depends only on the MoE input `x`, not on the router
or any routed expert. It was 62 ms/pass of pure serial time, run after every
read had finished and the compute team was idle. The expert-read window already
exists for the resident experts; this puts work that has to happen anyway into
it.

**Measured** as five interleaved pairs at `--budget 16 --gen 20`, same prompt,
alternating which build ran first (baseline first in rounds 1–3, new build first
in rounds 4–5):

| arm | runs (decode s/token) | median |
|---|---|---:|
| baseline | 0.895 / 0.885 / 0.885 / 0.900 / 0.885 | 0.885 |
| **shared expert overlapped** | 0.860 / 0.830 / 0.840 / 0.925 / 0.835 | **0.840** |

That is **−5.1%** on the median. One new-build run (0.925) was a thermal
outlier; the median of five is why it does not decide the result. Every round
produced identical generated ids, and all ten runs agreed.

Components, medians of the five runs:

| | baseline | overlapped | delta |
|---|---:|---:|---:|
| exposed expert I/O | 314.2 ms/pass | 259.3 ms/pass | **−54.9** |
| routed-expert matmul | 302.1 ms/pass | 293.8 ms/pass | −8.3 (noise) |
| shared expert | 62.3 ms/pass | 71.8 ms/pass | **+9.5** |

The shared expert itself got 15% slower: it now runs while the reader pool is
streaming into cache slots, so it pays the same DMA-versus-DRAM contention the
resident-expert matmuls already pay. The exposed I/O falls by more than that
contention costs, which is the whole trade.

**Verified:** generated ids identical across all ten runs; `bin/test_layer`
passes; the cache and trunk-bind gates still pass. The model/layer oracle
fixtures are not installed on this machine, so those were not run.

## Measurement rules this machine earned

- **Do not A/B two 60 s runs back to back and believe a 10% difference.** The
  machine heat-soaks. During these three runs the pinned configuration was
  measured immediately afterwards and drifted monotonically:

  | pinned rep | decode s/token | attention s |
  |---|---|---|
  | 1 | 2.100 | 26.2 |
  | 2 | 2.300 | 27.6 |
  | 3 | 2.480 | 30.1 |

  The first pinned run looks 9% faster than the baseline; the third looks 7%
  slower. That is the machine, not the affinity. Interleave the arms, take the
  median of at least three, and treat anything below ~10% as unresolved.
- **`OMP_NUM_THREADS` controls more than compute.** The expert cache sizes its
  reader pool and O_DIRECT staging buffers from `omp_get_max_threads()`
  (`src/cache/dsv4_cache.c`), capped at 6. Dropping to 4 threads also drops the
  reader pool, which is why the pinned runs above trade matmul time for exposed
  I/O instead of winning outright.
- **The reported "s/token" is not the decode rate** when prefill is non-trivial.
  Quote 2.315, not 2.75, or say which one is meant.

## What to watch when changing code

- Decode s/token (the number above) is the primary metric.
- After the FP8 table change, the **routed-expert FP4 matmul is the largest
  single component** (11.6 s of the 31.7 s median run), followed by attention
  (7.6 s) and exposed expert disk (6.7 s). That is where the next change should
  be aimed.
- The routed-expert disk and matmul rows overlap, so a change that serialises
  them shows up as disk time rather than matmul time.
- The generated ids must stay identical. If they change, the build is a
  different model and its timing is not comparable.
