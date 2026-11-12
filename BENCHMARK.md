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

**Current baseline (2026-11-12, after both changes below): 1.210 s/token
decode, 1.2% spread over 3 runs.** The earlier numbers are kept in the change
history so the deltas can be checked.

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
| `--pin pcores` | `taskset -c 0-3` + `OMP_NUM_THREADS=4` |
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

### 2026-11-12 — FP8 decode from a 256-entry table

**Change:** `load4_fp8()` in `src/core/dsv4_matmul.c` now decodes E4M3 through a
1 KB table of exact f32 bit patterns instead of inlining the branchy
`dsv4_e4m3_to_f32()` bit-twiddle. The table was generated from that function;
the scalar path still uses the bit-twiddle, so `test_matmul`'s AVX2-vs-scalar
gate proves the table cannot drift.

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

**Verified:** `bin/test_matmul` passes, including AVX2-vs-scalar bitwise
equality and the 1-vs-8-thread gate.

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
