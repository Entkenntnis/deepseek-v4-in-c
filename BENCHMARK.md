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
| `--synthetic-kv N` | benchmark-only: fill the KV state for an N-token context, start at position N, skip prefill |
| `--synthetic-experts` | benchmark-only: use one fixed FP4 expert in RAM, no routed-expert disk traffic |

To compare **builds** instead of measuring one, use `bench/ab.py`. It runs two
or more binaries interleaved, rotates the order every round, pairs each arm
against the reference within the same round, and checks the generated ids:

```sh
python3 bench/ab.py --bin base=bin/dsv4 --bin lto=bin/variants/dsv4.lto \
  --reps 5 --gen 50 --budget 8 --synthetic-kv 1024 --synthetic-experts \
  --extra "--cpu-pin p --threads 4 --readers 6" --json /tmp/ab.json
```

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

### 2026-11-13 — compiler matrix: `-O3` is the ceiling

**Question.** Is there anything left in the compiler flags? The hot loops are
hand-written AVX2 intrinsics, and `-ffp-contract=off` is a correctness
constraint, so the honest prior was "not much". This measures it instead of
assuming it.

**Harness.** `bench/ab.py` (new) runs two or more binaries interleaved, rotates
the order every round, compares each arm against the reference *in the same
round*, and checks the generated ids. It imports `bench_decode.parse()`, so the
TTFT subtraction and synthetic-KV handling are the same code as the one-shot
baseline. Workload: `--gen 50 --budget 8 --synthetic-kv 1024
--synthetic-experts --cpu-pin p --threads 4 --readers 6`. That is the 1k
compute-only control from the two entries above; the base binary measured
0.6475 s/token here against 0.628 in the cooler single run, so absolute numbers
are session-bound and only the paired deltas are compared.

**Variants.** Built from the same sources with `make -B bin/dsv4
CFLAGS="..."`, copied to `bin/variants/`, so `bin/dsv4` stayed the reference.
One warmup run, five rounds, random order per round.

| arm | flags on top of the baseline | median s/token | spread | paired delta | ids |
|---|---|---:|---:|---:|---|
| base | `-O3 -march=native -ffp-contract=off` | **0.6475** | 5.0% | — | — |
| lto | `-flto=auto` | 0.6605 | 2.6% | +0.11% | same |
| pgo | `-fprofile-generate/use`, 2×20-token profile | 0.6576 | 4.0% | −0.34% | same |
| fma | `-ffp-contract=fast` (upper bound) | 0.6625 | 1.7% | +1.33% | same |
| ofast | `-Ofast` (upper bound) | 0.6559 | 3.9% | +0.92% | same |

The 3-round screen also covered `-flto=auto -fno-semantic-interposition
-fno-plt` (−0.45%), `-falign-functions=32 -falign-loops=32` (+0.25%) and
`-O2` (**+3.96%**, clearly worse). PGO generation with
`-fprofile-update=atomic` was unusable (~35× slower, one 50-token run took
20 minutes); the profile above used the default update and
`-fprofile-correction` at use time.

**Result.** No shippable flag is faster than the current build. LTO and PGO
are inside the noise (−0.3% and +0.1%), and PGO would add a two-build bootstrap
and a machine-specific profile for nothing. More importantly, the two
deliberately non-bit-exact upper bounds are *not faster either*: `-ffp-contract=fast`
and `-Ofast` land at +1.3% and +0.9%, i.e. slower, so the remaining headroom is
not "GCC failed to emit FMA". The generated ids stayed identical over 50 tokens
in both arms, but that is not proof of bit-exactness and neither flag is
mergeable under the engine's contract.

**Decision:** keep `-O3 -march=native -ffp-contract=off` exactly as it is.
The next lever is source-level: the FP8 kernels still accumulate in `double`
(`BLOCK_DOT_V_8`), and FP8 is ~57% of this workload. That is an accuracy-policy
change, not a compiler flag.

```sh
python3 bench/ab.py --bin base=bin/variants/dsv4.base \
  --bin ofast=bin/variants/dsv4.ofast --reps 5 \
  --gen 50 --budget 8 --synthetic-kv 1024 --synthetic-experts \
  --extra "--cpu-pin p --threads 4 --readers 6"
```

### 2026-11-13 — hash-layer expert prefetch through the main cache queue

**Workload.** `--gen 20`, budget 16 GB, real routed experts, P-cores. Same
prompt as the decode baseline. Prefetch is automatic when the cache has room;
the A/B was run with a temporary flag before that flag was removed.

**Design.** Layers below `num_hash_layers` route from `tid2eid`, so all 18
experts (3 layers × top-6) are known from the token id alone. At the start of
each forward pass, `dsv4_cache_prefetch_hash_layers()` reads those rows and
queues all 18 as one batch in the **main cache's** reader FIFO. A normal
`begin()` that asks for a slot still in flight waits on that slot's `pending`
bit instead of issuing a second read. The prefetch is only used when the main
cache can hold 18 experts and still leave one layer's top-k (`nslot > 18 + 6`);
there is no dedicated area and no extra memory.

**What was rejected.** A first implementation used 18 dedicated prefetch slots
and one prefetch reader thread. It was consistently ~10% slower: the single
reader could not keep up, layer 0 waited on it, and the reads competed with the
main pool. That version was removed. The queue above shares the existing 16
readers (the pool defaults to `DSV4_MAX_TOPK` in the test harness; the CLI
passes 6), so the 18 jobs go out at queue depth instead of serially.

**Interleaved A/B** (`off on on off off on`), generated ids identical in all
six runs:

| arm | decode s/token | exposed expert I/O ms/pass |
|---|---|---:|
| off | 0.845 / 0.915 / 0.845 — median **0.845** | 277 / 332 / 284 — median **284** |
| on | 0.825 / 0.820 / 0.830 — median **0.825** | 250 / 255 / 267 — median **255** |

Decode improves **−2.4%** and exposed expert I/O falls **−29 ms/pass**. The
bytes read are unchanged (33.77 GB in both arms): the 359 hash-layer misses are
moved earlier, and the 55 experts already resident are skipped by the prefetch
scan. Routed-expert matmul and attention are flat to slightly better, so the
DMA contention did not eat the gain.

Prefetch counters on the three `on` runs: 359 requests, 359 loads, 0 failures,
4.47 GB, 552–610 ms total waiting across 23 passes (~24–27 ms/pass).

**Status:** automatic, no flag. `dsv4_cache_init()` enables the batch when a
reader pool exists, and every token-start call re-checks that
`nslot > num_hash_layers * topk + topk`. A cache too small to keep one layer's
misses alongside the 18 prefetched experts simply never prefetches. 2.4% is
above this machine's run-to-run noise in this A/B but not enough to be called
large; keep an eye on it in longer traces. Three automatic post-removal runs
measured 0.780 / 0.780 / 0.790 s/token, 359 loads, 0 failures, identical ids.

```sh
python3 bench/bench_decode.py --reps 3 --gen 20 --budget 16
```

### 2026-11-13 — bind: pre-resolved plan and cached pinned layers

**Workload.** `--synthetic-kv 1024 --synthetic-experts --gen 50 --budget 8
--pin p --threads 4 --readers 6`: 1k context, I/O-free control, 50 generated
tokens. The 8 GB budget pins all 43 layers, so this isolates the bind path
rather than trunk reads.

**What was actually slow.** The profile row was **10.6 ms/pass** at baseline.
`bin/bind_resolve` (new; real manifest, no reads or widening) separates the
lookup: the old linear `strcmp` scan over a layer's ~46 manifest entries cost
**51.8 µs/token**; the precomputed map costs **2.3 µs/token**. That is 0.05 ms,
not 10 ms. The rest was re-widening ~3.3 MB of BF16 mHC and compressor vectors
for every layer on every token. A pinned layer's run never moves, and neither
do those widened values, so the binding is now built once and returned from a
per-layer cache. Ring layers still bind per pass.

**Changes.**

1. `dsv4_bind_layer_plan()` exposes the order in which `plan_layer()` requests
   tensors. `DSV4MemSrc` gained an optional `find_idx` callback, and
   `dsv4_trunk_open()` resolves name → manifest index once. The per-token path
   indexes that array instead of running `strcmp`.
2. `dsv4_trunk_bind()` caches the complete `DSV4LayerBind` for pinned layers,
   including a per-layer widen area, instead of re-running the plan and the
   widening loop each token.

**Interleaved A/B** (`old new new old old new`), generated ids identical in all
six runs:

| arm | decode s/token | bind ms/pass |
|---|---|---:|
| old | 0.6202 / 0.6411 / 0.6401 — median **0.6401** | 10.5 / 10.9 / 10.9 |
| new | 0.6277 / 0.6473 / 0.6319 — median **0.6319** | 2.6 / 2.6 / 2.5 |

The median is **−1.3%**, consistent with removing 8.1 ms/pass from a 640 ms
token. Peak RSS went 8.32 → 8.45 GB: each of the 43 pinned layers now keeps its
own ~3.4 MB of widened vectors instead of sharing one transient buffer.

**Verified:** `test_trunk` (streamed bind byte-identical to the shard bind),
`test_bindmem`, `test_bind`; generated ids identical across every arm.

```sh
./bin/bind_resolve                      # lookup cost, linear vs map
python3 bench/bench_decode.py --reps 3 --gen 50 --budget 8 \
  --cpu-pin p --threads 4 --readers 6 \
  --synthetic-kv 1024 --synthetic-experts
```

### 2026-11-13 — context 2k A/B: indexer, top-k and the serial dot kernels

**Why a synthetic KV cache.** A real 2k prefill costs ~1 s per prompt token on
this machine (~34 minutes) and reads the routed experts 2,048 times. The
context-dependent kernels can be put at the same workload without any prefill:
every layer's `kv_cache` is filled with deterministic random values, the
compressor's open window keeps its `-INFINITY` scores, and `n_compressed` is set
to `start_pos / ratio`. The forward loop then starts at `position = 2048` and
generates normally. Output is meaningless but deterministic; correctness is
checked by comparing the generated ids between arms.

This is benchmark-only (`--synthetic-kv START_POS`, plus
`--synthetic-experts` for the I/O-free control); the production path is
unchanged when the flag is absent.

**Arms.** A = `HEAD` + harness; B = A + the heap top-k and parallel/AVX2
indexer (`dsv4_indexer.c` + `dsv4_dot.h`); C = B + the vectorised attention
score dot and mHC. Each arm was run twice, interleaved `A B C C B A`, 30
tokens, 2k context. Generated ids were **identical across all six runs** in
both the real-expert and synthetic-expert experiments.

**Direct kernel timers** (medians, ms/pass; `indexer` is a subset of
`attention`):

| arm | indexer | attention | mHC |
|---|---:|---:|---:|
| A old | 164.0 | 635.3 | 38.8 |
| B indexer-only | 17.7 | 482.3 | 38.9 |
| C full | 18.4 | 363.2 | 7.3 |

The indexer block — projections, score pass and top-k — falls from **164 ms to
~18 ms per pass** at 2k. Attention falls by the same 148 ms in B; C removes
another ~116 ms through the vectorised score dot. mHC falls 38.8 → 7.3 ms.

**Clean decode rate, I/O-free control.** The machine was under external I/O
load, so the real-expert wall clock was dominated by exposed disk (the same
7,998 requests / 828 hits / 7,170 misses in every arm, but 684–1,729 ms/pass
of exposed read time). A second matrix replaced the routed experts with one
fixed FP4 expert in RAM, leaving attention, indexer, router, shared expert and
the full layer stack real. This is CPU-bound and reproducible:

| arm | s/token | tok/s | vs A |
|---|---:|---:|---:|
| A old | 0.965 | 1.04 | — |
| B indexer-only | 0.815 | 1.23 | **+18.5 %** |
| C full | 0.671 | 1.49 | **+43.9 %** |

The indexer-only gain is ~15 % less wall time; the rest of the context package
adds the attention score dot and mHC. The real-expert run confirms the same
kernel deltas, but its wall clock is not a clean instrument while the disk is
shared with other work.

Commands:

```sh
# real experts, true end-to-end
./bin/dsv4 ... --prompt "The" --synthetic-kv 2048 --gen 30 \
  --budget 16 --cpu-pin p --threads 4 --readers 6

# I/O-free control: same model, one fixed FP4 expert in RAM
./bin/dsv4 ... --prompt "The" --synthetic-kv 2048 --synthetic-experts \
  --gen 30 --budget 8 --cpu-pin p --threads 4 --readers 6
```

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
(`--readers`, default 6), and the engine pins itself.

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

## FP8 float-inner block — 2026-11-13

**Change:** the production AVX2 FP8 GEMV now forms each 128-wide scale block's
dot product with 8-lane float FMAs and folds the block once in double — the same
shape FP4 adopted on 2026-11-12. The old double kernel now lives in
`bench/matmul_reference.c` as `dsv4_matmul_fp8_avx2`, so `bench/matmul_variants`
can A/B the two; the public `dsv4_matmul_fp8` is the new path. It is **not**
bitwise-equal to the scalar
double reference; `test_matmul` now bounds the L2 relative error at `1e-6`
(measured `8–10e-8`), exactly as it already did for FP4.

**Why:** `perf` on a real decode put **52% of all decode compute cycles in the
FP8 kernel** and 37–40% in FP4. Top-down over the decode window was
`tma_ports_utilized_3m` **79%** and `tma_memory_bound` only **9%**, so the
kernel was execution-port/conversion-bound, not DRAM-bound. `perf annotate`
attributed **33% of the FP8 kernel's cycles to `vcvtps2pd`/`vextractf128`**
(the per-weight f32→f64 widening), 18% to `vpgatherdd`, and 7% to the double
fold. The BF16 head streamed at ~50 GB/s while FP8 attention sat at ~15 GB/s,
which is what ruled out a memory ceiling.

**Kernel**, `bench/matmul_variants`, 4 P-cores, 10 interleaved reps, best of
run. The 128-wide specialisation removes the generic block loop and its tail:

| shape | double | float-inner | speedup |
|---|---:|---:|---:|
| fp8 attn wo_b 4096x8192 | 42.8 GF/s | **50.5 GF/s** | **1.18x** |
| fp8 attn wq_b 32768x1024 | 44.5 GF/s | **51.4 GF/s** | **1.16x** |

`matmul_bw` best of 15 on the same machine then reports 47.0/48.7 GF/s at L2
`8.6e-8`/`8.2e-8`; the absolute numbers move with the thermal state, the
direction does not.

**Full decode**, three interleaved pairs, same prompt, `--budget 16 --gen 20`.
Decode rate is `(wall - TTFT) / 20`, not the printed `s/token`:

| arm | runs (s/token) | median |
|---|---|---:|
| double | 0.755 / 0.870 / 0.795 | 0.795 |
| **float-inner** | 0.710 / 0.705 / 0.735 | **0.710** |

That is **−10.7%**, with identical ids across all six runs. A 50-token
follow-up pair also kept all 51 ids identical (`0.744 -> 0.692 s/token`,
−7.0%). Components from the profile, medians of the three 20-token runs:

| | double | float-inner | delta |
|---|---:|---:|---:|
| attention | 294.9 ms/pass | 246.6 ms/pass | **−16.4%** |
| shared expert | 69.2 ms/pass | 58.5 ms/pass | **−15.5%** |
| routed-expert matmul | 293.7 ms/pass | 287.8 ms/pass | −2.0% (noise) |
| exposed expert disk | 225.2 ms/pass | 223.7 ms/pass | flat |
| lm head + final norm | 21.3 ms/pass | 21.1 ms/pass | flat |

**Verified:** `bin/test_matmul` passes, including the all-256-E4M3-code gate
that is still bitwise (one non-zero term per row); every gate in `make test`
passes. The oracle test files and their fixture targets have been removed from
the tree, so there is no oracle gate to skip.

**Decision:** adopted as the production FP8 path. The FP8 kernel no longer
carries the bit-exact claim; the old double kernel remains available for a
direct A/B. If a longer real-checkpoint run shows id drift, reverting is one
dispatch change.

## FP8 hybrid decode and FP4 mask removal — 2026-11-13

**Question.** After the float-inner change, what is the FP8 kernel actually
waiting on? `perf` answered that before any code changed.

**Where the cycles were.** `perf record -F 1999` over a
`--synthetic-kv 1024 --synthetic-experts --cpu-pin p --threads 4` decode
attributes the compute time to three kernels, not to the model loop:

| symbol | cycles |
|---|---:|
| `dsv4_matmul_fp8_f32._omp_fn.0` | **48.6%** |
| `dsv4_matmul_fp4_avx2._omp_fn.0` | **40.6%** |
| `dsv4_matmul_bf16_avx2._omp_fn.0` | 6.5% |
| `dsv4_sparse_attn._omp_fn.0` | 2.9% |

Top-down on the same window put the FP8 kernel at **53.5% core-bound and 3.1%
memory-bound** (`TopdownL2`); FP4 was **40.1% core-bound and 0.9% memory-bound**.
Neither kernel is limited by DRAM. `perf annotate` then named the FP8 cost:

| instruction in `dsv4_matmul_fp8_f32` | cycles |
|---|---:|
| `vpgatherdd` (16 per 128-weight block) | **42.7%** |
| `vpmovzxbd` (the eight-byte load feeding each gather) | **19.7%** |
| the two `vhaddps` + `vcvtss2sd` horizontal reduction | ~16% |

The gather reads a 1 KB L1 table, so this was not a cache miss. The decisive
counter was `uops_dispatched.load`: **12.7 billion load uops for 300 calls**,
against 2.9 billion for an arithmetic-decode prototype. That is **1.33 load
uops per weight** — the kernel was saturating the load ports.

### FP8: a closed-form decode with an exact fallback

E4M3 has 256 codes, but only 18 of them need a table: the eight subnormal
magnitudes and the two NaNs (times sign). For every normal code the f32 bits are

    0x3C000000 + ((byte & 0x7F) << 20) | sign

so the common case is five integer ops and one load per eight weights, against
one eight-way gather per eight weights.

The model's weights almost never take the exception: in
`layers.0.attn.wo_b.weight`, **0.0145% of bytes are subnormal and none are
NaN**, so a 128-byte block is special-free **98.2%** of the time. The kernel
scans each block with four `vpcmpeqb` tests and takes the old gather path for
the other 1.8%. That makes the new path **bitwise identical** to the old one,
not merely inside its L2 bound.

| `matmul_variants 5 --realistic`, 4 P-cores | wo_b 4096x8192 | wq_b 32768x1024 |
|---|---:|---:|
| gather kernel (`f32inner`) | 49.8 GF/s | 51.2 GF/s |
| **hybrid decode** | **77.9 GF/s** | **78.7 GF/s** |
| **+ row-block scale precompute** | **81.2 GF/s** | **81.2 GF/s** |
| speedup | **1.63x** | **1.59x** |

Both hybrid variants compare **bitwise** against `f32inner` in the harness. The
row-block variant is also bitwise: FP8's scale grid is 128x128, so the same 64
scale bytes are reused by 128 output rows. Decoding them once per row block
rather than 128 times removes the `movzbl` that `perf annotate` had put at 15%
of the kernel's cycles.

**The fallback must not be inlined.** The first hybrid build measured
**74.4 GF/s**, not 83.5. `perf annotate` showed why: because the rare gather
body was `static inline`, GCC hoisted its `vpmovzxbd` loads and spilled eight
YMM registers to the stack *before* the special/not-special branch, so the
98.2% path paid for the 1.8% path on every block. Marking the fallback
`__attribute__((noinline))` raised the same kernel from **74.4 to 83.5 GF/s**
(real tensor, best of 300). The benchmark's committed `--realistic` numbers are
81.2 because that harness uses synthetic activations and scales; the direction
and the bitwise check are the same.

`--realistic` is a new mode in `bench/matmul_variants`. Uniform random bytes
have **6.25% subnormals**, so almost every block would take the fallback and the
benchmark would measure the wrong path. The mode reproduces the exponent
histogram measured from the checkpoint (33,554,432 bytes of `wo_b`).

### FP4: two masks that were never needed

`fp4_8` shifted the nibble into a 32-bit lane and then masked with `0x0F`, then
masked again with `7` before `vpermps`. Neither mask can change a result:
`vpermps` indexes with the low three bits, and the sign test reads bit 3, which
the `0x0F` mask left untouched. Removing both is two fewer vector ops per eight
weights and leaves every output bitwise identical.

| FP4 w2 4096x2048, best of 400 | GF/s |
|---|---:|
| before | 65.5 |
| **after** | **68.5** |

### Full decode, interleaved

`bench/ab.py`, `--gen 30 --budget 8 --synthetic-kv 1024 --synthetic-experts
--cpu-pin p --threads 4 --readers 6`, four rounds, random order, paired against
the baseline within each round:

| arm | median s/token | paired delta | ids |
|---|---:|---:|---|
| baseline | 0.5414 | — | — |
| hybrid FP8 | 0.4609 | **−16.15%** | same |
| **hybrid + row scales** | **0.4441** | **−19.49%** | **same** |

Component medians from the same runs (`ms/pass`):

| | baseline | hybrid | hybrid + row scales |
|---|---:|---:|---:|
| attention (FP8) | 244.7 | 190.2 | **169.5** (−30.7%) |
| shared expert (FP8) | 49.4 | 37.9 | **31.9** (−35.4%) |
| routed experts (FP4) | 219.8 | 221.2 | **213.8** (−2.7%) |
| lm head (BF16) | 21.5 | 22.2 | 22.1 |
| mHC + norms | 7.8 | 7.5 | 7.4 |

With the real expert cache (`--gen 20 --budget 16`, three rounds, random order):

| arm | median s/token | paired delta | ids |
|---|---:|---:|---|
| baseline | 0.7150 | — | — |
| **hybrid + row scales** | **0.6350** | **−10.27%** | **same** |

Attention fell **248.4 → 166.1 ms/pass** and the shared expert **59.3 → 40.0**
in that run. Exposed expert disk rose from 221.2 to 240.4 ms/pass because there
is less compute left to overlap it with; total time still fell.

**Verified:** `make test` — every gate passes, including the all-256-E4M3-code
gate, the 1-thread/8-thread bit-equality gate, and the L2 bound. The new kernel
is bitwise-identical to the previous production kernel on real weights, so the
A/B ids are the same for the right reason, not by luck.

### Measured and rejected

| idea | result | why not |
|---|---:|---|
| branchless arithmetic decode (no gather at all) | 48.3 vs 51.9 GF/s (**−7%**) | removes 4.3x the load uops but costs 2.7x the instructions (20.4B vs 7.6B for 300 calls); it becomes ALU-bound |
| `hsum_fast` (add/shuffle instead of two `vhaddps`) | +0.5–1.5% FP8; +1.8% FP4 | changes the summation order (L2 6e-8); not taken since the gain is small and the current path stays bitwise |
| `#pragma GCC unroll 2` on the 128-wide block loop | 85.1 vs 86.0 GF/s | no gain; the reduction latency is already hidden |

### Where the next change should look

The new profiles, not guesses:

- **FP8 is still 50% backend-bound** (`TopdownL2`: 46.4% retiring, 3.2%
  memory-bound). `perf annotate` now attributes ~16.5% to the two `vhaddps`
  and ~12% to the dependent `vcvtss2sd`. The `hsum_fast` experiment shows that
  replacing the instructions alone is worth little, so the next lever is
  **interleaving two output rows per block pass**: it would halve the activation
  loads and give the scheduler a second independent reduction chain. It has not
  been built, so no number is claimed for it.
- **FP4 is ALU-bound, not load-bound** (55.9% retiring, 42.6% backend, 1.3%
  memory). Its decode is already eight ops per eight weights. Further gains
  need fewer ops or a different representation; more readers or row blocking
  will not help a kernel with 0.28 load uops per weight.
- **Remove the per-block special scan** with a bitmap computed at bind time
  (one bit per 128-byte block, 256 bytes for `wo_b`). The scan itself is ~2.2%
  of the FP8 kernel now. It needs a field in `DSV4QMat` and one pass over the
  matrix when it is bound; not built.
- **BF16 head**: 6.5% of decode cycles, and `widen_x()` still mallocs per call.
  A thread-local widen buffer is a small, unmeasured candidate; the head is not
  hot enough to justify changing it without an A/B.

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
- After the FP8 float-inner change, the **routed-expert FP4 matmul is still the
  largest single compute component**; attention and the shared expert both fell
  ~16% in the A/B. Exposed expert disk is next. That is where the next change
  should be aimed.
- The routed-expert disk and matmul rows overlap, so a change that serialises
  them shows up as disk time rather than matmul time.
- The generated ids must stay identical. If they change, the build is a
  different model and its timing is not comparable.
