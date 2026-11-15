# src/60_cache — the streamed routed-expert cache

## Why LRU here and not for the trunk

The trunk is walked in a fixed cyclic order every token, which is the classic
LRU pathology: with fewer slots than layers the next thing needed is always the
thing just evicted, and the hit rate is zero however much memory is added. So
the trunk pins a prefix instead (see `src/50_loader`).

Routed experts are the opposite case. Which six of 256 fire is data dependent
and skewed, so recency genuinely predicts reuse and every extra gigabyte buys
hit rate.

## Measured sizes (DeepSeek-V4-Flash, real checkpoint)

```
one expert          13,369,344 B  (w1/w2/w3 weight + scale, six tensors)
per layer           256 experts = 3.19 GB
whole model         43 layers    = 137 GB, none of it resident
per token           6 x 43 x 13,369,344 = 3.21 GB read
```

Read size matters, but less than first claimed. Measured on the target NVMe
while the disk was also being written:

```
    4.46 MB (one matrix)      4.0 GB/s
   13.37 MB (one expert)      4.5 GB/s
   26.74 MB (two experts)     4.7 GB/s
   80.22 MB (a layer's six)   4.9 GB/s
```

Coalescing an expert's six tensors into one read is worth ~12%, batching a
whole layer's six experts ~22%. Worth doing, not decisive.

## Slot layout

SLOT LAYOUT MIRRORS DISK LAYOUT: all three scales, then all three weights. On
the released checkpoint an expert's six tensors are not six scattered blocks but
TWO contiguous runs, because the checkpoint groups by dtype:

```
layer 2, expert 0
  33,255,000 .. 33,779,288    0.75 MB  w1.scale w2.scale w3.scale
  [ ~325 MB of other tensors ]
 374,830,168 .. 387,413,080   12.00 MB w1.weight w2.weight w3.weight
```

Interleaving them as (w1.w, w1.s, w2.w, ...) would force six reads and a
scatter. Matching disk order lets each run land in one pread with no copy.

The coalescing is DISCOVERED, not assumed: tensor descriptors are sorted by
file offset and merged only where one ends exactly where the next begins. If a
future checkpoint scatters them, this degrades to six reads and stays correct.

## O_DIRECT, alignment and the copy that was not rounding error

The page cache is not merely unhelpful for these reads, it is harmful: a 148 GB
expert working set cannot be cached in 23 GB, so every buffered read evicts
pages that WOULD have been reused. Measured cold: 1.00 GB/s buffered against
4.4 GB/s raw O_DIRECT.

An O_DIRECT read is widened outward to a 4096-byte window and its payload starts
somewhere inside it. The runner-up design read into a staging buffer and then
memcpy'd into the slot, and that copy was measurably not free: 12.75 MB took
2.81 ms on the drive while the engine took 3.60 ms per expert — ~4 s of a 45 s
run.

So a slot is allocated 4096-aligned with slack, and each run is placed at the
offset whose residue matches its file offset. The widened window is read STRAIGHT
INTO THE SLOT. The staging buffer survives only as a fallback.

The placement must start at least `pad` bytes past the previous run, otherwise
the widened window reaches back and clobbers the tail of the run already
loaded. Reserving only "after the cursor" is not enough, and the damage is
silent: correct-looking logits drift, routing collapses onto a few experts, and
the cache hit rate goes UP.

## Reader pool

Threads whose entire job is to sit inside `pread()`. pthreads, not an OpenMP
team, deliberately: the caller of `dsv4_cache_get_many_begin` is expected to run
a full-width OpenMP matmul between begin and end, and an inner parallel region
opened from inside that would either collapse to one thread — taking reads back
to queue depth one — or oversubscribe every core against the matmul.

A pool, not a thread per batch: 43 layers times 26 tokens is ~1,100 batches in a
short run.

The pool width is `dsv4_threads_readers()` (the CLI's `--readers`), not
`omp_get_max_threads()`: dropping the compute team to the 4 P-cores must not
also drop the reader pool. One staging buffer per reader, plus `bounce[0]` for
synchronous `get()`.

Readers inherit the COMPUTE affinity mask, so `reader_main` re-pins itself to
the E-cores where it spends its life blocked in `pread()` instead of competing
with FP4.

## How wide? topk. The plausible argument for less is wrong.

The argument: this drive plateaus by QD2-4 and is flat to 32, so readers past
the plateau return no bandwidth — while expert matmul was measured 34% slower
during streaming, DMA contending with a memory-bound FP4 kernel. A narrower pool
should buy back matmul time for free.

It does not. Five interleaved rounds on the real checkpoint, `--gen 15
--budget 16`, varying `--readers`:

```
readers            1      2      3      4      6
exposed I/O ms  667.9  488.5  418.3  375.4  351.3
s/token          1.72   1.54   1.40   1.39   1.39
```

Widest wins, and nothing is bought back below it. The sweep is not wrong; it
answers a different question. It measures THROUGHPUT over 48 reads. What end()
waits for is the MAKESPAN of one batch of at most six, against a fixed amount of
hit-matmul to hide it behind. Halve the readers and a six-miss batch takes three
serial rounds instead of one; the drive is no busier, but the caller waits
longer. A queue-depth curve cannot see that, because it never has a deadline.

## Overlapped fetching

`get_many` without overlap is a hard barrier: read all six experts, THEN run all
six matmuls. Disk and expert matmul are the two largest movable components in
the profile and they were strictly serialised, while the hit rate at a realistic
budget is around 50%.

`begin()` does the entire serial decide pass: hits resolved, victims chosen,
stamps written, all in request order and before a byte is read. It fills `out[k]`
for every HIT, leaves NULL for misses, and hands the misses to the pool. `end()`
waits and fills the rest. Between the two calls the caller owns the CPU.

`streaming` goes one step further: each miss is published as its read lands and
the caller consumes it through `wait_any()`. `stream_cap` limits how many reads
are in flight, because issuing all top-k at once makes them finish together,
which leaves nothing to overlap; 2-3 keeps the drive at its plateau and staggers
completions.

DETERMINISM: slot selection, LRU stamps and eviction all happen serially in
request order before I/O starts; only reads run in parallel. Cache state after a
call does not depend on which read finished first.

## Batching a layer

Serialised one at a time, expert reads run at queue depth one, which does not
saturate the drive. `cache_bw <model> qd` sweeps depth 1..32 with expert-sized
O_DIRECT reads over distinct regions of the real checkpoint. Three consecutive
runs agree to within 2%:

```
QD    1     2     3     4     6     8    12    16    24    32
GB/s  3.95  4.98  4.56  4.66  4.97  4.81  4.94  4.70  4.61  4.77
```

Concurrency is worth about 25%, all of it arrives by QD2, and the curve is flat
from there to 32.

- An io_uring submission ring buys NOTHING here. Six preads already sit past the
  knee. Worth measuring before building it.
- More DRIVES is the only remaining lever on read bandwidth, and that is a
  hardware answer.

A live run moves 32.16 GB of experts in 7.8-10.5 s = 3.1-4.1 GB/s, which is
75-95% of what the sweep says the drive does at ANY depth. There is no large
in-situ gap left to explain.

## Cross-layer prefetch

Within one layer the top-k experts are chosen together and are mutually
independent, which is exactly the parallelism this takes. Cross-layer is three
separate questions:

- **EXACT, on a scored layer:** genuinely impossible. Layer L+1's router
  consumes layer L's output, so L+1's expert ids do not exist until L has
  finished.
- **EXACT, on a hash layer** (`layer < num_hash_layers`, 3 of 43): possible and
  implemented. `tid2eid` makes those experts a pure function of the token id, so
  the moment a token is sampled, the next token's layers 0..2 are known exactly.
  ~7% of expert traffic, but free of any guess.
- **SPECULATIVE, on any layer:** possible and unimplemented. A wrong guess costs
  a wasted read, not a wrong token, provided it prefetches into slots that
  cannot evict a live entry.

## Budget policy

A cache smaller than `topk` cannot serve a whole layer. The constraint belongs to
`get_many`, which will not evict a slot it already claimed for the same call;
single-expert use is fine with two slots. The budget planner reserves `topk`
experts before the trunk takes its share, and `moe()` aborts rather than emit a
token routed through fewer experts than the model specifies.

One forward pass touches `n_layers * topk` experts in a fixed cyclic order.
Under LRU a hit needs the entry to survive until the next visit, so if the cache
holds fewer than a full pass's working set, the hit rate is exactly zero no
matter how skewed routing is. At the old fixed 1/4 budget split, Flash ran 160
slots against a 258-expert working set and returned 0 hits on 2,580 requests
while reading 32 GB. Silently useless memory is worse than no memory, so init
names the threshold it is under.
