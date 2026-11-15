# src/80_prefill — batched prompt prefill

## Why this is a separate path

Decode is a GEMV arranged around one token: weights are read once per matmul and
the SSD is the floor. Prefill is the opposite problem: the same weight is needed
by every prompt token, so the right unit of work is a batch of tokens through a
layer. Keeping the batch code here means the decode path keeps its bit-exact
reduction order and its existing gates.

## What is batched

The dense matrices: q/kv/output projections, the shared expert and the router,
read once per block and used for every token in it. The routed experts are
batched too, by layer-level union: every distinct expert the block routes to is
requested once, and each expert's whole token bucket is run through one FP4
batch kernel.

## What is not batched, deliberately

mHC, RoPE, the compressor's sequential state, the CSA indexer and sparse
attention. They either carry a position-ordered dependency or have no
weight-reuse payoff. Only projections are GEMMs.

## Scheduling

The prompt is one block and is walked layer-major: every token of layer L runs
before L+1. That keeps L's dense weights and routed-expert working set resident
for the whole prompt. `dsv4_mmq_n()` splits the block into fixed kernel tiles
internally, so a very long block does not turn the dense GEMMs into a DRAM
stream.

## Expert union

The old path asked for six experts per token and waited at `end()` before the
next token, which exposed the read tail once per token. The union path asks for
every distinct expert the layer needs, in first-use order, and computes each
expert's whole token bucket as its read lands. The per-token k-order sum is
still done afterwards, so the arithmetic is unchanged. Sources without
begin/end keep the per-token loop.

## Files

| file | contents |
|---|---|
| `dsv4_prefill.c` | block loop, layer walk, trunk bind, prof folding |
| `dsv4_prefill_alloc.c` | block buffers, per-token scratch |
| `dsv4_prefill_pm.c` | the batched-matmul seam and timed wrappers |
| `dsv4_prefill_attn.c` | attention half |
| `dsv4_prefill_experts.c` | one expert, one token / one bucket |
| `dsv4_prefill_union.c` | layer-level request set, then expert-major |
| `dsv4_prefill_moe.c` | router, shared expert, routed dispatch |
| `dsv4_prefill_report.c` | progress callback and stage table |
| `dsv4_prefill_int.h` | block state and internal seams |
