# src/50_loader — weight plan, shard binding, trunk streaming

Two loaders over the same plan:

- `dsv4_bind_*` turns checkpoint tensors into `DSV4LayerW` / `DSV4ModelW`,
  either from the safetensors index (resident) or from a trunk run (streamed).
- `dsv4_trunk_*` streams the dense trunk from a packed `trunk.bin`.

## One plan, three storage classes, five layer shapes

`plan_layer()` declares exactly what one layer needs. Both binding paths walk
that same plan; two independent name lists would drift, and the drift would show
up as a transposed or truncated matrix rather than as an error.

Three storage classes, not two:

- **WIDE** — widened to fp32. Only the small vectors kernels index
  elementwise (norms, sinks, gate bias, mHC parameters), where a silently wrong
  element type is not a crash but a different model.
- **NARROW** — kept as the checkpoint's own bytes, widened inside the matmul.
- **QUANTISED PAIR** — an F8_E4M3 or packed-FP4 weight and its F8_E8M0 scale are
  one unit (`reqq`), never bound independently or half-bound.

Layers are not all the same shape (verified in the released checkpoint):

```
compress_ratio 0    no compressor, no indexer     Flash layers 0,1
compress_ratio 4    compressor + INDEXER          21 of Flash's layers
compress_ratio 128  compressor, no indexer        20 of Flash's layers
layer < num_hash_layers   ffn.gate.tid2eid, NO ffn.gate.bias
layer >= num_hash_layers  ffn.gate.bias, NO tid2eid
```

Requesting a tensor that should be absent is as much a bug as omitting one that
should be present. Routed experts are never bound: they are streamed per token.
The MTP block (`mtp.0.*`) is skipped; this engine does not implement speculative
decoding.

`plan_resolve()` compares every tensor's element count against what the CONFIG
implies before a single byte is read. The expectations are expressions of
config fields, never literals, so pointing the engine at another checkpoint
exercises the same code with different numbers.

## Binding from memory

The trunk holds a layer's tensors as a verbatim copy of their bytes. Binding
from it walks the SAME `plan_layer()`; only the source of the bytes differs.
Narrow matrices are pointed at directly, in place, so streaming a layer costs
one read and no unpacking. Only the small wide vectors are widened into a
per-slot area. Every widened tensor is BF16 or F32 — FP8/FP4 go through `reqq`
and stay narrow because they cannot be widened without their scale partner.

## Why the trunk is streamed

Flash's trunk is 6.37 GB (21 indexed layers at 165.47 MB, 20 compressed at
139.28, 2 dense at 131.03); Pro's is ~50 GB. Streaming turns that floor into a
dial.

**LRU would be the worst possible policy here.** The trunk is walked
`0, 1, ... N-1` on every token. That cyclic scan is the classic LRU pathology:
with fewer slots than layers, the next layer needed is always the one just
evicted, and the hit rate is zero however much memory is added. So the trunk
PINS a prefix and streams the rest through a small ring. Pin K layers and the
hit rate is exactly K/N, deterministically. The routed-expert cache is the
opposite case and does use LRU: which six of 256 experts fire is data
dependent, so recency genuinely predicts reuse.

The read is affordable because the access order is fixed, so the next read is
always known before it is needed. Measured here: one pread per layer from a
packed trunk runs at 4.80 GB/s with O_DIRECT.

## Layout

`tools/pack_trunk.py` writes:

- `trunk.bin` — each layer is ONE contiguous run at a 4096-aligned offset.
- `trunk.json` — every tensor's offset WITHIN its run. Offsets are recorded,
  not derived, because the source checkpoint splits a layer across two runs with
  the routed experts in between.

A pinned layer's run never moves and its widened vectors are token-independent,
so its binding is built once; rebuilding it per token re-widens ~3.3 MB of
mHC/compressor vectors every pass. A ring slot is rebound on every hit.
