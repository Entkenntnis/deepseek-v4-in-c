# src/40_io — safetensors bytes on disk

This layer only moves bytes. It does not know what a layer, an expert or a
token is; `dsv4_st_find`/`dsv4_st_read` are the whole interface above the file
format.

## Why a hand-written scanner

A safetensors header is machine generated with a rigid shape: one flat object
whose values are all small objects with exactly three known keys. Building a
general DOM for it costs an allocation per node across 78 MB of JSON and
497,220 tensors, for no benefit. The scanner in `dsv4_st_scan.c` walks the text
once and writes straight into the index.

The obvious objection is that a hand-written parser can be subtly wrong, so it
is not trusted: `tools/verify_st.py` reparses the same shard with Python's json
and compares dtype, shape and both offsets for every tensor. The parser is
checked against an external implementation on real data.

`str_()` does handle JSON escapes even though tensor names never contain any:
silently mangling one would corrupt a name and turn a lookup into a spurious
"missing weight".

## Reading

`dsv4_st_read()` is one coalesced `pread`, looped only because the kernel may
return short. This is what the streaming tier calls per expert: a single
contiguous span.

`dsv4_st_read_f32()` widens BF16/F16/U8 to float. It REFUSES FP8, packed FP4
and I64 rather than returning uninitialised output: an FP8 value's magnitude
comes from a block scale in a SEPARATE tensor this signature cannot reach, and
I64 is an index table. Callers dequantise those with both tensors in hand.

Two descriptors are kept per shard: a buffered one and an optional `O_DIRECT`
one for streamed expert reads that must not go through the page cache.
`dsv4_st_read_aligned()` widens a read out to the enclosing 4096-byte window
and reports how far the payload sits inside the buffer, which is what lets the
cache land an aligned read directly in a slot.

Widening is done in fixed chunks (4 MB) rather than into a full-size staging
buffer. `embed_tokens` and `lm_head` are 2.35 GB each as BF16; a transient of
that size on top of the destination is enough to get the process OOM-killed on
a machine the memory plan says has room.
