# src/decode — one decoder layer, decode path

This is the per-token path: one token enters, one residual stream leaves. It
does not know about the CLI, the tokenizer or the trunk format; it does know
`DSV4LayerW` and asks a `DSV4ExpertSrc` for routed experts.

## The block

The residual stream is `hc_mult` vectors wide, not one. Every layer boundary
carries `[hc_mult][hidden]`, and a block is

```
residual = x
x, post, comb = hc_pre(x, hc_attn)     hc_mult -> 1
x = attn(attn_norm(x))
x = hc_post(x, residual, post, comb)   1 -> hc_mult
residual = x
x, post, comb = hc_pre(x, hc_ffn)
x = ffn(ffn_norm(x))
x = hc_post(x, residual, post, comb)
```

The SECOND residual is taken after the attention half, not at the top of the
block. Reusing the block's input for both halves is the obvious reading and
quietly removes the attention contribution from the FFN's residual path.

## Three things inside attention that are not obvious

1. **q is normalised twice**, and the second one has no weight:

   ```
   qr = q_norm(wq_a(x))                learned RMSNorm on q_lora
   q  = wq_b(qr) -> [n_heads, head_dim]
   q *= rsqrt(q.square().mean(-1)+eps)  per head, UNWEIGHTED
   ```

   The second runs over `head_dim`, per head, with no parameter. It is not a
   repeat of `q_norm`.

2. **qr is reused by the indexer.** The indexer takes the pre-`wq_b` latent,
   which is why it shares `q_lora_rank` with attention.

3. **The output projection is grouped and low-rank.** `o` is viewed as
   `o_groups` slices, each multiplied by its own slice of `wo_a`, and only then
   does `wo_b` map back to hidden. Flattening `o` and applying `wo_a` as one
   matrix has the right shapes and mixes heads that must stay separate. The
   slice pointer advances by STORED BYTES, which is 2 for BF16 and 1 for FP8
   and packed FP4 (and a row is `cols/2` bytes for FP4).

## Scratch and state

`DSV4Scratch` is reused by every layer within one token. `DSV4LayerState` is
per-layer and persists across tokens — a streaming engine that evicts a
layer's weights must not evict this.

- `expert_acc` has one accumulator per k. With overlapped fetching, expert k's
  matmul runs as soon as k is available, which is not k order; each k needs
  somewhere of its own until the weighted sum can run in k order.
- `expert_acc` must never alias the MoE input (`x1` IS the input at every call
  site). Writing an expert's output there corrupts the input for later experts
  and for the shared expert.
- `shared_acc` is separate because the streamed experts overwrite `expert_acc`.
- `qr` must hold `max(q_lora, hidden)`: it carries the q latent and is reused
  for the block's hidden-width output.

## MoE: overlap without changing a bit

`src->begin()` returns the resident experts immediately and hands the misses to
the cache's reader pool. Their matmuls run while the drive works;
`src->end()` waits for the rest. `src->wait_any()` additionally lets a miss be
computed as soon as its read lands.

This is bit-exact because each k computes into its own accumulator and the
weighted sum into `out` runs afterwards, strictly in k order. Float addition is
not associative, so that ordering is the whole guarantee. The shared expert
runs between begin() and end() — it does not depend on routing — but its result
is added unweighted, after the routed sum.

This directory no longer holds the profiler; see `src/base/dsv4_prof.h`.
