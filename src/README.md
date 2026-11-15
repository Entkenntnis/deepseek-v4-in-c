# src — layers, numbered bottom-up

The numeric prefix is the dependency direction: **a file may include headers
from its own folder or from a LOWER number, never from a higher one.**
`make check` enforces both that rule and the 200-line limit.

```
00_base        profiler, clock, platform queries          (no deps)
10_compute     kernels: matmul, norm, rope, attn, MoE     (no deps)
20_runtime     thread affinity / OpenMP policy            (no deps)
30_tokenizer   byte-level BPE                             (no deps)
40_io          safetensors bytes on disk                  (no deps)
50_loader      weight plan, shard binding, trunk stream   (io)
60_cache       routed-expert LRU + reader pool            (io)
70_decode      one decoder layer, decode path             (compute, loader)
80_prefill     batched prompt prefill                     (decode, cache, loader)
90_cli         resident engine, JSON lines                (everything)
```

`10_compute` and `20_runtime` are also what `make bench` links: a benchmark must
be able to run without a model, so neither may ever reach upward.

`50_loader` and `60_cache` are peers (both sit on `40_io` and do not include
each other); numbering them 50/60 keeps the stack readable. `70_decode` is the
per-token path; `80_prefill` is the same model executed as a batch. Both use
the weight structs and the per-layer state, which is why those live in
`50_loader` and `70_decode` rather than in the CLI.

## Where the prose lives

Headers carry the *contract*; each directory's `README.md` carries the
reasoning, measurements and failure history. If a comment grows past a few
lines of invariant, it belongs in that README.

## Rules of thumb

- A layer never prints an event or reads a CLI argument.
- A layer never knows a file format it does not own: `70_decode` sees
  `DSV4LayerW`, not safetensors.
- No file may exceed 200 lines (`.c` and `.h` alike). If it does, the split is
  usually already visible as a section boundary.
- Run `make check` after moving a file. `make test` links the whole engine, so
  it catches missing symbols but not a newly upward include.
