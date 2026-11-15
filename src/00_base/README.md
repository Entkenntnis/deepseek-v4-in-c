# src/00_base — engine-wide primitives

No dependencies beyond libc. Nothing here knows about tensors, files or tokens.

- `dsv4_prof.h/.c` — the one `DSV4Prof` counter set and `dsv4_now_s()`. Decode,
  prefill and the CLI all write it, and none of them owns it, which is why it is
  not in a layer header. `expert_io` and `expert_mm` are kept apart because one
  is a disk read and the other is arithmetic; `idxer` is a subset of `attn` and
  must not be added twice.
- `dsv4_mem.h` — "how much RAM may this process still take?" for `--budget
  auto`. Reports MemAvailable, not MemFree: Linux's MemAvailable already
  accounts for reclaimable page cache, while MemFree would understate a laptop
  that has been reading model files all morning.
