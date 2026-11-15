# src/90_cli — the resident engine front end

The engine stays loaded: KV state survives from one turn to the next, and a slow
model has to show what it is doing WHILE it does it. This binary speaks one JSON
object per line on stdout. Python reads those lines and paints a TUI or a
one-shot diagnostic report. Nothing here is used by the correctness gates.

## Protocol

stdin commands:

```
{"cmd":"generate","prompt":"...","gen":128,"think":false,"layers":true}
{"cmd":"reset"}
{"cmd":"system","text":"..."}
{"cmd":"stats"}
{"cmd":"ping"}
{"cmd":"quit"}
```

SIGUSR1 cancels the running turn at the next token boundary; the engine emits
`done` with `"stop":"cancel"` and keeps the context usable for the next turn.
Generation is one long synchronous call, so a command on stdin cannot be read
until it returns; a signal flag lets the decode loop stop instead.

stdout events: `ready`, `start`, `layer`, `token`, `prefill`, `done`, `reset`,
`system`, `stats`, `error`.

- Token text is emitted as raw ORIGINAL BYTES in hex (`"hex"`). A BPE token can
  end in the middle of a UTF-8 character, so a JSON string would be invalid; the
  client buffers the bytes and decodes incrementally.
- All timing fields are seconds. `"t"` is relative to the turn's `"start"` event.
- `"sys"` is process-level truth (RSS, peak RSS, CPU, bytes read from disk),
  while `"cache"` and `"trunk"` are the engine's own accounting.

## The pending-token rule

When generation stops on EOS (or on the gen limit), the last token has been
appended to the sequence but NOT forwarded: its KV entry is not in the layer
state yet. The next turn must forward it before appending `<User>`, or the
attention window would have a hole at that position. The engine tracks this
with `pending_forward` and pays the cost at the start of the NEXT turn instead
of stalling the end of the previous one.

## Prompt handling

A turn's new tokens go through one block prefill (see `src/80_prefill`). The
head runs only on the final prompt token; earlier ones would cost a vocab-sized
matmul for a result nothing reads. A conversation's KV cache is extended rather
than rebuilt: `pos0` is where the previous turn stopped.

## Files

| file | contents |
|---|---|
| `dsv4_main.c` | args, signal, command loop |
| `dsv4_engine_init.c` | load model, size the budget, expert source |
| `dsv4_engine.c` | teardown, context reset, tokenizer append |
| `dsv4_run.c` | output head, one token through every layer |
| `dsv4_generate.c` | one turn: prefill then decode |
| `dsv4_emit.c` | JSON events |
| `dsv4_stats.c` | sys/cache/trunk/prof JSON fragments |
| `dsv4_util.c` | timing, process stats, JSON helpers, cancel flag |
| `dsv4_serve_int.h` | `Engine`, `TopK` and internal seams |
