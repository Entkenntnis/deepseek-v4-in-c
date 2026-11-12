# dsv4 TUI

A dependency-free Python TUI for `deepseek-v4-in-c`. It keeps the engine
resident and shows the conversation next to the numbers that decide whether the
run is behaving: per-layer progress, expert-cache hit rate, trunk residency,
RSS/peak RSS, live disk throughput, CPU, per-token latency and the engine's own
time breakdown.

```
make tui               # builds bin/dsv4-serve, then launches the TUI
python3 tui/dsv4_tui.py --demo   # no checkpoint required
```

The TUI uses only the Python standard library (`curses`), so there is nothing to
`pip install` for the plain-text mode.

## Markdown rendering (optional)

Assistant output can be rendered as Markdown. That needs one optional package:

```sh
python3 -m pip install -r tui/requirements.txt
# or with uv on a managed Python:
uv pip install --system -r tui/requirements.txt
```

If `rich` is not installed the TUI still works, falls back to plain wrapped text,
and says so once in the transcript. `--no-markdown` forces plain text, and
`/markdown [on|off]` toggles it at runtime.

The palette is deliberately close to opencode's:

| Markdown | Rendering |
|---|---|
| `*emphasis*` | green italic |
| `**strong**` | orange bold |
| `# heading` | orange bold (h3+ yellow/cyan) |
| `` `inline code` `` | cyan bold |
| fenced code blocks | monokai syntax colours on dark grey |
| `- bullet` | orange bullet |
| `1. item` | cyan number |
| `> quote` | magenta |
| `[link](…)` | blue underline |
| tables | cyan border, orange header |

Rich parses CommonMark, lays out the blocks and produces styled spans. The TUI
converts those spans to curses colour pairs and attributes; no ANSI escape
sequences reach the terminal. A growing stream is re-rendered from the same
message text, so an incomplete code fence simply renders as text until it
closes.

## Why a separate backend

`bin/dsv4` is deliberately one-shot: load, print ids, report totals, exit. The
TUI needs the opposite shape — KV state that survives across turns and events
emitted *during* a 1.6–2.2 s/token forward pass. `bin/dsv4-serve` is an additive
CLI that speaks JSON lines:

```
stdin
  {"cmd":"generate","prompt":"...","gen":128,"think":false,"layers":true}
  {"cmd":"reset"}
  {"cmd":"system","text":"..."}
  {"cmd":"stats"}
  {"cmd":"quit"}

stdout, one object per line
  ready  start  layer  token  done  reset  system  stats  error
```

Token text is emitted as the token's **original bytes in hex**, not as JSON text.
A byte-level BPE token can end in the middle of a UTF-8 character; the client
buffers the bytes and decodes incrementally. Special tokens are decoded as their
literal spelling and stripped by the TUI.

`--layers` (default on) emits one `layer` event after every decoder layer. That
is what makes the layer counter and the live cache/disk rates move during a
token. With `--no-layers` there are only per-token events, which is quieter on a
slow terminal or over a remote link.

The one-shot CLI and all correctness gates are unchanged. The only engine-side
additions needed for streaming were a tokenizer decoder (the old code could
encode but not decode token ids to bytes) and the new `dsv4_serve.c`.

## Defaults

The defaults match the development machine:

| flag | default |
|---|---|
| `--model` | `~/colibri-models/deepseek-v4-flash` |
| `--trunk` | `~/colibri-models/dsv4-trunk-flash` |
| `--tok` | `~/colibri-models/dsv4-tok-flash.bin` |
| `--binary` | `bin/dsv4-serve` (relative to this repo) |
| `--budget` | `10.0` GB (or `auto`) |
| `--ctx` | `4096` tokens |
| `--gen` | `256` tokens per turn |

Budget matters. The trunk is ~6.3 GB and one decode pass touches ~3.2 GB of
experts; a budget below roughly 9.5 GB leaves the expert cache with fewer slots
than a forward pass needs, so it misses on every request. At `--budget 10` the
cache is a little above that working set. Raise it if you have the RAM; the TUI
shows the hit rate directly, so the effect of the number is visible rather than
theoretical. `--budget auto` reads `MemAvailable` and subtracts the model's
embed/head weights, a 1 GiB reserve and 0.5 GiB of runtime, which is the right
choice on a machine whose free memory is not known in advance.

`--ctx` sizes the per-layer KV state and the RoPE tables. 4096 is cheap on this
model; raise it for longer conversations.

## Keys and commands

| key | action |
|---|---|
| `Enter` | send the message / run the command |
| `PgUp` / `PgDn` | scroll the transcript |
| `Up` / `Down` | scroll one line |
| `Ctrl-U` | clear the input line |
| `Esc` | clear the input line |
| `Ctrl-C` | quit (kills the engine process) |

| command | action |
|---|---|
| `/help` | list commands |
| `/reset` | clear model context (the expert cache stays warm) |
| `/clear` | clear the display only |
| `/system TEXT` | system prompt for the next reset |
| `/think [on|off]` | thinking mode for the next turn |
| `/gen N` | max tokens per turn |
| `/layers [on|off]` | per-layer events |
| `/markdown [on|off]` | Markdown rendering |
| `/stats` | ask the engine for a fresh snapshot |
| `/exit` | quit |

Only one turn generates at a time. The engine is greedy (the repository's
bit-exactness contract is built around that), so this is a continuation engine,
not a sampler playground.

## Reading the panel

Stats in the right-hand panel are sampled at **1 Hz**, plus a final snapshot at
the end of a turn. Layer progress and token text still arrive on every event, so
the transcript and the layer counter stay live; the numbers are deliberately not
repainted 27 times a second, and rates over a one-second window are more
meaningful than rates over a single layer.

* **PROCESS / RSS, peak** — from `/proc/self/status`; peak is `VmHWM`, not an
  estimate.
* **PROCESS / disk** — `/proc/self/io` `read_bytes`: actual device reads, which
  is larger than the engine's logical model-byte accounting when O_DIRECT
  alignment widens a read.
* **CPU** — process CPU time delta divided by wall delta, sampled at each event.
* **EXPERT CACHE / hit rate** — the LRU cache for routed experts. Below one
  forward pass's working set this is 0% by construction; that is the number to
  watch when choosing `--budget`.
* **TRUNK / pinned** — the trunk is a cyclic scan, so it pins a prefix and uses
  round-robin for the rest rather than LRU. With a 10 GB budget all 43 layers pin.
* **THIS TURN / speed** — rolling mean of the last 16 token intervals. TTFT
  includes prompt prefill.
* **PROFILE / TURN** — the engine's own `DSV4Prof` counters, reset per turn:
  expert I/O, expert matmul, attention, mHC+norms, shared expert, router.
* **NEXT-TOKEN TOP-3** — greedy choice and the runner-ups with softmax
  probabilities over the full vocab, useful for seeing how decided a token is.

## Caveats

This is a prototype. It assumes a local terminal, one model process, and a
checkpoint already packed with `tools/pack_trunk.py` and
`tools/pack_tokenizer.py`. It does not implement a cancel button (Ctrl-C kills
the whole TUI), speculative decoding, or a chat template beyond the single-turn
`BOS system <User> … <Assistant> </think>` encoding used by the existing CLI;
multi-turn context is kept by appending the official turn tokens and the EOS in
the engine. In thinking mode the generated reasoning stays in the context (the
reference encoder drops earlier reasoning by re-prefilling); use `/reset` to
start clean.
