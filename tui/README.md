# dsv4 TUI

A dependency-free Python TUI for `deepseek-v4-in-c`. It keeps the engine
resident and shows the conversation next to a live decode tok/s graph. The
graph appears only while tokens are decoding and carries tok/s and
seconds-per-token numbers in its header; live RSS, available memory, cache-hit
rate, disk-read, CPU-inflow, per-core CPU use and temperature stats sit below
it, and prefill progress is shown separately. The profiler split and top-k
counters are still sent by the engine but are not drawn yet.

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

SIGUSR1
  stop the running turn at the next token boundary; emits done with
  "stop":"cancel". The partial turn is closed like an EOS on the next generate,
  so the context stays valid.

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

Batched prefill is layer-major, so it does not emit one `layer` event per
prompt token. Instead the engine emits `prefill` progress events carrying the
stage (`chunk`, `layer`, `chunk_done`, `head`), chunk/layer counters, an
estimated progress fraction and the same live `sys`/`cache`/`trunk` blocks as
a layer event. While the prompt runs, the panel shows a **PREFILL** section with
the current stage, a progress bar on its own line and the percentage, elapsed
time and rate on the next line; MEMORY keeps updating from the same events, so
cache-hit and disk readout move during the prompt. The DECODE section appears
only once the first token has been decoded. The prompt-only timer also arrives on
the first `token` event, and the transcript gets a
`prefill: N tokens in T s (tok/s, s/token; batch N|per-token)` line before the
assistant bubble.

The one-shot CLI and all correctness gates are unchanged. The streaming
backend is `dsv4_serve.c`; it shares the same `src/prefill/` dense-prefill
module as the one-shot CLI, so both paths use identical math.

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
| `--prefill-batch-size` | `16` tokens per dense prefill chunk (`0` disables) |

The TUI runs the prompt through the dense prefill path in chunks of 16 by
default; `--prefill-batch-size 0` restores the per-token path. Routed experts
stay per token either way, and the generated token stream is identical.

Budget matters. The trunk is ~6.3 GB and one decode pass touches ~3.2 GB of
experts; a budget below roughly 9.5 GB leaves the expert cache with fewer slots
than a forward pass needs, so it misses on every request. At `--budget 10` the
cache is a little above that working set. Raise it if you have the RAM; the
panel's `hit` line shows the effect directly. `--budget auto` reads
`MemAvailable` and subtracts the model's embed/head weights, a 1 GiB reserve
and 0.5 GiB of runtime, which is the right choice on a machine whose free
memory is not known in advance.

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
| click `[cancel]` | stop generation at the next token (dark-red button at the end of the footer) |
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
not a sampler playground. Clicking `[cancel]` stops at the next token boundary;
the partial reply stays in the model context, so the next message continues the
conversation from there. During a long prompt prefill the cancel is honoured
after prefill completes, because the engine does not leave a half-forwarded
prompt in the KV state.

## Reading the panel

The right-hand panel keeps the live resource numbers that are useful while the
model runs, plus a decode graph while a turn is generating tokens:

* **DECODE** — a graph of **tok/s** for generated tokens only. It appears while
  the turn is decoding and disappears during prefill and while idle. The graph
  is right-aligned and scrolls left as new tokens arrive, and uses a linear axis
  with round labels (0, 0.25, 0.5, 0.75, 1.0, 1.25, …).
* **readout** — the latest decode speed sits on the `DECODE` line as
  `xx tok/s (xx s/tok)`.
* **decode-avg** — the mean of every decode sample in the current turn, shown
  directly under `DECODE` in the same `xx tok/s (xx s/tok)` form. The
  `tok/s` value is highlighted in pink because it is the number to watch.
* **MEMORY** — below the PREFILL/DECODE area:
  * **RSS / avail** — current resident set size and `MemAvailable` from
    `/proc/meminfo`, on the same line. Peak RSS is omitted because it barely
    moves after the checkpoint loads.
  * **cache-hit** — expert-cache hit rate.
  * **disk** — current device read rate from `/proc/self/io`.
  * **inflow** — estimated RAM→CPU traffic. Each token reads the dense trunk
    plus `layers × topk` expert slots from RAM, so the readout is
    `(trunk_gb + layers × topk × slot_mb / 1024) × current_tok/s`. This is an
    estimate of model-weight traffic, not a measured memory-bus counter.
* **CPU** — utilization of cores 0–3 from `/proc/stat` deltas, and the CPU
  package temperature from the first matching `hwmon` sensor (preferring
  `Package`, `Tctl`/`Tdie`, then `x86_pkg_temp`). Temperature shows `n/a` when
  no CPU sensor is exposed by sysfs.

The graph, memory and CPU numbers are refreshed at 1 Hz, with an immediate
graph update on each token event. While prefill runs, the PREFILL progress
lines update; the DECODE section appears once the first token is decoded. The
trunk hit rate, profiler split and top-k counters are still collected and sent
by `bin/dsv4-serve`, but are not drawn while this panel is being reworked.

## Caveats

This is a prototype. It assumes a local terminal, one model process, and a
checkpoint already packed with `tools/pack_trunk.py` and
`tools/pack_tokenizer.py`. It does not implement speculative decoding or a
chat template beyond the single-turn
`BOS system <User> … <Assistant> </think>` encoding used by the existing CLI;
multi-turn context is kept by appending the official turn tokens and the EOS in
the engine. In thinking mode the generated reasoning stays in the context (the
reference encoder drops earlier reasoning by re-prefilling); use `/reset` to
start clean.
