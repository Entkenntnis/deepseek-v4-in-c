#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read a dsv4-serve prefill trace and render the activity timeline.

`bench/prefill.py --trace FILE` (or `bench/e2e.py --trace FILE
--trace-phase prefill`) writes one "# prefill ..." block per call. Every line
below it is an event relative to the block start:

    t kind layer k expert slot reader bytes ms n ntodo tag

The interesting prefill kinds are:
    D  a named stage span (qkv, attn_core, gate, shared_2, ...)
    U  one routed-expert token bucket (tag=cpu|gpu|fill)
    E  one layer finished
and the same cache events decode traces use (B/H/Q/R/P/W/K/Z/F). Those show
when each layer's expert reads started, when they published, and how much
compute waited on them.

Examples:
    python3 tools/prefill_trace_view.py t.log --summary
    python3 tools/prefill_trace_view.py t.log --timeline --layer 12
    python3 tools/prefill_trace_view.py t.log --note --layer 12
    python3 tools/prefill_trace_view.py t.log --summary --all-calls
"""
import argparse
import collections
import sys

from trace_view import parse

HC = {"hc_attn", "hc_ffn", "hc_post_a", "hc_post_f"}
IO = {"trunk", "bind"}
ATTN = {
    "qkv_a", "kv", "qnorm", "qkv_b", "comp_kv", "comp_sc", "qhead",
    "kv_norm", "comp_step", "win_idx", "idx_q", "idx_w", "idx_rope",
    "idx_score", "attn_core", "derope", "oproj_pack", "oproj_a",
    "oproj_unpack", "oproj_b",
}
MOE = {"gate", "route", "shared_13", "shared_swiglu", "shared_2",
       "expert_sum"}


def stages(evs):
    out = []
    for e in evs:
        if e["kind"] == "D":
            out.append(e)
    return out


def buckets(evs):
    return [e for e in evs if e["kind"] == "U"]


def layer_events(call, layer):
    return [e for e in call["ev"] if e["layer"] == layer]


def layer_total(evs):
    for e in evs:
        if e["kind"] == "E":
            return e["ms"]
    if not evs:
        return 0.0
    return 1000.0 * (max(e["t"] + e["ms"] / 1000.0 for e in evs) -
                     min(e["t"] for e in evs))


def category(tag):
    if tag in HC:
        return "hc"
    if tag in IO:
        return "io"
    if tag in ATTN:
        return "attn"
    if tag in MOE:
        return "moe"
    return "other"


def summarize_layer(evs):
    st = stages(evs)
    bu = buckets(evs)
    cat = collections.Counter()
    tags = collections.Counter()
    for e in st:
        cat[category(e["tag"])] += e["ms"]
        tags[e["tag"]] += e["ms"]
    bucket_s = collections.Counter()
    for e in bu:
        bucket_s[e["tag"]] += e["ms"]
    reads = [e for e in evs if e["kind"] in ("R", "P")]
    waits = 0.0
    w_t = None
    for e in evs:
        if e["kind"] == "W":
            w_t = e["t"]
        elif e["kind"] in ("K", "Z") and w_t is not None:
            waits += (e["t"] - w_t) * 1000.0 if e["kind"] == "K" else e["ms"]
            w_t = None
    miss = sum(1 for e in evs if e["kind"] == "B")
    bytes_read = sum(e["bytes"] for e in evs if e["kind"] == "P")
    slowest = tags.most_common(1)[0] if tags else ("-", 0.0)
    return {
        "total": layer_total(evs),
        "hc": cat["hc"], "io": cat["io"], "attn": cat["attn"],
        "moe": cat["moe"], "other": cat["other"],
        "buckets": len(bu), "bucket_s": sum(bucket_s.values()),
        "gpu": sum(e["ms"] for e in bu if e["tag"] == "gpu"),
        "cpu": sum(e["ms"] for e in bu if e["tag"] == "cpu"),
        "fill": sum(e["ms"] for e in bu if e["tag"] == "fill"),
        "reads": len(reads), "bytes": bytes_read, "wait": waits,
        "batches": miss, "slowest": slowest,
    }


def print_summary(calls, every):
    print(f"{'call':>4} {'L':>3} {'total':>9} {'hc':>8} {'io':>8} "
          f"{'attn':>9} {'moe':>9} {'exp':>9} {'gpu':>8} {'wait':>8} "
          f"{'reads':>6} {'GB':>7}  slowest")
    for call in calls:
        if not every and call is not calls[0]:
            continue
        if call.get("dropped"):
            print(f"# call {call['token']}: {call['dropped']} events dropped")
        layers = collections.defaultdict(list)
        for e in call["ev"]:
            if e["layer"] >= 0:
                layers[e["layer"]].append(e)
        for L in sorted(layers):
            s = summarize_layer(layers[L])
            print(f"{call['token']:>4} {L:>3} {s['total']:9.3f} "
                  f"{s['hc']:8.3f} {s['io']:8.3f} {s['attn']:9.3f} "
                  f"{s['moe']:9.3f} {s['bucket_s']:9.3f} {s['gpu']:8.3f} "
                  f"{s['wait']:8.3f} {s['reads']:>6} "
                  f"{s['bytes'] / 1e9:7.2f}  {s['slowest'][0]} "
                  f"{s['slowest'][1]:.3f} ms")


def print_tag_totals(calls, every):
    tags = collections.Counter()
    counts = collections.Counter()
    for call in calls:
        if not every and call is not calls[0]:
            continue
        for e in call["ev"]:
            if e["kind"] == "D" and e["layer"] >= 0:
                tags[e["tag"]] += e["ms"]
                counts[e["tag"]] += 1
    if not tags:
        return
    print(f"\n{'stage':<18} {'total ms':>10} {'count':>8} {'avg ms':>9}")
    for tag, ms in tags.most_common():
        print(f"{tag:<18} {ms:10.3f} {counts[tag]:8d} "
              f"{ms / counts[tag]:9.3f}")


def pick(calls, idx, layer):
    call = calls[0]
    if idx is not None:
        sel = [c for c in calls if c["token"] == idx]
        if not sel:
            print(f"prefill_trace_view: call {idx} not in file "
                  f"(have {[c['token'] for c in calls]})", file=sys.stderr)
            return None, None
        call = sel[0]
    layers = sorted({e["layer"] for e in call["ev"] if e["layer"] >= 0})
    if not layers:
        return None, None
    if layer is None:
        return call, layers[0]
    return call, layer


def _col(t, t0, span, width):
    return max(0, min(width - 1, int((t - t0) / span * (width - 1))))


def print_timeline(calls, idx, layer, width=118):
    call, layer = pick(calls, idx, layer)
    if call is None:
        return
    evs = layer_events(call, layer)
    if not evs:
        print(f"prefill_trace_view: no events for call {call['token']} "
              f"layer {layer}", file=sys.stderr)
        return

    t0 = min(e["t"] for e in evs)
    tend = max(e["t"] + e["ms"] / 1000.0 for e in evs)
    span = max(tend - t0, 1e-9)
    rows = collections.defaultdict(lambda: [" "] * width)

    def put(row, t, ch):
        row[_col(t, t0, span, width)] = ch

    def fill(row, t, ms, ch):
        for c in range(_col(t, t0, span, width),
                       _col(t + ms / 1000.0, t0, span, width) + 1):
            row[c] = ch

    # Named stages: one row per tag, ordered by total time.
    by_tag = collections.defaultdict(list)
    for e in evs:
        if e["kind"] == "D":
            by_tag[e["tag"]].append(e)
    totals = {tag: sum(e["ms"] for e in es) for tag, es in by_tag.items()}
    for tag in sorted(by_tag, key=lambda k: (-totals[k], k)):
        row = rows["s:" + tag]
        for e in by_tag[tag]:
            fill(row, e["t"], e["ms"], "#")

    # Expert buckets get one lane per path; a GPU bucket can contain CPU fill
    # work, so those are separate lanes and may overlap.
    for e in evs:
        if e["kind"] == "U":
            fill(rows["b:" + e["tag"]], e["t"], e["ms"],
                 {"gpu": "G", "cpu": "C", "fill": "f"}.get(e["tag"], "?"))

    # Reads: each reader gets a lane between R and P.
    read_start = {}
    for e in evs:
        if e["kind"] == "R":
            read_start[(e["k"], e["expert"])] = e
        elif e["kind"] == "P" and e["reader"] >= 0:
            r = read_start.get((e["k"], e["expert"]))
            if r:
                fill(rows[f"r{e['reader']}"], r["t"], e["ms"], "~")

    # Main lane: waits and the end barrier.
    ordered = sorted(evs, key=lambda e: e["t"])
    for i, e in enumerate(ordered):
        if e["kind"] == "W":
            end = next((o for o in ordered[i + 1:]
                        if o["kind"] in ("K", "Z")), None)
            if end:
                fill(rows["main"], e["t"], (end["t"] - e["t"]) * 1000.0, "w")
        elif e["kind"] == "F":
            fill(rows["main"], e["t"], e["ms"], "F")
        elif e["kind"] == "E":
            put(rows["main"], e["t"], "E")

    axis = [" "] * width
    raw = span * 1000.0 / 8.0
    step = min((0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20,
                50, 100), key=lambda c: abs(c - raw))
    for i in range(int(span * 1000.0 / step) + 1):
        ms = i * step
        c = _col(t0 + ms / 1000.0, t0, span, width)
        lab = f"{ms:g}"
        if c + len(lab) >= width:
            continue
        if any(axis[j] != " " for j in range(c, c + len(lab))):
            continue
        for j, ch in enumerate(lab):
            axis[c + j] = ch

    print(f"call {call['token']} pos {call['pos']} "
          f"({call.get('tokens', 0)} tokens)  layer {layer}  "
          f"start +{t0 * 1000.0:.3f} ms  span {span * 1000.0:.3f} ms  "
          f"({width} cols, {(span * 1000.0) / width:.3f} ms/col)")
    print()
    labels = (sorted((k for k in rows if k.startswith("s:")),
                     key=lambda k: (-totals[k[2:]], k[2:])) +
              sorted(k for k in rows if k.startswith("b:")) +
              sorted(k for k in rows if k.startswith("r")) +
              (["main"] if "main" in rows else []))
    lw = max(len(k.replace("s:", "").replace("b:", "bucket ")) for k in labels)
    for key in labels:
        label = key.replace("s:", "").replace("b:", "bucket ")
        print(f"{label:>{lw}} |{''.join(rows[key])}|")
    print(f"{'ms':>{lw}} |{''.join(axis)}|")
    print()
    print("Legende  s:... = benannte Stage      bucket gpu G / cpu C / fill f")
    print("         rN = Reader N (R..P)         main: w = wait_any blockiert, "
          "F = end()-Barriere, E = Layer fertig")
    print("         Scale: # = Stage/Bucket, ~ = pread")


def fmt_event(e):
    k = e["kind"]
    t = f"{e['t'] * 1000.0:9.3f}"
    if k == "E":
        return f"{t}  layer span {e['ms']:.3f} ms"
    if k == "D":
        return f"{t}  stage {e['tag']:<14} {e['ms']:9.3f} ms"
    if k == "B":
        return f"{t}  batch  n={e['n']} miss={e['ntodo']} dup={e['k']} " \
               f"pf-wait={e['slot']}"
    if k == "H":
        return f"{t}  hit    k{e['k']} e{e['expert']} ({e['tag']})"
    if k == "Q":
        return f"{t}  queue  k{e['k']} e{e['expert']}"
    if k == "R":
        return f"{t}  read   k{e['k']} e{e['expert']} reader{e['reader']}"
    if k == "P":
        return f"{t}  pub    k{e['k']} e{e['expert']} reader{e['reader']} " \
               f"{e['bytes'] / 1e6:.2f} MB {e['ms']:.3f} ms ({e['tag']})"
    if k == "W":
        return f"{t}  wait   enter"
    if k == "K":
        return f"{t}  wake   k{e['k']} e{e['expert']}"
    if k == "Z":
        return f"{t}  wait   end {e['ms']:.3f} ms"
    if k == "F":
        return f"{t}  barrier {e['ms']:.3f} ms"
    return f"{t}  {k}"


def write_note(calls, idx, layer, out):
    call, layer = pick(calls, idx, layer)
    if call is None:
        return
    evs = sorted(layer_events(call, layer), key=lambda e: (e["t"], e["kind"]))
    print(f"call {call['token']} pos {call['pos']} layer {layer} "
          f"events {len(evs)}", file=out)
    for e in evs:
        if e["kind"] == "U":
            print(f"{e['t'] * 1000.0:9.3f}  bucket e{e['expert']:<4} "
                  f"n={e['n']:<4} {e['ms']:9.3f} ms ({e['tag']})", file=out)
        else:
            print(fmt_event(e), file=out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--call", type=int, default=None,
                   help="prefill call index to render (default: first)")
    p.add_argument("--layer", type=int, default=None,
                   help="layer for --timeline/--note (default: first)")
    p.add_argument("--summary", action="store_true",
                   help="per-layer activity table")
    p.add_argument("--all-calls", action="store_true",
                   help="summary over every traced call")
    p.add_argument("--tag-totals", action="store_true",
                   help="sum every named stage over the selected calls")
    p.add_argument("--timeline", action="store_true", help="ASCII Gantt")
    p.add_argument("--width", type=int, default=118,
                   help="columns for --timeline (default 118)")
    p.add_argument("--note", nargs="?", const="-", default=None,
                   metavar="FILE", help="chronological digest ('-' = stdout)")
    a = p.parse_args()

    calls = parse(a.file)
    if not calls:
        print("prefill_trace_view: no blocks in file", file=sys.stderr)
        return 1
    calls = [c for c in calls if c.get("kind") == "prefill"]
    if not calls:
        print("prefill_trace_view: no prefill blocks in file "
              "(decode traces belong to tools/trace_view.py)", file=sys.stderr)
        return 1
    if a.call is not None and not a.all_calls:
        sel = [c for c in calls if c["token"] == a.call]
        if not sel:
            print(f"prefill_trace_view: call {a.call} not in file "
                  f"(have {[c['token'] for c in calls]})", file=sys.stderr)
            return 1
        calls = sel

    did = False
    if a.summary or (not a.timeline and a.note is None):
        print_summary(calls, a.all_calls)
        did = True
    if a.tag_totals:
        print_tag_totals(calls, a.all_calls)
        did = True
    if a.timeline:
        print_timeline(calls, a.call, a.layer, max(40, a.width))
        did = True
    if a.note is not None:
        if a.note == "-":
            write_note(calls, a.call, a.layer, sys.stdout)
        else:
            with open(a.note, "w", encoding="utf-8") as f:
                write_note(calls, a.call, a.layer, f)
            print(f"prefill_trace_view: wrote {a.note}", file=sys.stderr)
        did = True
    return 0 if did else 1


if __name__ == "__main__":
    sys.exit(main())
