#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read a dsv4-serve --trace file and render the expert scheduling.

The trace records one decode token per "# token" block. Every line below it is
one event, relative to the token's start:

    t kind layer k expert slot reader bytes ms n ntodo tag

Examples:
    python3 tools/trace_view.py t.log --summary
    python3 tools/trace_view.py t.log --timeline --layer 12 --width 140
    python3 tools/trace_view.py t.log --breakdown --layer 12
    python3 tools/trace_view.py t.log --note order.txt
    python3 tools/trace_view.py t.log --all-tokens --summary
"""
import argparse
import collections
import sys

KIND_LABEL = {
    "B": "begin", "H": "hit", "Q": "queue", "R": "read", "P": "publish",
    "W": "wait", "K": "wake", "Z": "wait-end", "C": "compute", "M": "mm",
    "G": "swiglu", "c": "done", "S": "shared", "F": "barrier", "E": "layer",
    "V": "vk",
}

# Matmul tags: the CPU trio and the GPU stages share the same lane glyphs.
MM_GLYPH = {"w1": "1", "w3": "3", "w2": "2",
            "gw1": "1", "gw3": "3", "gw2": "2"}
VK_GLYPH = {"cin": "c", "rec": "r", "sub": "S", "wait": "w", "cout": "o"}


def parse(path):
    tokens = []
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                parts = s[1:].split()
                d = dict(zip(parts[0::2], parts[1::2]))
                cur = {
                    "token": int(d.get("token", -1)),
                    "pos": int(d.get("pos", -1)),
                    "phase": d.get("phase", "?"),
                    "events": int(d.get("events", 0)),
                    "dropped": int(d.get("dropped", 0)),
                    "ev": [],
                }
                tokens.append(cur)
                continue
            f = s.split()
            if cur is None or len(f) < 11:
                continue
            cur["ev"].append({
                "t": float(f[0]), "kind": f[1], "layer": int(f[2]),
                "k": int(f[3]), "expert": int(f[4]), "slot": int(f[5]),
                "reader": int(f[6]), "bytes": int(f[7]), "ms": float(f[8]),
                "n": int(f[9]), "ntodo": int(f[10]),
                "tag": f[11] if len(f) > 11 else "",
            })
    return tokens


def by_layer(ev):
    out = collections.defaultdict(list)
    for e in ev:
        if e["layer"] >= 0:
            out[e["layer"]].append(e)
    return out


def layer_stats(evs):
    hits = []
    pf_waits = []
    miss = 0
    dups = 0
    pfw_b = 0
    pubs = []
    last_pub = 0.0
    wait_ms = 0.0
    w_t = None
    shared_ms = 0.0
    mm_events = []
    mm_ms = 0.0
    gpu_ms = 0.0
    vk_total_ms = 0.0
    vk_host_ms = 0.0
    vk_wait_ms = 0.0
    tail_ms = 0.0
    layer_ms = 0.0

    for e in evs:
        k = e["kind"]
        if k == "B":
            miss = e["ntodo"]
            dups = e["k"]
            pfw_b = e["slot"]
        elif k == "H":
            hits.append(e)
            if e["tag"] == "pf-wait":
                pf_waits.append(e)
        elif k == "P" and e["tag"] != "prefetch":
            pubs.append(e)
            last_pub = max(last_pub, e["t"])
        elif k == "W":
            w_t = e["t"]
        elif k == "K":
            if w_t is not None:
                wait_ms += (e["t"] - w_t) * 1000.0
                w_t = None
        elif k == "Z":
            wait_ms += e["ms"]
            w_t = None
        elif k == "S":
            shared_ms = e["ms"]
        elif k == "M":
            if e["k"] >= 0:
                mm_events.append(e)
                mm_ms += e["ms"]
                if e["tag"] in ("gw1", "gw3", "gw2"):
                    gpu_ms += e["ms"]
        elif k == "G":
            if e["k"] >= 0 and e["tag"] == "gswiglu":
                gpu_ms += e["ms"]
        elif k == "V":
            tag = e["tag"]
            if tag == "tot":
                vk_total_ms += e["ms"]
            elif tag == "wait":
                vk_wait_ms += e["ms"]
                vk_host_ms += e["ms"]
            elif tag in ("cin", "rec", "sub", "cout"):
                vk_host_ms += e["ms"]
        elif k == "E":
            layer_ms = e["ms"]

    # Only arithmetic that starts after the last MISS of this layer is
    # exposed tail. With no miss there is nothing to overlap, so tail is 0.
    if pubs:
        tail_ms = sum(e["ms"] for e in mm_events if e["t"] >= last_pub)

    return {
        "hits": len(hits), "dups": dups,
        "pfw": pfw_b if pfw_b else len(pf_waits),
        "pf_wait_ms": sum(e["ms"] for e in pf_waits),
        "miss": miss, "last_pub": last_pub, "wait_ms": wait_ms,
        "mm_ms": mm_ms, "gpu_ms": gpu_ms, "tail_ms": tail_ms,
        "layer_ms": layer_ms, "shared_ms": shared_ms, "pubs": pubs,
        "hits_ev": hits, "vk_total_ms": vk_total_ms,
        "vk_host_ms": vk_host_ms, "vk_wait_ms": vk_wait_ms,
    }


def print_summary(tokens, every):
    print(f"{'tok':>4} {'L':>3} {'hit':>4} {'pfw':>4} {'miss':>4} "
          f"{'lastP':>8} {'wait':>8} {'mm':>8} {'gpu':>8} {'vk':>8} "
          f"{'tail':>8} {'moe':>8}")
    for tok in tokens:
        if not every and tok is not tokens[0]:
            continue
        if tok["dropped"]:
            print(f"# token {tok['token']}: {tok['dropped']} events dropped")
        layers = by_layer(tok["ev"])
        for L in sorted(layers):
            st = layer_stats(layers[L])
            print(f"{tok['token']:>4} {L:>3} {st['hits']:>4} {st['pfw']:>4} "
                  f"{st['miss']:>4} {1000.0 * st['last_pub']:8.3f} "
                  f"{st['wait_ms']:8.3f} {st['mm_ms']:8.3f} "
                  f"{st['gpu_ms']:8.3f} {st['vk_total_ms']:8.3f} "
                  f"{st['tail_ms']:8.3f} {st['layer_ms']:8.3f}")

def pick_layer(tokens, want):
    layers = sorted(by_layer(tokens[0]["ev"]))
    if not layers:
        return None
    return layers[0] if want is None else want


def _col(t, t0, span, width):
    return max(0, min(width - 1, int((t - t0) / span * (width - 1))))


def print_timeline(tokens, layer, width=118):
    toks = [t for t in tokens if by_layer(t["ev"]).get(layer)]
    if not toks:
        print(f"trace_view: no events for layer {layer}", file=sys.stderr)
        return
    tok = toks[0]
    evs = by_layer(tok["ev"])[layer]
    st = layer_stats(evs)
    t0 = min(e["t"] for e in evs)
    tend = max([e["t"] for e in evs] + [t0 + st["layer_ms"] / 1000.0])
    span = max(tend - t0, 1e-9)
    ks = sorted({e["k"] for e in evs if e["k"] >= 0})

    info = {}
    for k in ks:
        kev = [e for e in evs if e["k"] == k]
        hits = [e for e in kev if e["kind"] == "H"]
        pubs = [e for e in kev if e["kind"] == "P"
                and e["tag"] in ("read", "prefetch")]
        if hits:
            tags = {e["tag"] for e in hits}
            state = ("hit/pf-wait" if "pf-wait" in tags else
                     "hit/prefetch" if "prefetch" in tags else
                     "hit/dup" if "dup" in tags else "hit")
        else:
            state = "miss"
        info[k] = {
            "hits": hits, "pubs": pubs,
            "qs": [e for e in kev if e["kind"] == "Q"],
            "expert": next((e["expert"] for e in kev
                            if e["kind"] in ("H", "Q", "C")
                            and e["expert"] >= 0), -1),
            "reader": next((e["reader"] for e in pubs if e["reader"] >= 0), -1),
            "state": state,
            "R": next((e for e in kev if e["kind"] == "R"), None),
        }

    rows = collections.defaultdict(lambda: [" "] * width)
    main = rows["main"]

    def put(row, t, ch):
        row[_col(t, t0, span, width)] = ch

    # Reader lanes: which k each thread is serving between its R and P.
    for e in evs:
        if e["kind"] != "R":
            continue
        d = info[e["k"]]
        pe = next((p for p in d["pubs"] if p["reader"] == e["reader"]), None)
        end = pe["t"] if pe else e["t"] + 1e-6
        rr = rows[f"r{e['reader']}"]
        ch = "p" if (pe and pe["tag"] == "prefetch") else str(e["k"])
        for c in range(_col(e["t"], t0, span, width),
                       _col(end, t0, span, width) + 1):
            rr[c] = ch

    # Expert lanes: queue, read, then the three matmuls of its compute.
    for k in ks:
        d = info[k]
        rr = rows[f"k{k}"]
        if d["R"] and d["pubs"]:
            for c in range(_col(d["R"]["t"], t0, span, width),
                           _col(d["pubs"][0]["t"], t0, span, width) + 1):
                rr[c] = "~"
        for q in d["qs"]:
            put(rr, q["t"], "q")
        for e in evs:
            if e["kind"] == "M" and e["k"] == k:
                ch = MM_GLYPH.get(e["tag"], "#")
                for c in range(_col(e["t"], t0, span, width),
                               _col(e["t"] + e["ms"] / 1000.0,
                                    t0, span, width) + 1):
                    rr[c] = ch
            elif e["kind"] == "G" and e["k"] == k:
                put(rr, e["t"], "s")

    # P and H must survive a matmul that lands in the same column.
    for k in ks:
        for h in info[k]["hits"]:
            put(rows[f"k{k}"], h["t"], "H")
        for p in info[k]["pubs"]:
            put(rows[f"k{k}"], p["t"], "P")

    # Shared expert lane.
    srow = rows["shared"]
    for e in evs:
        if e["kind"] == "M" and e["k"] < 0:
            ch = MM_GLYPH.get(e["tag"], "#")
            for c in range(_col(e["t"], t0, span, width),
                           _col(e["t"] + e["ms"] / 1000.0,
                                t0, span, width) + 1):
                srow[c] = ch
        elif e["kind"] == "G" and e["k"] < 0:
            put(srow, e["t"], "s")
        elif e["kind"] == "S":
            # Old traces stamped S at the shared block's start, new ones at its
            # end. Place it after the matmuls either way.
            sm_end = max((m["t"] + m["ms"] / 1000.0 for m in evs
                          if m["kind"] == "M" and m["k"] < 0), default=e["t"])
            at = (e["t"] + e["ms"] / 1000.0
                  if e["t"] < sm_end - 1e-9 else e["t"])
            put(srow, at, "S")

    # Vulkan host phases: copy-in, record, submit, fence wait, copy-out.
    vkrow = rows["vk-host"]
    for e in evs:
        if e["kind"] != "V" or e["tag"] not in VK_GLYPH:
            continue
        ch = VK_GLYPH[e["tag"]]
        for c in range(_col(e["t"], t0, span, width),
                       _col(e["t"] + e["ms"] / 1000.0, t0, span, width) + 1):
            vkrow[c] = ch

    # Main lane: fill compute first, then draw the blocking windows on top.
    for e in evs:
        if e["kind"] == "M":
            for c in range(_col(e["t"], t0, span, width),
                           _col(e["t"] + e["ms"] / 1000.0,
                                t0, span, width) + 1):
                if main[c] == " ":
                    main[c] = "+"
    ordered = sorted(evs, key=lambda e: e["t"])
    for i, e in enumerate(ordered):
        if e["kind"] == "W":
            end = next((o for o in ordered[i + 1:]
                        if o["kind"] in ("K", "Z")), None)
            if end:
                for c in range(_col(e["t"], t0, span, width),
                               _col(end["t"], t0, span, width) + 1):
                    main[c] = "w"
                put(main, end["t"], "k" if end["kind"] == "K" else "z")
        elif e["kind"] == "F":
            put(main, e["t"], "F")
        elif e["kind"] == "E":
            put(main, e["t"], "E")

    # Time axis, with a step that keeps roughly eight labels.
    axis = [" "] * width
    raw = span * 1000.0 / 8.0
    step = min((0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100),
               key=lambda c: abs(c - raw))
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

    print(f"token {tok['token']} pos {tok['pos']}  layer {layer}  "
          f"start +{t0 * 1000.0:.3f} ms  span {span * 1000.0:.3f} ms  "
          f"({width} cols, {(span * 1000.0) / width:.3f} ms/col)")
    print()
    out = []
    for k in ks:
        d = info[k]
        out.append((f"k{k} e{d['expert']} {d['state']}", rows[f"k{k}"]))
    out.append(("shared expert", srow))
    out.append(("main (compute thread)", main))
    out.append(("vk host", vkrow))
    for r in sorted((x for x in rows if x.startswith("r")),
                    key=lambda s: int(s[1:])):
        out.append((r, rows[r]))
    out.append(("ms", axis))
    # Wide enough for the longest label ("k0 e115 hit/prefetch"), so every
    # row's opening '|' is in the same column.
    lw = max(len(label) for label, _ in out)
    for label, cells in out:
        print(f"{label:>{lw}} |{''.join(cells)}|")
    print()
    print("Legende  H = Hit, sofort nutzbar      q = Read in Queue            "
          "~ = pread laeuft")
    print("         P = Read fertig/publiziert   1/3/2 = w1/w3/w2-Matmul      "
          "s = SwiGLU")
    print("         S = Shared fertig            w = wartet in wait_any       "
          "k = wait_any liefert k")
    print("         z = wait_any leer            F = end()-Barriere            "
          "E = Layer fertig")
    print("         main: + = Compute-Thread rechnet (irgendein Expert/Shared)")
    print("         Reader-Zeile: Ziffer = welchen k dieser Thread liest, "
          "p = Prefetch-Read")
    print("         vk host: c = Copy-in, r = Command-Recording, S = Submit, "
          "w = Fence-Wait, o = Copy-out")
    print()

    print(f"{'k':>2} {'expert':>6} {'state':>12} {'reader':>6} "
          f"{'read start':>10} {'read end':>9} {'read ms':>8} "
          f"{'w1':>6} {'w3':>6} {'w2':>6} {'gpu':>8}")
    for k in ks:
        d = info[k]
        rs = (d["R"]["t"] - t0) * 1000.0 if d["R"] else None
        re = (d["pubs"][0]["t"] - t0) * 1000.0 if d["pubs"] else None
        s_rs = f"{rs:.3f}" if rs is not None else "-"
        s_re = f"{re:.3f}" if re is not None else "-"
        s_len = f"{re - rs:.3f}" if rs is not None and re is not None else "-"
        rdr = f"r{d['reader']}" if d["reader"] >= 0 else "-"
        def mm(which):
            for tag in (which, "g" + which):
                v = next((e["ms"] for e in evs if e["kind"] == "M"
                          and e["k"] == k and e["tag"] == tag), None)
                if v is not None:
                    return v
            return 0.0
        gpu = (mm('w1') + mm('w3') + mm('w2') +
               sum(e["ms"] for e in evs if e["kind"] == "G"
                   and e["k"] == k))
        print(f"{k:>2} {d['expert']:>6} {d['state']:>12} {rdr:>6} "
              f"{s_rs:>10} {s_re:>9} {s_len:>8} "
              f"{mm('w1'):>6.3f} {mm('w3'):>6.3f} {mm('w2'):>6.3f} "
              f"{gpu:>8.3f}")


def print_breakdown(tokens, layer):
    """Per-wave and per-expert split of host and device time for one layer."""
    toks = [t for t in tokens if by_layer(t["ev"]).get(layer)]
    if not toks:
        print(f"trace_view: no events for layer {layer}", file=sys.stderr)
        return
    tok = toks[0]
    evs = by_layer(tok["ev"])[layer]
    ks = sorted({e["k"] for e in evs if e["k"] >= 0})
    wave_tots = [e for e in evs
                 if e["kind"] == "V" and e["tag"] == "tot" and e["k"] < 0]

    print(f"token {tok['token']} pos {tok['pos']}  layer {layer}  "
          f"expert breakdown (ms)")

    if wave_tots:
        print(f"{'wave':>4} {'n':>2} {'gpu':>7} {'cin':>6} {'rec':>6} "
              f"{'sub':>6} {'wait':>7} {'cout':>6} {'host':>7} "
              f"{'total':>7} {'conc':>6}")
        tot = collections.defaultdict(float)
        for wt in wave_tots:
            slot = wt["slot"]
            v = collections.defaultdict(float)
            for e in evs:
                if e["kind"] == "V" and e["slot"] == slot:
                    v[e["tag"]] += e["ms"]
            gpu = sum(e["ms"] for e in evs
                      if e["slot"] == slot and e["k"] >= 0
                      and (e["kind"] == "M" or
                           (e["kind"] == "G" and e["tag"] == "gswiglu")))
            host = v["cin"] + v["rec"] + v["sub"] + v["wait"] + v["cout"]
            total = v["tot"] if v["tot"] > 0.0 else gpu + host
            conc = gpu / total if total > 0.0 else 0.0
            for name in ("cin", "rec", "sub", "wait", "cout"):
                tot[name] += v[name]
            tot["gpu"] += gpu
            tot["host"] += host
            tot["total"] += total
            print(f"{slot:>4} {wt['n']:>2} {gpu:>7.3f} {v['cin']:>6.3f} "
                  f"{v['rec']:>6.3f} {v['sub']:>6.3f} {v['wait']:>7.3f} "
                  f"{v['cout']:>6.3f} {host:>7.3f} {total:>7.3f} "
                  f"{conc:>5.2f}x")
        conc = tot["gpu"] / tot["total"] if tot["total"] > 0.0 else 0.0
        print(f"{'sum':>4} {'':>2} {tot['gpu']:>7.3f} {tot['cin']:>6.3f} "
              f"{tot['rec']:>6.3f} {tot['sub']:>6.3f} {tot['wait']:>7.3f} "
              f"{tot['cout']:>6.3f} {tot['host']:>7.3f} {tot['total']:>7.3f} "
              f"{conc:>5.2f}x")
        print()
        print("conc = Summe der GPU-Stages / Wave-Busy-Zeit. >1 heißt: "
              "Experten liefen gleichzeitig.")
        print("total = cin+rec+sub+gpu_span+cout; wait = tatsächliche "
              "Host-Blockzeit beim Einsammeln der Wave.")
        print()
        print(f"{'wave':>4} {'k':>2} {'expert':>6} {'start':>9} "
              f"{'w1':>7} {'w3':>7} {'silu':>7} {'w2':>7} {'gpu':>7}")
        for wt in wave_tots:
            slot = wt["slot"]
            for k in sorted({e["k"] for e in evs
                             if e["slot"] == slot and e["k"] >= 0
                             and e["kind"] in ("M", "G")}):
                kev = [e for e in evs if e["slot"] == slot and e["k"] == k]
                expert = next((e["expert"] for e in evs
                               if e["k"] == k and e["expert"] >= 0), -1)
                def mm(tag):
                    return sum(e["ms"] for e in kev
                               if e["kind"] == "M" and e["tag"] == tag)
                silu = sum(e["ms"] for e in kev
                           if e["kind"] == "G" and e["tag"] == "gswiglu")
                start = next((e["t"] for e in kev
                              if e["kind"] == "M" and e["tag"] == "gw1"), 0.0)
                gpu = mm("gw1") + mm("gw3") + silu + mm("gw2")
                print(f"{slot:>4} {k:>2} {expert:>6} {start * 1000.0:>9.3f} "
                      f"{mm('gw1'):>7.3f} {mm('gw3'):>7.3f} {silu:>7.3f} "
                      f"{mm('gw2'):>7.3f} {gpu:>7.3f}")
        return

    has_vk = any(e["kind"] == "V" for e in evs)
    if not has_vk:
        print("CPU path: no Vulkan host stages; total = matmuls + swiglu")
    print(f"{'k':>2} {'expert':>6} {'w1':>7} {'w3':>7} {'silu':>7} "
          f"{'w2':>7} {'gpu':>7} {'cin':>6} {'rec':>6} {'sub':>6} "
          f"{'wait':>7} {'cout':>6} {'host':>7} {'total':>7} {'busy':>6}")

    tot = collections.defaultdict(float)
    for k in ks:
        kev = [e for e in evs if e["k"] == k]
        expert = next((e["expert"] for e in kev
                       if e["kind"] in ("C", "H", "Q") and e["expert"] >= 0),
                      -1)
        v = collections.defaultdict(float)
        for e in kev:
            if e["kind"] == "V":
                v[e["tag"]] += e["ms"]
        gw = {}
        for tag in ("gw1", "gw3", "gw2"):
            gw[tag] = sum(e["ms"] for e in kev
                          if e["kind"] == "M" and e["tag"] == tag)
        gsilu = sum(e["ms"] for e in kev
                    if e["kind"] == "G" and e["tag"] == "gswiglu")
        gpu = gw["gw1"] + gw["gw3"] + gsilu + gw["gw2"]
        if gpu == 0.0:
            # CPU fallback: the same three matmuls and swiglu, no Vulkan.
            gw["gw1"] = sum(e["ms"] for e in kev
                            if e["kind"] == "M" and e["tag"] == "w1")
            gw["gw3"] = sum(e["ms"] for e in kev
                            if e["kind"] == "M" and e["tag"] == "w3")
            gw["gw2"] = sum(e["ms"] for e in kev
                            if e["kind"] == "M" and e["tag"] == "w2")
            gsilu = sum(e["ms"] for e in kev if e["kind"] == "G")
            gpu = gw["gw1"] + gw["gw3"] + gsilu + gw["gw2"]
        host = (v["cin"] + v["rec"] + v["sub"] + v["wait"] + v["cout"])
        total = v["tot"] if v["tot"] > 0.0 else gpu + host
        busy = 100.0 * gpu / total if total > 0.0 else 0.0

        for name in ("gw1", "gw3", "gw2"):
            tot[name] += gw[name]
        tot["gsilu"] += gsilu
        tot["gpu"] += gpu
        tot["host"] += host
        tot["total"] += total
        for name in ("cin", "rec", "sub", "wait", "cout"):
            tot[name] += v[name]

        print(f"{k:>2} {expert:>6} {gw['gw1']:>7.3f} {gw['gw3']:>7.3f} "
              f"{gsilu:>7.3f} {gw['gw2']:>7.3f} {gpu:>7.3f} "
              f"{v['cin']:>6.3f} {v['rec']:>6.3f} {v['sub']:>6.3f} "
              f"{v['wait']:>7.3f} {v['cout']:>6.3f} {host:>7.3f} "
              f"{total:>7.3f} {busy:>5.1f}%")

    if ks:
        busy = 100.0 * tot["gpu"] / tot["total"] if tot["total"] > 0.0 else 0.0
        print(f"{'':>2} {'sum':>6} {tot['gw1']:>7.3f} {tot['gw3']:>7.3f} "
              f"{tot['gsilu']:>7.3f} {tot['gw2']:>7.3f} {tot['gpu']:>7.3f} "
              f"{tot['cin']:>6.3f} {tot['rec']:>6.3f} {tot['sub']:>6.3f} "
              f"{tot['wait']:>7.3f} {tot['cout']:>6.3f} {tot['host']:>7.3f} "
              f"{tot['total']:>7.3f} {busy:>5.1f}%")
    print()
    print("busy = GPU-Dispatches / gesamte Host-Wall-Zeit des Experten; "
          "host = cin+rec+sub+wait+cout")


def fmt_event(e):
    k = e["kind"]
    t = f"{e['t'] * 1000.0:8.3f}"
    if k == "B":
        return f"{t}  begin     {e['n']} requested, {e['ntodo']} miss, " \
               f"{e['k']} dup"
    if k == "H":
        extra = (f" waited {e['ms']:.3f} ms" if e["tag"] == "pf-wait"
                 else "")
        return f"{t}  hit       k{e['k']} e{e['expert']} slot{e['slot']} " \
               f"({e['tag']}){extra}"
    if k == "Q":
        return f"{t}  queue     k{e['k']} e{e['expert']} slot{e['slot']}" \
               f"{' prefetch' if e['tag'] == 'pf' else ''}"
    if k == "R":
        return f"{t}  read      k{e['k']} e{e['expert']} slot{e['slot']} " \
               f"reader{e['reader']}"
    if k == "P":
        return f"{t}  publish   k{e['k']} e{e['expert']} slot{e['slot']} " \
               f"reader{e['reader']} {e['bytes'] / 1e6:.2f} MB in " \
               f"{e['ms']:.3f} ms ({e['tag']})"
    if k == "W":
        return f"{t}  wait      enter"
    if k == "K":
        return f"{t}  wake      k{e['k']} e{e['expert']}"
    if k == "Z":
        return f"{t}  wait      done (blocked {e['ms']:.3f} ms)"
    if k == "C":
        return f"{t}  compute   k{e['k']} e{e['expert']} ({e['tag']})"
    if k == "M":
        return f"{t}    {e['tag']:<6} {e['ms']:.3f} ms"
    if k == "G":
        return f"{t}    swiglu {e['ms']:.3f} ms" \
               f"{' (gpu)' if e['tag'] == 'gswiglu' else ''}"
    if k == "V":
        extra = (f"  wave{e['slot']} n{e['n']}" if e["tag"] == "tot"
                 and e["k"] < 0 else "")
        return f"{t}    vk {e['tag']:<4} {e['ms']:.3f} ms{extra}"
    if k == "c":
        return f"{t}  done      k{e['k']} e{e['expert']}"
    if k == "S":
        return f"{t}  shared    {e['ms']:.3f} ms"
    if k == "F":
        return f"{t}  barrier   waited {e['ms']:.3f} ms"
    if k == "E":
        return f"{t}  layer     {e['ms']:.3f} ms (moe)"
    return f"{t}  {k}"


def write_note(tokens, out):
    for tok in tokens:
        out.write(f"token {tok['token']} pos {tok['pos']} phase {tok['phase']} "
                  f"events {tok['events']} dropped {tok['dropped']}\n")
        layers = by_layer(tok["ev"])
        for L in sorted(layers):
            evs = sorted(layers[L], key=lambda e: e["t"])
            st = layer_stats(evs)
            out.write(f"\nL{L:02d}  moe {st['layer_ms']:.3f} ms   "
                      f"{st['hits']} hit / {st['miss']} miss")
            if st["pfw"]:
                out.write(f" / {st['pfw']} prefetch")
            out.write(f"   wait {st['wait_ms']:.3f} ms   "
                      f"tail {st['tail_ms']:.3f} ms\n")
            if st["pfw"]:
                if st["pf_wait_ms"] > 0.0:
                    out.write(f"  prefetch hits that blocked begin(): "
                              f"{st['pfw']} ({st['pf_wait_ms']:.3f} ms)\n")
                else:
                    out.write(f"  prefetch slots streamed by wait_any(): "
                              f"{st['pfw']}\n")
            for e in evs:
                out.write("  " + fmt_event(e) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--token", type=int, default=None,
                   help="traced token index to render (default: the first)")
    p.add_argument("--layer", type=int, default=None,
                   help="layer for --timeline (default: the first)")
    p.add_argument("--summary", action="store_true", help="per-layer table")
    p.add_argument("--all-tokens", action="store_true",
                   help="summary over every traced token")
    p.add_argument("--timeline", action="store_true", help="ASCII interleaving")
    p.add_argument("--breakdown", action="store_true",
                   help="per-expert Vulkan host/GPU time table for --layer")
    p.add_argument("--width", type=int, default=118,
                   help="columns for --timeline (default 118)")
    p.add_argument("--note", nargs="?", const="-", default=None,
                   metavar="FILE", help="chronological digest ('-' = stdout)")
    a = p.parse_args()

    tokens = parse(a.file)
    if not tokens:
        print("trace_view: no token blocks in file", file=sys.stderr)
        return 1
    if a.token is not None:
        sel = [t for t in tokens if t["token"] == a.token]
        if not sel:
            print(f"trace_view: token {a.token} not in file "
                  f"(have {[t['token'] for t in tokens]})", file=sys.stderr)
            return 1
        sel = sel[0]
    else:
        sel = tokens[0]

    did = False
    if a.summary or (not a.timeline and not a.breakdown and a.note is None):
        print_summary(tokens, a.all_tokens)
        did = True
    if a.timeline:
        layer = pick_layer([sel], a.layer)
        if layer is not None:
            print_timeline([sel], layer, max(40, a.width))
            did = True
    if a.breakdown:
        layer = pick_layer([sel], a.layer)
        if layer is not None:
            print_breakdown([sel], layer)
            did = True
    if a.note is not None:
        if a.note == "-":
            write_note([sel], sys.stdout)
        else:
            with open(a.note, "w", encoding="utf-8") as f:
                write_note([sel], f)
            print(f"trace_view: wrote {a.note}", file=sys.stderr)
        did = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
