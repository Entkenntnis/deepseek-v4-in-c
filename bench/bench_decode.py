#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""bench_decode.py - repeatable decode-rate measurement for dsv4.

Prefill is NOT the number this reports. dsv4's own summary folds TTFT into
"tokens in T s, s/token", which makes the decode rate look worse on a short
prompt and moves whenever the trunk has to be read from disk. This script
subtracts TTFT and divides by the tokens actually decoded:

    decode_s_per_token = (wall - ttft) / (ntokens - 1)

(ntokens includes the one token the prefill head picks, hence the -1.)

WHY REPEATS. On this laptop a single run swings by 20% with thermal and disk
state. The median of several interleaved runs is the number worth comparing;
the min/max spread says whether a difference is real.

The generated ids are also compared across runs. They should be identical every
time: the route and cache counters already are, and a differing id means the
build is not deterministic and no timing from it is trustworthy.
"""
import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
PDIR = os.path.join(HOME, "colibri-models")

PROFILES = [
    "routed experts (matmul)",
    "routed experts (disk)",
    "attention",
    "mHC + norms",
    "shared expert",
    "router",
]

NUM = r"([0-9]+(?:\.[0-9]+)?)"


def parse(log):
    def first(pattern, cast=float):
        m = re.search(pattern, log)
        return cast(m.group(1)) if m else None

    d = {}
    d["ntok"] = first(r"(\d+) tokens in [0-9.]+ s", int)
    d["wall"] = first(rf"\d+ tokens in {NUM} s")
    d["ttft"] = first(rf"time to first token: {NUM} s")
    d["reported"] = first(rf"tokens in [0-9.]+ s, {NUM} s/token")
    d["prompt_tokens"] = first(r"over (\d+) prompt tokens", int)
    d["rss"] = first(rf"PEAK RSS for the whole run: {NUM} GB")
    d["trunk_gb"] = first(rf"\n  {NUM} GB read in")

    d["slots"] = first(rf"expert cache run: (\d+) slots", int)
    m = re.search(r"expert cache run:.*?\n\s*(\d+) requests, (\d+) hits "
                  r"\(([0-9.]+)%\), (\d+) misses, (\d+) evictions\n"
                  rf"\s*{NUM} GB read from disk", log, re.S)
    if m:
        d["requests"] = int(m.group(1))
        d["hits"] = int(m.group(2))
        d["hit_rate"] = float(m.group(3))
        d["misses"] = int(m.group(4))
        d["evictions"] = int(m.group(5))
        d["disk_gb"] = float(m.group(6))

    d["prof"] = {}
    for name in PROFILES:
        m = re.search(rf"^  {re.escape(name)}\s+{NUM} s\s+([0-9.]+)%\s+"
                      rf"{NUM} ms/pass", log, re.M)
        if m:
            d["prof"][name] = (float(m.group(1)), float(m.group(3)))

    body = log.split("--- generated ids ---", 1)
    ids = []
    if len(body) == 2:
        text = body[1].split("\n\n", 1)[0]
        text = text.replace("<eos>", " ")
        ids = [int(x) for x in re.findall(r"-?\d+", text)]
    d["ids"] = ids

    if d["ntok"] and d["wall"] is not None and d["ttft"] is not None:
        n = d["ntok"] - 1
        d["decode"] = (d["wall"] - d["ttft"]) / n if n > 0 else None
    else:
        d["decode"] = None
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get(
        "DSV4_MODEL", os.path.join(PDIR, "deepseek-v4-flash")))
    ap.add_argument("--trunk", default=os.environ.get(
        "DSV4_TRUNK", os.path.join(PDIR, "dsv4-trunk-flash")))
    ap.add_argument("--tok", default=os.environ.get(
        "DSV4_TOK", os.path.join(PDIR, "dsv4-tok-flash.bin")))
    ap.add_argument("--bin", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "dsv4"))
    ap.add_argument("--prompt", default="Das Leben ist")
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--budget", type=float, default=16.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pin", choices=["none", "pcores"], default="none",
                    help="'pcores' runs 4 threads pinned to CPU 0-3, which is "
                         "the fastest configuration on a 4P+4E hybrid")
    ap.add_argument("--json", metavar="FILE", help="write all parsed reps as JSON")
    ap.add_argument("--keep", metavar="DIR", help="keep raw logs in DIR")
    args = ap.parse_args()

    if not os.path.isfile(args.bin):
        sys.exit(f"{args.bin} not found; run `make bin/dsv4` first")

    cmd = [args.bin, args.model, "--trunk", args.trunk, "--tok", args.tok,
           "--prompt", args.prompt, "--gen", str(args.gen),
           "--budget", str(args.budget)]
    env = dict(os.environ)
    prefix = []
    if args.pin == "pcores":
        prefix = ["taskset", "-c", "0-3"]
        env["OMP_NUM_THREADS"] = "4"
    full = prefix + cmd

    if args.keep:
        os.makedirs(args.keep, exist_ok=True)

    print(f"binary : {args.bin}")
    print(f"model  : {args.model}")
    print(f"prompt : {args.prompt!r} -> {args.gen} generated tokens")
    print(f"budget : {args.budget} GB   pin: {args.pin}   reps: {args.reps}")
    print(f"cmd    : {' '.join(full)}")
    print()

    reps = []
    for i in range(args.reps):
        print(f"rep {i + 1}/{args.reps} ...", end=" ", flush=True)
        t0 = time.time()
        p = subprocess.run(full, env=env, capture_output=True, text=True)
        log = p.stdout + p.stderr
        if args.keep:
            with open(os.path.join(args.keep, f"rep{i + 1}.log"), "w") as f:
                f.write(log)
        if p.returncode != 0:
            print(f"FAILED rc={p.returncode}")
            print(log[-4000:])
            sys.exit(1)
        d = parse(log)
        d["rep"] = i + 1
        d["real_wall"] = time.time() - t0
        reps.append(d)
        print(f"{d['decode']:.3f} s/token decode "
              f"(wall {d['wall']:.1f} s, ttft {d['ttft']:.1f} s)")

    dec = [r["decode"] for r in reps]
    med = statistics.median(dec)
    lo, hi = min(dec), max(dec)
    spread = 100.0 * (hi - lo) / med if med else 0.0

    print()
    print(f"decode s/token : median {med:.3f}   min {lo:.3f}   max {hi:.3f}   "
          f"spread {spread:.1f}%")
    print(f"reported s/tok : {reps[0]['reported']:.2f} (includes prefill; ignore)")
    print(f"ttft           : {statistics.median([r['ttft'] for r in reps]):.1f} s")
    print(f"peak RSS       : {statistics.median([r['rss'] for r in reps]):.2f} GB")
    print(f"expert cache   : {reps[0]['slots']} slots, "
          f"{reps[0]['hit_rate']:.1f}% hits, "
          f"{reps[0]['disk_gb']:.1f} GB read from disk")

    med_rep = min(reps, key=lambda r: abs(r["decode"] - med))
    print(f"\nprofile of median run (ms/pass, includes prefill in wall):")
    for name in PROFILES:
        if name in med_rep["prof"]:
            s, ms = med_rep["prof"][name]
            print(f"  {name:24s} {s:6.1f} s  {ms:7.1f} ms/pass")

    ok = all(set(r["ids"]) == set(reps[0]["ids"]) and
             r["ids"] == reps[0]["ids"] for r in reps)
    print(f"\nids: {len(reps[0]['ids'])} tokens, "
          f"{'identical across all reps' if ok else 'DIFFER BETWEEN REPS'}")

    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(reps, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
