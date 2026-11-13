#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ab.py - interleaved A/B timing for two or more dsv4 binaries.

The one-shot bench_decode.py answers "how fast is this build". This answers
"is build B faster than build A", which is a different question on a laptop
that heat-soaks: runs are interleaved, the order rotates every round, and the
comparison is a per-round paired delta against the reference binary.

It parses the engine's own log with bench_decode.parse(), so it inherits the
TTFT subtraction and the synthetic-KV handling instead of growing a second
parser that can disagree with the first.

Example:

    python3 bench/ab.py \
        --bin base=bin/dsv4 --bin lto=bin/variants/dsv4.lto \
        --reps 5 --gen 50 --budget 8 --synthetic-kv 1024 --synthetic-experts \
        --extra "--cpu-pin p --threads 4 --readers 6" --json /tmp/ab.json

The generated ids are compared across every run. A differing id means the
build is a different model, and the timing is not comparable; pass
--allow-id-mismatch only for a deliberately non-bit-exact upper-bound arm.
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_decode import parse  # noqa: E402

HOME = os.path.expanduser("~")
PDIR = os.path.join(HOME, "colibri-models")


def label_and_path(spec):
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        path = spec
        label = os.path.basename(path)
    return label, path


def run_one(binary, args):
    t0 = time.time()
    p = subprocess.run(binary + args, capture_output=True, text=True)
    log = p.stdout + p.stderr
    wall = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{binary[0]} failed rc={p.returncode}\n{log[-4000:]}")
    d = parse(log)
    if d["decode"] is None:
        raise RuntimeError(f"could not parse a decode rate from:\n{log[-4000:]}")
    d["real_wall"] = wall
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", action="append", required=True, metavar="LABEL=PATH",
                    help="binary to measure; repeat for each arm")
    ap.add_argument("--model", default=os.environ.get(
        "DSV4_MODEL", os.path.join(PDIR, "deepseek-v4-flash")))
    ap.add_argument("--trunk", default=os.environ.get(
        "DSV4_TRUNK", os.path.join(PDIR, "dsv4-trunk-flash")))
    ap.add_argument("--tok", default=os.environ.get(
        "DSV4_TOK", os.path.join(PDIR, "dsv4-tok-flash.bin")))
    ap.add_argument("--prompt", default="Das Leben ist")
    ap.add_argument("--gen", type=int, default=50)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--synthetic-kv", type=int, default=0)
    ap.add_argument("--synthetic-experts", action="store_true")
    ap.add_argument("--extra", default="",
                    help="extra engine arguments, shell-split naively")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded runs of the first binary (default 1)")
    ap.add_argument("--order", choices=["rotate", "random"], default="rotate")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--allow-id-mismatch", action="store_true")
    ap.add_argument("--json", metavar="FILE")
    args = ap.parse_args()

    arms = [label_and_path(s) for s in args.bin]
    if len(arms) < 2:
        sys.exit("need at least two --bin arms")
    labels = [a[0] for a in arms]
    if len(set(labels)) != len(labels):
        sys.exit("arm labels must be unique")

    common = [args.model, "--trunk", args.trunk, "--tok", args.tok,
              "--prompt", args.prompt, "--gen", str(args.gen),
              "--budget", str(args.budget)]
    if args.synthetic_kv > 0:
        common += ["--synthetic-kv", str(args.synthetic_kv)]
    if args.synthetic_experts:
        common += ["--synthetic-experts"]
    common += args.extra.split()

    print(f"model  : {args.model}")
    print(f"work   : gen={args.gen} budget={args.budget} "
          f"synth_kv={args.synthetic_kv} synth_experts={args.synthetic_experts}")
    print(f"extra  : {args.extra or '(none)'}")
    print(f"reps   : {args.reps}   warmup: {args.warmup}   order: {args.order}")
    for label, path in arms:
        print(f"  {label:10s} {path}")
    print()

    for _ in range(args.warmup):
        run_one([arms[0][1]], common)
        print(f"warmup done ({arms[0][0]})")

    rng = random.Random(args.seed)
    order = list(range(len(arms)))
    results = {label: [] for label in labels}
    for r in range(args.reps):
        if args.order == "random":
            rng.shuffle(order)
        else:
            order = order[r % len(order):] + order[:r % len(order)]
        for i in order:
            label, path = arms[i]
            d = run_one([path], common)
            d["round"] = r + 1
            results[label].append(d)
            print(f"round {r + 1}/{args.reps}  {label:10s} "
                  f"{d['decode']:.4f} s/token  ttft {d['ttft']:.1f} s")
        print()

    ref = labels[0]
    ref_ids = results[ref][0]["ids"]
    print(f"{'arm':10s} {'n':>2s} {'median':>9s} {'min':>9s} {'max':>9s} "
          f"{'spread':>7s} {'vs ref':>8s}  ids")
    summary = {}
    for label in labels:
        dec = [d["decode"] for d in results[label]]
        med = statistics.median(dec)
        lo, hi = min(dec), max(dec)
        spread = 100.0 * (hi - lo) / med if med else 0.0
        # Paired against the reference in the same round. This is the number
        # to trust when the machine drifts; the absolute medians are not.
        deltas = sorted(
            (results[label][i]["decode"] - results[ref][i]["decode"])
            / results[ref][i]["decode"] * 100.0
            for i in range(args.reps))
        pairs = statistics.median(deltas)
        ids_ok = all(d["ids"] == ref_ids for d in results[label])
        ids_txt = "same" if ids_ok else "DIFFERENT"
        summary[label] = {
            "median": med, "min": lo, "max": hi, "spread_pct": spread,
            "paired_delta_pct": pairs, "paired_deltas_pct": deltas,
            "ids_same": ids_ok,
            "decode": dec,
            "profile": {k: v[1] for k, v in results[label][
                len(results[label]) // 2]["prof"].items()},
        }
        print(f"{label:10s} {len(dec):2d} {med:9.4f} {lo:9.4f} {hi:9.4f} "
              f"{spread:6.1f}% {pairs:+7.2f}%  {ids_txt}")

    bad = [l for l in labels if not summary[l]["ids_same"]]
    if bad and not args.allow_id_mismatch:
        print(f"\nWARNING: ids differ for {', '.join(bad)}; "
              "timings are not comparable to the reference model")
    elif bad:
        print(f"\nnote: ids differ for {', '.join(bad)} "
              "(allowed: upper-bound arm)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "arms": arms,
                       "summary": summary}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
