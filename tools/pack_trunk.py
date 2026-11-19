#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pack the dense trunk and build the three index files the C engine reads.

WHY THIS EXISTS
    A decode step walks layers 0..N-1 in a fixed order, every token. If each
    layer's weights are scattered across a 160 GB checkpoint, that walk is
    thousands of small reads. Rewritten into trunk.bin, one layer is ONE pread
    at a known offset, and the memory requirement becomes a dial: pin a prefix,
    stream the rest.

    The C engine no longer parses safetensors JSON. This script already reads
    every header, so it writes the complete layout instead:

        trunk.idx    dense layer runs, tensors in slot order
        experts.idx  one fixed record per routed-expert tensor
        model.idx    embed, final norm, head and the head's mHC

THE ASSUMPTION K3 MADE THAT IS FALSE HERE
    kimi-k3-in-c's packer takes each layer's trunk to be a single contiguous run
    in its shard, and derives every tensor's in-slot offset as
    (absolute shard offset - run start).

    Measured on DeepSeek-V4-Flash, that is wrong. The checkpoint groups tensors
    by dtype, so a layer's trunk is TWO runs with all 256 routed experts between
    them:

        trunk x22  ->  EXPERT x768  ->  trunk x11  ->  EXPERT x768

    So this packer reads each tensor individually and records where it actually
    landed in the destination. The destination is still one contiguous run per
    layer, which is the only property the streamer needs.

WHAT IS DELIBERATELY NOT PACKED
    The routed experts. They are most of the checkpoint, they are streamed on
    demand by the LRU cache, and copying them here would defeat the point.
    experts.idx records their absolute locations instead.

SLOT IDS
    C and this script share a numbering. The C side has one descriptor per slot
    saying where it belongs in DSV4LayerW; this script never needs to know that,
    it only has to write the right slot id. A mismatch is caught at bind time
    (an unknown slot is a hard error).

FORMAT
    trunk.idx is line based:

        DSV4TRUNK 2
        n_layers 43
        layout 1
        layer <L> <file_off> <nbytes> <nt> <widen_bytes>
        slot <slot> <off> <nbytes> <dtype> <rows> <cols> <wdt> <blk_r> <blk_c>
        ...

    experts.idx and model.idx are little-endian records; see the C readers.
"""
import argparse
import json
import os
import struct
import sys

ALIGN = 4096          # O_DIRECT needs offset, length and buffer all aligned

# Dtype codes, exactly DSV4Dtype in src/40_io/dsv4_st.h.
DT = {"U8": 1, "BF16": 2, "F16": 3, "F32": 4,
      "F8_E4M3": 5, "F8_E8M0": 6, "I8": 7, "I64": 8}

# Weight-format codes, exactly the DSV4_W* enum in src/00_base/dsv4.h.
WDT = {"F32": 0, "BF16": 1, "FP8": 2, "FP4": 3}

# Layer slots. Keep in step with src/50_loader/dsv4_bind.c.
L_ATTN_NORM = 1
L_FFN_NORM = 2
L_ATTN_SINK = 3
L_Q_NORM = 4
L_KV_NORM = 5
L_WQ_A = 10
L_WQ_A_S = 11
L_WQ_B = 12
L_WQ_B_S = 13
L_WKV = 14
L_WKV_S = 15
L_WO_A = 16
L_WO_A_S = 17
L_WO_B = 18
L_WO_B_S = 19
L_COMP_APE = 20
L_COMP_NORM = 21
L_COMP_WKV = 22
L_COMP_WGATE = 23
L_IDX_WQ_B = 24
L_IDX_WQ_B_S = 25
L_IDX_WP = 26
L_IDX_COMP_APE = 27
L_IDX_COMP_NORM = 28
L_IDX_COMP_WKV = 29
L_IDX_COMP_WGATE = 30
L_HC_ATTN_FN = 31
L_HC_ATTN_BASE = 32
L_HC_ATTN_SCALE = 33
L_HC_FFN_FN = 34
L_HC_FFN_BASE = 35
L_HC_FFN_SCALE = 36
L_GATE = 40
L_TID2EID = 41
L_BIAS = 42
L_SH_W1 = 50
L_SH_W1_S = 51
L_SH_W2 = 52
L_SH_W2_S = 53
L_SH_W3 = 54
L_SH_W3_S = 55

# suffix -> (slot, kind, optional scale slot, qmat format)
# kind: vec, i64, qmat, scale
LAYER_TENSORS = {
    "attn_norm.weight": (L_ATTN_NORM, "vec", None, None),
    "ffn_norm.weight": (L_FFN_NORM, "vec", None, None),
    "attn.attn_sink": (L_ATTN_SINK, "vec", None, None),
    "attn.q_norm.weight": (L_Q_NORM, "vec", None, None),
    "attn.kv_norm.weight": (L_KV_NORM, "vec", None, None),

    "attn.wq_a.weight": (L_WQ_A, "qmat", L_WQ_A_S, "FP8"),
    "attn.wq_a.scale": (L_WQ_A_S, "scale", None, None),
    "attn.wq_b.weight": (L_WQ_B, "qmat", L_WQ_B_S, "FP8"),
    "attn.wq_b.scale": (L_WQ_B_S, "scale", None, None),
    "attn.wkv.weight": (L_WKV, "qmat", L_WKV_S, "FP8"),
    "attn.wkv.scale": (L_WKV_S, "scale", None, None),
    "attn.wo_a.weight": (L_WO_A, "qmat", L_WO_A_S, "FP8"),
    "attn.wo_a.scale": (L_WO_A_S, "scale", None, None),
    "attn.wo_b.weight": (L_WO_B, "qmat", L_WO_B_S, "FP8"),
    "attn.wo_b.scale": (L_WO_B_S, "scale", None, None),

    "attn.compressor.ape": (L_COMP_APE, "vec", None, None),
    "attn.compressor.norm.weight": (L_COMP_NORM, "vec", None, None),
    "attn.compressor.wkv.weight": (L_COMP_WKV, "qmat", None, "BF16"),
    "attn.compressor.wgate.weight": (L_COMP_WGATE, "qmat", None, "BF16"),

    "attn.indexer.wq_b.weight": (L_IDX_WQ_B, "qmat", L_IDX_WQ_B_S, "FP8"),
    "attn.indexer.wq_b.scale": (L_IDX_WQ_B_S, "scale", None, None),
    "attn.indexer.weights_proj.weight": (L_IDX_WP, "qmat", None, "BF16"),
    "attn.indexer.compressor.ape": (L_IDX_COMP_APE, "vec", None, None),
    "attn.indexer.compressor.norm.weight": (L_IDX_COMP_NORM, "vec", None, None),
    "attn.indexer.compressor.wkv.weight": (L_IDX_COMP_WKV, "qmat", None, "BF16"),
    "attn.indexer.compressor.wgate.weight": (L_IDX_COMP_WGATE, "qmat", None, "BF16"),

    "hc_attn_fn": (L_HC_ATTN_FN, "vec", None, None),
    "hc_attn_base": (L_HC_ATTN_BASE, "vec", None, None),
    "hc_attn_scale": (L_HC_ATTN_SCALE, "vec", None, None),
    "hc_ffn_fn": (L_HC_FFN_FN, "vec", None, None),
    "hc_ffn_base": (L_HC_FFN_BASE, "vec", None, None),
    "hc_ffn_scale": (L_HC_FFN_SCALE, "vec", None, None),

    "ffn.gate.weight": (L_GATE, "qmat", None, "BF16"),
    "ffn.gate.tid2eid": (L_TID2EID, "i64", None, None),
    "ffn.gate.bias": (L_BIAS, "vec", None, None),

    "ffn.shared_experts.w1.weight": (L_SH_W1, "qmat", L_SH_W1_S, "FP8"),
    "ffn.shared_experts.w1.scale": (L_SH_W1_S, "scale", None, None),
    "ffn.shared_experts.w2.weight": (L_SH_W2, "qmat", L_SH_W2_S, "FP8"),
    "ffn.shared_experts.w2.scale": (L_SH_W2_S, "scale", None, None),
    "ffn.shared_experts.w3.weight": (L_SH_W3, "qmat", L_SH_W3_S, "FP8"),
    "ffn.shared_experts.w3.scale": (L_SH_W3_S, "scale", None, None),
}

# Model slots. Keep in step with src/50_loader/dsv4_bind.c.
M_MODEL = {
    "embed.weight": 1,
    "norm.weight": 2,
    "head.weight": 3,
    "hc_head_fn": 4,
    "hc_head_base": 5,
    "hc_head_scale": 6,
}

# Expert tensor order. The C cache reads these six slots per expert.
EXPERT_SUFFIX = [
    "ffn.experts.{e}.w1.weight",
    "ffn.experts.{e}.w1.scale",
    "ffn.experts.{e}.w2.weight",
    "ffn.experts.{e}.w2.scale",
    "ffn.experts.{e}.w3.weight",
    "ffn.experts.{e}.w3.scale",
]


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def is_expert(name):
    return ".ffn.experts." in name


def index_checkpoint(model_dir):
    """name -> (shard, absolute_offset, nbytes, dtype, shape)."""
    idx = {}
    shards = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if not shards:
        sys.exit(f"no .safetensors in {model_dir}")
    for si, s in enumerate(shards):
        p = os.path.join(model_dir, s)
        try:
            hdr, base = read_header(p)
        except Exception as e:
            print(f"  skipping {s}: {e}")
            continue
        for name, v in hdr.items():
            a, b = v["data_offsets"]
            idx[name] = (si, base + a, b - a, v["dtype"], v.get("shape", []))
    return idx, shards


def qmat_meta(dtype, shape, fmt):
    """Return (rows, cols, wdt, blk_r, blk_c) for a weight tensor."""
    if fmt == "FP8":
        return shape[0], shape[1], WDT["FP8"], 128, 128
    if fmt == "BF16":
        return shape[0], shape[1], WDT["BF16"], 0, 0
    if fmt == "FP4":
        return shape[0], shape[1] * 2, WDT["FP4"], 1, 32
    raise ValueError(fmt)


def layer_entries(L, idx):
    """All known trunk tensors of layer L, in slot order.

    Returns a list of dicts with name, slot, kind, qmat format and location.
    A tensor the checkpoint does not carry is skipped; a missing tensor the
    layer needs will make the C bind fail loudly rather than read zeroes.
    """
    pfx = f"layers.{L}."
    out = []
    for name, (shard, off, nb, dt, shape) in idx.items():
        if not name.startswith(pfx) or is_expert(name):
            continue
        suffix = name[len(pfx):]
        ent = LAYER_TENSORS.get(suffix)
        if not ent:
            continue
        slot, kind, scale_slot, fmt = ent
        e = {"name": name, "suffix": suffix, "slot": slot, "kind": kind,
             "scale_slot": scale_slot, "fmt": fmt, "shard": shard,
             "off": off, "nbytes": nb, "dtype": DT[dt], "shape": shape}
        out.append(e)
    out.sort(key=lambda e: e["slot"])
    return out


BASE_SLOTS = {L_ATTN_NORM, L_FFN_NORM, L_ATTN_SINK, L_Q_NORM, L_KV_NORM,
              L_WQ_A, L_WQ_A_S, L_WQ_B, L_WQ_B_S, L_WKV, L_WKV_S,
              L_WO_A, L_WO_A_S, L_WO_B, L_WO_B_S,
              L_HC_ATTN_FN, L_HC_ATTN_BASE, L_HC_ATTN_SCALE,
              L_HC_FFN_FN, L_HC_FFN_BASE, L_HC_FFN_SCALE,
              L_GATE, L_SH_W1, L_SH_W1_S, L_SH_W2, L_SH_W2_S,
              L_SH_W3, L_SH_W3_S}
COMP_SLOTS = {L_COMP_APE, L_COMP_NORM, L_COMP_WKV, L_COMP_WGATE}
IDX_SLOTS = {L_IDX_WQ_B, L_IDX_WQ_B_S, L_IDX_WP,
             L_IDX_COMP_APE, L_IDX_COMP_NORM,
             L_IDX_COMP_WKV, L_IDX_COMP_WGATE}


def validate_layer(entries, L):
    """Refuse to write an index a layer cannot be bound from.

    The C binder trusts the slot set; a missing tensor would become a NULL
    weight pointer and a plausible crash much later.
    """
    slots = {e["slot"] for e in entries}
    need = set(BASE_SLOTS)
    if slots & COMP_SLOTS:
        need |= COMP_SLOTS
    if slots & IDX_SLOTS:
        need |= IDX_SLOTS
    need.add(L_TID2EID if L_TID2EID in slots else L_BIAS)
    missing = need - slots
    if missing:
        sys.exit(f"layer {L}: missing slot(s) {sorted(missing)}")
    if L_TID2EID in slots and L_BIAS in slots:
        sys.exit(f"layer {L}: both hash and scored gate tensors")


def widen_bytes(entries):
    total = 0
    for e in entries:
        if e["kind"] in ("vec",) and e["dtype"] != DT["F32"]:
            esz = {1: 1, 2: 2, 3: 2, 4: 4, 8: 8}.get(e["dtype"], 0)
            if esz:
                total = (total + (e["nbytes"] // esz) * 4 + 7) & ~7
    return total


def write_trunk_index(path, n_layers, layers, widen):
    with open(path, "w", encoding="utf-8") as f:
        f.write("DSV4TRUNK 2\n")
        f.write(f"n_layers {n_layers}\n")
        f.write("layout 1\n")
        for L, lay in enumerate(layers):
            if lay is None:
                f.write(f"layer {L} 0 0 0 0\n")
                continue
            ents = lay["entries"]
            f.write(f"layer {L} {lay['file_off']} {lay['nbytes']} "
                    f"{len(ents)} {widen[L]}\n")
            for e in ents:
                rows, cols, wdt, br, bc = (0, 0, 0, 0, 0)
                if e["kind"] == "qmat":
                    rows, cols, wdt, br, bc = qmat_meta(
                        e["dtype"], e["shape"], e["fmt"])
                f.write(f"slot {e['slot']} {e['off']} {e['nbytes']} "
                        f"{e['dtype']} {rows} {cols} {wdt} {br} {bc}\n")


def write_model_index(path, idx):
    entries = []
    for name, slot in sorted(M_MODEL.items(), key=lambda kv: kv[1]):
        if name not in idx:
            sys.exit(f"model tensor {name} is missing")
        shard, off, nb, dt, _ = idx[name]
        entries.append((slot, shard, off, nb, DT[dt]))
    with open(path, "wb") as f:
        f.write(b"DSV4MDL1")
        f.write(struct.pack("<ii", len(entries), 0))
        for slot, shard, off, nb, dt in entries:
            f.write(struct.pack("<iiqqii", slot, shard, off, nb, dt, 0))


def write_expert_index(path, n_layers, n_experts, idx):
    with open(path, "wb") as f:
        f.write(b"DSV4EXP1")
        f.write(struct.pack("<ii", n_layers, n_experts))
        for L in range(n_layers):
            pfx = f"layers.{L}."
            for E in range(n_experts):
                for tpl in EXPERT_SUFFIX:
                    name = pfx + tpl.format(e=E)
                    if name not in idx:
                        sys.exit(f"expert tensor {name} is missing")
                    shard, off, nb, _dt, _shape = idx[name]
                    # int32 shard, int32 pad, int64 off, int64 nbytes
                    f.write(struct.pack("<iiqq", shard, 0, off, nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--layers", type=int, default=None,
                    help="pack only the first N layers (for testing)")
    ap.add_argument("--reindex", action="store_true",
                    help="rewrite the index files from existing trunk.bin")
    args = ap.parse_args()

    model_dir = os.path.expanduser(args.model_dir)
    out_dir = os.path.expanduser(args.out_dir)
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    n_layers_cfg = cfg["num_hidden_layers"]
    n_layers = min(n_layers_cfg, args.layers) if args.layers else n_layers_cfg

    idx, shards = index_checkpoint(model_dir)

    # The expert count is in the checkpoint itself; no config key guessing.
    max_e = -1
    for name in idx:
        if ".ffn.experts." not in name:
            continue
        parts = name.split(".")
        try:
            e = int(parts[parts.index("experts") + 1])
        except (ValueError, IndexError):
            continue
        if e > max_e:
            max_e = e
    n_experts = max_e + 1
    if n_experts <= 0:
        sys.exit("no routed expert tensors found")
    print(f"indexed {len(idx):,} tensors across {len(shards)} shards; "
          f"{n_experts} experts/layer")

    layers = []
    widen = []
    if args.reindex:
        manifest = json.load(open(os.path.join(out_dir, "trunk.json"),
                                  encoding="utf-8"))
        if manifest["n_layers"] != n_layers:
            n_layers = manifest["n_layers"]
        for L, lay in enumerate(manifest["layers"]):
            if lay is None:
                layers.append(None)
                widen.append(0)
                continue
            entries = []
            for t in lay["tensors"]:
                suffix = t["name"][len(f"layers.{L}."):]
                ent = LAYER_TENSORS.get(suffix)
                if not ent:
                    continue
                slot, kind, scale_slot, fmt = ent
                shape = idx[t["name"]][4] if t["name"] in idx else []
                entries.append({"name": t["name"], "suffix": suffix,
                                "slot": slot, "kind": kind,
                                "scale_slot": scale_slot, "fmt": fmt,
                                "shard": 0, "off": t["off"],
                                "nbytes": t["nbytes"], "dtype": DT[t["dtype"]],
                                "shape": shape})
            entries.sort(key=lambda e: e["slot"])
            validate_layer(entries, L)
            layers.append({"file_off": lay["file_off"],
                           "nbytes": lay["nbytes"], "entries": entries})
            widen.append(widen_bytes(entries))
        print(f"reindexed {len(layers)} layers from trunk.json")
    else:
        os.makedirs(out_dir, exist_ok=True)
        bin_path = os.path.join(out_dir, "trunk.bin")
        packed = skipped = 0
        with open(bin_path, "wb") as out:
            for L in range(n_layers):
                entries = layer_entries(L, idx)
                if not entries:
                    skipped += 1
                    layers.append(None)
                    widen.append(0)
                    continue
                validate_layer(entries, L)

                # Pad the layer's start so every run begins O_DIRECT-aligned.
                out.seek((out.tell() + ALIGN - 1) // ALIGN * ALIGN)
                run_start = out.tell()

                for e in entries:
                    shard, off, nb = e["shard"], e["off"], e["nbytes"]
                    with open(os.path.join(model_dir, shards[shard]), "rb") as f:
                        f.seek(off)
                        buf = f.read(nb)
                    if len(buf) != nb:
                        sys.exit(f"short read on {e['name']}")
                    e["off"] = out.tell() - run_start
                    out.write(buf)
                    pad = (-out.tell()) % 8
                    if pad:
                        out.write(b"\0" * pad)

                run_bytes = out.tell() - run_start
                layers.append({"file_off": run_start,
                               "nbytes": run_bytes, "entries": entries})
                widen.append(widen_bytes(entries))
                packed += 1
                print(f"  layer {L:>3}: {len(entries):>3} tensors, "
                      f"{run_bytes / 1048576:8.2f} MB at {run_start:,}")

            # Pad the file itself so an O_DIRECT read of the last run cannot run
            # off the end when it is widened out to an aligned window.
            out.seek((out.tell() + ALIGN - 1) // ALIGN * ALIGN - 1)
            out.write(b"\0")

        manifest = {"n_layers": n_layers, "align": ALIGN,
                    "layers": [None if lay is None else
                               {"file_off": lay["file_off"],
                                "nbytes": lay["nbytes"],
                                "tensors": [{"name": e["name"],
                                             "off": e["off"],
                                             "nbytes": e["nbytes"],
                                             "dtype": next(
                                                 k for k, v in DT.items()
                                                 if v == e["dtype"])}
                                            for e in lay["entries"]]}
                               for lay in layers]}
        json_path = os.path.join(out_dir, "trunk.json")
        with open(json_path, "w") as f:
            json.dump(manifest, f)
        total = os.path.getsize(bin_path)
        print(f"\npacked {packed} layers ({skipped} absent) -> {bin_path}")
        print(f"  {total:,} bytes = {total / 1073741824:.2f} GB")

    write_trunk_index(os.path.join(out_dir, "trunk.idx"), n_layers,
                      layers, widen)
    write_model_index(os.path.join(out_dir, "model.idx"), idx)
    write_expert_index(os.path.join(out_dir, "experts.idx"),
                       n_layers, n_experts, idx)
    print(f"wrote trunk.idx, model.idx and experts.idx in {out_dir}")


if __name__ == "__main__":
    main()
