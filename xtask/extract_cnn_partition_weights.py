#!/usr/bin/env python3
"""Extract the intra CNN partition model weights from libaom into a Rust file.

Reads reference/libaom/av1/encoder/partition_cnn_weights.h and emits
crates/aom-encode/src/cnn_partition/weights.rs.

Each float value is carried through VERBATIM: the exact decimal digits written
in the C source (e.g. `0.604356`, `-5.97783`, `-1.34495e-05`) with only the
trailing `f` suffix dropped, then an `f32` suffix appended. No reformatting of
the number — this guarantees identical round-to-nearest as the C compiler.

Integer arrays (quad_to_linear_*) are emitted as bare i32 literals.

FAILS LOUD (nonzero exit) if any extracted array length does not match the
expected size baked into the target list below.
"""

import re
import sys

SRC = "/root/aom-rs/reference/libaom/av1/encoder/partition_cnn_weights.h"
DST = "/root/aom-rs/crates/aom-encode/src/cnn_partition/weights.rs"

# num_features feeding each branch's dnn_layer_0 (from the header's NN_CONFIG /
# feature-assembly comments): branch b has num_features[b] inputs, and
# dnn_layer_0_kernel is num_features[b] * 16.
BRANCH_NUM_FEATURES = [37, 25, 25, 41]

# (rust_name, c_name, expected_len, is_int)
FLOAT_TARGETS = []

# --- CNN conv layers ---------------------------------------------------------
CNN_LAYERS = [
    # (idx, kernel_len, bias_len)
    (0, 500, 20),
    (1, 1600, 20),
    (2, 1600, 20),
    (3, 320, 4),
    (4, 320, 20),
]
for idx, kn, bn in CNN_LAYERS:
    FLOAT_TARGETS.append((
        f"CNN_LAYER_{idx}_KERNEL",
        f"av1_intra_mode_cnn_partition_cnn_layer_{idx}_kernel",
        kn, False,
    ))
    FLOAT_TARGETS.append((
        f"CNN_LAYER_{idx}_BIAS",
        f"av1_intra_mode_cnn_partition_cnn_layer_{idx}_bias",
        bn, False,
    ))

# --- branch DNNs -------------------------------------------------------------
for b in range(4):
    nf = BRANCH_NUM_FEATURES[b]
    pfx = f"av1_intra_mode_cnn_partition_branch_{b}"
    FLOAT_TARGETS.append((
        f"BRANCH_{b}_DNN_LAYER_0_KERNEL", f"{pfx}_dnn_layer_0_kernel",
        nf * 16, False,
    ))
    FLOAT_TARGETS.append((
        f"BRANCH_{b}_DNN_LAYER_0_BIAS", f"{pfx}_dnn_layer_0_bias",
        16, False,
    ))
    FLOAT_TARGETS.append((
        f"BRANCH_{b}_DNN_LAYER_1_KERNEL", f"{pfx}_dnn_layer_1_kernel",
        16 * 24, False,
    ))
    FLOAT_TARGETS.append((
        f"BRANCH_{b}_DNN_LAYER_1_BIAS", f"{pfx}_dnn_layer_1_bias",
        24, False,
    ))
    FLOAT_TARGETS.append((
        f"BRANCH_{b}_LOGITS_KERNEL", f"{pfx}_logits_kernel",
        24 * 1, False,
    ))
    FLOAT_TARGETS.append((
        f"BRANCH_{b}_LOGITS_BIAS", f"{pfx}_logits_bias",
        1, False,
    ))

# --- threshold / normalization arrays ---------------------------------------
for res in ("hdres", "midres", "lowres"):
    FLOAT_TARGETS.append((
        f"SPLIT_THRESH_{res.upper()}",
        f"av1_intra_mode_cnn_partition_split_thresh_{res}", 5, False,
    ))
    FLOAT_TARGETS.append((
        f"NO_SPLIT_THRESH_{res.upper()}",
        f"av1_intra_mode_cnn_partition_no_split_thresh_{res}", 5, False,
    ))
FLOAT_TARGETS.append((
    "MEAN", "av1_intra_mode_cnn_partition_mean", 1, False,
))
FLOAT_TARGETS.append((
    "STD", "av1_intra_mode_cnn_partition_std", 1, False,
))

# --- integer arrays ----------------------------------------------------------
INT_TARGETS = [
    ("QUAD_TO_LINEAR_1", "quad_to_linear_1", 4, True),
    ("QUAD_TO_LINEAR_2", "quad_to_linear_2", 16, True),
    ("QUAD_TO_LINEAR_3", "quad_to_linear_3", 64, True),
]

TARGETS = FLOAT_TARGETS + INT_TARGETS


def extract_body(src, c_name):
    """Return the raw text between the declaration's `= {` and its `}`.

    Matches the *declaration* specifically: `NAME[...] = {`. References to the
    same symbol inside CNN_CONFIG/NN_CONFIG structs are `NAME,` (no `[` `=` `{`)
    and are therefore not matched.
    """
    pat = re.compile(re.escape(c_name) + r"\s*\[[^\]]*\]\s*=\s*\{", re.DOTALL)
    m = pat.search(src)
    if not m:
        sys.exit(f"ERROR: declaration for `{c_name}` not found")
    start = m.end() - 1  # index of the opening '{'
    end = src.find("}", start)
    if end == -1:
        sys.exit(f"ERROR: no closing brace for `{c_name}`")
    return src[start + 1:end]


def parse_values(body, is_int, c_name):
    values = []
    for tok in body.split(","):
        t = tok.strip()
        if not t:
            continue
        if is_int:
            if not re.fullmatch(r"-?\d+", t):
                sys.exit(f"ERROR: non-integer token {t!r} in `{c_name}`")
            values.append(t)
        else:
            if not t.endswith("f"):
                sys.exit(
                    f"ERROR: float token {t!r} in `{c_name}` lacks `f` suffix"
                )
            values.append(t[:-1])  # drop trailing 'f', keep verbatim decimal
    return values


def emit_array(rust_name, values, is_int, per_line=8):
    ty = "i32" if is_int else "f32"
    suffix = "" if is_int else "f32"
    n = len(values)
    lines = ["#[rustfmt::skip]", f"pub static {rust_name}: [{ty}; {n}] = ["]
    for i in range(0, n, per_line):
        chunk = values[i:i + per_line]
        lines.append("    " + " ".join(f"{v}{suffix}," for v in chunk))
    lines.append("];")
    return "\n".join(lines)


def main():
    src = open(SRC).read()

    header = (
        "//! GENERATED by xtask/extract_cnn_partition_weights.py from\n"
        "//! reference/libaom/av1/encoder/partition_cnn_weights.h "
        "(libaom v3.14.1).\n"
        "//! Verbatim decimal literals -> f32 (identical round-to-nearest "
        "as the C).\n"
        "//! Do not edit by hand; re-run the extractor.\n"
        "#![allow(clippy::all)]\n"
        "#![allow(clippy::excessive_precision)]\n"
    )

    chunks = [header]
    counts = []
    mismatches = []

    for rust_name, c_name, expected, is_int in TARGETS:
        body = extract_body(src, c_name)
        values = parse_values(body, is_int, c_name)
        got = len(values)
        counts.append((rust_name, got, expected))
        if got != expected:
            mismatches.append((rust_name, got, expected))
        chunks.append(emit_array(rust_name, values, is_int))

    # Report every count to stderr.
    for rust_name, got, expected in counts:
        status = "OK" if got == expected else "MISMATCH"
        print(f"  {status:8} {rust_name:28} got={got:5} expect={expected}",
              file=sys.stderr)

    if mismatches:
        print("\nFATAL: size mismatch(es), refusing to write:", file=sys.stderr)
        for rust_name, got, expected in mismatches:
            print(f"  {rust_name}: got {got}, expected {expected}",
                  file=sys.stderr)
        sys.exit(1)

    out = "\n\n".join(chunks) + "\n"
    open(DST, "w").write(out)
    print(f"\nwrote {DST} ({len(out.encode('utf-8'))} bytes, "
          f"{len(TARGETS)} arrays)", file=sys.stderr)


if __name__ == "__main__":
    main()
