#!/usr/bin/env python3
"""Transcribe the AB-partition pruning NN (ml_prune_ab_partition,
partition_strategy.c:1223, the NN behind PARTITION_HORZ_A/HORZ_B/VERT_A/
VERT_B candidate pruning) weight tables from libaom's
partition_model_weights.h into Rust.

Unlike the sibling 4-way NN (transcribe_part4_nn.py), this NN has only ONE
weight variant (no ml_model_index branch), no mean/std feature
normalization, and a uniform hidden-layer size (64) across all 4 reachable
block sizes (16/32/64/128 -- AB partitions ARE reachable at BLOCK_128X128,
unlike the 4-way NN which tops out at 64x64). FEATURE_SIZE=10, LABEL_SIZE=16
(16-way bitmask over {HORZ_A,HORZ_B,VERT_A,VERT_B} allowed/disallowed
combinations) -- these are the file's TOP-OF-HEADER `#define FEATURE_SIZE`/
`#define LABEL_SIZE` values (partition_model_weights.h:25-26); the header
redefines both macros further down for OTHER unrelated NN sections in the
same file (e.g. line 1318 redefines them to 18/4 for the 4-way section), so
they are hardcoded as Python constants here rather than regex-extracted.

Faithful mechanical transcription; correctness is enforced by the
table-shape assertions here plus the differential partition-search gate
tests in aom-encode/tests/.
"""
import re

SRC = "reference/libaom/av1/encoder/partition_model_weights.h"
src = open(SRC).read()


def floats(body):
    return [float(v[:-1]) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?f", body)]


def named_array(name, expected_len=None):
    m = re.search(
        r"static const float\s+" + re.escape(name) + r"\[[^\]]*\]\s*=\s*\{(.*?)\};",
        src, re.S,
    )
    assert m, f"array not found: {name}"
    vals = floats(m.group(1))
    if expected_len is not None:
        assert len(vals) == expected_len, (name, len(vals), expected_len)
    return vals


FEATURE_SIZE = 10
LABEL_SIZE = 16
HIDDEN = 64  # uniform across all 4 bsizes (verified by reading the NN_CONFIG
             # blocks directly: av1_ab_partition_nnconfig_{16,32,64,128} all
             # declare `1, // num_hidden_layers` + `64, // num_hidden_nodes`)

bsizes = [16, 32, 64, 128]
tables = {}
for bs in bsizes:
    tables[bs] = {
        "w0": named_array(f"av1_ab_partition_nn_weights_{bs}_layer0", FEATURE_SIZE * HIDDEN),
        "b0": named_array(f"av1_ab_partition_nn_bias_{bs}_layer0", HIDDEN),
        "w1": named_array(f"av1_ab_partition_nn_weights_{bs}_layer1", HIDDEN * LABEL_SIZE),
        "b1": named_array(f"av1_ab_partition_nn_bias_{bs}_layer1", LABEL_SIZE),
    }


def rust_arr(vals, per_line=6):
    lines = []
    for i in range(0, len(vals), per_line):
        row = ", ".join(repr(v) for v in vals[i : i + per_line])
        lines.append(f"    {row},")
    return "\n".join(lines)


out = []
out.append("//! `ml_prune_ab_partition`'s NN weight tables (the pruning NN behind")
out.append("//! `PARTITION_HORZ_A`/`HORZ_B`/`VERT_A`/`VERT_B` candidate selection),")
out.append("//! transcribed from libaom v3.14.1")
out.append("//! `av1/encoder/partition_model_weights.h` by")
out.append("//! `xtask/transcribe_ab_nn.py`. Unlike the sibling 4-way NN")
out.append("//! (`part4_nn_weights.rs`), this NN has a single weight variant (no")
out.append("//! ml_model_index branch), no mean/std feature normalization, and a")
out.append("//! uniform hidden-layer size across all 4 reachable block sizes -- see")
out.append("//! the script docstring for details. Do not hand-edit -- rerun the")
out.append("//! script against the checked-in libaom reference source instead.")
out.append("#![allow(clippy::excessive_precision)]")
out.append("")
out.append(f"pub const FEATURE_SIZE: usize = {FEATURE_SIZE};")
out.append(f"pub const LABEL_SIZE: usize = {LABEL_SIZE};")
out.append(f"pub const HIDDEN: usize = {HIDDEN};")
out.append("")
for bs in bsizes:
    t = tables[bs]
    out.append(f"#[rustfmt::skip]")
    out.append(f"pub static W0_{bs}: [f32; FEATURE_SIZE * HIDDEN] = [\n{rust_arr(t['w0'])}\n];")
    out.append(f"#[rustfmt::skip]")
    out.append(f"pub static B0_{bs}: [f32; HIDDEN] = [\n{rust_arr(t['b0'])}\n];")
    out.append(f"#[rustfmt::skip]")
    out.append(f"pub static W1_{bs}: [f32; HIDDEN * LABEL_SIZE] = [\n{rust_arr(t['w1'])}\n];")
    out.append(f"#[rustfmt::skip]")
    out.append(f"pub static B1_{bs}: [f32; LABEL_SIZE] = [\n{rust_arr(t['b1'])}\n];")
    out.append("")

open("crates/aom-encode/src/ab_nn_weights.rs", "w").write("\n".join(out) + "\n")
total = sum(len(t["w0"]) + len(t["b0"]) + len(t["w1"]) + len(t["b1"]) for t in tables.values())
print(f"wrote {total} weight/bias floats across {len(bsizes)} block sizes")
