#!/usr/bin/env python3
"""Transcribe the PARTITION_HORZ_4/VERT_4 pruning NN (av1_ml_prune_4_partition,
partition_strategy.c) weight tables from libaom's partition_model_weights.h
into Rust.

Only the ml_model_index==1 ("hd_", softmax/NEW_LABEL_SIZE=3) variant is
transcribed: `ml_4_partition_search_level_index` stays 0 at speed 0 for BOTH
GOOD and ALLINTRA (verified against speed_features.c -- every override is
gated `if (speed >= 1)` or higher), and `ml_model_index =
(ml_4_partition_search_level_index < 3)` is therefore always 1 in our
speed-0 envelope, so the ml_model_index==0 (LABEL_SIZE=4, non-softmax)
weight variant is dead code for this port and is NOT transcribed.

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


FEATURE_SIZE = 18
NEW_LABEL_SIZE = 3
HIDDEN = {16: 24, 32: 32, 64: 24}

bsizes = [16, 32, 64]
tables = {}
for bs in bsizes:
    h = HIDDEN[bs]
    tables[bs] = {
        "mean": named_array(f"av1_partition4_nn_mean_{bs}", FEATURE_SIZE),
        "std": named_array(f"av1_partition4_nn_std_{bs}", FEATURE_SIZE),
        "w0": named_array(f"av1_hd_4_partition_nn_weights_{bs}_layer0", FEATURE_SIZE * h),
        "b0": named_array(f"av1_hd_4_partition_nn_bias_{bs}_layer0", h),
        "w1": named_array(f"av1_hd_4_partition_nn_weights_{bs}_layer1", h * NEW_LABEL_SIZE),
        "b1": named_array(f"av1_hd_4_partition_nn_bias_{bs}_layer1", NEW_LABEL_SIZE),
    }

# av1_partition4_search_thresh[6][3][5] / av1_partition4_not_search_thresh[6][3][5]
# (aggressiveness=6 x res_idx=3 x bsize_idx=5). Only row [0] (aggressiveness ==
# ml_4_partition_search_level_index == 0, the speed-0 value both usages) is
# actually read by this port, but the whole (tiny, 90-float) table is
# transcribed for shape-fidelity / future speed>0 use.
def thresh_table(name):
    m = re.search(
        r"static const float\s+" + re.escape(name) + r"\[6\]\[3\]\[5\]\s*=\s*\{(.*?)\n\};",
        src, re.S,
    )
    assert m, name
    vals = floats(m.group(1))
    assert len(vals) == 6 * 3 * 5, (name, len(vals))
    return vals


search_thresh = thresh_table("av1_partition4_search_thresh")
not_search_thresh = thresh_table("av1_partition4_not_search_thresh")


def rust_arr(vals, per_line=6):
    lines = []
    for i in range(0, len(vals), per_line):
        row = ", ".join(repr(v) for v in vals[i : i + per_line])
        lines.append(f"    {row},")
    return "\n".join(lines)


out = []
out.append("//! `av1_ml_prune_4_partition`'s NN weight tables, transcribed from libaom")
out.append("//! v3.14.1 `av1/encoder/partition_model_weights.h` by")
out.append("//! `xtask/transcribe_part4_nn.py`. Only the `ml_model_index == 1`")
out.append("//! (\"hd_\", softmax / NEW_LABEL_SIZE=3) variant is present -- see the")
out.append("//! script docstring for why the other variant is dead at speed 0.")
out.append("//! Do not hand-edit -- rerun the script against the checked-in libaom")
out.append("//! reference source instead.")
out.append("#![allow(clippy::excessive_precision)]")
out.append("")
out.append("pub const FEATURE_SIZE: usize = 18;")
out.append("pub const NEW_LABEL_SIZE: usize = 3;")
out.append("")
for bs in bsizes:
    t = tables[bs]
    h = HIDDEN[bs]
    out.append(f"pub const HIDDEN_{bs}: usize = {h};")
    out.append(f"#[rustfmt::skip]")
    out.append(f"pub static MEAN_{bs}: [f32; FEATURE_SIZE] = [\n{rust_arr(t['mean'])}\n];")
    out.append(f"#[rustfmt::skip]")
    out.append(f"pub static STD_{bs}: [f32; FEATURE_SIZE] = [\n{rust_arr(t['std'])}\n];")
    out.append(f"#[rustfmt::skip]")
    out.append(f"pub static W0_{bs}: [f32; FEATURE_SIZE * HIDDEN_{bs}] = [\n{rust_arr(t['w0'])}\n];")
    out.append(f"#[rustfmt::skip]")
    out.append(f"pub static B0_{bs}: [f32; HIDDEN_{bs}] = [\n{rust_arr(t['b0'])}\n];")
    out.append(f"#[rustfmt::skip]")
    out.append(f"pub static W1_{bs}: [f32; HIDDEN_{bs} * NEW_LABEL_SIZE] = [\n{rust_arr(t['w1'])}\n];")
    out.append(f"#[rustfmt::skip]")
    out.append(f"pub static B1_{bs}: [f32; NEW_LABEL_SIZE] = [\n{rust_arr(t['b1'])}\n];")
    out.append("")
out.append("/// `av1_partition4_search_thresh[aggressiveness=6][res_idx=3][bsize_idx=5]`.")
out.append("#[rustfmt::skip]")
out.append(f"pub static SEARCH_THRESH: [f32; 6 * 3 * 5] = [\n{rust_arr(search_thresh)}\n];")
out.append("/// `av1_partition4_not_search_thresh[aggressiveness=6][res_idx=3][bsize_idx=5]`.")
out.append("#[rustfmt::skip]")
out.append(f"pub static NOT_SEARCH_THRESH: [f32; 6 * 3 * 5] = [\n{rust_arr(not_search_thresh)}\n];")
out.append("")

open("crates/aom-encode/src/part4_nn_weights.rs", "w").write("\n".join(out) + "\n")
total = sum(len(t["mean"]) + len(t["std"]) + len(t["w0"]) + len(t["b0"]) + len(t["w1"]) + len(t["b1"]) for t in tables.values())
print(f"wrote {total} weight/mean/std floats + {len(search_thresh)+len(not_search_thresh)} threshold floats")
