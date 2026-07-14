#!/usr/bin/env python3
"""Generate crates/aom-entropy/src/default_cdfs.rs from libaom's default-CDF
initializers (av1/common/entropymode.c, entropymv.c, token_cdfs.h).

Parsing model (aom_dsp/prob.h): AOM_CDFn(a0..a_{n-2}) expands to n+1 row slots
[32768-a0, ..., 32768-a_{n-2}, AOM_ICDF(CDF_PROB_TOP)=0, count=0]; innermost
brace groups are rows zero-padded to the declared CDF_SIZE; outer groups
zero-pad missing children. The generated tables are diffed BYTE-IDENTICAL
against the COMPILED defaults (shim_dump_default_kf_fc, which drives the real
av1_setup_past_independence) in crates/aom-entropy/tests/default_cdfs_diff.rs —
that closes the loop over this parse, the Rust struct mapping, and the aom-txb
arena packing.

Usage: python3 xtask/gen_default_cdfs.py && cargo fmt -p aom-entropy  (repo root)
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reference/libaom/av1/common"
OUT = ROOT / "crates/aom-entropy/src/default_cdfs.rs"

CONSTS = {
    "TOKEN_CDF_Q_CTXS": 4, "TX_SIZES": 5, "PLANE_TYPES": 2,
    "TXB_SKIP_CONTEXTS": 13, "EOB_COEF_CONTEXTS": 9,
    "SIG_COEF_CONTEXTS_EOB": 4, "SIG_COEF_CONTEXTS": 42, "LEVEL_CONTEXTS": 21,
    "BR_CDF_SIZE": 4, "DC_SIGN_CONTEXTS": 3, "KF_MODE_CONTEXTS": 5,
    "INTRA_MODES": 13, "UV_INTRA_MODES": 14, "CFL_ALLOWED_TYPES": 2,
    "DIRECTIONAL_MODES": 8, "SKIP_CONTEXTS": 3, "SPATIAL_PREDICTION_PROBS": 3,
    "MAX_SEGMENTS": 8, "PARTITION_CONTEXTS": 20, "EXT_PARTITION_TYPES": 10,
    "PALATTE_BSIZE_CTXS": 7, "PALETTE_Y_MODE_CONTEXTS": 3,
    "PALETTE_UV_MODE_CONTEXTS": 2, "PALETTE_SIZES": 7, "BLOCK_SIZES_ALL": 22,
    "PALETTE_COLOR_INDEX_CONTEXTS": 5, "PALETTE_COLORS": 8,
    "RESTORE_SWITCHABLE_TYPES": 3,
    "FILTER_INTRA_MODES": 5, "CFL_JOINT_SIGNS": 8, "CFL_ALPHA_CONTEXTS": 6,
    "CFL_ALPHABET_SIZE": 16, "DELTA_Q_PROBS": 3, "FRAME_LF_COUNT": 4,
    "DELTA_LF_PROBS": 3, "MAX_TX_CATS": 4, "TX_SIZE_CONTEXTS": 3,
    "MAX_TX_DEPTH": 2, "EXT_TX_SETS_INTRA": 3, "EXT_TX_SETS_INTER": 4, "EXT_TX_SIZES": 4,
    "TX_TYPES": 16, "MV_JOINTS": 4, "MV_CLASSES": 11, "CLASS0_SIZE": 2,
    "MV_FP_SIZE": 4, "MV_OFFSET_BITS": 10, "MAX_ANGLE_DELTA": 3,
    "NUM_BASE_LEVELS": 2, "EOB_MAX_SYMS": 11,
}


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def eval_dim(expr):
    expr = " ".join(expr.split())  # collapse line breaks inside brackets
    m = re.fullmatch(r"CDF_SIZE\((.+)\)", expr)
    if m:
        return eval_dim(m.group(1)) + 1
    for name, val in CONSTS.items():
        expr = re.sub(rf"\b{name}\b", str(val), expr)
    if not re.fullmatch(r"[0-9+\-* ()]+", expr):
        sys.exit(f"unparseable dim expr: {expr!r}")
    return eval(expr)  # arithmetic on ints only (validated above)


def expand_macros(body):
    def repl(m):
        n = int(m.group(1))
        args = [a.strip() for a in m.group(2).split(",")]
        if len(args) != n - 1:
            sys.exit(f"AOM_CDF{n} arg count {len(args)}")
        vals = []
        for a in args:
            if not re.fullmatch(r"[0-9+\-* ()/]+", a):
                sys.exit(f"non-arithmetic AOM_CDF arg: {a!r}")
            vals.append(32768 - eval(a))  # integer arithmetic only (validated)
        vals += [0, 0]  # AOM_ICDF(CDF_PROB_TOP) value slot + count slot
        return ", ".join(str(v) for v in vals)

    out = re.sub(r"AOM_CDF(\d+)\s*\(([^)]*)\)", repl, body)
    if "AOM_CDF" in out or "AOM_ICDF" in out:
        sys.exit("unexpanded CDF macro remains")
    return out


def parse_braces(s, i=0):
    """Parse a { ... } group at s[i] into nested lists of ints."""
    assert s[i] == "{"
    i += 1
    out = []
    num = ""
    while True:
        ch = s[i]
        if ch == "{":
            child, i = parse_braces(s, i)
            out.append(child)
        elif ch == "}":
            if num.strip():
                out.append(int(num))
            return out, i + 1
        elif ch == ",":
            if num.strip():
                out.append(int(num))
            num = ""
            i += 1
            continue
        elif ch.isspace():
            i += 1
            continue
        else:
            num += ch
            i += 1
            continue
        num = ""


def flatten(node, dims):
    """Zero-padded row-major flatten of a brace tree against declared dims."""
    if len(dims) == 1:
        vals = [v for v in node if isinstance(v, int)]
        if any(isinstance(v, list) for v in node):
            sys.exit(f"unexpected nesting at leaf (dims {dims})")
        if len(vals) > dims[0]:
            sys.exit(f"row overflow: {len(vals)} > {dims[0]}")
        return vals + [0] * (dims[0] - len(vals))
    kids = list(node)
    if len(kids) > dims[0]:
        sys.exit(f"group overflow: {len(kids)} > {dims[0]}")
    flat = []
    for k in kids:
        if not isinstance(k, list):
            sys.exit(f"expected braced child at dims {dims}")
        flat += flatten(k, dims[1:])
    inner = 1
    for d in dims[1:]:
        inner *= d
    flat += [0] * ((dims[0] - len(kids)) * inner)
    return flat


def extract(text, name):
    """Find `name[..][..] = { ... };` -> (dims, flat values)."""
    m = re.search(rf"\b{name}\b((?:\s*\[[^\]]*\])+)\s*=\s*", text)
    if not m:
        sys.exit(f"table {name} not found")
    dims = [eval_dim(d) for d in re.findall(r"\[([^\]]*)\]", m.group(1))]
    i = text.index("{", m.end())
    tree, _ = parse_braces(text, i)
    return dims, flatten(tree, dims)


def rust_array(flat, dims):
    if len(dims) == 1:
        return "[" + ", ".join(str(v) for v in flat) + "]"
    inner = 1
    for d in dims[1:]:
        inner *= d
    parts = [rust_array(flat[i * inner:(i + 1) * inner], dims[1:]) for i in range(dims[0])]
    return "[" + ", ".join(parts) + "]"


def rust_type(dims):
    t = "u16"
    for d in reversed(dims):
        t = f"[{t}; {d}]"
    return t


def main():
    mode_c = strip_comments((SRC / "entropymode.c").read_text())
    mv_c = strip_comments((SRC / "entropymv.c").read_text())
    tok_h = strip_comments((SRC / "token_cdfs.h").read_text())
    mode_c = expand_macros(mode_c)
    mv_c = expand_macros(mv_c)
    tok_h = expand_macros(tok_h)

    out = []
    out.append("//! GENERATED by xtask/gen_default_cdfs.py — DO NOT EDIT.")
    out.append("//!")
    out.append("//! libaom v3.14.1 default CDF tables (entropymode.c / entropymv.c /")
    out.append("//! token_cdfs.h), parsed from the C initializers: AOM_CDFn -> the n+1")
    out.append("//! row slots `[32768-a0, .., 32768-a_{n-2}, 0, 0]`, rows zero-padded to")
    out.append("//! the declared CDF_SIZE, ext-tx sets sliced to their exact-sized")
    out.append("//! alphabets, and the coefficient tables PRE-PACKED per qindex band into")
    out.append("//! the aom-txb 4045-u16 arena layout. Diffed byte-identical vs the")
    out.append("//! COMPILED defaults (`av1_setup_past_independence` via")
    out.append("//! `shim_dump_default_kf_fc`) in tests/default_cdfs_diff.rs.")
    out.append("")

    def emit(rust_name, dims, flat, doc):
        out.append(f"/// {doc}")
        # clippy::large_const_arrays: big tables become statics.
        kw = "static" if len(flat) > 512 else "const"
        out.append(f"pub {kw} {rust_name}: {rust_type(dims)} = {rust_array(flat, dims)};")
        out.append("")

    # --- mode tables, exact-sized ---
    d, f = extract(mode_c, "default_kf_y_mode_cdf")
    assert d == [5, 5, 14], d
    emit("DEFAULT_KF_Y", d, f, "`default_kf_y_mode_cdf[KF_MODE_CONTEXTS][KF_MODE_CONTEXTS]`.")
    d, f = extract(mode_c, "default_uv_mode_cdf")
    assert d == [2, 13, 15], d
    emit("DEFAULT_UV_MODE", d, f, "`default_uv_mode_cdf[CFL_ALLOWED_TYPES][INTRA_MODES]`.")
    d, f = extract(mode_c, "default_angle_delta_cdf")
    assert d == [8, 8], d
    emit("DEFAULT_ANGLE_DELTA", d, f, "`default_angle_delta_cdf[DIRECTIONAL_MODES]`.")
    d, f = extract(mode_c, "default_skip_txfm_cdfs")
    assert d == [3, 3], d
    emit("DEFAULT_SKIP", d, f, "`default_skip_txfm_cdfs[SKIP_CONTEXTS]`.")
    d, f = extract(mode_c, "default_spatial_pred_seg_tree_cdf")
    assert d == [3, 9], d
    emit("DEFAULT_SEG_SPATIAL", d, f, "`default_spatial_pred_seg_tree_cdf[SPATIAL_PREDICTION_PROBS]`.")
    d, f = extract(mode_c, "default_partition_cdf")
    assert d == [20, 11], d
    emit("DEFAULT_PARTITION", d, f, "`default_partition_cdf[PARTITION_CONTEXTS]` (ns-symbol rows).")
    d, f = extract(mode_c, "default_palette_y_mode_cdf")
    assert d == [7, 3, 3], d
    emit("DEFAULT_PALETTE_Y_MODE", d, f, "`default_palette_y_mode_cdf[PALATTE_BSIZE_CTXS][PALETTE_Y_MODE_CONTEXTS]`.")
    d, f = extract(mode_c, "default_palette_uv_mode_cdf")
    assert d == [2, 3], d
    emit("DEFAULT_PALETTE_UV_MODE", d, f, "`default_palette_uv_mode_cdf[PALETTE_UV_MODE_CONTEXTS]`.")
    d, f = extract(mode_c, "default_palette_y_size_cdf")
    assert d == [7, 8], d
    emit("DEFAULT_PALETTE_Y_SIZE", d, f, "`default_palette_y_size_cdf[PALATTE_BSIZE_CTXS]`.")
    d, f = extract(mode_c, "default_palette_uv_size_cdf")
    assert d == [7, 8], d
    emit("DEFAULT_PALETTE_UV_SIZE", d, f, "`default_palette_uv_size_cdf[PALATTE_BSIZE_CTXS]`.")
    d, f = extract(mode_c, "default_palette_y_color_index_cdf")
    assert d == [7, 5, 9], d
    emit("DEFAULT_PALETTE_Y_COLOR_INDEX", d, f,
         "`default_palette_y_color_index_cdf[PALETTE_SIZES][PALETTE_COLOR_INDEX_CONTEXTS]`.")
    d, f = extract(mode_c, "default_palette_uv_color_index_cdf")
    assert d == [7, 5, 9], d
    emit("DEFAULT_PALETTE_UV_COLOR_INDEX", d, f,
         "`default_palette_uv_color_index_cdf[PALETTE_SIZES][PALETTE_COLOR_INDEX_CONTEXTS]`.")
    d, f = extract(mode_c, "default_filter_intra_cdfs")
    assert d == [22, 3], d
    emit("DEFAULT_FILTER_INTRA", d, f, "`default_filter_intra_cdfs[BLOCK_SIZES_ALL]`.")
    d, f = extract(mode_c, "default_filter_intra_mode_cdf")
    assert d == [6], d
    emit("DEFAULT_FILTER_INTRA_MODE", d, f, "`default_filter_intra_mode_cdf`.")
    d, f = extract(mode_c, "default_cfl_sign_cdf")
    assert d == [9], d
    emit("DEFAULT_CFL_SIGN", d, f, "`default_cfl_sign_cdf`.")
    d, f = extract(mode_c, "default_cfl_alpha_cdf")
    assert d == [6, 17], d
    emit("DEFAULT_CFL_ALPHA", d, f, "`default_cfl_alpha_cdf[CFL_ALPHA_CONTEXTS]`.")
    d, f = extract(mode_c, "default_delta_q_cdf")
    assert d == [5], d
    emit("DEFAULT_DELTA_Q", d, f, "`default_delta_q_cdf`.")
    d, f = extract(mode_c, "default_delta_lf_multi_cdf")
    assert d == [4, 5], d
    emit("DEFAULT_DELTA_LF_MULTI", d, f, "`default_delta_lf_multi_cdf[FRAME_LF_COUNT]`.")
    d, f = extract(mode_c, "default_delta_lf_cdf")
    assert d == [5], d
    emit("DEFAULT_DELTA_LF", d, f, "`default_delta_lf_cdf`.")
    d, f = extract(mode_c, "default_intrabc_cdf")
    assert d == [3], d
    emit("DEFAULT_INTRABC", d, f, "`default_intrabc_cdf`.")
    d, f = extract(mode_c, "default_tx_size_cdf")
    assert d == [4, 3, 4], d
    emit("DEFAULT_TX_SIZE", d, f, "`default_tx_size_cdf[MAX_TX_CATS][TX_SIZE_CONTEXTS]` (ns-symbol rows: cat 0 codes 2 symbols).")

    # --- intra ext-tx: slice padded [3][4][13][17] to the exact-sized sets ---
    d, f = extract(mode_c, "default_intra_ext_tx_cdf")
    assert d == [3, 4, 13, 17], d
    def slice_set(set_idx, keep):
        vals = []
        for sz in range(4):
            for m in range(13):
                base = ((set_idx * 4 + sz) * 13 + m) * 17
                row = f[base:base + 17]
                assert all(v == 0 for v in row[keep:]), "nonzero beyond sliced alphabet"
                vals += row[:keep]
        return vals
    # set 0 (DCT-only) codes nothing — assert it is all-zero in the C table.
    assert all(v == 0 for v in f[: 4 * 13 * 17]), "ext-tx set 0 not all-zero"
    emit("DEFAULT_EXT_TX_1DDCT", [4, 13, 8], slice_set(1, 8),
         "`default_intra_ext_tx_cdf[1]` (EXT_TX_SET_DTT4_IDTX_1DDCT, 7-symbol) sliced to 8 slots.")
    emit("DEFAULT_EXT_TX_DTT4", [4, 13, 6], slice_set(2, 6),
         "`default_intra_ext_tx_cdf[2]` (EXT_TX_SET_DTT4_IDTX, 5-symbol) sliced to 6 slots.")

    # --- inter ext-tx: full padded [4][4][17] table, C-faithful (no slicing).
    # intrabc blocks are is_inter, so av1_read_tx_type selects their tx-type
    # CDF from inter_ext_tx_cdf[eset][square_tx_size]; the reader codes each
    # set's exact alphabet, leaving the padding slots untouched. Emitting the
    # raw table keeps the byte-for-byte comparison against the compiled C
    # fc->inter_ext_tx_cdf trivial (see shim_dump_default_inter_ext_tx).
    d, f = extract(mode_c, "default_inter_ext_tx_cdf")
    assert d == [4, 4, 17], d
    emit("DEFAULT_INTER_EXT_TX", d, f,
         "`default_inter_ext_tx_cdf[EXT_TX_SETS_INTER][EXT_TX_SIZES][CDF_SIZE(TX_TYPES)]`, "
         "the full padded table (set 0 is DCT-only / all-zero; sets 1-3 fill their "
         "alphabet's leading slots). Selected by (eset, square tx size) for intrabc tx-type.")

    # --- loop-restoration mode CDFs (single instances) ---
    d, f = extract(mode_c, "default_switchable_restore_cdf")
    assert d == [4], d
    emit("DEFAULT_SWITCHABLE_RESTORE", d, f, "`default_switchable_restore_cdf`.")
    d, f = extract(mode_c, "default_wiener_restore_cdf")
    assert d == [3], d
    emit("DEFAULT_WIENER_RESTORE", d, f, "`default_wiener_restore_cdf`.")
    d, f = extract(mode_c, "default_sgrproj_restore_cdf")
    assert d == [3], d
    emit("DEFAULT_SGRPROJ_RESTORE", d, f, "`default_sgrproj_restore_cdf`.")

    # --- nmv (entropymv.c): struct initializer -> joints + 2 packed comps ---
    m = re.search(r"default_nmv_context\s*=\s*", mv_c)
    if not m:
        sys.exit("default_nmv_context not found")
    tree, _ = parse_braces(mv_c, mv_c.index("{", m.end()))
    joints = flatten(tree[0], [5])
    comps = tree[1]
    assert len(comps) == 2, "expected comps[2]"
    packed = []
    for comp in comps:
        classes, class0_fp, fp, sign, class0_hp, hp, class0, bits = comp
        blob = (
            flatten(sign, [3]) + flatten(classes, [12]) + flatten(class0, [3])
            + flatten(bits, [10, 3]) + flatten(class0_fp, [2, 5]) + flatten(fp, [5])
            + flatten(class0_hp, [3]) + flatten(hp, [3])
        )
        assert len(blob) == 69
        packed += blob
    emit("DEFAULT_NMV_JOINTS", [5], joints, "`default_nmv_context.joints_cdf`.")
    emit("DEFAULT_NMV_COMPS", [2, 69], packed,
         "`default_nmv_context.comps` in aom-entropy's 69-u16 packing (sign/classes/class0/bits/class0_fp/fp/class0_hp/hp).")

    # --- coefficient tables: pack each qindex band into the aom-txb arena ---
    coeff = {}
    for name, dims in [
        ("av1_default_txb_skip_cdfs", [4, 5, 13, 3]),
        ("av1_default_eob_extra_cdfs", [4, 5, 2, 9, 3]),
        ("av1_default_dc_sign_cdfs", [4, 2, 3, 3]),
        ("av1_default_coeff_lps_multi_cdfs", [4, 5, 2, 21, 5]),
        ("av1_default_coeff_base_multi_cdfs", [4, 5, 2, 42, 5]),
        ("av1_default_coeff_base_eob_multi_cdfs", [4, 5, 2, 4, 4]),
        ("av1_default_eob_multi16_cdfs", [4, 2, 2, 6]),
        ("av1_default_eob_multi32_cdfs", [4, 2, 2, 7]),
        ("av1_default_eob_multi64_cdfs", [4, 2, 2, 8]),
        ("av1_default_eob_multi128_cdfs", [4, 2, 2, 9]),
        ("av1_default_eob_multi256_cdfs", [4, 2, 2, 10]),
        ("av1_default_eob_multi512_cdfs", [4, 2, 2, 11]),
        ("av1_default_eob_multi1024_cdfs", [4, 2, 2, 12]),
    ]:
        d, f = extract(tok_h, name)
        assert d == dims, (name, d)
        per = len(f) // 4
        coeff[name] = [f[i * per:(i + 1) * per] for i in range(4)]

    # aom-txb CdfArena regions (write.rs): offset -> (table, per-band length).
    arena_layout = [
        (0, "av1_default_txb_skip_cdfs", 195),
        (195, "av1_default_eob_multi16_cdfs", 24),
        (219, "av1_default_eob_multi32_cdfs", 28),
        (247, "av1_default_eob_multi64_cdfs", 32),
        (279, "av1_default_eob_multi128_cdfs", 36),
        (315, "av1_default_eob_multi256_cdfs", 40),
        (355, "av1_default_eob_multi512_cdfs", 44),
        (399, "av1_default_eob_multi1024_cdfs", 48),
        (447, "av1_default_eob_extra_cdfs", 270),
        (717, "av1_default_coeff_base_eob_multi_cdfs", 160),
        (877, "av1_default_coeff_base_multi_cdfs", 2100),
        (2977, "av1_default_coeff_lps_multi_cdfs", 1050),
        (4027, "av1_default_dc_sign_cdfs", 18),
    ]
    bands = []
    for b in range(4):
        arena = [0] * 4045
        for off, name, ln in arena_layout:
            src = coeff[name][b]
            assert len(src) == ln, (name, len(src), ln)
            arena[off:off + ln] = src
        bands.append(arena)
    emit("DEFAULT_COEFF_ARENA", [4, 4045], [v for b in bands for v in b],
         "The four TOKEN_CDF_Q_CTXS coefficient-CDF bands (qindex <=20 / <=60 / <=120 / >120), each pre-packed into aom-txb's 4045-u16 arena layout.")

    OUT.write_text("\n".join(out))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
