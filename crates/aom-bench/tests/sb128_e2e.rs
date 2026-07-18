//! SB128 (`--sb-size=128`) ENCODE byte-exact gates (PARITY C8).
//!
//! The decoder + entropy layers are SB-size-generic and byte-exact at
//! `--sb-size=128` (the `real_bitstream` SB128 gate). This suite brings the
//! ENCODER search+pack up to the same envelope: the port reads
//! `use_128x128_superblock` from the bootstrap seq header and walks the frame
//! in 128x128 superblocks (32-mi step, BLOCK_128X128 partition root), then
//! byte-matches real `aomenc --sb-size=128` on real-image content >= 128px.
//!
//! Chunking (frugal, smallest byte-exact chunk first):
//!   * MILESTONE A (this landing) — `--sb-size=128 --max-partition-size=64`.
//!     The 128 root is forced to PARTITION_SPLIT (BLOCK_128X128 >
//!     max_partition_size => `av1_set_square_split_only`, partition_search.c
//!     path), so every coded leaf is <= 64x64 (the proven SB64 machinery). The
//!     ONLY new code exercised is the 128-superblock geometry: the 32-mi SB
//!     grid step, the BLOCK_128X128 partition symbol (8-way CDF, no
//!     HORZ_4/VERT_4) + its context group, and the pack walk emitting
//!     PARTITION_SPLIT at the 128 root then recursing. No >64-sized coding
//!     block, so no `av1_write_intra_coeffs_mb` 64-chunk interleave yet.
//!
//! The C reference is driven through the generic-ctrls shim path
//! (`c_encode_ctrls`): `AV1E_SET_SUPERBLOCK_SIZE = AOM_SUPERBLOCK_SIZE_128X128`
//! (+ `AV1E_SET_MAX_PARTITION_SIZE = 64` for milestone A). The port bootstraps
//! `sb_size_128` from that stream's seq header and threads the SB geometry
//! itself; `max_partition_size_px` rides the `ToggleKnobs` (the same C8 knob
//! the toggle gates use).
//!
//! Anti-vacuity: `--sb-size=128` must genuinely change the C stream vs
//! `--sb-size=64` (else a silent SB64 fallback in the port would "pass"
//! vacuously). Asserted per gate.
//!
//! Run: `cargo test -p aom-bench --test sb128_e2e -- --nocapture`

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;
use c::cx_ctrl::{
    AOM_SUPERBLOCK_SIZE_128X128, AV1E_SET_MAX_PARTITION_SIZE, AV1E_SET_SUPERBLOCK_SIZE,
};

/// A 352x288 photographic vector (bd8 4:2:0) — the multi-SB real-content
/// source the KB-6 / toggle gates ride. Crops of it give clean whole-SB128
/// frames (128 = 1 SB, 256 = 2x2 SBs).
const VPHOTO: &str = "av1-1-b8-00-quantizer-00";

/// `(label, crop (w,h,ox,oy), cq)`. 128x128 = a single 128-superblock; 256x256
/// = a 2x2 grid of 128-superblocks (exercises SB-row/col context carry).
const GRID_A: &[(&str, (usize, usize, usize, usize), i32)] = &[
    ("128_cq12", (128, 128, 0, 0), 12),
    ("128_cq32", (128, 128, 0, 0), 32),
    ("128_cq63", (128, 128, 0, 0), 63),
    ("256_cq12", (256, 256, 0, 0), 12),
    ("256_cq32", (256, 256, 0, 0), 32),
    ("256_cq63", (256, 256, 0, 0), 63),
];

/// One forced-SPLIT SB128 cell (milestone A): C encode with
/// `--sb-size=128 --max-partition-size=64`, port encode with the sb128
/// bootstrap + the max-partition-size=64 knob, byte-compare the frame OBU
/// payloads. Returns `(byte_exact, sb128_changed_vs_sb64)`.
fn run_forced_split_cell(
    label: &str,
    crop: (usize, usize, usize, usize),
    cq: i32,
) -> (bool, bool) {
    c::ref_init();
    let cell = EncodeCell::real_content(label, VPHOTO, Some(crop), cq, 0);

    // C reference: --sb-size=128 --max-partition-size=64 (forced SPLIT at the
    // 128 root). max_partition_size overrides only the SEARCH space; the coded
    // partition symbol at the 128 root is still PARTITION_SPLIT.
    let sb128_ctrls = [
        (AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_128X128),
        (AV1E_SET_MAX_PARTITION_SIZE, 64),
    ];
    let c_ref = cell.c_encode_ctrls(&sb128_ctrls);
    assert!(!c_ref.is_empty(), "{label}: C sb128 encode failed");

    // Anti-vacuity: sb128 must change the stream vs sb64 (same max-partition).
    let sb64_ctrls = [(AV1E_SET_MAX_PARTITION_SIZE, 64)];
    let c_sb64 = cell.c_encode_ctrls(&sb64_ctrls);
    let sb128_changed = c_ref != c_sb64;

    // Port: reads sb_size_128 from the bootstrap seq header; max-partition-size
    // rides the C8 knob (mirrors the C control exactly).
    let knobs = ToggleKnobs {
        max_partition_size_px: 64,
        ..Default::default()
    };
    let port_payload = cell.port_encode_with(&c_ref, &knobs);
    let real_payload = EncodeCell::frame_obu_payload(&c_ref);

    let byte_exact = port_payload == real_payload;
    if !byte_exact {
        let n = port_payload.len().min(real_payload.len());
        let first_diff = (0..n).find(|&i| port_payload[i] != real_payload[i]);
        eprintln!(
            "  SB128-A {label}: MISMATCH port {} B vs real {} B, first diff at {:?}",
            port_payload.len(),
            real_payload.len(),
            first_diff
        );
    }
    (byte_exact, sb128_changed)
}

/// MILESTONE A — `--sb-size=128 --max-partition-size=64` (forced SPLIT at the
/// 128 root; every coded leaf <= 64x64). Full byte-identity vs real aomenc on
/// the real-image grid, plus the sb128-vs-sb64 anti-vacuity witness.
#[test]
fn sb128_forced_split_e2e() {
    let mut any_changed = false;
    let mut failures = Vec::new();
    for &(cell_tag, crop, cq) in GRID_A {
        let label = format!("sb128A_{cell_tag}");
        let (byte_exact, changed) = run_forced_split_cell(&label, crop, cq);
        any_changed |= changed;
        if !byte_exact {
            failures.push(label);
        }
    }
    assert!(
        any_changed,
        "--sb-size=128 did not change the C stream vs --sb-size=64 on ANY \
         cell — the byte-exact verdicts would be vacuous (a silent SB64 \
         fallback would pass)"
    );
    assert!(
        failures.is_empty(),
        "SB128 forced-split (milestone A) cells NOT byte-exact vs real \
         aomenc --sb-size=128 --max-partition-size=64: {failures:?}"
    );
}

/// One NATURAL SB128 cell (milestone B): C encode with plain `--sb-size=128`
/// (default max-partition=128, so the RD search evaluates the 128x128 NONE
/// candidate and may code a 128-sized leaf), port encode with the sb128
/// bootstrap + default knobs, byte-compare. Returns `(byte_exact, changed)`.
fn run_natural_cell(label: &str, crop: (usize, usize, usize, usize), cq: i32) -> (bool, bool) {
    c::ref_init();
    let cell = EncodeCell::real_content(label, VPHOTO, Some(crop), cq, 0);

    let sb128_ctrls = [(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_128X128)];
    let c_ref = cell.c_encode_ctrls(&sb128_ctrls);
    assert!(!c_ref.is_empty(), "{label}: C sb128 encode failed");
    let sb128_changed = c_ref != cell.c_encode();

    let port_payload = cell.port_encode_with(&c_ref, &ToggleKnobs::default());
    let real_payload = EncodeCell::frame_obu_payload(&c_ref);

    let byte_exact = port_payload == real_payload;
    if !byte_exact {
        let n = port_payload.len().min(real_payload.len());
        let first_diff = (0..n).find(|&i| port_payload[i] != real_payload[i]);
        eprintln!(
            "  SB128-B {label}: MISMATCH port {} B vs real {} B, first diff at {:?}",
            port_payload.len(),
            real_payload.len(),
            first_diff
        );
    }
    (byte_exact, sb128_changed)
}

/// MILESTONE B (coded 128-leaf arm) — content that actually codes a 128-sized
/// leaf under real `aomenc --sb-size=128`, so the port's
/// `av1_write_intra_coeffs_mb` L/U/V 64-chunk **interleave** in `pack_leaf` +
/// the >64 re-encode (`encode_b_intra_dry`) are exercised, not merely the
/// search's evaluation of the 128 NONE candidate.
///
/// A smooth diagonal ramp (`synthetic_diag`) at 256² cq55/cq63 is the content
/// real aomenc resolves to 128-level partitions (a directional pred fits the
/// ramp with few coeffs, so NONE/HORZ/VERT at the 128 root beats SPLIT on at
/// least one SB — the `quantizer-00` photographic crops split to ≤64 even at
/// cq63, so they do NOT reach this path). Each cell is (1) anti-vacuity-checked
/// that it genuinely codes a 128-level partition (natural `--sb-size=128`
/// DIFFERS from forced-SPLIT `--sb-size=128 --max-partition-size=64` — the cap
/// removes exactly the 128-level non-SPLIT partitions), then (2) byte-matched
/// port vs real aomenc `--sb-size=128`.
#[test]
fn sb128_coded_128_leaf_e2e() {
    c::ref_init();
    let mut failures = Vec::new();
    let mut coded_a_128_leaf = false;
    for cq in [55i32, 63] {
        let label = format!("diag256_cq{cq}");
        let cell = EncodeCell::synthetic_diag(&label, 256, 256, cq, 0);

        let sb128 = [(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_128X128)];
        let c_ref = cell.c_encode_ctrls(&sb128);
        assert!(!c_ref.is_empty(), "{label}: C sb128 encode failed");

        // Anti-vacuity: this cell must actually code a 128-level partition (a
        // >=128 leaf), else the pack interleave is not exercised.
        let forced_split = cell.c_encode_ctrls(&[
            (AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_128X128),
            (AV1E_SET_MAX_PARTITION_SIZE, 64),
        ]);
        let codes_128 = c_ref != forced_split;
        coded_a_128_leaf |= codes_128;

        let port_payload = cell.port_encode_with(&c_ref, &ToggleKnobs::default());
        let real_payload = EncodeCell::frame_obu_payload(&c_ref);
        if port_payload != real_payload {
            let n = port_payload.len().min(real_payload.len());
            let first_diff = (0..n).find(|&i| port_payload[i] != real_payload[i]);
            eprintln!(
                "  SB128-128leaf {label}: codes_128={codes_128} MISMATCH port {} B vs real {} B, first diff at {:?}",
                port_payload.len(),
                real_payload.len(),
                first_diff
            );
            failures.push(label);
        }
    }
    assert!(
        coded_a_128_leaf,
        "neither diag256 cell coded a 128-level partition — the pack coeff \
         interleave / >64 re-encode would be unexercised (vacuous); \
         pick smoother content / higher cq"
    );
    assert!(
        failures.is_empty(),
        "SB128 coded-128-leaf cells NOT byte-exact vs real aomenc \
         --sb-size=128 (the pack L/U/V 64-chunk interleave / >64 re-encode \
         path): {failures:?}"
    );
}

/// MILESTONE B (real-image arm) — plain `--sb-size=128` on real photographic
/// content (`quantizer-00` crops). The RD search EVALUATES the 128x128 NONE /
/// 128x64 / 64x128 candidates at the 128 root (exercising the mu-64 chunk walk
/// in the SEARCH tx walks — reconstructing the 128 block to score it), then
/// resolves to SPLIT / ≤64 leaves (this content is textured enough that the
/// 128-level partitions lose even at cq63 — see `sb128_coded_128_leaf_e2e` for
/// the content that WINS a 128-leaf and exercises the pack coeff interleave).
/// Full byte-identity vs real aomenc on the real-image grid.
#[test]
fn sb128_natural_e2e() {
    let mut any_changed = false;
    let mut failures = Vec::new();
    for &(cell_tag, crop, cq) in GRID_A {
        let label = format!("sb128B_{cell_tag}");
        let (byte_exact, changed) = run_natural_cell(&label, crop, cq);
        any_changed |= changed;
        if !byte_exact {
            failures.push(label);
        }
    }
    assert!(
        any_changed,
        "--sb-size=128 did not change the C stream vs the default --sb-size=64 \
         on ANY cell — the byte-exact verdicts would be vacuous"
    );
    assert!(
        failures.is_empty(),
        "SB128 natural (milestone B) cells NOT byte-exact vs real aomenc \
         --sb-size=128: {failures:?}"
    );
}
