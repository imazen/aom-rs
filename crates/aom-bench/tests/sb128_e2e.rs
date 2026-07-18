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
