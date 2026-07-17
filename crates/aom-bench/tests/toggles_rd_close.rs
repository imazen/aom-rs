//! TOGGLE SWEEP — RD-closeness gates for the C8-C11 CLI-toggle families
//! (PARITY.md): each test encodes the SAME real-content source with the port
//! (knob threaded into the search config) and with real aomenc (same knob via
//! `aom_codec_control`), then compares size + zensim through
//! `aom_bench::rd_close`. Byte-identical cells are recorded EXACT — a toggle
//! that only shrinks the search space deterministically should come out
//! bit-identical if the port's gating matches C's; cells measured EXACT are
//! then HARD-PINNED (`assert!(bit_identical)`) so a later regression cannot
//! hide inside the closeness bands.
//!
//! Anti-vacuity: every non-default knob set must CHANGE the C encoder's
//! output on at least one grid cell (asserted per test). An EXACT verdict on
//! a cell the knob doesn't even reach would prove nothing; the witness
//! guarantees the toggle's disable arm genuinely fires on this content.
//!
//! Run with the per-cell table:
//! `cargo test -p aom-bench --test toggles_rd_close -- --nocapture`
//!
//! No-bootstrap-leak note (PARITY.md rule 4): every knob here is a pure
//! encoder SEARCH gate — none is signalled in the frame header the port
//! bootstraps from the C stream — except where a test says otherwise (those
//! assert the parsed header bit explicitly). The port derives all per-block
//! decisions itself.

use aom_bench::rd_close::{self, RdBands, RdCellResult};
use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

/// The two real-content sources: a 1-SB frame and a multi-SB crop (both
/// landed KB-6 byte-match content).
const V64: &str = "av1-1-b8-01-size-64x64";
const V128: &str = "av1-1-b8-00-quantizer-00";
const CROP128: Option<(usize, usize, usize, usize)> = Some((128, 128, 64, 64));

/// The standard toggle grid: 1-SB mid-q + 1-SB aggressive-q (where large
/// partitions/large tx win by default) + multi-SB low-q (where small
/// partitions dominate) — covers both ends every knob can bite on.
const GRID: [(&str, &str, Option<(usize, usize, usize, usize)>, i32); 3] = [
    ("64_cq32", V64, None, 32),
    ("64_cq63", V64, None, 63),
    ("128_cq12", V128, CROP128, 12),
];

/// One toggle cell: C encode with the knob's ctrl pairs, port encode with
/// the same knobs, rd_close comparison. Returns the result plus whether the
/// toggle CHANGED the C stream vs the default-knobs encode (the per-cell
/// anti-vacuity signal).
fn run_toggle_cell(
    label: &str,
    vector: &str,
    crop: Option<(usize, usize, usize, usize)>,
    cq: i32,
    knobs: &ToggleKnobs,
) -> (RdCellResult, bool) {
    c::ref_init();
    let cell = EncodeCell::real_content(label, vector, crop, cq, 0);
    let ctrls = knobs.c_ctrls();
    let c_tu = cell.c_encode_ctrls(&ctrls);
    assert!(!c_tu.is_empty(), "{label}: C encode failed");
    let knob_changed_c = if ctrls.is_empty() {
        false
    } else {
        cell.c_encode() != c_tu
    };
    let port_payload = cell.port_encode_with(&c_tu, knobs);
    let port_tu = rd_close::splice_frame_obu(&c_tu, &port_payload);
    (
        rd_close::compare_cell(label, &cell, &port_tu, &c_tu),
        knob_changed_c,
    )
}

/// Run one knob set over [`GRID`], gate every cell within the default bands,
/// hard-pin EXACT cells, and (for non-default knobs) assert the toggle
/// genuinely changed the C encoder's output somewhere on the grid.
fn run_grid_and_gate(tag: &str, knobs: &ToggleKnobs, expect_exact: bool) {
    let mut results = Vec::new();
    let mut knob_live = false;
    for (cell_tag, vector, crop, cq) in GRID {
        let (r, changed) = run_toggle_cell(&format!("{tag}_{cell_tag}"), vector, crop, cq, knobs);
        results.push(r);
        knob_live |= changed;
    }
    rd_close::assert_rd_close(&results, &RdBands::default());
    if !knobs.c_ctrls().is_empty() {
        assert!(
            knob_live,
            "{tag}: the toggle did not change the C encoder's output on ANY \
             grid cell — the EXACT verdicts are vacuous; pick content/cq the \
             knob actually reaches"
        );
    }
    if expect_exact {
        for r in &results {
            assert!(
                r.bit_identical,
                "{}: measured EXACT at landing and is now merely close — a \
                 toggle-gating regression is hiding inside the bands",
                r.label
            );
        }
    }
}

/// Harness-faithfulness control: DEFAULT knobs through the toggle path must
/// reproduce the stock proven-byte-exact envelope (c_encode_ctrls(&[]) ==
/// c_encode config; port_encode_with(default) == port_encode).
#[test]
fn toggles_control_default_knobs_exact() {
    let knobs = ToggleKnobs::default();
    assert!(knobs.c_ctrls().is_empty(), "default knobs emit no ctrls");
    run_grid_and_gate("ctl_default", &knobs, true);
}

/// C8 `--enable-rect-partitions=0` (AV1E_SET_ENABLE_RECT_PARTITIONS): kills
/// HORZ/VERT (and transitively AB) partition arms in the search.
#[test]
fn toggles_c8_rect_partitions_off() {
    let knobs = ToggleKnobs {
        enable_rect_partitions: false,
        ..Default::default()
    };
    run_grid_and_gate("c8_rect0", &knobs, true);
}

/// C8 `--enable-ab-partitions=0` (AV1E_SET_ENABLE_AB_PARTITIONS): kills
/// HORZ_A/HORZ_B/VERT_A/VERT_B.
#[test]
fn toggles_c8_ab_partitions_off() {
    let knobs = ToggleKnobs {
        enable_ab_partitions: false,
        ..Default::default()
    };
    run_grid_and_gate("c8_ab0", &knobs, true);
}

/// C8 `--enable-1to4-partitions=0` (AV1E_SET_ENABLE_1TO4_PARTITIONS): kills
/// HORZ_4/VERT_4.
#[test]
fn toggles_c8_1to4_partitions_off() {
    let knobs = ToggleKnobs {
        enable_1to4_partitions: false,
        ..Default::default()
    };
    run_grid_and_gate("c8_1to4_0", &knobs, true);
}

/// C8 `--min-partition-size=16` (AV1E_SET_MIN_PARTITION_SIZE, pixels):
/// raises the leaf floor — no partitions below 16×16.
#[test]
fn toggles_c8_min_partition_16() {
    let knobs = ToggleKnobs {
        min_partition_size_px: 16,
        ..Default::default()
    };
    run_grid_and_gate("c8_min16", &knobs, true);
}

/// C8 `--max-partition-size=32` (AV1E_SET_MAX_PARTITION_SIZE, pixels):
/// lowers the root cap — every 64×64 SB is forced to split. The 64² cq63
/// grid cell is the live witness (aggressive q keeps 64×64 leaves by
/// default; the cq32/cq12 cells split below 32 on their own).
#[test]
fn toggles_c8_max_partition_32() {
    let knobs = ToggleKnobs {
        max_partition_size_px: 32,
        ..Default::default()
    };
    run_grid_and_gate("c8_max32", &knobs, true);
}

/// C8 interaction arm: square-only search (rect + AB + 1to4 all off) with a
/// tightened 8..32 size band — the strongest simultaneous shrink of the
/// partition space.
#[test]
fn toggles_c8_square_only_banded() {
    let knobs = ToggleKnobs {
        enable_rect_partitions: false,
        enable_ab_partitions: false,
        enable_1to4_partitions: false,
        min_partition_size_px: 8,
        max_partition_size_px: 32,
        ..Default::default()
    };
    run_grid_and_gate("c8_sq_band", &knobs, true);
}

// ---------------------------------------------------------------------------
// C10 — intra mode toggles (oxcf.intra_mode_cfg; PARITY.md section C10)
// ---------------------------------------------------------------------------

/// C10 `--enable-smooth-intra=0` (AV1E_SET_ENABLE_SMOOTH_INTRA): kills
/// SMOOTH/SMOOTH_V/SMOOTH_H in both the luma and chroma mode loops. Also
/// makes every neighbour non-SMOOTH, so the per-block intra edge filter
/// type (KB-2/KB-6) is structurally 0 on both sides.
#[test]
fn toggles_c10_smooth_intra_off() {
    let knobs = ToggleKnobs {
        enable_smooth_intra: false,
        ..Default::default()
    };
    run_grid_and_gate("c10_smooth0", &knobs, true);
}

/// C10 `--enable-paeth-intra=0` (AV1E_SET_ENABLE_PAETH_INTRA): kills PAETH
/// in both loops.
#[test]
fn toggles_c10_paeth_intra_off() {
    let knobs = ToggleKnobs {
        enable_paeth_intra: false,
        ..Default::default()
    };
    run_grid_and_gate("c10_paeth0", &knobs, true);
}

/// C10 `--enable-cfl-intra=0` (AV1E_SET_ENABLE_CFL_INTRA): kills UV_CFL_PRED
/// in the chroma loop (the grid content is 4:2:0, so the chroma loop runs).
#[test]
fn toggles_c10_cfl_intra_off() {
    let knobs = ToggleKnobs {
        enable_cfl_intra: false,
        ..Default::default()
    };
    run_grid_and_gate("c10_cfl0", &knobs, true);
}

/// C10 `--enable-diagonal-intra=0` (AV1E_SET_ENABLE_DIAGONAL_INTRA): kills
/// D45..D203 (both loops), keeping V/H + deltas.
#[test]
fn toggles_c10_diagonal_intra_off() {
    let knobs = ToggleKnobs {
        enable_diagonal_intra: false,
        ..Default::default()
    };
    run_grid_and_gate("c10_diag0", &knobs, true);
}

/// C10 `--enable-directional-intra=0` (AV1E_SET_ENABLE_DIRECTIONAL_INTRA):
/// kills EVERY directional mode + all angle deltas (both loops) — the
/// strongest single intra shrink (13 -> 5 luma modes).
#[test]
fn toggles_c10_directional_intra_off() {
    let knobs = ToggleKnobs {
        enable_directional_intra: false,
        ..Default::default()
    };
    run_grid_and_gate("c10_dir0", &knobs, true);
}

/// C10 `--enable-angle-delta=0` (AV1E_SET_ENABLE_ANGLE_DELTA): directional
/// modes search only delta 0 (both loops); the delta-0 symbol is still
/// coded, so this is a pure search shrink.
#[test]
fn toggles_c10_angle_delta_off() {
    let knobs = ToggleKnobs {
        enable_angle_delta: false,
        ..Default::default()
    };
    run_grid_and_gate("c10_adelta0", &knobs, true);
}

/// C10 `--enable-filter-intra=0` (AV1E_SET_ENABLE_FILTER_INTRA): a
/// SEQUENCE-header bit — the filter-intra flag disappears from the mode
/// syntax and the search never evaluates filter-intra candidates.
/// `port_encode_with` asserts the C stream's seq header matches the knob
/// (the port side is knob-driven, not bootstrap-driven).
#[test]
fn toggles_c10_filter_intra_off() {
    let knobs = ToggleKnobs {
        enable_filter_intra: false,
        ..Default::default()
    };
    run_grid_and_gate("c10_fi0", &knobs, true);
}

/// C10 `--enable-intra-edge-filter=0` (AV1E_SET_ENABLE_INTRA_EDGE_FILTER): a
/// SEQUENCE-header bit — directional predictions skip the edge
/// filter/upsample stage in prediction (search + re-encode + decode all
/// follow the seq bit; seq header asserted against the knob).
#[test]
fn toggles_c10_intra_edge_filter_off() {
    let knobs = ToggleKnobs {
        enable_intra_edge_filter: false,
        ..Default::default()
    };
    run_grid_and_gate("c10_ief0", &knobs, true);
}

// ---------------------------------------------------------------------------
// C9 — transform controls (oxcf.txfm_cfg; PARITY.md section C9)
// ---------------------------------------------------------------------------

/// C9 `--enable-tx64=0` (AV1E_SET_ENABLE_TX64): caps the tx-size search at
/// 32 (the `choose_largest`/depth tables demote 64-pt sizes).
#[test]
fn toggles_c9_tx64_off() {
    let knobs = ToggleKnobs {
        enable_tx64: false,
        ..Default::default()
    };
    run_grid_and_gate("c9_tx64_0", &knobs, true);
}

/// C9 `--enable-rect-tx=0` (AV1E_SET_ENABLE_RECT_TX): square-only tx sizes.
#[test]
fn toggles_c9_rect_tx_off() {
    let knobs = ToggleKnobs {
        enable_rect_tx: false,
        ..Default::default()
    };
    run_grid_and_gate("c9_recttx0", &knobs, true);
}

/// C9 `--enable-flip-idtx=0` (AV1E_SET_ENABLE_FLIP_IDTX): masks the
/// FLIPADST/IDTX tx-type family out of every ext-tx set
/// (`DCT_ADST_TX_MASK` in `get_tx_mask`).
#[test]
fn toggles_c9_flip_idtx_off() {
    let knobs = ToggleKnobs {
        enable_flip_idtx: false,
        ..Default::default()
    };
    run_grid_and_gate("c9_flip0", &knobs, true);
}

/// C9 `--use-intra-dct-only=1` (AV1E_SET_INTRA_DCT_ONLY) — PINNED-OPEN
/// characterization (KB-5/KB-10 pattern): the LUMA side is byte-faithful
/// (probe: Y recon identical on the divergent cell), but the CHROMA side
/// diverges from real aomenc in the UV MODE-LOOP layer.
///
/// Measured (2026-07-17): 64_cq32 OUT of band (+2.23% size, zensim drop
/// 3.588 — the port's recon is worse); 64_cq63 EXACT; 128_cq12 CLOSE
/// (−1.40%, drop 0.333). Localization so far (decode-both, kb6 recipe):
/// first divergent leaf mi(0,0) bsize 32×32 — real picks uv D45/aduv2
/// (eob 1) where the port picks uv V/aduv0 (eob 78); real's winners across
/// the frame are derived-type==DCT modes (D45/DC/CFL — the DCT-forced-
/// search signature), the port's V rd (1872917) beats DC (2157931) with
/// D45/CFL gated as never-evaluated. The port's UV txb eval + UV mode loop
/// BOTH match the C-pieces oracle chain under the knob (txfm_uvrd_diff /
/// intra_sbuv_mode_loop_diff sweep it green — the port forces DCT on the
/// chroma mask exactly as `get_tx_mask` reads, verified against the REAL
/// facade incl. the PAETH reduced-set empty-mask reset), so the residual
/// gap is a shared port+oracle mis-model of the REAL UV loop under
/// dct_only — next step is a sibling-C instrumented dump of the (0,0) UV
/// candidate rds (KB-2/KB-7 method).
///
/// This test pins the CURRENT state: it FAILS when the divergent cell
/// starts matching (→ promote to `run_grid_and_gate(expect_exact)`), and
/// FAILS if the two in-band cells regress.
#[test]
fn toggles_c9_intra_dct_only_pinned_open() {
    let knobs = ToggleKnobs {
        use_intra_dct_only: true,
        ..Default::default()
    };
    let mut results = Vec::new();
    let mut knob_live = false;
    for (cell_tag, vector, crop, cq) in GRID {
        let (r, changed) =
            run_toggle_cell(&format!("c9_dct1_{cell_tag}"), vector, crop, cq, &knobs);
        results.push(r);
        knob_live |= changed;
    }
    println!(
        "{}",
        rd_close::render_table(&results, &RdBands::default())
    );
    assert!(knob_live, "c9_dct1: knob must change the C stream");
    // Pinned state (fails on ANY movement, in either direction):
    let r64_32 = &results[0];
    assert!(
        !r64_32.bit_identical && !r64_32.within(&RdBands::default()),
        "c9_dct1_64_cq32 now within bands ({}% / {}) — the UV-loop divergence \
         moved; re-measure and promote this test toward run_grid_and_gate",
        r64_32.size_delta_pct,
        r64_32.zensim_drop
    );
    assert!(
        results[1].bit_identical,
        "c9_dct1_64_cq63 was EXACT at landing and regressed"
    );
    assert!(
        results[2].within(&RdBands::default()),
        "c9_dct1_128_cq12 was within-band at landing and regressed"
    );
}

/// C9 `--use-intra-default-tx-only=1` (AV1E_SET_INTRA_DEFAULT_TX_ONLY): each
/// luma intra txb searches only its mode's default tx type
/// (`get_default_tx_type`; the MODE_EVAL `use_default_intra_tx_type` OR-arm,
/// rdopt_utils.h:579-581).
#[test]
fn toggles_c9_intra_default_tx_only() {
    let knobs = ToggleKnobs {
        use_intra_default_tx_only: true,
        ..Default::default()
    };
    run_grid_and_gate("c9_deftx1", &knobs, true);
}

/// C9/C11 `--reduced-tx-type-set=1` (AV1E_SET_REDUCED_TX_TYPE_SET): a
/// FRAME-header bit (`reduced_tx_set_used`) — the search's ext-tx sets AND
/// the coded tx-type signalling both shrink. `port_encode_with` asserts the
/// bootstrapped frame-header bit equals the knob.
#[test]
fn toggles_c9_reduced_tx_type_set() {
    let knobs = ToggleKnobs {
        reduced_tx_type_set: true,
        ..Default::default()
    };
    run_grid_and_gate("c9_redtx1", &knobs, true);
}

// ---------------------------------------------------------------------------
// C9 tail + C11 — tx-size-search / cdf-update (PARITY.md C9/C11)
// ---------------------------------------------------------------------------

/// C9 `--enable-tx-size-search=0` (AV1E_SET_ENABLE_TX_SIZE_SEARCH): forces
/// `tx_size_search_level = 3` → USE_LARGESTALL at every eval stage
/// (speed_features.c:2726) and the frame codes `tx_mode = TX_MODE_LARGEST`
/// (asserted vs the bootstrap header). C forbids combining with
/// `--enable-tx64=0` (encodeframe.c:2461 assert), so this arm runs alone.
#[test]
fn toggles_c9_tx_size_search_off() {
    let knobs = ToggleKnobs {
        enable_tx_size_search: false,
        ..Default::default()
    };
    run_grid_and_gate("c9_txss0", &knobs, true);
}

/// C11 `--cdf-update-mode=0` (AV1E_SET_CDF_UPDATE_MODE): the KEY header
/// codes `disable_cdf_update = 1` (asserted vs the bootstrap) and the pack
/// runs the entropy coder with symbol adaptation OFF
/// (`PackCfg::allow_update_cdf`, threaded from the parsed header). The
/// decoder-track gate landed 1dfbcc3; this is the ENCODER-side e2e byte
/// gate PARITY.md C11 lists as absent.
#[test]
fn toggles_c11_cdf_update_mode_0() {
    let knobs = ToggleKnobs {
        cdf_update_mode: 0,
        ..Default::default()
    };
    run_grid_and_gate("c11_cdf0", &knobs, true);
}
