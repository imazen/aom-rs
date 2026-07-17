//! Self-consistency gates for the per-unit LR search + frame-level pick
//! (`restoration_search` / `av1_pick_filter_restoration` port in
//! `aom_restore::pick`). These are NOT C-decision differentials — the
//! decision layer's true oracle is the end-to-end encoder gate (the C
//! search has no narrow export); what IS asserted here:
//!
//! - a perfect reconstruction (src == recon) resolves to RESTORE_NONE on
//!   every plane (the RD can never justify coding params for zero SSE);
//! - on a noisy reconstruction, whenever the search picks a non-NONE frame
//!   type, APPLYING its decision through the (already C-proven) frame walk
//!   strictly reduces the total SSE vs the unfiltered recon — the invariant
//!   the C per-unit RD guarantees (a filter only wins when its SSE beats
//!   NONE: rate is never negative);
//! - the decision is deterministic;
//! - the speed-feature disables (wiener/sgr/chroma/luma) constrain the
//!   outcome exactly;
//! - multi-unit geometry (>256px frames, odd dims) walks without panicking
//!   and covers every unit of the chosen grid.
//!
//! Costs are derived the way the encoder wiring does (`av1_fill_lr_rates`
//! == `cost_tokens_from_cdf` over the frame-init LR CDFs).

use aom_entropy::lr::{
    LrFrameConfig, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER,
};
use aom_entropy::partition::KfFrameContext;
use aom_restore::frame::{loop_restoration_filter_frame, LrPlaneInput};
use aom_restore::pick::{
    pick_filter_restoration, LrPlanePixels, LrSearchInput, LrSearchOutcome, LrSearchSf,
};
use aom_txb::cost_tokens_from_cdf;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i32
    }
}

/// `av1_fill_lr_rates`: the three restore-cost arrays from the frame-init
/// CDFs.
fn lr_costs() -> ([i32; 2], [i32; 2], [i32; 3]) {
    let fc = KfFrameContext::default_for_qindex(60);
    let mut sw = [0i32; 3];
    let mut wn = [0i32; 2];
    let mut sg = [0i32; 2];
    cost_tokens_from_cdf(&mut sw, &fc.switchable_restore, None);
    cost_tokens_from_cdf(&mut wn, &fc.wiener_restore, None);
    cost_tokens_from_cdf(&mut sg, &fc.sgrproj_restore, None);
    (wn, sg, sw)
}

fn walk_plane(rng: &mut Rng, w: usize, h: usize, stride: usize, bd: i32) -> Vec<u16> {
    let maxv = (1i32 << bd) - 1;
    let mut out = vec![0u16; stride * h];
    let mut v = rng.range(0, maxv);
    for r in 0..h {
        for c in 0..w {
            v = (v + rng.range(-8, 8)).clamp(0, maxv);
            out[r * stride + c] = v as u16;
        }
    }
    out
}

fn noisy(rng: &mut Rng, src: &[u16], amp: i32, bd: i32) -> Vec<u16> {
    let maxv = (1i32 << bd) - 1;
    src.iter()
        .map(|&v| (v as i32 + rng.range(-amp, amp)).clamp(0, maxv) as u16)
        .collect()
}

fn plane_sse(a: &[u16], b: &[u16], w: usize, h: usize, stride: usize) -> i64 {
    let mut sse = 0i64;
    for r in 0..h {
        for c in 0..w {
            let e = a[r * stride + c] as i64 - b[r * stride + c] as i64;
            sse += e * e;
        }
    }
    sse
}

#[allow(clippy::too_many_arguments)]
fn build_input<'a>(
    src: &'a [Vec<u16>],
    recon: &'a [Vec<u16>],
    stride: usize,
    w: i32,
    h: i32,
    ss: (usize, usize),
    bd: i32,
    rdmult: i64,
    sf: LrSearchSf,
) -> LrSearchInput<'a> {
    let (wn, sg, sw) = lr_costs();
    let mi_cols = ((w + 7) & !7) >> 2;
    let mi_rows = ((h + 7) & !7) >> 2;
    let sb_cols = (mi_cols + 15) >> 4;
    let sb_rows = (mi_rows + 15) >> 4;
    LrSearchInput {
        planes: (0..src.len())
            .map(|p| LrPlanePixels {
                src: &src[p],
                deblocked: &recon[p],
                cur: &recon[p],
                stride,
            })
            .collect(),
        crop_width: w,
        crop_height: h,
        ss_x: ss.0,
        ss_y: ss.1,
        bit_depth: bd,
        highbd: bd > 8,
        rdmult,
        dc_quant_qtx: 0,
        mib_size_log2: 4,
        mi_rows,
        mi_cols,
        tile_sb_rows: vec![(0, sb_rows)],
        tile_sb_cols: vec![(0, sb_cols)],
        wiener_restore_cost: wn,
        sgrproj_restore_cost: sg,
        switchable_restore_cost: sw,
        sf,
    }
}

/// Apply the outcome through the C-proven frame walk; return per-plane SSE
/// of (src, restored).
fn apply_and_sse(
    outcome: &LrSearchOutcome,
    src: &[Vec<u16>],
    recon: &[Vec<u16>],
    stride: usize,
    w: i32,
    h: i32,
    ss: (usize, usize),
    bd: i32,
) -> Vec<i64> {
    let lr = LrFrameConfig {
        frame_restoration_type: outcome.frame_restoration_type,
        unit_size: [outcome.unit_size; 3],
        crop_width: w,
        crop_height: h,
        superres_denom: 0,
    };
    let mut restored: Vec<Vec<u16>> = recon.to_vec();
    let deblocked = recon.to_vec();
    {
        let mut planes: Vec<LrPlaneInput<'_>> = restored
            .iter_mut()
            .enumerate()
            .map(|(p, cur)| LrPlaneInput {
                cur,
                deblocked: &deblocked[p],
                stride,
                units: &outcome.units[p],
            })
            .collect();
        // Encoder-final apply: optimized_lr = 0 (av1_loop_restoration_filter
        // _frame(cm, 0)).
        loop_restoration_filter_frame(&mut planes, &lr, ss.0, ss.1, bd, false);
    }
    (0..src.len())
        .map(|p| {
            let (sx, sy) = if p > 0 { ss } else { (0, 0) };
            let pw = ((w + (1 << sx) - 1) >> sx) as usize;
            let ph = ((h + (1 << sy) - 1) >> sy) as usize;
            plane_sse(&src[p], &restored[p], pw, ph, stride)
        })
        .collect()
}

#[test]
fn perfect_recon_resolves_to_none() {
    let mut rng = Rng(0xA11_C0DE_0001);
    for &(w, h) in &[(64i32, 64i32), (192, 130)] {
        let stride = w as usize + 16;
        let src: Vec<Vec<u16>> = (0..3)
            .map(|_| walk_plane(&mut rng, w as usize, h as usize, stride, 8))
            .collect();
        let input = build_input(
            &src,
            &src,
            stride,
            w,
            h,
            (1, 1),
            8,
            70000,
            LrSearchSf::default(),
        );
        let outcome = pick_filter_restoration(&input);
        assert_eq!(
            outcome.frame_restoration_type,
            [RESTORE_NONE; 3],
            "{w}x{h}: perfect recon must resolve to NONE"
        );
        assert!(outcome.units.iter().all(|u| u.is_empty()));
    }
}

#[test]
fn noisy_recon_search_improves_sse_and_is_deterministic() {
    let mut rng = Rng(0xA11_C0DE_0002);
    for &bd in &[8i32, 10] {
        // >256 luma => multi-unit grid at every candidate size; odd height.
        let (w, h) = (320i32, 130i32);
        let stride = w as usize + 16;
        let src: Vec<Vec<u16>> = (0..3)
            .map(|_| walk_plane(&mut rng, w as usize, h as usize, stride, bd))
            .collect();
        let amp = if bd == 8 { 9 } else { 24 };
        let recon: Vec<Vec<u16>> = src.iter().map(|p| noisy(&mut rng, p, amp, bd)).collect();
        // rdmult in the ballpark the encoder uses for mid-q (KB-3: 68796 at
        // qindex 128 bd8); scaled up for bd10 like av1_compute_rd_mult.
        let rdmult: i64 = if bd == 8 { 70000 } else { 70000 * 16 };

        let input = build_input(
            &src,
            &recon,
            stride,
            w,
            h,
            (1, 1),
            bd,
            rdmult,
            LrSearchSf::default(),
        );
        let outcome = pick_filter_restoration(&input);
        let outcome2 = pick_filter_restoration(&input);
        assert_eq!(
            outcome.frame_restoration_type,
            outcome2.frame_restoration_type
        );
        assert_eq!(outcome.unit_size, outcome2.unit_size);
        for p in 0..3 {
            assert_eq!(outcome.units[p], outcome2.units[p], "determinism p{p}");
        }

        // Heavy uniform noise on structured content: the luma search must
        // find SOME restoration gain (this is the exact regime LR exists
        // for). If this ever fails the search is broken, not the content.
        assert_ne!(
            outcome.frame_restoration_type[0], RESTORE_NONE,
            "bd{bd}: luma search found no restoration on noisy recon"
        );

        // Applying the decision strictly reduces total SSE (per-unit RD only
        // ever selects a filter whose unit SSE beats NONE).
        let sse_before: Vec<i64> = (0..3)
            .map(|p| {
                let (sx, sy) = if p > 0 { (1usize, 1usize) } else { (0, 0) };
                let pw = ((w + (1 << sx) - 1) >> sx) as usize;
                let ph = ((h + (1 << sy) - 1) >> sy) as usize;
                plane_sse(&src[p], &recon[p], pw, ph, stride)
            })
            .collect();
        let sse_after = apply_and_sse(&outcome, &src, &recon, stride, w, h, (1, 1), bd);
        for p in 0..3 {
            match outcome.frame_restoration_type[p] {
                RESTORE_NONE => assert_eq!(sse_after[p], sse_before[p], "p{p} NONE must be a no-op"),
                _ => assert!(
                    sse_after[p] < sse_before[p],
                    "bd{bd} p{p}: applied restoration must reduce SSE ({} -> {})",
                    sse_before[p],
                    sse_after[p]
                ),
            }
        }

        // Unit-grid coverage: the outcome's unit vectors match the chosen
        // grid exactly.
        let lr = LrFrameConfig {
            frame_restoration_type: outcome.frame_restoration_type,
            unit_size: [outcome.unit_size; 3],
            crop_width: w,
            crop_height: h,
            superres_denom: 0,
        };
        for p in 0..3 {
            if outcome.frame_restoration_type[p] != RESTORE_NONE {
                let (hu, vu) = lr.plane_units(p, 1, 1);
                assert_eq!(outcome.units[p].len(), (hu * vu) as usize, "grid p{p}");
                // In a WIENER/SGRPROJ frame the per-unit types are only that
                // type or NONE; SWITCHABLE units never exceed SGRPROJ.
                for u in &outcome.units[p] {
                    match outcome.frame_restoration_type[p] {
                        RESTORE_WIENER => assert!(
                            u.restoration_type == RESTORE_NONE
                                || u.restoration_type == RESTORE_WIENER
                        ),
                        RESTORE_SGRPROJ => assert!(
                            u.restoration_type == RESTORE_NONE
                                || u.restoration_type == RESTORE_SGRPROJ
                        ),
                        RESTORE_SWITCHABLE => assert!(u.restoration_type <= RESTORE_SGRPROJ),
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
}

#[test]
fn sf_disables_constrain_the_outcome() {
    let mut rng = Rng(0xA11_C0DE_0003);
    let (w, h) = (128i32, 96i32);
    let stride = w as usize + 16;
    let src: Vec<Vec<u16>> = (0..3)
        .map(|_| walk_plane(&mut rng, w as usize, h as usize, stride, 8))
        .collect();
    let recon: Vec<Vec<u16>> = src.iter().map(|p| noisy(&mut rng, p, 9, 8)).collect();

    // Both filters disabled -> all NONE without searching.
    let sf_off = LrSearchSf {
        disable_wiener_filter: true,
        disable_sgr_filter: true,
        ..Default::default()
    };
    let outcome = pick_filter_restoration(&build_input(
        &src, &recon, stride, w, h, (1, 1), 8, 70000, sf_off,
    ));
    assert_eq!(outcome.frame_restoration_type, [RESTORE_NONE; 3]);

    // Wiener disabled -> no WIENER anywhere (frame or unit level).
    let sf_no_wiener = LrSearchSf {
        disable_wiener_filter: true,
        ..Default::default()
    };
    let outcome = pick_filter_restoration(&build_input(
        &src,
        &recon,
        stride,
        w,
        h,
        (1, 1),
        8,
        70000,
        sf_no_wiener,
    ));
    for p in 0..3 {
        assert_ne!(outcome.frame_restoration_type[p], RESTORE_WIENER);
        assert_ne!(outcome.frame_restoration_type[p], RESTORE_SWITCHABLE);
        for u in &outcome.units[p] {
            assert_ne!(u.restoration_type, RESTORE_WIENER);
        }
    }

    // Chroma disabled -> planes 1/2 NONE.
    let sf_no_chroma = LrSearchSf {
        disable_loop_restoration_chroma: true,
        ..Default::default()
    };
    let outcome = pick_filter_restoration(&build_input(
        &src,
        &recon,
        stride,
        w,
        h,
        (1, 1),
        8,
        70000,
        sf_no_chroma,
    ));
    assert_eq!(outcome.frame_restoration_type[1], RESTORE_NONE);
    assert_eq!(outcome.frame_restoration_type[2], RESTORE_NONE);

    // Luma disabled -> plane 0 NONE.
    let sf_no_luma = LrSearchSf {
        disable_loop_restoration_luma: true,
        ..Default::default()
    };
    let outcome = pick_filter_restoration(&build_input(
        &src,
        &recon,
        stride,
        w,
        h,
        (1, 1),
        8,
        70000,
        sf_no_luma,
    ));
    assert_eq!(outcome.frame_restoration_type[0], RESTORE_NONE);
}
