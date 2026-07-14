//! Loop-restoration RU-params syntax vs C: random unit-parameter sequences
//! are written by the REAL C arithmetic writer (`ref_lr_units_roundtrip`,
//! transcribed encoder control flow over EXPORTED
//! `aom_write_primitive_refsubexpfin` / `aom_write_symbol` on the REAL
//! default LR CDFs), then decoded by `aom_entropy::lr::read_lr_unit` from the
//! produced bitstream. Asserts, per unit: type + parameters equal both the
//! written intent AND the C reader's read-back, and the final adapted CDFs
//! (switchable/wiener/sgrproj) match the C reader's byte-for-byte.

use aom_entropy::dec::OdEcDec;
use aom_entropy::lr::{
    self, LrRefState, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER,
    SGRPROJ_PRJ_MAX0, SGRPROJ_PRJ_MAX1, SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MIN1, SGR_PARAMS_R,
};
use aom_entropy::partition::KfFrameContext;
use aom_sys_ref as c;

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
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u64) as i32
    }
}

/// Generate one encoder-legal unit intent in the shim's LRU packing.
fn gen_unit(rng: &mut Rng, plane: usize, frame_rtype: u8) -> [i32; c::LRU_WORDS] {
    let mut u = [0i32; c::LRU_WORDS];
    u[0] = plane as i32;
    u[1] = frame_rtype as i32;
    let rtype = match frame_rtype {
        RESTORE_SWITCHABLE => rng.below(3) as u8, // NONE / WIENER / SGRPROJ
        t => {
            if rng.below(3) == 0 {
                RESTORE_NONE
            } else {
                t
            }
        }
    };
    u[2] = rtype as i32;
    match rtype {
        RESTORE_WIENER => {
            // Taps within the coded ranges; chroma windows zero tap 0.
            for d in 0..2 {
                u[3 + 3 * d] = if plane > 0 { 0 } else { rng.range(-5, 10) };
                u[4 + 3 * d] = rng.range(-23, 8);
                u[5 + 3 * d] = rng.range(-17, 46);
            }
        }
        RESTORE_SGRPROJ => {
            let ep = rng.below(16) as i32;
            u[9] = ep;
            let r = SGR_PARAMS_R[ep as usize];
            // Encoder-legal xqd per the parameter set's radii; the r[1]==0
            // case's xqd[1] is DERIVED by the reader (not coded), so the
            // intent must carry the derived value for the roundtrip compare.
            if r[0] == 0 {
                u[10] = 0;
                u[11] = rng.range(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
            } else if r[1] == 0 {
                u[10] = rng.range(SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0);
                u[11] = (128 - u[10]).clamp(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
            } else {
                u[10] = rng.range(SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0);
                u[11] = rng.range(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
            }
        }
        _ => {}
    }
    u
}

#[test]
fn lr_unit_params_read_matches_c() {
    let mut rng = Rng(0x1EE7_5EED_0BAD_F00D);
    for case in 0..400 {
        // Per-plane frame restoration types for this case; at least one
        // non-NONE (a NONE plane codes no units).
        let mut frame_rtype = [RESTORE_NONE; 3];
        loop {
            for t in frame_rtype.iter_mut() {
                *t = rng.below(4) as u8;
            }
            if frame_rtype.iter().any(|&t| t != RESTORE_NONE) {
                break;
            }
        }
        let n_units = 1 + rng.below(40) as usize;
        let mut units = Vec::with_capacity(n_units * c::LRU_WORDS);
        for _ in 0..n_units {
            // Pick any plane whose frame type codes units.
            let plane = loop {
                let p = rng.below(3) as usize;
                if frame_rtype[p] != RESTORE_NONE {
                    break p;
                }
            };
            units.extend_from_slice(&gen_unit(&mut rng, plane, frame_rtype[plane]));
        }

        let (stream, readback, c_cdfs) = c::ref_lr_units_roundtrip(&units);

        // Rust decode over the same stream with the default LR CDFs.
        let fc = KfFrameContext::default_for_qindex(60);
        let mut sw = fc.switchable_restore;
        let mut wn = fc.wiener_restore;
        let mut sg = fc.sgrproj_restore;
        let mut refs = LrRefState::default();
        let mut dec = OdEcDec::new(&stream);
        for i in 0..n_units {
            let u = &units[i * c::LRU_WORDS..(i + 1) * c::LRU_WORDS];
            let rb = &readback[i * c::LRU_WORDS..(i + 1) * c::LRU_WORDS];
            let plane = u[0] as usize;
            let got = lr::read_lr_unit(
                &mut dec, u[1] as u8, plane, &mut refs, &mut sw, &mut wn, &mut sg,
            );
            assert_eq!(
                got.restoration_type, u[2] as u8,
                "case {case} unit {i}: type intent"
            );
            assert_eq!(
                got.restoration_type, rb[2] as u8,
                "case {case} unit {i}: type C-readback"
            );
            match got.restoration_type {
                RESTORE_WIENER => {
                    let w = &got.wiener;
                    let taps = [
                        w.vfilter[0] as i32,
                        w.vfilter[1] as i32,
                        w.vfilter[2] as i32,
                        w.hfilter[0] as i32,
                        w.hfilter[1] as i32,
                        w.hfilter[2] as i32,
                    ];
                    assert_eq!(taps, u[3..9], "case {case} unit {i}: wiener intent");
                    assert_eq!(taps, rb[3..9], "case {case} unit {i}: wiener C-readback");
                    // Derived slots: centre + mirror + zero slot 7.
                    assert_eq!(
                        w.vfilter[3],
                        -2 * (w.vfilter[0] + w.vfilter[1] + w.vfilter[2])
                    );
                    assert_eq!(
                        [w.vfilter[4], w.vfilter[5], w.vfilter[6], w.vfilter[7]],
                        [w.vfilter[2], w.vfilter[1], w.vfilter[0], 0]
                    );
                }
                RESTORE_SGRPROJ => {
                    let s = &got.sgrproj;
                    assert_eq!(
                        [s.ep, s.xqd[0], s.xqd[1]],
                        [u[9], u[10], u[11]],
                        "case {case} unit {i}: sgrproj intent"
                    );
                    assert_eq!(
                        [s.ep, s.xqd[0], s.xqd[1]],
                        [rb[9], rb[10], rb[11]],
                        "case {case} unit {i}: sgrproj C-readback"
                    );
                }
                _ => assert_eq!(got.restoration_type, RESTORE_NONE),
            }
        }
        // Adapted CDFs must match the C reader's final state exactly.
        assert_eq!(sw.as_slice(), &c_cdfs[0..4], "case {case}: switchable CDF");
        assert_eq!(wn.as_slice(), &c_cdfs[4..7], "case {case}: wiener CDF");
        assert_eq!(sg.as_slice(), &c_cdfs[7..10], "case {case}: sgrproj CDF");
    }
}

/// The default LR reference state matches C's `set_default_wiener` /
/// `set_default_sgrproj` (spot values traced from restoration.h).
#[test]
fn lr_default_refs_match_c_formulas() {
    let r = LrRefState::default();
    for p in 0..3 {
        assert_eq!(r.wiener[p].vfilter, [3, -7, 15, -22, 15, -7, 3, 0]);
        assert_eq!(r.wiener[p].hfilter, [3, -7, 15, -22, 15, -7, 3, 0]);
        assert_eq!(r.sgrproj[p].ep, 0);
        assert_eq!(r.sgrproj[p].xqd, [(-96 + 31) / 2, (-32 + 95) / 2]);
    }
}

/// `lr_corners_in_sb` + unit-grid geometry vs the REAL C
/// (`av1_alloc_restoration_struct` counts + `av1_loop_restoration_corners_in_sb`),
/// over frame dims incl. non-multiples of 64, all subsamplings, all unit
/// sizes, every 64x64 SB of each frame. Also asserts full coverage: over all
/// SBs the hit rectangles partition the unit grid exactly (every RU's params
/// are coded exactly once).
#[test]
fn lr_corners_and_unit_grid_match_c() {
    const BLOCK_64X64: usize = 12;
    for &(w, h) in &[
        (64, 64),
        (65, 65),
        (128, 96),
        (176, 144),
        (320, 240),
        (127, 250),
        (4, 4),
        (36, 20),
        (256, 256),
        (448, 232),
    ] {
        for &(ss_x, ss_y) in &[(0usize, 0usize), (1, 0), (1, 1)] {
            for &ys in &[64i32, 128, 256] {
                // Chroma size: same, or halved when both axes subsampled
                // (the only coded choices; 64 luma can't halve — the C
                // reader derives chroma from the coded luma size).
                let chroma_choices: &[i32] = if ss_x.min(ss_y) == 1 && ys > 64 {
                    &[0, 1]
                } else {
                    &[0]
                };
                for &half in chroma_choices {
                    let cs = ys >> half;
                    let unit_size = [ys, cs, cs];
                    let lr = aom_entropy::lr::LrFrameConfig {
                        frame_restoration_type: [RESTORE_WIENER; 3],
                        unit_size,
                        crop_width: w,
                        crop_height: h,
                    };
                    let mi_rows = ((h + 7) & !7) >> 2;
                    let mi_cols = ((w + 7) & !7) >> 2;
                    for plane in 0..3 {
                        let (hu, vu) = lr.plane_units(plane, ss_x, ss_y);
                        let mut covered = vec![0u32; (hu * vu) as usize];
                        let mut mi_row = 0;
                        while mi_row < mi_rows {
                            let mut mi_col = 0;
                            while mi_col < mi_cols {
                                let (c_hit, c_hu, c_vu, c_r) = c::ref_lr_corners_in_sb(
                                    w,
                                    h,
                                    ss_x as i32,
                                    ss_y as i32,
                                    unit_size,
                                    plane,
                                    mi_row,
                                    mi_col,
                                    BLOCK_64X64,
                                );
                                assert_eq!(
                                    (hu, vu),
                                    (c_hu, c_vu),
                                    "unit grid {w}x{h} ss({ss_x},{ss_y}) us{unit_size:?} p{plane}"
                                );
                                let r = aom_entropy::lr::lr_corners_in_sb(
                                    &lr, plane, ss_x, ss_y, mi_row, mi_col, 16, 16,
                                );
                                match r {
                                    Some((rc0, rc1, rr0, rr1)) => {
                                        assert!(c_hit, "rust hit, C miss @({mi_row},{mi_col}) p{plane} {w}x{h}");
                                        assert_eq!([rc0, rc1, rr0, rr1], c_r, "corners @({mi_row},{mi_col}) p{plane} {w}x{h} ss({ss_x},{ss_y}) us{unit_size:?}");
                                        for rr in rr0..rr1 {
                                            for rc in rc0..rc1 {
                                                covered[(rr * hu + rc) as usize] += 1;
                                            }
                                        }
                                    }
                                    None => assert!(
                                        !c_hit,
                                        "C hit, rust miss @({mi_row},{mi_col}) p{plane} {w}x{h}"
                                    ),
                                }
                                mi_col += 16;
                            }
                            mi_row += 16;
                        }
                        assert!(
                            covered.iter().all(|&c| c == 1),
                            "unit coverage not exactly-once: {w}x{h} ss({ss_x},{ss_y}) us{unit_size:?} p{plane} {covered:?}"
                        );
                    }
                }
            }
        }
    }
}
