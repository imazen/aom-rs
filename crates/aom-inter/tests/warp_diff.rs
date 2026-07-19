//! Differential harness for the decoder local-warped-motion core (chunk 5 —
//! `WARPED_CAUSAL`) vs the **real exported C** libaom v3.14.1. Three independent
//! byte/value-identity locks:
//!
//!  1. `warp_affine_matches_c` — the bd8 non-compound affine warp filter
//!     (`aom_inter::warp::warp_affine`) vs the real C `av1_warp_affine_c`
//!     (`aom_sys_ref::ref_warp_affine`, with the decoder's single-ref
//!     `ConvolveParams`), over many valid models × block shapes × subsampling ×
//!     reference planes / positions.
//!  2. `find_projection_matches_c` — the least-squares AFFINE model derivation
//!     (`find_projection`) vs the real C `av1_find_projection`
//!     (`ref_find_projection`): return value + `wmmat[0..6]` + shear params.
//!  3. `get_shear_params_matches_c` — the standalone shear derivation
//!     (`get_shear_params`) vs the real C `av1_get_shear_params`.
//!
//! These lock the entire warp arithmetic surface (model + kernel) against C.
//! The `av1_findSamples` neighbour gather that feeds (2) is a mode-info-grid walk
//! that lives in the decoder driver and is locked by the frame-MD5 / census-match
//! gate there.

use aom_inter::warp::{find_projection, get_shear_params, warp_affine, WarpedMotionParams, AFFINE};
use aom_sys_ref::{ref_find_projection, ref_get_shear_params, ref_warp_affine};

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
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
    fn byte(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
    /// uniform in `lo..=hi`
    fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u32) as i32
    }
}

// (bsize enum value, bw, bh) for the warp-eligible (min dim >= 8) sizes.
const WARP_BSIZES: [(i32, i32, i32); 8] = [
    (3, 8, 8),
    (4, 8, 16),
    (5, 16, 8),
    (6, 16, 16),
    (9, 32, 32),
    (12, 64, 64),
    (18, 8, 32),
    (20, 16, 64),
];

/// The kernel: real decoder warp models fed to both the port `warp_affine` and
/// the real C `av1_warp_affine_c`, byte-identical.
#[test]
fn warp_affine_matches_c() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let iters = 20_000;

    let mut n_valid = 0u32;
    let mut n_real_shear = 0u32; // beta or gamma != 0 (actual shear, not just zoom)
    let mut n_ss = [0u32; 4];

    for _ in 0..iters {
        // Reference plane.
        let width = *[24usize, 32, 48, 64].get(rng.below(4) as usize).unwrap();
        let height = *[24usize, 32, 48, 64].get(rng.below(4) as usize).unwrap();
        let pad = rng.below(3) as usize; // stride padding 0..2
        let stride = width + pad;
        let mut refp: Vec<u16> = vec![0u16; stride * height];
        for v in refp.iter_mut() {
            *v = rng.byte() as u16;
        }
        let refp_u8: Vec<u8> = refp.iter().map(|&v| v as u8).collect();

        // Block shape (include non-multiple-of-8 to exercise the edge crop).
        let p_width = *[8usize, 12, 16, 32].get(rng.below(4) as usize).unwrap();
        let p_height = *[8usize, 12, 16, 32].get(rng.below(4) as usize).unwrap();
        let (ss_x, ss_y) = match rng.below(4) {
            0 => (0usize, 0usize),
            1 => (1, 1),
            2 => (1, 0),
            _ => (0, 1),
        };
        let p_col = rng.range_i32(0, (width as i32 - 1).max(0));
        let p_row = rng.range_i32(0, (height as i32 - 1).max(0));

        // A near-identity AFFINE model + random translation.
        let mut wm = WarpedMotionParams {
            wmtype: AFFINE,
            ..Default::default()
        };
        wm.wmmat[2] = 65536 + rng.range_i32(-3000, 3000);
        wm.wmmat[5] = 65536 + rng.range_i32(-3000, 3000);
        wm.wmmat[3] = rng.range_i32(-1500, 1500);
        wm.wmmat[4] = rng.range_i32(-1500, 1500);
        wm.wmmat[0] = rng.range_i32(-(1 << 17), 1 << 17);
        wm.wmmat[1] = rng.range_i32(-(1 << 17), 1 << 17);

        // Derive the shear via the REAL C (guaranteed correct); only a model the
        // fast warp filter accepts (`ret == 1`) is ever handed to av1_warp_plane.
        let m = ref_get_shear_params(&wm.wmmat);
        if m.ret != 1 {
            continue;
        }
        n_valid += 1;
        if m.beta != 0 || m.gamma != 0 {
            n_real_shear += 1;
        }
        n_ss[ss_y * 2 + ss_x] += 1;

        // Port kernel.
        let mut port = vec![0u16; p_width * p_height];
        warp_affine(
            &wm.wmmat, &refp, width, height, stride, &mut port, 0, p_width, p_col, p_row, p_width,
            p_height, ss_x, ss_y, m.alpha, m.beta, m.gamma, m.delta,
        );

        // Real C kernel.
        let c = ref_warp_affine(
            &wm.wmmat, &refp_u8, width, height, stride, p_col, p_row, p_width, p_height, ss_x,
            ss_y, m.alpha, m.beta, m.gamma, m.delta,
        );

        for i in 0..p_width * p_height {
            assert_eq!(
                port[i] as u8,
                c[i],
                "warp_affine px {i} mismatch: port {} vs C {} \
                 (wmmat={:?} abgd=({},{},{},{}) p=({},{}) sz={}x{} ss=({},{}) ref={}x{})",
                port[i],
                c[i],
                wm.wmmat,
                m.alpha,
                m.beta,
                m.gamma,
                m.delta,
                p_col,
                p_row,
                p_width,
                p_height,
                ss_x,
                ss_y,
                width,
                height,
            );
        }
    }

    // Anti-vacuity: enough valid models, real shear exercised, all subsamplings hit.
    assert!(
        n_valid > 5_000,
        "too few valid warp models tested: {n_valid}"
    );
    assert!(
        n_real_shear > 500,
        "shear (beta/gamma != 0) barely exercised: {n_real_shear}"
    );
    for (i, &c) in n_ss.iter().enumerate() {
        assert!(c > 200, "subsampling case {i} under-exercised: {c}");
    }
}

/// The AFFINE model derivation from least-squares sample points.
#[test]
fn find_projection_matches_c() {
    let mut rng = Rng(0xdead_beef_0bad_f00d);
    let iters = 40_000;

    let mut n_valid = 0u32; // ret == 0
    let mut n_invalid = 0u32; // ret == 1

    for _ in 0..iters {
        let (bsize, bw, bh) = WARP_BSIZES[rng.below(WARP_BSIZES.len() as u32) as usize];
        let np = rng.range_i32(1, 8) as usize;
        let mvy = rng.range_i32(-256, 256);
        let mvx = rng.range_i32(-256, 256);
        let mi_row = rng.range_i32(0, 63);
        let mi_col = rng.range_i32(0, 63);

        // Sample points: source near a block-center-ish spread (1/8-pel), the
        // in-reference points = source + mv + noise (mirrors record_samples +
        // the affine spread). Small noise -> many valid models; large -> many
        // degenerate/invalid, so both branches are covered.
        let spread = *[8i32, 64, 256, 1024].get(rng.below(4) as usize).unwrap();
        let mut pts1 = vec![0i32; np * 2];
        let mut pts2 = vec![0i32; np * 2];
        for i in 0..np {
            let sx = rng.range_i32(-spread, spread);
            let sy = rng.range_i32(-spread, spread);
            pts1[i * 2] = sx;
            pts1[i * 2 + 1] = sy;
            pts2[i * 2] = sx + mvx + rng.range_i32(-spread, spread);
            pts2[i * 2 + 1] = sy + mvy + rng.range_i32(-spread, spread);
        }

        let mut wm = WarpedMotionParams {
            wmtype: AFFINE,
            ..Default::default()
        };
        let ret = find_projection(np, &pts1, &pts2, bw, bh, mvy, mvx, &mut wm, mi_row, mi_col);
        let c = ref_find_projection(np, &pts1, &pts2, bsize, mvy, mvx, mi_row, mi_col);

        assert_eq!(
            ret, c.ret,
            "find_projection ret mismatch: port {ret} vs C {} (np={np} bsize={bsize} \
             mv=({mvy},{mvx}) mi=({mi_row},{mi_col}) pts1={pts1:?} pts2={pts2:?})",
            c.ret
        );
        if ret == 0 {
            n_valid += 1;
            assert_eq!(
                wm.wmmat, c.wmmat,
                "wmmat mismatch (np={np} bsize={bsize} mv=({mvy},{mvx}) mi=({mi_row},{mi_col}) \
                 pts1={pts1:?} pts2={pts2:?})"
            );
            assert_eq!(
                (wm.alpha, wm.beta, wm.gamma, wm.delta),
                (c.alpha, c.beta, c.gamma, c.delta),
                "shear mismatch (np={np} bsize={bsize} mv=({mvy},{mvx}))"
            );
        } else {
            n_invalid += 1;
        }
    }

    assert!(n_valid > 1_000, "too few valid models: {n_valid}");
    assert!(n_invalid > 1_000, "too few invalid models: {n_invalid}");
}

/// The standalone shear derivation from a model.
#[test]
fn get_shear_params_matches_c() {
    let mut rng = Rng(0x00c0_ffee_1337_babe);
    let iters = 40_000;

    let mut n_ok = 0u32; // C ret == 1 (usable)
    let mut n_bad = 0u32; // C ret == 0

    for _ in 0..iters {
        let mut wmmat = [0i32; 6];
        // Bias toward near-identity so a good fraction pass the shear check, but
        // widen occasionally to exercise the invalid / clamp / mat[2]<=0 paths.
        let wide = rng.below(4) == 0;
        let d = if wide { 1 << 17 } else { 6000 };
        wmmat[0] = rng.range_i32(-(1 << 20), 1 << 20);
        wmmat[1] = rng.range_i32(-(1 << 20), 1 << 20);
        wmmat[2] = if wide {
            rng.range_i32(-(1 << 17), 1 << 17)
        } else {
            65536 + rng.range_i32(-d, d)
        };
        wmmat[3] = rng.range_i32(-d, d);
        wmmat[4] = rng.range_i32(-d, d);
        wmmat[5] = 65536 + rng.range_i32(-d, d);

        let mut wm = WarpedMotionParams {
            wmmat,
            wmtype: AFFINE,
            ..Default::default()
        };
        let ok = get_shear_params(&mut wm);
        let c = ref_get_shear_params(&wmmat);

        assert_eq!(
            ok as i32, c.ret,
            "get_shear_params ret mismatch: port {ok} vs C {} (wmmat={wmmat:?})",
            c.ret
        );
        // C sets alpha/beta unconditionally when mat[2] > 0; on the mat[2] <= 0
        // early-out neither side computes them, so only compare when valid.
        if c.ret == 1 {
            n_ok += 1;
            assert_eq!(
                (wm.alpha, wm.beta, wm.gamma, wm.delta),
                (c.alpha, c.beta, c.gamma, c.delta),
                "shear mismatch (wmmat={wmmat:?})"
            );
        } else {
            n_bad += 1;
        }
    }

    assert!(n_ok > 1_000, "too few usable models: {n_ok}");
    assert!(n_bad > 1_000, "too few rejected models: {n_bad}");
}
