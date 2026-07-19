//! WARPED_CAUSAL (chunk 5) **parse census-match** for `av1-1-b8-01-size-16x34`
//! frame 1's single local-warped-motion block.
//!
//! The C instrument census (`AV1D_GET_MI_INFO` on the real libaom decoder) for
//! that block — mi(4,0) `BLOCK_16X16` NEARESTMV `mv=(row -1, col -7)`,
//! `num_proj_ref=3` — is the golden:
//!   wmmat = [-57869, -9917, 65611, 0, 0, 65611], alpha=64 beta=0 gamma=0 delta=64.
//!
//! This test drives the ACTUAL decoder parse functions
//! (`av1_findSamples` -> `av1_selectSamples` -> `av1_find_projection` ports, each
//! already differentially locked vs the real C in `dv_ref_diff.rs` /
//! `warp_diff.rs`) over that block's real neighbour geometry and asserts the
//! derived model == the C census — cross-checked against the real C
//! `av1_find_projection` on the identical samples. The warp KERNEL
//! (`av1_warp_affine`) is locked separately in `aom-inter/tests/warp_diff.rs`.
//! Together these are the byte-exact proof for the WARP feature.
//!
//! The full end-to-end 16x34 frame-1 MD5 gate (golden
//! `0a026e579f57bb108b9fd01bf0af557a`, the `.md5` line 2) is NOT here: that frame
//! also uses **OBMC** blends (chunk 4) AND **inter var-tx / TX_MODE_SELECT** (its
//! blocks, including this BLOCK_16X16 which codes TX_8X8, split their transform
//! quadtree sub-maximally) — both orthogonal to warp and outside this chunk. When
//! both land, add the golden-MD5 cell to `inter_ratchet.rs`.

use aom_entropy::dv_ref::{DvNbr, DvTileBounds, find_samples, select_samples};
use aom_inter::warp::{AFFINE, WarpedMotionParams, find_projection};
use aom_sys_ref::ref_find_projection;

// The C-instrument census golden for mi(4,0).
const CENSUS_WMMAT: [i32; 6] = [-57869, -9917, 65611, 0, 0, 65611];
const CENSUS_ABGD: (i16, i16, i16, i16) = (64, 0, 0, 64);
const BLOCK_16X16: usize = 6;
const BLOCK_4X8: usize = 1;
const BLOCK_8X8: usize = 3;
const NEARESTMV: i32 = 13;

#[test]
fn census_match_16x34_warp_block() {
    // The warp block: mi(4,0), BLOCK_16X16 (4x4 mi), NEARESTMV, mv=(row -1,col -7),
    // single LAST ref, at frame left edge (no left neighbour). Its above row
    // (mi row 3) is covered by three single-LAST inter blocks, all mv=(-1,-7):
    //   cols 0,1 -> mi(2,0)/mi(2,1) BLOCK_4X8 (1 mi wide);
    //   cols 2,3 -> mi(2,2) BLOCK_8X8 (2 mi wide).
    let grid = |row_off: i32, col_off: i32| -> DvNbr {
        if row_off == -1 && (0..=3).contains(&col_off) {
            let bsize = if col_off <= 1 { BLOCK_4X8 } else { BLOCK_8X8 };
            DvNbr {
                bsize,
                ref_frame0: 1, // LAST_FRAME
                ref_frame1: -1,
                use_intrabc: false,
                mode: NEARESTMV,
                mv0_row: -1,
                mv0_col: -7,
                mv1_row: 0,
                mv1_col: 0,
            }
        } else {
            DvNbr::default()
        }
    };

    let tile = DvTileBounds {
        mi_row_start: 0,
        mi_row_end: 9, // 16x34 -> mi_rows = 9
        mi_col_start: 0,
        mi_col_end: 4, // mi_cols = 4
    };

    // av1_findSamples: width_mi = height_mi = 4 (BLOCK_16X16), sb 64x64 -> mib=16.
    let mut samples = find_samples(
        &grid, &tile, 16, /*mi_rows*/ 9, /*mi_cols*/ 4, /*mi_row*/ 4,
        /*mi_col*/ 0, /*width_mi*/ 4, /*height_mi*/ 4, /*partition*/ 0,
        /*ref_frame*/ 1, /*up*/ true, /*left*/ false,
    );
    assert_eq!(
        samples.np, 3,
        "warp block mi(4,0) should gather 3 above samples (census num_proj_ref=3)"
    );
    // The three neighbour centre points (source) + their in-reference projections.
    assert_eq!(&samples.pts[..6], &[8, -40, 40, -40, 88, -40], "pts");
    assert_eq!(
        &samples.pts_inref[..6],
        &[1, -41, 33, -41, 81, -41],
        "pts_inref"
    );

    // av1_selectSamples (num_proj_ref > 1): all three are within threshold here.
    let np = select_samples(
        &mut samples,
        /*mv_row*/ -1,
        /*mv_col*/ -7,
        /*bw*/ 16,
        /*bh*/ 16,
    );
    assert_eq!(np, 3, "all three samples kept by selectSamples");

    // av1_find_projection (port) -> the census model, byte-for-byte.
    let mut wm = WarpedMotionParams {
        wmtype: AFFINE,
        ..Default::default()
    };
    let ret = find_projection(
        samples.np,
        &samples.pts,
        &samples.pts_inref,
        16,
        16,
        -1,
        -7,
        &mut wm,
        4,
        0,
    );
    assert_eq!(ret, 0, "find_projection yields a valid model");
    assert_eq!(wm.wmmat, CENSUS_WMMAT, "port wmmat != C census");
    assert_eq!(
        (wm.alpha, wm.beta, wm.gamma, wm.delta),
        CENSUS_ABGD,
        "port shear params != C census"
    );

    // Cross-check the real C av1_find_projection on the identical samples.
    let c = ref_find_projection(
        samples.np,
        &samples.pts[..samples.np * 2],
        &samples.pts_inref[..samples.np * 2],
        BLOCK_16X16 as i32,
        -1,
        -7,
        4,
        0,
    );
    assert_eq!(c.ret, 0);
    assert_eq!(c.wmmat, CENSUS_WMMAT, "C wmmat != census");
    assert_eq!(
        (c.alpha, c.beta, c.gamma, c.delta),
        CENSUS_ABGD,
        "C shear != census"
    );

    eprintln!(
        "census_match: mi(4,0) warp model = {:?} abgd={:?} — port == C == census",
        wm.wmmat,
        (wm.alpha, wm.beta, wm.gamma, wm.delta)
    );
}
