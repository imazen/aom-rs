//! Differential harnesses for the chroma-intra-RD building blocks:
//! - `av1_get_tx_type` PLANE_TYPE_UV intra arm (`uv_intra_tx_type`) vs the
//!   REAL `av1_get_tx_type` over a marshalled MACROBLOCKD;
//! - `get_tx_mask` chroma arm (`get_tx_mask_uv_intra`) vs the C transcription
//!   over the REAL av1_get_tx_type + real blockd.h tables;
//! - the CfL cost slice of `av1_fill_mode_rates` + `palette_uv_mode_cost` vs
//!   the REAL `av1_cost_tokens_from_cdf` fills;
//! - `intra_mode_info_cost_uv` vs the transcription over real header gates;
//! - the CfL store/predict pipeline (aom-intra `CflCtx`) vs the REAL EXPORTED
//!   `cfl_store_tx` + `av1_cfl_predict_block` (the first direct C diff of the
//!   CfL kernels — the decoder track validated them by hand-traced vectors +
//!   roundtrip only, with the shared-misread risk documented).

use aom_encode::mode_costs::{
    CflCosts, IntraModeCosts, fill_cfl_costs, fill_palette_uv_mode_costs, intra_mode_info_cost_uv,
};
use aom_encode::tx_search::{TxMaskParams, get_tx_mask_uv_intra, uv_intra_tx_type};
use aom_intra::cfl::{CflCtx, cfl_predict_block, cfl_store_tx};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

fn cdf_row(rng: &mut Rng, nsymbs: usize, padded: usize) -> Vec<u16> {
    let mut row = vec![0u16; padded];
    let mut acc: u32 = 0;
    for e in row.iter_mut().take(nsymbs - 1) {
        acc += rng.range(1, (32000 / nsymbs as i32).max(2)) as u32;
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    row[nsymbs - 1] = 0;
    row
}

#[test]
fn uv_intra_tx_type_matches_real_c() {
    // Exhaustive: 14 uv modes x 19 tx sizes x lossless x reduced, vs the REAL
    // av1_get_tx_type (PLANE_TYPE_UV, intra).
    let mut non_dct = 0usize;
    let mut demoted = 0usize;
    for uv_mode in 0..14usize {
        for tx_size in 0..19usize {
            for cfg in 0..4usize {
                let lossless = cfg & 1 != 0;
                let reduced = cfg & 2 != 0;
                let t = uv_intra_tx_type(uv_mode, lossless, tx_size, reduced);
                let t_c = c::ref_get_tx_type_uv_intra(uv_mode, lossless, tx_size, reduced);
                assert_eq!(t, t_c, "uv_mode={uv_mode} ts={tx_size} cfg={cfg}");
                if t != 0 {
                    non_dct += 1;
                }
                // Demotion coverage: default type nonzero but result DCT.
                let mode = aom_entropy::partition::get_uv_mode(uv_mode) as usize;
                if !lossless && aom_encode::tx_search::INTRA_MODE_TO_TX_TYPE[mode] != 0 && t == 0 {
                    demoted += 1;
                }
            }
        }
    }
    assert!(
        non_dct > 150,
        "non-DCT uv tx types under-exercised: {non_dct}"
    );
    assert!(
        demoted > 100,
        "ext-tx-set demotion under-exercised: {demoted}"
    );
}

#[test]
fn tx_mask_uv_intra_matches_c() {
    // Exhaustive: 19 tx sizes x 14 uv modes x 13 luma modes x fi(off + 5) x
    // {lossless, reduced, sf 0/1/2, flip_idtx} — every branch of the chroma
    // arm, incl. the empty-mask reset that KEEPS uv_tx_type.
    let mut reset_hits = 0usize;
    let mut non_dct = 0usize;
    for tx_size in 0..19usize {
        for uv_mode in 0..14usize {
            for luma_mode in 0..13usize {
                for fi in [5usize, 0, 2, 4] {
                    let (use_fi, fi_mode) = if fi == 5 { (false, 0) } else { (true, fi) };
                    for cfg in 0..24usize {
                        let lossless = cfg & 1 != 0;
                        let reduced = cfg & 2 != 0;
                        let use_reduced_txset = (cfg >> 2) % 3;
                        let flip_idtx = cfg & 16 == 0;
                        let p = TxMaskParams {
                            use_reduced_intra_txset: use_reduced_txset as u8,
                            use_derived_intra_tx_type_set: false,
                            enable_flip_idtx: flip_idtx,
                            use_intra_dct_only: false,
                        };
                        let (mask, txk) = get_tx_mask_uv_intra(
                            tx_size, uv_mode, luma_mode, use_fi, fi_mode, lossless, reduced, &p,
                        );
                        let (mask_c, txk_c) = c::ref_get_tx_mask_uv_intra(
                            tx_size,
                            uv_mode,
                            luma_mode,
                            use_fi,
                            fi_mode,
                            lossless,
                            reduced,
                            use_reduced_txset as u8,
                            flip_idtx,
                            false,
                        );
                        assert_eq!(
                            (mask, txk),
                            (mask_c, txk_c),
                            "ts={tx_size} uv={uv_mode} y={luma_mode} fi={use_fi}/{fi_mode} cfg={cfg}",
                        );
                        // The chroma reset: uv type outside the (reduced) set.
                        let uv_t = uv_intra_tx_type(uv_mode, lossless, tx_size, reduced);
                        if txk == uv_t && uv_t != 0 && use_reduced_txset == 1 {
                            non_dct += 1;
                        }
                        if mask == 1 << txk && uv_t != 0 && txk == uv_t {
                            reset_hits += 1; // single-type by construction
                        }
                    }
                }
            }
        }
    }
    assert!(
        non_dct > 1000,
        "reduced-set non-DCT under-exercised: {non_dct}"
    );
    assert!(reset_hits > 1000);
}

#[test]
fn cfl_and_palette_uv_costs_match_c() {
    let mut rng = Rng(0xcf1c_0575_0000_0001);
    for it in 0..200 {
        let sign_cdf = cdf_row(&mut rng, 8, 9);
        let mut alpha_cdf = Vec::new();
        for _ in 0..6 {
            alpha_cdf.extend(cdf_row(&mut rng, 16, 17));
        }
        let mut cfl = CflCosts::zeroed();
        fill_cfl_costs(&mut cfl, &sign_cdf, &alpha_cdf);
        let c_out = c::ref_fill_cfl_costs(&sign_cdf, &alpha_cdf);
        for js in 0..8 {
            for pl in 0..2 {
                for a in 0..16 {
                    assert_eq!(
                        cfl.0[js][pl][a],
                        c_out[(js * 2 + pl) * 16 + a],
                        "it={it} js={js} pl={pl} a={a}",
                    );
                }
            }
        }

        let pal_cdf: Vec<u16> = (0..2).flat_map(|_| cdf_row(&mut rng, 2, 3)).collect();
        let mut costs = IntraModeCosts::zeroed();
        fill_palette_uv_mode_costs(&mut costs, &pal_cdf);
        let c_pal = c::ref_fill_palette_uv_mode_costs(&pal_cdf);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    costs.palette_uv_mode_cost[i][j],
                    c_pal[i * 2 + j],
                    "it={it}"
                );
            }
        }
    }
}

#[test]
fn intra_mode_info_cost_uv_matches_c() {
    let mut rng = Rng(0x0057_0cf1_0000_0002);
    let mut angle_hits = 0usize;
    let mut palette_hits = 0usize;
    for it in 0..4000 {
        let mut costs = IntraModeCosts::zeroed();
        for row in costs.angle_delta_cost.iter_mut() {
            for e in row.iter_mut() {
                *e = rng.range(0, 20 << 9);
            }
        }
        for row in costs.palette_uv_mode_cost.iter_mut() {
            for e in row.iter_mut() {
                *e = rng.range(0, 20 << 9);
            }
        }
        let angle_flat: Vec<i32> = costs.angle_delta_cost.iter().flatten().copied().collect();
        let pal_flat: Vec<i32> = costs
            .palette_uv_mode_cost
            .iter()
            .flatten()
            .copied()
            .collect();

        let uv_mode = (rng.next() % 14) as usize;
        let bsize = (rng.next() % 22) as usize;
        let angle_delta_uv = rng.range(-3, 4);
        let mode_cost = rng.range(0, 20 << 9);
        let try_palette = rng.next() & 1 != 0;
        let y_palette_active = rng.next() & 1 != 0;

        let r = intra_mode_info_cost_uv(
            &costs,
            mode_cost,
            uv_mode,
            bsize,
            angle_delta_uv,
            try_palette,
            y_palette_active,
        );
        let r_c = c::ref_intra_mode_info_cost_uv(
            &angle_flat,
            &pal_flat,
            mode_cost,
            uv_mode,
            bsize,
            angle_delta_uv,
            try_palette,
            y_palette_active,
        );
        assert_eq!(
            r, r_c,
            "it={it} uv={uv_mode} bsize={bsize} ad={angle_delta_uv}"
        );
        let im = aom_entropy::partition::get_uv_mode(uv_mode);
        if (1..=8).contains(&im) && r != mode_cost {
            angle_hits += 1;
        }
        if try_palette && uv_mode == 0 {
            palette_hits += 1;
        }
    }
    assert!(angle_hits > 400, "angle arm under-exercised: {angle_hits}");
    assert!(
        palette_hits > 100,
        "palette flag arm under-exercised: {palette_hits}"
    );
}

/// The CfL store/predict pipeline vs the REAL EXPORTED C functions:
/// random luma recon -> `cfl_store_tx` both sides (recon_buf_q3 + surface
/// tracking equal) -> DC-filled dst -> `cfl_predict_block` both sides (padded
/// AC + alpha-scaled prediction equal), across 420/422/444, bd 8/10/12, all
/// CfL-legal tx sizes, all joint signs, and the sub-8x8 shared-chroma offset
/// adjustment.
#[test]
fn cfl_store_predict_matches_real_c() {
    c::ref_init();
    let mut rng = Rng(0xcf15_70de_0000_0003);
    let mut sub8_hits = 0usize;
    let mut pad_hits = 0usize;
    let mut sign_seen = [false; 8];
    for it in 0..2000 {
        let (ss_x, ss_y) = [(1i32, 1i32), (1, 0), (0, 0)][it % 3];
        let bd = [8u8, 10, 12][(it / 3) % 3];
        // Luma block bsize: CfL-legal (<= 32x32), covering sub-8x8 shapes.
        // BLOCK_4X4=0, 4X8=1, 8X4=2, 8X8=3, 8X16=4, 16X8=5, 16X16=6,
        // 16X32=7, 32X16=8, 32X32=9, 4X16=16, 16X4=17, 8X32=18, 32X8=19.
        const BSIZES: [usize; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17];
        const BLK_W: [usize; 22] = [
            4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
        ];
        const BLK_H: [usize; 22] = [
            4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
        ];
        let bsize = BSIZES[(rng.next() as usize) % BSIZES.len()];
        // 4:2:2 rejects tall sub-8x8 chroma shapes (ss_size_lookup invalid):
        // keep 4xN luma out of 4:2:2, and require a valid plane block.
        let plane_bsize =
            aom_entropy::partition::get_plane_block_size(bsize, ss_x as usize, ss_y as usize);
        if plane_bsize >= 22 {
            continue;
        }
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        // mi position: odd/even to exercise sub8x8_adjust_offset.
        let mi_row = rng.range(0, 4);
        let mi_col = rng.range(0, 4);

        // Luma recon plane (u16 at every bd).
        let stride = 96usize;
        let maxv = (1i32 << bd) - 1;
        let luma: Vec<u16> = (0..stride * 96)
            .map(|_| rng.range(0, maxv + 1) as u16)
            .collect();
        let block_off = 5 * stride + 7;

        // Store: whole block as one luma txb (tx dims == block dims — the
        // uniform luma tx at CfL sizes), or split into 4 quadrant txbs to
        // exercise cumulative surface tracking.
        const TX_OF_DIMS: fn(usize, usize) -> usize = |w, h| match (w, h) {
            (4, 4) => 0,
            (8, 8) => 1,
            (16, 16) => 2,
            (32, 32) => 3,
            (4, 8) => 5,
            (8, 4) => 6,
            (8, 16) => 7,
            (16, 8) => 8,
            (16, 32) => 9,
            (32, 16) => 10,
            (4, 16) => 13,
            (16, 4) => 14,
            (8, 32) => 15,
            (32, 8) => 16,
            _ => unreachable!(),
        };
        let mut st_c = c::RefCflState::default();
        let mut cfl = CflCtx::new(ss_x, ss_y);
        let split = bw >= 8 && bh >= 8 && rng.next() & 1 != 0;
        if split {
            let (hw, hh) = (bw / 2, bh / 2);
            let sub_tx = TX_OF_DIMS(hw, hh);
            for (r, c_) in [(0, 0), (0, hw / 4), (hh / 4, 0), (hh / 4, hw / 4)] {
                cfl_store_tx(
                    &mut cfl, &luma, block_off, stride, r as i32, c_ as i32, sub_tx, bsize, mi_row,
                    mi_col,
                );
                c::ref_cfl_store_tx(
                    &mut st_c, &luma, block_off, stride, r as i32, c_ as i32, sub_tx, bsize,
                    mi_row, mi_col, ss_x, ss_y, bd,
                );
            }
        } else {
            let tx = TX_OF_DIMS(bw, bh);
            cfl_store_tx(
                &mut cfl, &luma, block_off, stride, 0, 0, tx, bsize, mi_row, mi_col,
            );
            c::ref_cfl_store_tx(
                &mut st_c, &luma, block_off, stride, 0, 0, tx, bsize, mi_row, mi_col, ss_x, ss_y,
                bd,
            );
        }
        assert_eq!(
            &cfl.recon_buf_q3[..],
            &st_c.recon_q3[..],
            "it={it} store recon_q3"
        );
        assert_eq!(
            (cfl.buf_width, cfl.buf_height),
            (st_c.buf_w, st_c.buf_h),
            "it={it}"
        );
        if (mi_row & 1 != 0 && ss_y != 0) || (mi_col & 1 != 0 && ss_x != 0) {
            sub8_hits += 1;
        }

        // Chroma tx block = the plane block (single-txb CfL invariant).
        let (pw, ph) = (BLK_W[plane_bsize], BLK_H[plane_bsize]);
        let uv_tx = TX_OF_DIMS(pw, ph);
        // Exercise cfl_pad: pretend the stored luma surface undershoots by
        // clamping the tracked extent (frame-boundary chroma overrun).
        if rng.next() & 3 == 0 && cfl.buf_width > 4 {
            cfl.buf_width -= 2;
            cfl.buf_height = (cfl.buf_height - 1).max(1);
            st_c.buf_w = cfl.buf_width;
            st_c.buf_h = cfl.buf_height;
            pad_hits += 1;
        }

        let joint_sign = (rng.next() % 8) as i32;
        let alpha_idx = (rng.next() % 256) as i32;
        let plane = 1 + (rng.next() as usize % 2);
        sign_seen[joint_sign as usize] = true;

        // dst holds a DC prediction — BLOCK-CONSTANT by construction
        // (DC_PRED yields one value for the whole block). This is a REAL
        // production invariant: the RTCD cfl_predict_{lbd,hbd} SIMD kernels
        // broadcast `*dst` as the DC for the entire block
        // (cfl_ssse3.c:318 `_mm_set1_epi16(*dst)`), so non-constant dst would
        // diverge from the scalar per-pixel add — production never produces
        // one.
        let dst_stride = 40usize;
        let dc_val = rng.range(0, maxv + 1) as u16;
        let dst0: Vec<u16> = vec![dc_val; dst_stride * 40];
        let dst_off = 3 * dst_stride + 5;
        let mut dst_r = dst0.clone();
        let mut dst_c = dst0.clone();
        cfl_predict_block(
            &mut cfl,
            &mut dst_r,
            dst_off,
            dst_stride,
            uv_tx,
            plane,
            alpha_idx,
            joint_sign,
            i32::from(bd),
        );
        c::ref_cfl_predict_block(
            &mut st_c, &mut dst_c, dst_off, dst_stride, uv_tx, plane, alpha_idx, joint_sign, bsize,
            false, ss_x, ss_y, bd,
        );
        assert_eq!(&cfl.ac_buf_q3[..], &st_c.ac_q3[..], "it={it} ac_buf");
        assert_eq!(
            dst_r, dst_c,
            "it={it} predict bsize={bsize} ss=({ss_x},{ss_y}) bd={bd} plane={plane} js={joint_sign} idx={alpha_idx} split={split}",
        );
        assert!(cfl.are_parameters_computed && st_c.params_computed);

        // Second predict on the same store (params cached) — other plane.
        let plane2 = 3 - plane;
        let mut dst_r2 = dst0.clone();
        let mut dst_c2 = dst0;
        cfl_predict_block(
            &mut cfl,
            &mut dst_r2,
            dst_off,
            dst_stride,
            uv_tx,
            plane2,
            alpha_idx,
            joint_sign,
            i32::from(bd),
        );
        c::ref_cfl_predict_block(
            &mut st_c,
            &mut dst_c2,
            dst_off,
            dst_stride,
            uv_tx,
            plane2,
            alpha_idx,
            joint_sign,
            bsize,
            false,
            ss_x,
            ss_y,
            bd,
        );
        assert_eq!(dst_r2, dst_c2, "it={it} predict plane2");
    }
    assert!(
        sub8_hits > 100,
        "sub-8x8 offset adjustment under-exercised: {sub8_hits}"
    );
    assert!(pad_hits > 100, "cfl_pad under-exercised: {pad_hits}");
    assert!(
        sign_seen.iter().all(|&s| s),
        "all joint signs must be exercised"
    );
}
