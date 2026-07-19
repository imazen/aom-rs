//! Hand-traced vectors for the CfL pipeline (av1/common/cfl.c) — every expected
//! value below was computed BY HAND from the C source arithmetic, independently
//! of the Rust implementation, so a shared misreading of the C cannot satisfy
//! these (unlike the encode↔decode roundtrip, whose two sides share the port).
//!
//! Stages covered: 420/422/444 hbd luma subsampling (Q3 semantics + CFL_BUF_LINE
//! row advance), cfl_pad column/row replication, subtract_average rounding,
//! cfl_idx_to_alpha joint-sign decode (all 8 joint signs), cfl_predict_hbd
//! rounding + clipping at bd 8/10/12, cfl_store_tx offset/adjust/surface
//! tracking (sub-8x8 parity shifts, buffer reset vs extend), and the
//! compute-once parameter laziness of cfl_predict_block.
//!
//! Direct differential tests against the exported C kernels
//! (cfl_subsample_*_c / cfl_subtract_average_*_c / cfl_predict_hbd_*_c) are
//! DEFERRED until aom-sys-ref is free to take new extern declarations.

use aom_dsp::intra::cfl::{cfl_idx_to_alpha, cfl_predict_block, cfl_store_tx, CflCtx, CFL_BUF_LINE};

const TX_4X4: usize = 0;
const TX_8X8: usize = 1;
const BLOCK_4X4: usize = 0;
const BLOCK_8X8: usize = 3;

/// A 8-stride luma plane whose top-left 4x4 is
/// 10 20 30 40 / 50 60 70 80 / 90 100 110 120 / 130 140 150 160.
fn plane_4x4() -> Vec<u16> {
    let mut p = vec![0u16; 8 * 8];
    for r in 0..4 {
        for c in 0..4 {
            p[r * 8 + c] = (10 + 40 * r + 10 * c) as u16;
        }
    }
    p
}

// ---- subsampling ------------------------------------------------------------------

#[test]
fn subsample_420_hand_traced() {
    // (a+b+c+d) << 1 per 2x2 quad; output advances one CFL_BUF_LINE per luma
    // row PAIR. Quads: (10,20,50,60)=140, (30,40,70,80)=220,
    // (90,100,130,140)=460, (110,120,150,160)=540.
    let mut cfl = CflCtx::new(1, 1);
    let plane = plane_4x4();
    cfl_store_tx(&mut cfl, &plane, 0, 8, 0, 0, TX_4X4, BLOCK_8X8, 0, 0);
    assert_eq!(cfl.recon_buf_q3[0], 280);
    assert_eq!(cfl.recon_buf_q3[1], 440);
    assert_eq!(cfl.recon_buf_q3[CFL_BUF_LINE], 920);
    assert_eq!(cfl.recon_buf_q3[CFL_BUF_LINE + 1], 1080);
    // untouched elsewhere
    assert_eq!(cfl.recon_buf_q3[2], 0);
    assert_eq!(cfl.recon_buf_q3[2 * CFL_BUF_LINE], 0);
    // surface = store dims (2x2 chroma px), reset by the (0,0) store
    assert_eq!((cfl.buf_width, cfl.buf_height), (2, 2));
    assert!(!cfl.are_parameters_computed);
}

#[test]
fn subsample_422_hand_traced() {
    // (a+b) << 2 per horizontal pair; output advances one CFL_BUF_LINE per
    // luma row. Pairs row0: 30->120, 70->280; row1: 110->440, 150->600;
    // row2: 190->760, 230->920; row3: 270->1080, 310->1240.
    let mut cfl = CflCtx::new(1, 0);
    let plane = plane_4x4();
    cfl_store_tx(&mut cfl, &plane, 0, 8, 0, 0, TX_4X4, BLOCK_8X8, 0, 0);
    let expect = [[120, 280], [440, 600], [760, 920], [1080, 1240]];
    for (r, row) in expect.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(cfl.recon_buf_q3[r * CFL_BUF_LINE + c], v, "({r},{c})");
        }
    }
    assert_eq!((cfl.buf_width, cfl.buf_height), (2, 4));
}

#[test]
fn subsample_444_hand_traced() {
    // value << 3, straight copy.
    let mut cfl = CflCtx::new(0, 0);
    let plane = plane_4x4();
    cfl_store_tx(&mut cfl, &plane, 0, 8, 0, 0, TX_4X4, BLOCK_8X8, 0, 0);
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(
                cfl.recon_buf_q3[r * CFL_BUF_LINE + c],
                (10 + 40 * r as u16 + 10 * c as u16) << 3,
                "({r},{c})"
            );
        }
    }
    assert_eq!((cfl.buf_width, cfl.buf_height), (4, 4));
}

#[test]
fn subsample_input_offset_math() {
    // cfl_store_tx reads the tx block at block_off + ((row*stride + col) << 2):
    // an 8x8 block's (row,col)=(1,1) 4x4 txb reads pixels starting at (4,4).
    let mut plane = vec![0u16; 16 * 16];
    for r in 0..4 {
        for c in 0..4 {
            plane[(4 + r) * 16 + (4 + c)] = 1000 + (r * 4 + c) as u16;
        }
    }
    let mut cfl = CflCtx::new(0, 0); // 4:4:4: store lands at (4,4) in the buffer
    cfl_store_tx(&mut cfl, &plane, 0, 16, 1, 1, TX_4X4, BLOCK_8X8, 0, 0);
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(
                cfl.recon_buf_q3[(4 + r) * CFL_BUF_LINE + 4 + c],
                (1000 + (r * 4 + c) as u16) << 3,
                "({r},{c})"
            );
        }
    }
    // (row,col) != (0,0): surface EXTENDS (from garbage-free zero) to 8x8.
    assert_eq!((cfl.buf_width, cfl.buf_height), (8, 8));
}

// ---- padding + average ------------------------------------------------------------

/// Drive pad+average through cfl_predict_block on a 2x2-stored surface asked to
/// predict a 4x4 tx: pad replicates the last column right (rows 0..2 only —
/// min_height excludes the to-be-padded rows), then the last row down over the
/// full padded width.
#[test]
fn pad_and_subtract_average_hand_traced() {
    let mut cfl = CflCtx::new(1, 1);
    // Store one 4x4 luma txb of a BLOCK_4X4 at even mi position -> 2x2 chroma
    // surface: quads chosen to produce 100 200 / 300 400.
    let mut plane = vec![0u16; 8 * 8];
    // top-left 2x2 luma quad sums: (a+b+c+d)<<1 = 100 -> quad sum 50.
    let quads = [[50u16, 100], [150, 200]]; // -> Q3 values 100 200 / 300 400
    for (qr, qrow) in quads.iter().enumerate() {
        for (qc, &qsum) in qrow.iter().enumerate() {
            // put the whole sum in one pixel of the quad
            plane[(qr * 2) * 8 + qc * 2] = qsum;
        }
    }
    cfl_store_tx(&mut cfl, &plane, 0, 8, 0, 0, TX_4X4, BLOCK_4X4, 0, 0);
    assert_eq!(cfl.recon_buf_q3[0], 100);
    assert_eq!(cfl.recon_buf_q3[1], 200);
    assert_eq!(cfl.recon_buf_q3[CFL_BUF_LINE], 300);
    assert_eq!(cfl.recon_buf_q3[CFL_BUF_LINE + 1], 400);
    assert_eq!((cfl.buf_width, cfl.buf_height), (2, 2));

    // Predict a 4x4 chroma tx with alpha 0 (joint sign 0 U-component): the
    // prediction adds nothing, but parameters get computed: pad to 4x4 --
    // padded surface rows: [100 200 200 200] [300 400 400 400] x3 --
    // avg = (5200 + 8) >> 4 = 325; ac = value - 325.
    let mut dst = vec![500u16; 8 * 8];
    cfl_predict_block(&mut cfl, &mut dst, 0, 8, TX_4X4, 1, 0, 0, 10);
    assert!(cfl.are_parameters_computed);
    let pad_expect = [
        [100u16, 200, 200, 200],
        [300, 400, 400, 400],
        [300, 400, 400, 400],
        [300, 400, 400, 400],
    ];
    let ac_expect = [
        [-225i16, -125, -125, -125],
        [-25, 75, 75, 75],
        [-25, 75, 75, 75],
        [-25, 75, 75, 75],
    ];
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(
                cfl.recon_buf_q3[r * CFL_BUF_LINE + c],
                pad_expect[r][c],
                "pad ({r},{c})"
            );
            assert_eq!(
                cfl.ac_buf_q3[r * CFL_BUF_LINE + c],
                ac_expect[r][c],
                "ac ({r},{c})"
            );
        }
    }
    // alpha 0: dst unchanged
    assert!(dst.iter().all(|&v| v == 500));
    assert_eq!((cfl.buf_width, cfl.buf_height), (4, 4));
}

// ---- cfl_idx_to_alpha -------------------------------------------------------------

#[test]
fn idx_to_alpha_all_joint_signs() {
    // CFL_SIGN_U(js) = ((js+1)*11)>>5, CFL_SIGN_V(js) = (js+1) - 3*U.
    // (u_sign, v_sign) over js 0..8: (0,1) (0,2) (1,0) (1,1) (1,2) (2,0) (2,1) (2,2)
    // sign 0 -> 0; sign 2 (POS) -> +(abs+1); sign 1 (NEG) -> -(abs+1).
    let idx = (5 << 4) | 9; // abs_u = 5, abs_v = 9
    let expect_u = [0, 0, -6, -6, -6, 6, 6, 6];
    let expect_v = [-10, 10, 0, -10, 10, 0, -10, 10];
    for js in 0..8 {
        assert_eq!(
            cfl_idx_to_alpha(idx, js, 1),
            expect_u[js as usize],
            "U js={js}"
        );
        assert_eq!(
            cfl_idx_to_alpha(idx, js, 2),
            expect_v[js as usize],
            "V js={js}"
        );
    }
    // Alpha magnitude extremes: abs 0 -> ±1, abs 15 -> ±16.
    assert_eq!(cfl_idx_to_alpha(0, 7, 1), 1);
    assert_eq!(cfl_idx_to_alpha(0xF0, 2, 1), -16);
    assert_eq!(cfl_idx_to_alpha(0x0F, 7, 2), 16);
}

// ---- predict ----------------------------------------------------------------------

/// Hand-traced ROUND_POWER_OF_TWO_SIGNED(alpha * ac, 6) + DC, with clipping.
#[test]
fn predict_rounding_and_clip_hand_traced() {
    // Build a context whose 4x4 AC is the pad_and_subtract vector's:
    // row0 = [-225 -125 -125 -125], rows 1..4 = [-25 75 75 75].
    let mut cfl = CflCtx::new(1, 1);
    let mut plane = vec![0u16; 8 * 8];
    for (qr, qrow) in [[50u16, 100], [150, 200]].iter().enumerate() {
        for (qc, &qsum) in qrow.iter().enumerate() {
            plane[(qr * 2) * 8 + qc * 2] = qsum;
        }
    }
    cfl_store_tx(&mut cfl, &plane, 0, 8, 0, 0, TX_4X4, BLOCK_4X4, 0, 0);

    // alpha_q3 = -6 (js=2 -> U sign NEG; abs_u = 0x50 >> 4 = 5 -> -(5+1)):
    // scaled = RPOT_SIGNED(-6*ac, 6):
    // ac=-225 -> +1350 -> (1382)>>6 = 21; ac=-125 -> 750 -> 782>>6 = 12;
    // ac=-25 -> 150 -> 182>>6 = 2;  ac=75 -> -450 -> -(482>>6) = -7.
    let mut dst = vec![500u16; 8 * 8];
    cfl_predict_block(&mut cfl, &mut dst, 0, 8, TX_4X4, 1, 0x50, 2, 10);
    let expect = [
        [521u16, 512, 512, 512],
        [502, 493, 493, 493],
        [502, 493, 493, 493],
        [502, 493, 493, 493],
    ];
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(dst[r * 8 + c], expect[r][c], "({r},{c})");
        }
    }

    // Clip at bd=8: alpha +16 on ac=-225 (via js=7 pos U, abs 15):
    // scaled = RPOT_SIGNED(16 * -225, 6) = -((3600+32)>>6) = -56 -> 3-56 clips to 0;
    // and on ac=+75: (1200+32)>>6 = 19 -> 250+19 = 269 clips to 255.
    let mut cfl2 = cfl.clone();
    cfl2.are_parameters_computed = true; // reuse the same AC
    let mut dst2 = vec![0u16; 8 * 8];
    dst2[0] = 3; // pairs with ac=-225
    dst2[1] = 250;
    // put ac=+75 under (0,1) for the clip-high case
    cfl2.ac_buf_q3[1] = 75;
    cfl_predict_block(&mut cfl2, &mut dst2, 0, 8, TX_4X4, 1, 0xF0, 7, 8);
    assert_eq!(dst2[0], 0, "clip low");
    assert_eq!(dst2[1], 255, "clip high");

    // bd=12 headroom: DC 4000 + 21 = 4021 (no clip at 4095).
    let mut cfl3 = cfl.clone();
    cfl3.are_parameters_computed = true;
    let mut dst3 = vec![4000u16; 8 * 8];
    cfl_predict_block(&mut cfl3, &mut dst3, 0, 8, TX_4X4, 1, 0x50, 2, 12);
    assert_eq!(dst3[0], 4021);
}

// ---- sub-8x8 shared-chroma store offsets ------------------------------------------

#[test]
fn sub8x8_store_offsets_and_surface_tracking() {
    // 4:2:0 SPLIT-to-4x4 group at mi (2,4)..(3,5): each member stores its 4x4
    // luma; the odd-position members land at parity-shifted buffer offsets.
    let mut cfl = CflCtx::new(1, 1);
    // Four distinct planes so each member's quad sums differ: constant-value
    // planes give quad Q3 = (4*v)<<1 = v*8.
    let mk = |v: u16| vec![v; 8 * 8];
    let (p00, p01, p10, p11) = (mk(10), mk(20), mk(30), mk(40));

    // (even,even) member: store offset (0,0) -- RESETS the surface to 2x2.
    cfl.buf_width = 31; // garbage to prove the reset
    cfl.buf_height = 31;
    cfl_store_tx(&mut cfl, &p00, 0, 8, 0, 0, TX_4X4, BLOCK_4X4, 2, 4);
    assert_eq!((cfl.buf_width, cfl.buf_height), (2, 2));
    // (even,odd): col adjusted 0->1 -> store_col = 2; surface 2x4.
    cfl_store_tx(&mut cfl, &p01, 0, 8, 0, 0, TX_4X4, BLOCK_4X4, 2, 5);
    assert_eq!((cfl.buf_width, cfl.buf_height), (4, 2));
    // (odd,even): row adjusted -> store_row = 2; surface 4x4.
    cfl_store_tx(&mut cfl, &p10, 0, 8, 0, 0, TX_4X4, BLOCK_4X4, 3, 4);
    assert_eq!((cfl.buf_width, cfl.buf_height), (4, 4));
    // (odd,odd): both adjusted.
    cfl_store_tx(&mut cfl, &p11, 0, 8, 0, 0, TX_4X4, BLOCK_4X4, 3, 5);
    assert_eq!((cfl.buf_width, cfl.buf_height), (4, 4));

    // Each 2x2 quadrant of the buffer holds its member's constant Q3 (= v*8).
    for r in 0..4 {
        for c in 0..4 {
            let v = match (r >= 2, c >= 2) {
                (false, false) => 80,
                (false, true) => 160,
                (true, false) => 240,
                (true, true) => 320,
            };
            assert_eq!(cfl.recon_buf_q3[r * CFL_BUF_LINE + c], v, "({r},{c})");
        }
    }

    // 4:2:2 col-only adjustment: an odd-col 4x4 shifts columns, not rows.
    let mut cfl422 = CflCtx::new(1, 0);
    cfl_store_tx(&mut cfl422, &p00, 0, 8, 0, 0, TX_4X4, BLOCK_4X4, 2, 4);
    assert_eq!((cfl422.buf_width, cfl422.buf_height), (2, 4));
    cfl_store_tx(&mut cfl422, &p01, 0, 8, 0, 0, TX_4X4, BLOCK_4X4, 2, 5);
    assert_eq!((cfl422.buf_width, cfl422.buf_height), (4, 4));
    // 422 Q3 of constant v: (v+v)<<2 = v*8, columns 2..4 from the odd member.
    assert_eq!(cfl422.recon_buf_q3[0], 80);
    assert_eq!(cfl422.recon_buf_q3[2], 160);
    assert_eq!(cfl422.recon_buf_q3[3 * CFL_BUF_LINE + 3], 160);
}

/// A store invalidates previously computed parameters (the C
/// `cfl->are_parameters_computed = 0` in cfl_store), and predict recomputes.
#[test]
fn store_invalidates_parameters() {
    let mut cfl = CflCtx::new(0, 0);
    let plane = plane_4x4();
    cfl_store_tx(&mut cfl, &plane, 0, 8, 0, 0, TX_4X4, BLOCK_8X8, 0, 0);
    let mut dst = vec![100u16; 8 * 8];
    cfl_predict_block(&mut cfl, &mut dst, 0, 8, TX_4X4, 1, 0, 0, 8);
    assert!(cfl.are_parameters_computed);
    let ac_before = cfl.ac_buf_q3;
    // V-plane predict reuses the computed AC (no recompute; C asserts on one).
    cfl_predict_block(&mut cfl, &mut dst, 0, 8, TX_4X4, 2, 0, 0, 8);
    assert_eq!(ac_before, cfl.ac_buf_q3);
    // A new store invalidates.
    cfl_store_tx(&mut cfl, &plane, 0, 8, 0, 0, TX_8X8, BLOCK_8X8, 0, 0);
    assert!(!cfl.are_parameters_computed);
}
