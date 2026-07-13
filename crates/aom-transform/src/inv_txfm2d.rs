//! Inverse 2-D transform + reconstruction, bit-exact port of libaom v3.14.1
//! `av1/common/av1_inv_txfm2d.c` (`inv_txfm2d_add_c` + facade + entry points).
//!
//! Row-first then column, with `NewInvSqrt2` rectangular scaling, per-stage
//! `clamp_value` (stage_range = a bd-dependent constant — both branches of
//! `av1_gen_inv_stage_range` assign the same `opt_range`), input `clamp_buf`
//! stages, `av1_round_shift_array` shifts, and `highbd_clip_pixel_add`
//! reconstruction onto the destination. Depends on `bd` (8/10/12).

use crate::cospi::{NEW_SQRT2_BITS, NEW_INV_SQRT2};
use crate::fdct::{clamp_value, round_shift};
use crate::txfm2d::{
    get_rect_tx_log_ratio, log2_idx, FLIP_CFG, HTX_TAB, TXFM_TYPE_LS, TX_SIZE_HIGH, TX_SIZE_WIDE,
    VTX_TAB,
};
use crate::{
    av1_iadst16, av1_iadst4, av1_iadst8, av1_idct16, av1_idct32, av1_idct4, av1_idct64, av1_idct8,
    av1_iidentity16, av1_iidentity32, av1_iidentity4, av1_iidentity8,
};

type Txfm1d = fn(&[i32], &mut [i32], i32, &[i8]);

const INV_COS_BIT: i32 = 12;

// av1_inv_txfm_shift_ls[tx_size][0..2]
#[rustfmt::skip]
static INV_SHIFT: [[i8; 2]; 19] = [
    [0, -4], [-1, -4], [-2, -4], [-2, -4], [-2, -4],
    [0, -4], [0, -4], [-1, -4], [-1, -4], [-1, -4],
    [-1, -4], [-1, -4], [-1, -4], [-1, -4], [-1, -4],
    [-2, -4], [-2, -4], [-2, -4], [-2, -4],
];

fn inv_txfm_func(txfm_type: i32) -> Txfm1d {
    match txfm_type {
        0 => av1_idct4,
        1 => av1_idct8,
        2 => av1_idct16,
        3 => av1_idct32,
        4 => av1_idct64,
        5 => av1_iadst4,
        6 => av1_iadst8,
        7 => av1_iadst16,
        8 => av1_iidentity4,
        9 => av1_iidentity8,
        10 => av1_iidentity16,
        11 => av1_iidentity32,
        _ => panic!("invalid inv txfm_type {txfm_type}"),
    }
}

/// (opt_range_col, opt_range_row) from `av1_gen_inv_stage_range` — the only
/// output-affecting product of that function (both if-branches assign the same
/// value; the assert-only `real_range` path is disabled).
fn opt_range(bd: i32) -> (i8, i8) {
    match bd {
        8 => (16, 16),
        10 => (16, 18),
        12 => (18, 20),
        _ => panic!("bd must be 8/10/12"),
    }
}

struct Cfg {
    shift: [i8; 2],
    func_col: Txfm1d,
    func_row: Txfm1d,
    ud_flip: bool,
    lr_flip: bool,
    valid: bool,
}

fn get_inv_txfm_cfg(tx_type: usize, tx_size: usize) -> Cfg {
    let (ud_flip, lr_flip) = FLIP_CFG[tx_type];
    let txw_idx = log2_idx(TX_SIZE_WIDE[tx_size]);
    let txh_idx = log2_idx(TX_SIZE_HIGH[tx_size]);
    let txfm_type_col = TXFM_TYPE_LS[txh_idx][VTX_TAB[tx_type]];
    let txfm_type_row = TXFM_TYPE_LS[txw_idx][HTX_TAB[tx_type]];
    let valid = txfm_type_col != -1 && txfm_type_row != -1;
    Cfg {
        shift: INV_SHIFT[tx_size],
        func_col: if valid { inv_txfm_func(txfm_type_col) } else { av1_idct4 },
        func_row: if valid { inv_txfm_func(txfm_type_row) } else { av1_idct4 },
        ud_flip,
        lr_flip,
        valid,
    }
}

/// Is `(tx_type, tx_size)` a supported inverse-transform combination?
pub fn inv_txfm_valid(tx_type: usize, tx_size: usize) -> bool {
    get_inv_txfm_cfg(tx_type, tx_size).valid
}

fn round_shift_array(arr: &mut [i32], bit: i32) {
    if bit == 0 {
        return;
    }
    if bit > 0 {
        for v in arr.iter_mut() {
            *v = round_shift(*v as i64, bit);
        }
    } else {
        for v in arr.iter_mut() {
            let widened = (1i64 << (-bit)) * (*v as i64);
            *v = widened.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }
}

#[inline]
fn clamp_buf(buf: &mut [i32], bit: i8) {
    for v in buf.iter_mut() {
        *v = clamp_value(*v, bit);
    }
}

#[inline]
fn highbd_clip_pixel_add(dest: u16, trans: i32, bd: i32) -> u16 {
    let hi = (1i32 << bd) - 1;
    ((dest as i32).wrapping_add(trans)).clamp(0, hi) as u16
}

/// Remap the (possibly 32-capped) coefficient `input` into the full
/// `col_n*row_n` buffer with zeros, matching the C entry points for the 5
/// large sizes. Returns the full modified input buffer.
fn remap_input(input: &[i32], tx_size: usize, col_n: usize, row_n: usize) -> Vec<i32> {
    let mut mod_input = vec![0i32; col_n * row_n];
    match tx_size {
        4 => {
            // 64x64: 32x32 -> 64x64
            for col in 0..32 {
                mod_input[col * 64..col * 64 + 32].copy_from_slice(&input[col * 32..col * 32 + 32]);
            }
        }
        11 => {
            // 32x64: 32x32 -> mod[col*64..+32]
            for col in 0..32 {
                mod_input[col * 64..col * 64 + 32].copy_from_slice(&input[col * 32..col * 32 + 32]);
            }
        }
        12 => {
            // 64x32: 32x32 contiguous -> first half
            mod_input[..32 * 32].copy_from_slice(&input[..32 * 32]);
        }
        17 => {
            // 16x64: 16x32 -> mod[col*64..+32]
            for col in 0..16 {
                mod_input[col * 64..col * 64 + 32].copy_from_slice(&input[col * 32..col * 32 + 32]);
            }
        }
        18 => {
            // 64x16: 32x16 contiguous -> first half
            mod_input[..16 * 32].copy_from_slice(&input[..16 * 32]);
        }
        _ => {
            mod_input.copy_from_slice(input);
        }
    }
    mod_input
}

/// Expected packed coefficient input length for a given tx_size (what the C
/// `av1_inv_txfm2d_add_*_c` entry point consumes).
pub fn inv_input_len(tx_size: usize) -> usize {
    match tx_size {
        4 | 11 | 12 => 32 * 32,
        17 | 18 => 16 * 32,
        _ => TX_SIZE_WIDE[tx_size] * TX_SIZE_HIGH[tx_size],
    }
}

/// Public inverse 2-D transform + add. `output` is a `bd`-bit pixel buffer of
/// at least `row_n*stride`; residuals are reconstructed onto it in place.
pub fn av1_inv_txfm2d_add(
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    tx_type: usize,
    tx_size: usize,
    bd: i32,
) {
    let cfg = get_inv_txfm_cfg(tx_type, tx_size);
    assert!(cfg.valid, "unsupported inverse (tx_type={tx_type}, tx_size={tx_size})");
    let col_n = TX_SIZE_WIDE[tx_size];
    let row_n = TX_SIZE_HIGH[tx_size];
    let shift = cfg.shift;
    let rect_type = get_rect_tx_log_ratio(col_n as i64, row_n as i64);
    let (opt_range_col, opt_range_row) = opt_range(bd);
    let stage_range_row = [opt_range_row; 12];
    let stage_range_col = [opt_range_col; 12];

    let mod_input = remap_input(input, tx_size, col_n, row_n);

    let mut buf = vec![0i32; col_n * row_n];
    let mut temp_in = [0i32; 64];
    let mut temp_out = [0i32; 64];

    // Rows
    for r in 0..row_n {
        let ti = &mut temp_in[0..col_n];
        if rect_type.abs() == 1 {
            for c in 0..col_n {
                ti[c] = round_shift(
                    mod_input[c * row_n + r] as i64 * NEW_INV_SQRT2 as i64,
                    NEW_SQRT2_BITS,
                );
            }
        } else {
            for c in 0..col_n {
                ti[c] = mod_input[c * row_n + r];
            }
        }
        clamp_buf(ti, (bd + 8) as i8);
        (cfg.func_row)(ti, &mut buf[r * col_n..r * col_n + col_n], INV_COS_BIT, &stage_range_row);
        round_shift_array(&mut buf[r * col_n..r * col_n + col_n], -(shift[0] as i32));
    }

    // Columns
    let col_clamp = (bd + 6).max(16) as i8;
    for c in 0..col_n {
        let ti = &mut temp_in[0..row_n];
        for r in 0..row_n {
            let cc = if cfg.lr_flip { col_n - c - 1 } else { c };
            ti[r] = buf[r * col_n + cc];
        }
        clamp_buf(ti, col_clamp);
        let to = &mut temp_out[0..row_n];
        (cfg.func_col)(ti, to, INV_COS_BIT, &stage_range_col);
        round_shift_array(to, -(shift[1] as i32));
        for r in 0..row_n {
            let src = if cfg.ud_flip { to[row_n - r - 1] } else { to[r] };
            let idx = r * stride + c;
            output[idx] = highbd_clip_pixel_add(output[idx], src, bd);
        }
    }
}
