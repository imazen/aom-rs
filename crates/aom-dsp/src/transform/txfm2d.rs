//! Forward 2-D transform, bit-exact port of libaom v3.14.1
//! `av1/encoder/av1_fwd_txfm2d.c` (+ config tables from `av1_txfm.c`,
//! `common_data.h`, `av1_txfm.h`). Composes the 1-D kernels with per-size
//! shift/round stages, up/down and left/right flips, rectangular Sqrt2 scaling,
//! a transpose, and the 64-point coefficient repacking.
//!
//! `bd` (bit depth) is intentionally absent: in libaom it only feeds
//! `av1_gen_fwd_stage_range`, which drives the (disabled) range checker and has
//! no effect on output.

use crate::transform::cospi::{NEW_SQRT2, NEW_SQRT2_BITS};
use crate::transform::fdct::round_shift;
use crate::transform::{
    av1_fadst16, av1_fadst4, av1_fadst8, av1_fdct16, av1_fdct32, av1_fdct4, av1_fdct64, av1_fdct8,
    av1_fidentity16, av1_fidentity32, av1_fidentity4, av1_fidentity8,
};

type Txfm1d = fn(&[i32], &mut [i32], i32, &[i8]);

// ---- TX_SIZE ordering (0..19), per libaom enums.h -------------------------
// 0:4x4 1:8x8 2:16x16 3:32x32 4:64x64 5:4x8 6:8x4 7:8x16 8:16x8 9:16x32
// 10:32x16 11:32x64 12:64x32 13:4x16 14:16x4 15:8x32 16:32x8 17:16x64 18:64x16
pub const TX_SIZES_ALL: usize = 19;

#[rustfmt::skip]
pub(crate) static TX_SIZE_WIDE: [usize; TX_SIZES_ALL] =
    [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
#[rustfmt::skip]
pub(crate) static TX_SIZE_HIGH: [usize; TX_SIZES_ALL] =
    [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

// av1_fwd_txfm_shift_ls[tx_size][0..3]
#[rustfmt::skip]
static FWD_SHIFT: [[i8; 3]; TX_SIZES_ALL] = [
    [2, 0, 0], [2, -1, 0], [2, -2, 0], [2, -4, 0], [0, -2, -2],
    [2, -1, 0], [2, -1, 0], [2, -2, 0], [2, -2, 0], [2, -4, 0],
    [2, -4, 0], [0, -2, -2], [2, -4, -2], [2, -1, 0], [2, -1, 0],
    [2, -2, 0], [2, -2, 0], [0, -2, 0], [2, -4, 0],
];

// av1_fwd_cos_bit_col / _row [txw_idx][txh_idx]
#[rustfmt::skip]
static COS_BIT_COL: [[i8; 5]; 5] = [
    [13, 13, 13, 0, 0], [13, 13, 13, 12, 0], [13, 13, 13, 12, 13],
    [0, 13, 13, 12, 13], [0, 0, 13, 12, 13],
];
#[rustfmt::skip]
static COS_BIT_ROW: [[i8; 5]; 5] = [
    [13, 13, 12, 0, 0], [13, 13, 13, 12, 0], [13, 13, 12, 13, 12],
    [0, 12, 13, 12, 11], [0, 0, 12, 11, 10],
];

// ---- TX_TYPE ordering (0..16) --------------------------------------------
// 0:DCT_DCT 1:ADST_DCT 2:DCT_ADST 3:ADST_ADST 4:FLIPADST_DCT 5:DCT_FLIPADST
// 6:FLIPADST_FLIPADST 7:ADST_FLIPADST 8:FLIPADST_ADST 9:IDTX 10:V_DCT 11:H_DCT
// 12:V_ADST 13:H_ADST 14:V_FLIPADST 15:H_FLIPADST
pub const TX_TYPES: usize = 16;

// TX_TYPE_1D: 0:DCT 1:ADST 2:FLIPADST 3:IDTX
#[rustfmt::skip]
pub(crate) static VTX_TAB: [usize; TX_TYPES] = [0,1,0,1,2,0,2,1,2,3,0,3,1,3,2,3];
#[rustfmt::skip]
pub(crate) static HTX_TAB: [usize; TX_TYPES] = [0,0,1,1,0,2,2,2,1,3,3,0,3,1,3,2];

// (ud_flip, lr_flip) per tx_type
#[rustfmt::skip]
pub(crate) static FLIP_CFG: [(bool, bool); TX_TYPES] = [
    (false,false),(false,false),(false,false),(false,false),
    (true,false),(false,true),(true,true),(false,true),
    (true,false),(false,false),(false,false),(false,false),
    (false,false),(false,false),(true,false),(false,true),
];

// TXFM_TYPE: 0:DCT4 1:DCT8 2:DCT16 3:DCT32 4:DCT64 5:ADST4 6:ADST8 7:ADST16
//            8:IDTX4 9:IDTX8 10:IDTX16 11:IDTX32 ; -1 = INVALID
// av1_txfm_type_ls[size_idx][tx_type_1d]
#[rustfmt::skip]
pub(crate) static TXFM_TYPE_LS: [[i32; 4]; 5] = [
    [0, 5, 5, 8],
    [1, 6, 6, 9],
    [2, 7, 7, 10],
    [3, -1, -1, 11],
    [4, -1, -1, -1],
];

fn txfm_func(txfm_type: i32) -> Txfm1d {
    match txfm_type {
        0 => av1_fdct4,
        1 => av1_fdct8,
        2 => av1_fdct16,
        3 => av1_fdct32,
        4 => av1_fdct64,
        5 => av1_fadst4,
        6 => av1_fadst8,
        7 => av1_fadst16,
        8 => av1_fidentity4,
        9 => av1_fidentity8,
        10 => av1_fidentity16,
        11 => av1_fidentity32,
        _ => panic!("invalid txfm_type {txfm_type}"),
    }
}

#[inline]
pub(crate) fn log2_idx(n: usize) -> usize {
    match n {
        4 => 0,
        8 => 1,
        16 => 2,
        32 => 3,
        64 => 4,
        _ => unreachable!(),
    }
}

pub(crate) fn get_rect_tx_log_ratio(col: i64, row: i64) -> i32 {
    if col == row {
        return 0;
    }
    if col > row {
        if col == row * 2 {
            return 1;
        }
        if col == row * 4 {
            return 2;
        }
    } else {
        if row == col * 2 {
            return -1;
        }
        if row == col * 4 {
            return -2;
        }
    }
    0
}

struct Cfg {
    tx_size: usize,
    shift: [i8; 3],
    cos_bit_col: i8,
    cos_bit_row: i8,
    func_col: Txfm1d,
    func_row: Txfm1d,
    /// Raw TXFM_TYPE ids (0..=11) — the SIMD per-kernel dispatch keys.
    txfm_type_col: i32,
    txfm_type_row: i32,
    ud_flip: bool,
    lr_flip: bool,
    valid: bool,
}

fn get_fwd_txfm_cfg(tx_type: usize, tx_size: usize) -> Cfg {
    let (ud_flip, lr_flip) = FLIP_CFG[tx_type];
    let tx_type_1d_col = VTX_TAB[tx_type];
    let tx_type_1d_row = HTX_TAB[tx_type];
    let txw_idx = log2_idx(TX_SIZE_WIDE[tx_size]);
    let txh_idx = log2_idx(TX_SIZE_HIGH[tx_size]);
    let txfm_type_col = TXFM_TYPE_LS[txh_idx][tx_type_1d_col];
    let txfm_type_row = TXFM_TYPE_LS[txw_idx][tx_type_1d_row];
    let valid = txfm_type_col != -1 && txfm_type_row != -1;
    Cfg {
        tx_size,
        shift: FWD_SHIFT[tx_size],
        cos_bit_col: COS_BIT_COL[txw_idx][txh_idx],
        cos_bit_row: COS_BIT_ROW[txw_idx][txh_idx],
        func_col: if valid { txfm_func(txfm_type_col) } else { av1_fdct4 },
        func_row: if valid { txfm_func(txfm_type_row) } else { av1_fdct4 },
        txfm_type_col,
        txfm_type_row,
        ud_flip,
        lr_flip,
        valid,
    }
}

/// Is `(tx_type, tx_size)` a supported forward-transform combination?
pub fn fwd_txfm_valid(tx_type: usize, tx_size: usize) -> bool {
    get_fwd_txfm_cfg(tx_type, tx_size).valid
}

/// libaom `av1_round_shift_array_c` — bit-exact.
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

const SR: [i8; 12] = [0; 12]; // stage_range: ignored by the 1-D kernels

/// Core composition, bit-exact port of `fwd_txfm2d_c`.
fn fwd_txfm2d_core(input: &[i16], output: &mut [i32], stride: usize, cfg: &Cfg) {
    let col_n = TX_SIZE_WIDE[cfg.tx_size];
    let row_n = TX_SIZE_HIGH[cfg.tx_size];
    let shift = cfg.shift;
    let rect_type = get_rect_tx_log_ratio(col_n as i64, row_n as i64);

    let mut buf = vec![0i32; col_n * row_n];

    // Columns — the SIMD column pass (8-column lane batches) is bit-identical
    // to this scalar loop (crate::transform::simd docs + differentials); it declines
    // (false) when the col kernel isn't ported / col_n < 8 / SIMD unavailable
    // or pinned off. Use `output` as scratch on the scalar path only.
    #[cfg(target_arch = "x86_64")]
    let cols_done = crate::transform::simd::try_fwd_col_pass(
        cfg.txfm_type_col,
        input,
        &mut buf,
        stride,
        col_n,
        row_n,
        shift[0] as i32,
        -(shift[1] as i32),
        cfg.cos_bit_col as i32,
        cfg.ud_flip,
        cfg.lr_flip,
    );
    #[cfg(not(target_arch = "x86_64"))]
    let cols_done = false;
    if !cols_done {
        // Scalar: temp_in = output[0..row], temp_out = output[row..2row].
        for c in 0..col_n {
            {
                let (temp_in, rest) = output.split_at_mut(row_n);
                let temp_out = &mut rest[0..row_n];
                for r in 0..row_n {
                    let src_r = if cfg.ud_flip { row_n - r - 1 } else { r };
                    temp_in[r] = input[src_r * stride + c] as i32;
                }
                round_shift_array(temp_in, -(shift[0] as i32));
                (cfg.func_col)(temp_in, temp_out, cfg.cos_bit_col as i32, &SR);
                round_shift_array(temp_out, -(shift[1] as i32));
                for r in 0..row_n {
                    let dst_c = if cfg.lr_flip { col_n - c - 1 } else { c };
                    buf[r * col_n + dst_c] = temp_out[r];
                }
            }
        }
    }

    // Rows — same contract: the SIMD row pass (8-row lane batches) is
    // bit-identical to the scalar loop and declines when not applicable.
    #[cfg(target_arch = "x86_64")]
    let rows_done = crate::transform::simd::try_fwd_row_pass(
        cfg.txfm_type_row,
        &buf,
        output,
        col_n,
        row_n,
        -(shift[2] as i32),
        cfg.cos_bit_row as i32,
        rect_type.abs() == 1,
    );
    #[cfg(not(target_arch = "x86_64"))]
    let rows_done = false;
    if !rows_done {
        let mut row_buffer = [0i32; 64];
        for r in 0..row_n {
            let rb = &mut row_buffer[0..col_n];
            (cfg.func_row)(&buf[r * col_n..r * col_n + col_n], rb, cfg.cos_bit_row as i32, &SR);
            round_shift_array(rb, -(shift[2] as i32));
            if rect_type.abs() == 1 {
                for v in rb.iter_mut() {
                    *v = round_shift(*v as i64 * NEW_SQRT2 as i64, NEW_SQRT2_BITS);
                }
            }
            for c in 0..col_n {
                output[c * row_n + r] = row_buffer[c];
            }
        }
    }
}

/// Public forward 2-D transform. `output` must have length `wide*high` of the
/// given `tx_size`. Mirrors the C `av1_fwd_txfm2d_<size>_c` entry points,
/// including the 64-point coefficient zeroing/repacking.
pub fn av1_fwd_txfm2d(input: &[i16], output: &mut [i32], stride: usize, tx_type: usize, tx_size: usize) {
    let cfg = get_fwd_txfm_cfg(tx_type, tx_size);
    assert!(cfg.valid, "unsupported (tx_type={tx_type}, tx_size={tx_size})");
    fwd_txfm2d_core(input, output, stride, &cfg);

    // Post-process for the transforms whose active area is capped at 32.
    match tx_size {
        4 => {
            // TX_64X64: zero top-right 32x32 of each of first 32 cols; zero
            // bottom 64x32; repack first 32x32.
            for col in 0..32 {
                for i in 32..64 {
                    output[col * 64 + i] = 0;
                }
            }
            for i in (32 * 64)..(64 * 64) {
                output[i] = 0;
            }
            for col in 1..32 {
                output.copy_within(col * 64..col * 64 + 32, col * 32);
            }
        }
        11 => {
            // TX_32X64: zero right 32x32; repack.
            for col in 0..32 {
                for i in 32..64 {
                    output[col * 64 + i] = 0;
                }
            }
            for col in 1..32 {
                output.copy_within(col * 64..col * 64 + 32, col * 32);
            }
        }
        12 => {
            // TX_64X32: zero bottom 32x32.
            for i in (32 * 32)..(64 * 32) {
                output[i] = 0;
            }
        }
        17 => {
            // TX_16X64: zero right 32x16; repack.
            for row in 0..16 {
                for i in 32..64 {
                    output[row * 64 + i] = 0;
                }
            }
            for row in 1..16 {
                output.copy_within(row * 64..row * 64 + 32, row * 32);
            }
        }
        18 => {
            // TX_64X16: zero bottom 16x32.
            for i in (16 * 32)..(64 * 16) {
                output[i] = 0;
            }
        }
        _ => {}
    }
}
