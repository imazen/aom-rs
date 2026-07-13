//! aom-txb — bit-exact AV1 transform-block coefficient-coding kernels (port of
//! libaom v3.14.1 `av1/encoder/encodetxb.c` + `av1/common/txb_common.{h,c}`).
//!
//! Encoder speed-0 critical path: `av1_txb_init_levels` (coefficient buffer →
//! padded absolute-level map) and `av1_get_nz_map_contexts` (per-coefficient
//! entropy contexts for the whole txb) feed both `av1_write_coeffs_txb`
//! (bitstream packing) and `av1_cost_coeffs_txb` (RD cost). Both are
//! RTCD-dispatched (SSE/AVX2/NEON) in libaom.
//!
//! Layout note: the coefficient buffer and context indices use libaom's
//! **transposed** raster order — `coeff_idx = col * height + row` with
//! `bhl = log2(adjusted height)` — and the levels buffer stride is
//! `height + TX_PAD_HOR`.

#![forbid(unsafe_code)]

mod tables;
pub use tables::nz_map_ctx_offset;
mod scan;
pub use scan::{iscan, scan, SCAN_ORDERS};
mod write;
pub use write::{txsize_entropy_ctx, write_coeffs_txb, write_coeffs_txb_full, CDF_ARENA_LEN};
mod read;
pub use read::{read_coeffs_txb, read_coeffs_txb_full};
mod cost;
pub use cost::{cost_coeffs_txb, CoeffCostTables};
mod prob_cost;
pub use prob_cost::{cost_symbol, cost_tokens_from_cdf};
mod fill;
pub use fill::{fill_lv_map_coeff_cost, LvMapCoeffCost};
mod ext_tx;
pub use ext_tx::{ext_tx_derive, ext_tx_set, ext_tx_set_type, read_tx_type, write_tx_type, ExtTxDeriv};
mod trellis_cost;
pub use trellis_cost::{br_cost_with_diff, coeff_cost_eob, coeff_cost_general, two_coeff_cost_simple};
mod optimize;
pub use optimize::{optimize_txb, optimize_txb_qm, OptimizeResult};
mod entropy_ctx;
pub use entropy_ctx::{get_txb_ctx, txb_entropy_context};

/// `TX_PAD_HOR` (enums.h): horizontal padding of the levels buffer.
pub const TX_PAD_HOR: usize = 4;
const TX_PAD_HOR_LOG2: u32 = 2;
/// `TX_PAD_BOTTOM` (enums.h). `TX_PAD_TOP` is 0, so `set_levels` is identity.
pub const TX_PAD_BOTTOM: usize = 4;
/// `TX_PAD_END` (enums.h): tail padding.
pub const TX_PAD_END: usize = 16;
/// `TX_PAD_2D` (enums.h): total padded levels-buffer size, (32+4)*(32+4)+16.
pub const TX_PAD_2D: usize = (32 + TX_PAD_HOR) * (32 + TX_PAD_BOTTOM) + TX_PAD_END;

/// `TX_CLASS` (entropy.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxClass {
    /// TX_CLASS_2D = 0
    TwoD = 0,
    /// TX_CLASS_HORIZ = 1
    Horiz = 1,
    /// TX_CLASS_VERT = 2
    Vert = 2,
}

impl TxClass {
    /// Map a libaom `TX_CLASS` integer (0/1/2).
    pub fn from_c(v: i32) -> TxClass {
        match v {
            1 => TxClass::Horiz,
            2 => TxClass::Vert,
            _ => TxClass::TwoD,
        }
    }
}

/// `tx_type_to_class[TX_TYPES]` (txb_common.h), indexed by libaom `TX_TYPE`.
pub const TX_TYPE_TO_CLASS: [TxClass; 16] = [
    TxClass::TwoD,  // DCT_DCT
    TxClass::TwoD,  // ADST_DCT
    TxClass::TwoD,  // DCT_ADST
    TxClass::TwoD,  // ADST_ADST
    TxClass::TwoD,  // FLIPADST_DCT
    TxClass::TwoD,  // DCT_FLIPADST
    TxClass::TwoD,  // FLIPADST_FLIPADST
    TxClass::TwoD,  // ADST_FLIPADST
    TxClass::TwoD,  // FLIPADST_ADST
    TxClass::TwoD,  // IDTX
    TxClass::Vert,  // V_DCT
    TxClass::Horiz, // H_DCT
    TxClass::Vert,  // V_ADST
    TxClass::Horiz, // H_ADST
    TxClass::Vert,  // V_FLIPADST
    TxClass::Horiz, // H_FLIPADST
];

// Per-TX_SIZE dimensions (enums.h tables), index order = libaom TX_SIZE 0..18.
const TX_SIZE_WIDE: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TX_SIZE_HIGH: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

/// `av1_get_adjusted_tx_size` (av1_common_int.h): 64-point sizes cap to 32.
pub fn adjusted_tx_size(tx_size: usize) -> usize {
    match tx_size {
        4 => 3,        // TX_64X64 -> TX_32X32
        12 => 3,       // TX_64X32 -> TX_32X32
        11 => 3,       // TX_32X64 -> TX_32X32
        18 => 10,      // TX_64X16 -> TX_32X16
        17 => 9,       // TX_16X64 -> TX_16X32
        other => other,
    }
}

/// `get_txb_wide`: adjusted transform-block width.
pub fn txb_wide(tx_size: usize) -> usize {
    TX_SIZE_WIDE[adjusted_tx_size(tx_size)]
}

/// `get_txb_high`: adjusted transform-block height.
pub fn txb_high(tx_size: usize) -> usize {
    TX_SIZE_HIGH[adjusted_tx_size(tx_size)]
}

/// `get_txb_bhl`: log2 of the adjusted transform-block height.
pub fn txb_bhl(tx_size: usize) -> u32 {
    TX_SIZE_HIGH[adjusted_tx_size(tx_size)].trailing_zeros()
}

/// `av1_txb_init_levels_c`: build the padded absolute-level map from the
/// (transposed-layout) coefficient buffer. `levels` must be at least
/// `TX_PAD_2D` bytes; only the region libaom writes is touched — callers may
/// rely on the exact write footprint (verified by the differential harness).
pub fn txb_init_levels(coeff: &[i32], width: usize, height: usize, levels: &mut [u8]) {
    let stride = height + TX_PAD_HOR;
    // memset(levels + stride*width, 0, TX_PAD_BOTTOM*stride + TX_PAD_END)
    let tail = stride * width;
    levels[tail..tail + TX_PAD_BOTTOM * stride + TX_PAD_END].fill(0);

    let mut ls = 0usize;
    for i in 0..width {
        for j in 0..height {
            levels[ls] = coeff[i * height + j].unsigned_abs().min(i8::MAX as u32) as u8;
            ls += 1;
        }
        for _ in 0..TX_PAD_HOR {
            levels[ls] = 0;
            ls += 1;
        }
    }
}

/// `get_padded_idx`: transposed raster index → padded levels-buffer index.
#[inline]
fn get_padded_idx(idx: usize, bhl: u32) -> usize {
    idx + ((idx >> bhl) << TX_PAD_HOR_LOG2)
}

#[inline]
fn min3(v: u8) -> i32 {
    (v as i32).min(3)
}

/// `get_nz_mag`: neighbour-magnitude sum for the base-level context, reading
/// the padded levels buffer at the coefficient's padded position.
#[inline]
fn get_nz_mag(levels: &[u8], base: usize, bhl: u32, tx_class: TxClass) -> i32 {
    let bhl = bhl as usize;
    // { 0, 1 } then { 1, 0 }
    let mut mag = min3(levels[base + (1 << bhl) + TX_PAD_HOR]);
    mag += min3(levels[base + 1]);
    match tx_class {
        TxClass::TwoD => {
            mag += min3(levels[base + (1 << bhl) + TX_PAD_HOR + 1]); // { 1, 1 }
            mag += min3(levels[base + (2 << bhl) + (2 << TX_PAD_HOR_LOG2)]); // { 0, 2 }
            mag += min3(levels[base + 2]); // { 2, 0 }
        }
        TxClass::Vert => {
            mag += min3(levels[base + 2]); // { 2, 0 }
            mag += min3(levels[base + 3]); // { 3, 0 }
            mag += min3(levels[base + 4]); // { 4, 0 }
        }
        TxClass::Horiz => {
            mag += min3(levels[base + (2 << bhl) + (2 << TX_PAD_HOR_LOG2)]); // { 0, 2 }
            mag += min3(levels[base + (3 << bhl) + (3 << TX_PAD_HOR_LOG2)]); // { 0, 3 }
            mag += min3(levels[base + (4 << bhl) + (4 << TX_PAD_HOR_LOG2)]); // { 0, 4 }
        }
    }
    mag
}

/// `get_br_ctx` (txb_common.h): coefficient base-range context, transposed
/// layout. `c` is the transposed raster index; reads the padded levels buffer.
pub fn get_br_ctx(levels: &[u8], c: usize, bhl: u32, tx_class: TxClass) -> i32 {
    let col = c >> bhl;
    let row = c - (col << bhl);
    let stride = (1usize << bhl) + TX_PAD_HOR;
    let pos = col * stride + row;
    let mut mag = levels[pos + 1] as i32 + levels[pos + stride] as i32;
    match tx_class {
        TxClass::TwoD => {
            mag += levels[pos + stride + 1] as i32;
            mag = ((mag + 1) >> 1).min(6);
            if c == 0 {
                return mag;
            }
            if row < 2 && col < 2 {
                return mag + 7;
            }
        }
        TxClass::Horiz => {
            mag += levels[pos + (stride << 1)] as i32;
            mag = ((mag + 1) >> 1).min(6);
            if c == 0 {
                return mag;
            }
            if col == 0 {
                return mag + 7;
            }
        }
        TxClass::Vert => {
            mag += levels[pos + 2] as i32;
            mag = ((mag + 1) >> 1).min(6);
            if c == 0 {
                return mag;
            }
            if row == 0 {
                return mag + 7;
            }
        }
    }
    mag + 14
}

/// `nz_map_ctx_offset_1d` (txb_common.h): SIG_COEF_CONTEXTS_2D=26 based.
const NZ_MAP_CTX_0: i32 = 26;
const NZ_MAP_CTX_5: i32 = NZ_MAP_CTX_0 + 5;
const NZ_MAP_CTX_10: i32 = NZ_MAP_CTX_0 + 10;

#[inline]
fn nz_map_ctx_offset_1d(i: usize) -> i32 {
    match i {
        0 => NZ_MAP_CTX_0,
        1 => NZ_MAP_CTX_5,
        _ => NZ_MAP_CTX_10,
    }
}

/// `get_nz_map_ctx_from_stats`.
#[inline]
fn get_nz_map_ctx_from_stats(
    stats: i32,
    coeff_idx: usize,
    bhl: u32,
    tx_size: usize,
    tx_class: TxClass,
) -> i32 {
    if tx_class == TxClass::TwoD && coeff_idx == 0 {
        return 0;
    }
    let ctx = ((stats + 1) >> 1).min(4);
    match tx_class {
        TxClass::TwoD => ctx + tables::nz_map_ctx_offset(tx_size)[coeff_idx] as i32,
        TxClass::Horiz => {
            let col = coeff_idx >> bhl;
            ctx + nz_map_ctx_offset_1d(col)
        }
        TxClass::Vert => {
            let col = coeff_idx >> bhl;
            let row = coeff_idx - (col << bhl);
            ctx + nz_map_ctx_offset_1d(row)
        }
    }
}

/// `get_nz_map_ctx` (encodetxb.c): per-coefficient context, with the EOB
/// coefficient using the 4 scan-position buckets instead of neighbour stats.
#[inline]
#[allow(clippy::too_many_arguments)]
fn get_nz_map_ctx(
    levels: &[u8],
    coeff_idx: usize,
    bhl: u32,
    width: usize,
    scan_idx: usize,
    is_eob: bool,
    tx_size: usize,
    tx_class: TxClass,
) -> i32 {
    if is_eob {
        if scan_idx == 0 {
            return 0;
        }
        if scan_idx <= (width << bhl) / 8 {
            return 1;
        }
        if scan_idx <= (width << bhl) / 4 {
            return 2;
        }
        return 3;
    }
    let stats = get_nz_mag(levels, get_padded_idx(coeff_idx, bhl), bhl, tx_class);
    get_nz_map_ctx_from_stats(stats, coeff_idx, bhl, tx_size, tx_class)
}


/// `get_lower_levels_ctx` (txb_common.h): base context from neighbour stats.
pub(crate) fn get_lower_levels_ctx(levels: &[u8], coeff_idx: usize, bhl: u32, tx_size: usize, tx_class: TxClass) -> i32 {
    let stats = get_nz_mag(levels, get_padded_idx(coeff_idx, bhl), bhl, tx_class);
    get_nz_map_ctx_from_stats(stats, coeff_idx, bhl, tx_size, tx_class)
}

/// `get_lower_levels_ctx_eob`.
pub(crate) fn get_lower_levels_ctx_eob(bhl: u32, width: usize, scan_idx: usize) -> i32 {
    if scan_idx == 0 { return 0; }
    if scan_idx <= (width << bhl) / 8 { return 1; }
    if scan_idx <= (width << bhl) / 4 { return 2; }
    3
}

/// `get_lower_levels_ctx_general`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_lower_levels_ctx_general(is_last: bool, scan_idx: usize, bhl: u32, width: usize, levels: &[u8], coeff_idx: usize, tx_size: usize, tx_class: TxClass) -> i32 {
    if is_last {
        return get_lower_levels_ctx_eob(bhl, width, scan_idx);
    }
    get_lower_levels_ctx(levels, coeff_idx, bhl, tx_size, tx_class)
}

/// `get_padded_idx` re-export for the optimize module.
pub(crate) fn padded_idx(idx: usize, bhl: u32) -> usize { get_padded_idx(idx, bhl) }

/// `av1_get_nz_map_contexts_c`: compute the entropy context for every coded
/// coefficient of a transform block. `scan` is the (tx_size, tx_type) scan
/// order (transposed positions); writes `coeff_contexts[scan[i]]` for
/// `i < eob` and touches nothing else.
pub fn get_nz_map_contexts(
    levels: &[u8],
    scan: &[i16],
    eob: usize,
    tx_size: usize,
    tx_class: TxClass,
    coeff_contexts: &mut [i8],
) {
    let bhl = txb_bhl(tx_size);
    let width = txb_wide(tx_size);
    for (i, &sc) in scan[..eob].iter().enumerate() {
        let pos = sc as usize;
        coeff_contexts[pos] =
            get_nz_map_ctx(levels, pos, bhl, width, i, i == eob - 1, tx_size, tx_class) as i8;
    }
}

/// `av1_eob_group_start[12]` (txb_common.c).
pub const EOB_GROUP_START: [i16; 12] = [0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513];
/// `av1_eob_offset_bits[12]` (txb_common.c).
pub const EOB_OFFSET_BITS: [i16; 12] = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

const EOB_TO_POS_SMALL: [i8; 33] = [
    0, 1, 2, // 0-2
    3, 3, // 3-4
    4, 4, 4, 4, // 5-8
    5, 5, 5, 5, 5, 5, 5, 5, // 9-16
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, // 17-32
];

const EOB_TO_POS_LARGE: [i8; 17] = [
    6, // placeholder
    7, // 33-64
    8, 8, // 65-128
    9, 9, 9, 9, // 129-256
    10, 10, 10, 10, 10, 10, 10, 10, // 257-512
    11, // 513-
];

/// `av1_get_eob_pos_token`: EOB → (group token, extra offset within group).
pub fn get_eob_pos_token(eob: i32) -> (i32, i32) {
    let t = if eob < 33 {
        EOB_TO_POS_SMALL[eob as usize] as i32
    } else {
        let e = ((eob - 1) >> 5).min(16);
        EOB_TO_POS_LARGE[e as usize] as i32
    };
    (t, eob - EOB_GROUP_START[t as usize] as i32)
}
