//! Extended-transform set derivation + `av1_write_tx_type` (libaom
//! `av1/common/blockd.h`, `av1/common/entropymode.h`,
//! `av1/encoder/bitstream.c`): select and signal a block's transform type — the
//! plane-0 step the coefficient writer/cost functions deliberately leave out.
//!
//! The derivation (set type / eset / symbol index / arity) is transcribed from
//! libaom's tables and verified exhaustively vs C. The symbol emission reuses
//! the bit-exact `aom_write_symbol` (aom-entropy).

use aom_entropy::cdf::write_symbol;
use aom_entropy::enc::OdEcEnc;

/// `TxSetType` (0..5). `av1_num_ext_tx_set = {1,2,5,7,12,16}`.
const NUM_EXT_TX_SET: [i32; 6] = [1, 2, 5, 7, 12, 16];

/// `av1_ext_tx_ind[EXT_TX_SET_TYPES][TX_TYPES]` — TX_TYPE -> transmitted symbol.
#[rustfmt::skip]
const EXT_TX_IND: [[i32; 16]; 6] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 5, 6, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
    [3, 4, 5, 8, 6, 7, 9, 10, 11, 0, 1, 2, 0, 0, 0, 0],
    [7, 8, 9, 12, 10, 11, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6],
];

/// `av1_ext_tx_used[EXT_TX_SET_TYPES][TX_TYPES]` — is this TX_TYPE in the set.
#[rustfmt::skip]
const EXT_TX_USED: [[i32; 16]; 6] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

/// `av1_ext_tx_set_lookup[2][2]` — [is_inter][tx_size_sqr==TX_16X16] -> TxSetType.
const EXT_TX_SET_LOOKUP: [[i32; 2]; 2] = [[3, 2], [5, 4]];

/// `ext_tx_set_index[2][EXT_TX_SET_TYPES]` — TxSetType -> CDF set index (or -1).
const EXT_TX_SET_INDEX: [[i32; 6]; 2] = [[0, -1, 2, 1, -1, -1], [0, 3, -1, -1, 2, 1]];

/// `fimode_to_intradir[FILTER_INTRA_MODES]` (DC_PRED=0, V=1, H=2, D157=6, DC=0).
const FIMODE_TO_INTRADIR: [i32; 5] = [0, 1, 2, 6, 0];

// txsize_sqr_map / txsize_sqr_up_map (common_data.h), TX_SIZE 0..18 -> 0..4.
const TXSIZE_SQR: [usize; 19] = [0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2];
const TXSIZE_SQR_UP: [usize; 19] = [0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4];

const TX_16X16: usize = 2;
const TX_32X32: usize = 3;

/// `av1_get_ext_tx_set_type`.
pub fn ext_tx_set_type(tx_size: usize, is_inter: bool, reduced: bool) -> usize {
    let up = TXSIZE_SQR_UP[tx_size];
    if up > TX_32X32 {
        return 0; // EXT_TX_SET_DCTONLY
    }
    if up == TX_32X32 {
        return if is_inter { 1 } else { 0 }; // DCT_IDTX : DCTONLY
    }
    if reduced {
        return if is_inter { 1 } else { 2 }; // DCT_IDTX : DTT4_IDTX
    }
    let sqr = TXSIZE_SQR[tx_size];
    EXT_TX_SET_LOOKUP[is_inter as usize][(sqr == TX_16X16) as usize] as usize
}

/// `get_ext_tx_set`: CDF set index (`eset`); -1 / 0 for DCT-only.
pub fn ext_tx_set(tx_size: usize, is_inter: bool, reduced: bool) -> i32 {
    let st = ext_tx_set_type(tx_size, is_inter, reduced);
    EXT_TX_SET_INDEX[is_inter as usize][st]
}

/// Derived signaling parameters (mirror of the C harness): set type, symbol
/// arity `num`, CDF set `eset`, `square_tx_size`, transmitted `symb`, `used`
/// flag, and the intra direction used to index the intra CDF.
#[derive(Debug, PartialEq, Eq)]
pub struct ExtTxDeriv {
    pub set_type: i32,
    pub num: i32,
    pub eset: i32,
    pub square: i32,
    pub symb: i32,
    pub used: i32,
    pub intra_dir: i32,
}

/// Compute the ext-tx derivation for `av1_write_tx_type`.
#[allow(clippy::too_many_arguments)]
pub fn ext_tx_derive(
    tx_size: usize,
    is_inter: bool,
    reduced: bool,
    tx_type: usize,
    use_filter_intra: bool,
    fi_mode: usize,
    mode: usize,
) -> ExtTxDeriv {
    let st = ext_tx_set_type(tx_size, is_inter, reduced);
    ExtTxDeriv {
        set_type: st as i32,
        num: NUM_EXT_TX_SET[st],
        eset: EXT_TX_SET_INDEX[is_inter as usize][st],
        square: TXSIZE_SQR[tx_size] as i32,
        symb: EXT_TX_IND[st][tx_type],
        used: EXT_TX_USED[st][tx_type],
        intra_dir: if use_filter_intra {
            FIMODE_TO_INTRADIR[fi_mode]
        } else {
            mode as i32
        },
    }
}

/// `av1_write_tx_type` core: when the block's ext-tx set carries more than one
/// type (and the frame gate `signal_gate` passes — qindex>0, not skip/seg-skip),
/// emit the tx_type symbol into the appropriate CDF. `cdf` is the pre-selected
/// slot `intra_ext_tx_cdf[eset][square][intra_dir]` (intra) or
/// `inter_ext_tx_cdf[eset][square]` (inter), length ≥ num+1.
#[allow(clippy::too_many_arguments)]
pub fn write_tx_type(
    enc: &mut OdEcEnc,
    cdf: &mut [u16],
    tx_size: usize,
    is_inter: bool,
    reduced: bool,
    tx_type: usize,
    use_filter_intra: bool,
    fi_mode: usize,
    mode: usize,
    signal_gate: bool,
) {
    let d = ext_tx_derive(tx_size, is_inter, reduced, tx_type, use_filter_intra, fi_mode, mode);
    if d.num > 1 && signal_gate {
        write_symbol(enc, d.symb, &mut cdf[..d.num as usize + 1], d.num as usize);
    }
}
