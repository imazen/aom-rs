//! Extended-transform set derivation + `av1_write_tx_type` (libaom
//! `av1/common/blockd.h`, `av1/common/entropymode.h`,
//! `av1/encoder/bitstream.c`): select and signal a block's transform type — the
//! plane-0 step the coefficient writer/cost functions deliberately leave out.
//!
//! The derivation (set type / eset / symbol index / arity) is transcribed from
//! libaom's tables and verified exhaustively vs C. The symbol emission reuses
//! the bit-exact `aom_write_symbol` (aom-entropy).

use crate::cost_tokens_from_cdf;
use aom_entropy::cdf::{read_symbol, write_symbol};
use aom_entropy::dec::OdEcDec;
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

/// Invert `EXT_TX_IND[st]`: the transmitted symbol back to its `TX_TYPE`. Within a
/// set the used tx_types map bijectively onto `[0, num)`, so the unique used
/// tx_type with `EXT_TX_IND[st][t] == symb` is the answer.
fn ext_tx_inv(st: usize, symb: i32) -> usize {
    for t in 0..16 {
        if EXT_TX_USED[st][t] == 1 && EXT_TX_IND[st][t] == symb {
            return t;
        }
    }
    0 // DCT_DCT — unreachable for a valid symbol
}

/// `av1_read_tx_type` core — inverse of [`write_tx_type`]. When the block's ext-tx
/// set carries more than one type and `signal_gate` passes, read the symbol on the
/// pre-selected CDF slot and map it back to the `TX_TYPE`; otherwise the type is
/// the inferred `DCT_DCT` (0). `cdf` is the same caller-selected slot the writer
/// used (`intra_ext_tx_cdf[eset][square][intra_dir]` or `inter_ext_tx_cdf[eset][square]`).
pub fn read_tx_type(
    dec: &mut OdEcDec,
    cdf: &mut [u16],
    tx_size: usize,
    is_inter: bool,
    reduced: bool,
    signal_gate: bool,
) -> usize {
    let st = ext_tx_set_type(tx_size, is_inter, reduced);
    let num = NUM_EXT_TX_SET[st];
    if num > 1 && signal_gate {
        let symb = read_symbol(dec, &mut cdf[..num as usize + 1], num as usize);
        ext_tx_inv(st, symb)
    } else {
        0 // DCT_DCT
    }
}

// ---- tx-type signaling cost (RD rate) ---------------------------------------

/// `EXT_TX_SIZES` (enums.h): number of square tx sizes using extended transforms.
pub const EXT_TX_SIZES: usize = 4;
/// `EXT_TX_SETS_INTRA` (enums.h).
pub const EXT_TX_SETS_INTRA: usize = 3;
/// `EXT_TX_SETS_INTER` (enums.h).
pub const EXT_TX_SETS_INTER: usize = 4;
/// `TX_TYPES` (enums.h).
pub const TX_TYPES: usize = 16;
/// `INTRA_MODES` (enums.h).
pub const INTRA_MODES: usize = 13;

/// `use_intra_ext_tx_for_txsize[EXT_TX_SETS_INTRA][EXT_TX_SIZES]` (rd.c):
/// which (cdf set, square tx size) intra combos get cost tables filled.
const USE_INTRA_EXT_TX_FOR_TXSIZE: [[i32; EXT_TX_SIZES]; EXT_TX_SETS_INTRA] = [
    [1, 1, 1, 1], // unused
    [1, 1, 0, 0],
    [0, 0, 1, 0],
];

/// `use_inter_ext_tx_for_txsize[EXT_TX_SETS_INTER][EXT_TX_SIZES]` (rd.c).
const USE_INTER_EXT_TX_FOR_TXSIZE: [[i32; EXT_TX_SIZES]; EXT_TX_SETS_INTER] = [
    [1, 1, 1, 1], // unused
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 1, 1],
];

/// `av1_ext_tx_set_idx_to_type[2][max(EXT_TX_SETS_INTRA, EXT_TX_SETS_INTER)]`
/// (rd.c): CDF set index -> TxSetType. Intra: DCTONLY, DTT4_IDTX_1DDCT,
/// DTT4_IDTX; inter: DCTONLY, ALL16, DTT9_IDTX_1DDCT, DCT_IDTX.
const EXT_TX_SET_IDX_TO_TYPE: [[usize; 4]; 2] = [[0, 3, 2, 0], [0, 5, 4, 1]];

/// `av1_ext_tx_inv[EXT_TX_SET_TYPES][TX_TYPES]` (entropymode.h) — transmitted
/// symbol -> TX_TYPE (the `inv_map` for cost-table fill).
#[rustfmt::skip]
const AV1_EXT_TX_INV: [[i32; 16]; 6] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 0, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 0, 10, 11, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 10, 11, 0, 1, 2, 4, 5, 3, 6, 7, 8, 0, 0, 0, 0],
    [9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 4, 5, 3, 6, 7, 8],
];

/// The tx-type slice of `MODE_COSTS` (block.h): per-symbol signaling rates,
/// indexed `[cdf set][square tx size][(intra dir)][tx_type]`. Entries for
/// combos gated off by `use_*_ext_tx_for_txsize` stay zero.
pub struct TxTypeCosts {
    pub intra: [[[[i32; TX_TYPES]; INTRA_MODES]; EXT_TX_SIZES]; EXT_TX_SETS_INTRA],
    pub inter: [[[i32; TX_TYPES]; EXT_TX_SIZES]; EXT_TX_SETS_INTER],
}

impl TxTypeCosts {
    /// All-zero tables (filled by [`fill_tx_type_costs`]).
    pub fn zeroed() -> Box<Self> {
        Box::new(Self {
            intra: [[[[0; TX_TYPES]; INTRA_MODES]; EXT_TX_SIZES]; EXT_TX_SETS_INTRA],
            inter: [[[0; TX_TYPES]; EXT_TX_SIZES]; EXT_TX_SETS_INTER],
        })
    }
}

/// Bit-exact port of the tx-type slice of `av1_fill_mode_rates` (rd.c): fill
/// [`TxTypeCosts`] from the frame's ext-tx CDFs via [`cost_tokens_from_cdf`]
/// with the `av1_ext_tx_inv` symbol->tx_type map.
///
/// `intra_cdf` is flat `[EXT_TX_SETS_INTRA][EXT_TX_SIZES][INTRA_MODES]
/// [TX_TYPES+1]` (matching `FRAME_CONTEXT::intra_ext_tx_cdf`; each row is an
/// inverse-CDF terminated at its set's symbol count); `inter_cdf` is flat
/// `[EXT_TX_SETS_INTER][EXT_TX_SIZES][TX_TYPES+1]`.
pub fn fill_tx_type_costs(costs: &mut TxTypeCosts, intra_cdf: &[u16], inter_cdf: &[u16]) {
    assert_eq!(
        intra_cdf.len(),
        EXT_TX_SETS_INTRA * EXT_TX_SIZES * INTRA_MODES * (TX_TYPES + 1)
    );
    assert_eq!(inter_cdf.len(), EXT_TX_SETS_INTER * EXT_TX_SIZES * (TX_TYPES + 1));
    for i in 0..EXT_TX_SIZES {
        // TX_4X4 == 0
        for s in 1..EXT_TX_SETS_INTER {
            if USE_INTER_EXT_TX_FOR_TXSIZE[s][i] != 0 {
                let off = (s * EXT_TX_SIZES + i) * (TX_TYPES + 1);
                cost_tokens_from_cdf(
                    &mut costs.inter[s][i],
                    &inter_cdf[off..off + TX_TYPES + 1],
                    Some(&AV1_EXT_TX_INV[EXT_TX_SET_IDX_TO_TYPE[1][s]]),
                );
            }
        }
        for s in 1..EXT_TX_SETS_INTRA {
            if USE_INTRA_EXT_TX_FOR_TXSIZE[s][i] != 0 {
                for j in 0..INTRA_MODES {
                    let off = ((s * EXT_TX_SIZES + i) * INTRA_MODES + j) * (TX_TYPES + 1);
                    cost_tokens_from_cdf(
                        &mut costs.intra[s][i][j],
                        &intra_cdf[off..off + TX_TYPES + 1],
                        Some(&AV1_EXT_TX_INV[EXT_TX_SET_IDX_TO_TYPE[0][s]]),
                    );
                }
            }
        }
    }
}

/// Bit-exact port of `get_tx_type_cost` (av1/encoder/txb_rdopt.c): the
/// tx_type signaling rate for one txb. Zero for chroma planes, DCT-only sets,
/// lossless segments, and set index 0. `lossless` is
/// `xd->lossless[mbmi->segment_id]`; `mode`/`filter_intra_mode` select the
/// intra CDF direction (`fimode_to_intradir` when filter-intra is used).
#[allow(clippy::too_many_arguments)]
pub fn get_tx_type_cost(
    costs: &TxTypeCosts,
    plane: usize,
    tx_size: usize,
    tx_type: usize,
    is_inter: bool,
    reduced_tx_set_used: bool,
    lossless: bool,
    use_filter_intra: bool,
    filter_intra_mode: usize,
    mode: usize,
) -> i32 {
    if plane > 0 {
        return 0;
    }

    let square_tx_size = TXSIZE_SQR[tx_size];

    let set_type = ext_tx_set_type(tx_size, is_inter, reduced_tx_set_used);
    if NUM_EXT_TX_SET[set_type] > 1 && !lossless {
        let ext_tx_set = EXT_TX_SET_INDEX[is_inter as usize][set_type];
        if is_inter {
            if ext_tx_set > 0 {
                return costs.inter[ext_tx_set as usize][square_tx_size][tx_type];
            }
        } else if ext_tx_set > 0 {
            let intra_dir = if use_filter_intra {
                FIMODE_TO_INTRADIR[filter_intra_mode] as usize
            } else {
                mode
            };
            return costs.intra[ext_tx_set as usize][square_tx_size][intra_dir][tx_type];
        }
    }
    0
}
