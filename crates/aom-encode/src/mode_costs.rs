//! Intra mode-info signaling costs (the intra slices of `MODE_COSTS`,
//! av1/encoder/block.h): cost tables derived from the frame's CDFs by
//! `av1_fill_mode_rates` (rd.c), and `intra_mode_info_cost_y`
//! (intra_mode_search_utils.h) — the per-mode signaling rate the intra RD
//! search adds on top of the coefficient rate. Bit-exact ports of libaom
//! v3.14.1.

use aom_entropy::partition::{is_directional_mode, use_angle_delta};
use aom_txb::cost_tokens_from_cdf;

/// `KF_MODE_CONTEXTS` (entropymode.h): KEY-frame Y-mode neighbour contexts.
pub const KF_MODE_CONTEXTS: usize = 5;
/// `BLOCK_SIZE_GROUPS` (entropymode.h).
pub const BLOCK_SIZE_GROUPS: usize = 4;
/// `INTRA_MODES` (enums.h).
pub const INTRA_MODES: usize = 13;
/// `UV_INTRA_MODES` (enums.h) — includes `UV_CFL_PRED`.
pub const UV_INTRA_MODES: usize = 14;
/// `CFL_ALLOWED_TYPES` (enums.h).
pub const CFL_ALLOWED_TYPES: usize = 2;
/// `FILTER_INTRA_MODES` (enums.h).
pub const FILTER_INTRA_MODES: usize = 5;
/// `DIRECTIONAL_MODES` (enums.h).
pub const DIRECTIONAL_MODES: usize = 8;
/// `MAX_ANGLE_DELTA` (enums.h).
pub const MAX_ANGLE_DELTA: usize = 3;
/// `BLOCK_SIZES_ALL` (enums.h).
pub const BLOCK_SIZES_ALL: usize = 22;
/// `PALATTE_BSIZE_CTXS` (entropymode.h; libaom's spelling).
pub const PALETTE_BSIZE_CTXS: usize = 7;
/// `PALETTE_Y_MODE_CONTEXTS` (entropymode.h).
pub const PALETTE_Y_MODE_CONTEXTS: usize = 3;

/// `DC_PRED` / `V_PRED` (enums.h).
const DC_PRED: usize = 0;
const V_PRED: usize = 1;

/// The intra slices of `MODE_COSTS` (block.h): per-symbol signaling rates in
/// `1<<9`-per-bit units, filled from the frame CDFs by
/// [`fill_intra_mode_costs`]. `y_mode_costs` holds only the
/// `[KF_MODE_CONTEXTS][KF_MODE_CONTEXTS]` slice the fill writes (the C
/// declares `[INTRA_MODES][INTRA_MODES][..]` but only ever fills/reads the
/// first 5x5 contexts). Rows gated off (filter-intra-ineligible block sizes)
/// stay zero.
pub struct IntraModeCosts {
    /// KEY-frame Y mode: `[above_ctx][left_ctx][mode]`, contexts from
    /// `intra_mode_context[above/left mode]`.
    pub y_mode_costs: [[[i32; INTRA_MODES]; KF_MODE_CONTEXTS]; KF_MODE_CONTEXTS],
    /// Non-KEY intra Y mode: `[size_group][mode]`.
    pub mbmode_cost: [[i32; INTRA_MODES]; BLOCK_SIZE_GROUPS],
    /// UV mode: `[cfl_allowed][y_mode][uv_mode]`.
    pub intra_uv_mode_cost: [[[i32; UV_INTRA_MODES]; INTRA_MODES]; CFL_ALLOWED_TYPES],
    /// Filter-intra mode (given the flag): `[filter_intra_mode]`.
    pub filter_intra_mode_cost: [i32; FILTER_INTRA_MODES],
    /// Filter-intra flag: `[bsize][use_filter_intra]`.
    pub filter_intra_cost: [[i32; 2]; BLOCK_SIZES_ALL],
    /// Angle delta: `[dir_mode - V_PRED][MAX_ANGLE_DELTA + delta]`.
    pub angle_delta_cost: [[i32; 2 * MAX_ANGLE_DELTA + 1]; DIRECTIONAL_MODES],
    /// Intrabc flag: `[use_intrabc]`.
    pub intrabc_cost: [i32; 2],
    /// Palette-Y flag: `[bsize_ctx][mode_ctx][use_palette]`.
    pub palette_y_mode_cost: [[[i32; 2]; PALETTE_Y_MODE_CONTEXTS]; PALETTE_BSIZE_CTXS],
    /// Palette-UV flag: `[y_palette_active][use_palette]`
    /// (`PALETTE_UV_MODE_CONTEXTS` = 2; filled by [`fill_palette_uv_mode_costs`]).
    pub palette_uv_mode_cost: [[i32; 2]; 2],
}

impl IntraModeCosts {
    /// All-zero tables (filled by [`fill_intra_mode_costs`]).
    pub fn zeroed() -> Box<Self> {
        Box::new(Self {
            y_mode_costs: [[[0; INTRA_MODES]; KF_MODE_CONTEXTS]; KF_MODE_CONTEXTS],
            mbmode_cost: [[0; INTRA_MODES]; BLOCK_SIZE_GROUPS],
            intra_uv_mode_cost: [[[0; UV_INTRA_MODES]; INTRA_MODES]; CFL_ALLOWED_TYPES],
            filter_intra_mode_cost: [0; FILTER_INTRA_MODES],
            filter_intra_cost: [[0; 2]; BLOCK_SIZES_ALL],
            angle_delta_cost: [[0; 2 * MAX_ANGLE_DELTA + 1]; DIRECTIONAL_MODES],
            intrabc_cost: [0; 2],
            palette_y_mode_cost: [[[0; 2]; PALETTE_Y_MODE_CONTEXTS]; PALETTE_BSIZE_CTXS],
            palette_uv_mode_cost: [[0; 2]; 2],
        })
    }
}

/// The palette-UV-flag slice of `av1_fill_mode_rates` (rd.c):
/// `av1_cost_tokens_from_cdf(palette_uv_mode_cost[i], palette_uv_mode_cdf[i])`
/// for the 2 y-palette-active contexts. `palette_uv_mode_cdf` is flat
/// `[2][3]` (2 symbols + padding).
pub fn fill_palette_uv_mode_costs(costs: &mut IntraModeCosts, palette_uv_mode_cdf: &[u16]) {
    assert_eq!(palette_uv_mode_cdf.len(), 2 * 3);
    for i in 0..2 {
        cost_tokens_from_cdf(
            &mut costs.palette_uv_mode_cost[i],
            &palette_uv_mode_cdf[i * 3..i * 3 + 3],
            None,
        );
    }
}

/// `av1_filter_intra_allowed_bsize` (reconintra.h): filter intra is available
/// when enabled in the sequence header and the block is at most 32x32.
pub fn filter_intra_allowed_bsize(enable_filter_intra: bool, bsize: usize) -> bool {
    if !enable_filter_intra {
        return false;
    }
    crate::BLK_W[bsize] <= 32 && crate::BLK_H[bsize] <= 32
}

/// Bit-exact port of the intra-mode slices of `av1_fill_mode_rates` (rd.c).
/// Every CDF input is flat with `nsymbs+1`-padded rows in the
/// `FRAME_CONTEXT` layouts: `kf_y_cdf [5][5][14]`, `y_mode_cdf [4][14]`,
/// `uv_mode_cdf [2][13][15]` (13 symbols when cfl is not allowed, 14 when it
/// is), `filter_intra_mode_cdf [6]`, `filter_intra_cdfs [22][3]`,
/// `palette_y_mode_cdf [7][3][3]`, `angle_delta_cdf [8][8]`,
/// `intrabc_cdf [3]`. `filter_intra_cost` rows are filled only for
/// filter-intra-eligible block sizes (matching the C's
/// `av1_filter_intra_allowed_bsize` gate).
#[allow(clippy::too_many_arguments)]
pub fn fill_intra_mode_costs(
    costs: &mut IntraModeCosts,
    kf_y_cdf: &[u16],
    y_mode_cdf: &[u16],
    uv_mode_cdf: &[u16],
    filter_intra_mode_cdf: &[u16],
    filter_intra_cdfs: &[u16],
    palette_y_mode_cdf: &[u16],
    angle_delta_cdf: &[u16],
    intrabc_cdf: &[u16],
    enable_filter_intra: bool,
) {
    assert_eq!(kf_y_cdf.len(), KF_MODE_CONTEXTS * KF_MODE_CONTEXTS * (INTRA_MODES + 1));
    assert_eq!(y_mode_cdf.len(), BLOCK_SIZE_GROUPS * (INTRA_MODES + 1));
    assert_eq!(uv_mode_cdf.len(), CFL_ALLOWED_TYPES * INTRA_MODES * (UV_INTRA_MODES + 1));
    assert_eq!(filter_intra_mode_cdf.len(), FILTER_INTRA_MODES + 1);
    assert_eq!(filter_intra_cdfs.len(), BLOCK_SIZES_ALL * 3);
    assert_eq!(palette_y_mode_cdf.len(), PALETTE_BSIZE_CTXS * PALETTE_Y_MODE_CONTEXTS * 3);
    assert_eq!(angle_delta_cdf.len(), DIRECTIONAL_MODES * (2 * MAX_ANGLE_DELTA + 2));
    assert_eq!(intrabc_cdf.len(), 3);

    for i in 0..KF_MODE_CONTEXTS {
        for j in 0..KF_MODE_CONTEXTS {
            let off = (i * KF_MODE_CONTEXTS + j) * (INTRA_MODES + 1);
            cost_tokens_from_cdf(
                &mut costs.y_mode_costs[i][j],
                &kf_y_cdf[off..off + INTRA_MODES + 1],
                None,
            );
        }
    }

    for i in 0..BLOCK_SIZE_GROUPS {
        let off = i * (INTRA_MODES + 1);
        cost_tokens_from_cdf(
            &mut costs.mbmode_cost[i],
            &y_mode_cdf[off..off + INTRA_MODES + 1],
            None,
        );
    }
    for i in 0..CFL_ALLOWED_TYPES {
        for j in 0..INTRA_MODES {
            let off = (i * INTRA_MODES + j) * (UV_INTRA_MODES + 1);
            cost_tokens_from_cdf(
                &mut costs.intra_uv_mode_cost[i][j],
                &uv_mode_cdf[off..off + UV_INTRA_MODES + 1],
                None,
            );
        }
    }

    cost_tokens_from_cdf(&mut costs.filter_intra_mode_cost, filter_intra_mode_cdf, None);
    for i in 0..BLOCK_SIZES_ALL {
        if filter_intra_allowed_bsize(enable_filter_intra, i) {
            cost_tokens_from_cdf(
                &mut costs.filter_intra_cost[i],
                &filter_intra_cdfs[i * 3..i * 3 + 3],
                None,
            );
        }
    }

    for i in 0..PALETTE_BSIZE_CTXS {
        for j in 0..PALETTE_Y_MODE_CONTEXTS {
            let off = (i * PALETTE_Y_MODE_CONTEXTS + j) * 3;
            cost_tokens_from_cdf(
                &mut costs.palette_y_mode_cost[i][j],
                &palette_y_mode_cdf[off..off + 3],
                None,
            );
        }
    }

    for i in 0..DIRECTIONAL_MODES {
        let off = i * (2 * MAX_ANGLE_DELTA + 2);
        cost_tokens_from_cdf(
            &mut costs.angle_delta_cost[i],
            &angle_delta_cdf[off..off + 2 * MAX_ANGLE_DELTA + 2],
            None,
        );
    }
    cost_tokens_from_cdf(&mut costs.intrabc_cost, intrabc_cdf, None);
}

/// Bit-exact port of `intra_mode_info_cost_y`
/// (av1/encoder/intra_mode_search_utils.h) for the **`palette_size[0] == 0`**
/// path: the Y mode-info signaling rate = `mode_cost` (the caller-selected
/// `y_mode_costs[above_ctx][left_ctx][mode]` / `mbmode_cost[group][mode]`
/// entry) + the no-palette flag bit (when palette is allowed and the mode is
/// `DC_PRED`) + the filter-intra flag/mode + the Y angle delta + the intrabc
/// flag. The palette-USE branch (size/color/map rate) is out of scope here —
/// it belongs to the palette search.
///
/// At most one of `mode != DC_PRED`, `use_intrabc`, `use_filter_intra` may
/// hold (the C asserts this; mirrored here).
#[allow(clippy::too_many_arguments)]
pub fn intra_mode_info_cost_y(
    costs: &IntraModeCosts,
    mode_cost: i32,
    mode: usize,
    bsize: usize,
    angle_delta_y: i32,
    use_filter_intra: bool,
    filter_intra_mode: usize,
    use_intrabc: bool,
    try_palette: bool,
    palette_bsize_ctx: usize,
    palette_mode_ctx: usize,
    enable_filter_intra: bool,
    allow_intrabc: bool,
) -> i32 {
    let mut total_rate = mode_cost;
    let use_palette = 0usize; // scope: palette_size[0] == 0
    // Can only activate one mode.
    assert!(
        usize::from(mode != DC_PRED)
            + use_palette
            + usize::from(use_intrabc)
            + usize::from(use_filter_intra)
            <= 1
    );
    if try_palette && mode == DC_PRED {
        total_rate += costs.palette_y_mode_cost[palette_bsize_ctx][palette_mode_ctx][use_palette];
    }
    // av1_filter_intra_allowed(cm, mbmi), with palette_size[0] == 0.
    if mode == DC_PRED && filter_intra_allowed_bsize(enable_filter_intra, bsize) {
        total_rate += costs.filter_intra_cost[bsize][usize::from(use_filter_intra)];
        if use_filter_intra {
            total_rate += costs.filter_intra_mode_cost[filter_intra_mode];
        }
    }
    if is_directional_mode(mode as i32) && use_angle_delta(bsize) {
        total_rate += costs.angle_delta_cost[mode - V_PRED]
            [(MAX_ANGLE_DELTA as i32 + angle_delta_y) as usize];
    }
    if allow_intrabc {
        total_rate += costs.intrabc_cost[usize::from(use_intrabc)];
    }
    total_rate
}

// ---------------------------------------------------------------------------
// CfL alpha signaling cost (ModeCosts.cfl_cost) + intra_mode_info_cost_uv
// ---------------------------------------------------------------------------

/// `CFL_JOINT_SIGNS` (enums.h): `CFL_SIGNS * CFL_SIGNS - 1`.
pub const CFL_JOINT_SIGNS: usize = 8;
/// `CFL_PRED_PLANES` (enums.h): U and V.
pub const CFL_PRED_PLANES: usize = 2;
/// `CFL_ALPHABET_SIZE` (enums.h): 16 coded alpha magnitudes.
pub const CFL_ALPHABET_SIZE: usize = 16;
/// `CFL_ALPHA_CONTEXTS` (enums.h): `CFL_JOINT_SIGNS + 1 - CFL_SIGNS`.
pub const CFL_ALPHA_CONTEXTS: usize = 6;

/// `CFL_SIGN_U(js)` (enums.h): `((js + 1) * 11) >> 5`.
#[inline]
pub fn cfl_sign_u(js: usize) -> usize {
    ((js + 1) * 11) >> 5
}
/// `CFL_SIGN_V(js)` (enums.h): `(js + 1) - CFL_SIGNS * CFL_SIGN_U(js)`.
#[inline]
pub fn cfl_sign_v(js: usize) -> usize {
    (js + 1) - 3 * cfl_sign_u(js)
}
/// `CFL_CONTEXT_U(js)` (enums.h): `js + 1 - CFL_SIGNS`.
#[inline]
pub fn cfl_context_u(js: usize) -> usize {
    js + 1 - 3
}
/// `CFL_CONTEXT_V(js)` (enums.h): `CFL_SIGN_V(js) * CFL_SIGNS + CFL_SIGN_U(js) - CFL_SIGNS`.
#[inline]
pub fn cfl_context_v(js: usize) -> usize {
    cfl_sign_v(js) * 3 + cfl_sign_u(js) - 3
}

/// `ModeCosts.cfl_cost[CFL_JOINT_SIGNS][CFL_PRED_PLANES][CFL_ALPHABET_SIZE]`
/// (block.h): the CfL alpha-magnitude signaling rate per joint sign and
/// plane, with the joint-sign symbol cost folded into every U-plane entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CflCosts(pub [[[i32; CFL_ALPHABET_SIZE]; CFL_PRED_PLANES]; CFL_JOINT_SIGNS]);

impl CflCosts {
    /// All-zero table, filled by [`fill_cfl_costs`].
    pub fn zeroed() -> Self {
        CflCosts([[[0; CFL_ALPHABET_SIZE]; CFL_PRED_PLANES]; CFL_JOINT_SIGNS])
    }
}

/// The CfL slice of `av1_fill_mode_rates` (rd.c 154-172): per joint sign,
/// alpha-magnitude costs from `cfl_alpha_cdf[CFL_CONTEXT_U/V(js)]` (all-zero
/// rows when that plane's sign is `CFL_SIGN_ZERO`), then the joint-sign
/// symbol cost (`cfl_sign_cdf`, 8 symbols) added into every U entry.
/// `cfl_sign_cdf` is one padded row (`8+1`); `cfl_alpha_cdf` is flat
/// `[CFL_ALPHA_CONTEXTS][CFL_ALPHABET_SIZE + 1]`.
pub fn fill_cfl_costs(out: &mut CflCosts, cfl_sign_cdf: &[u16], cfl_alpha_cdf: &[u16]) {
    assert_eq!(cfl_sign_cdf.len(), CFL_JOINT_SIGNS + 1);
    assert_eq!(cfl_alpha_cdf.len(), CFL_ALPHA_CONTEXTS * (CFL_ALPHABET_SIZE + 1));
    let mut sign_cost = [0i32; CFL_JOINT_SIGNS];
    cost_tokens_from_cdf(&mut sign_cost, cfl_sign_cdf, None);
    #[allow(clippy::needless_range_loop)] // js drives the CFL_* context macros
    for js in 0..CFL_JOINT_SIGNS {
        // CFL_SIGN_ZERO == 0.
        if cfl_sign_u(js) == 0 {
            out.0[js][0] = [0; CFL_ALPHABET_SIZE];
        } else {
            let off = cfl_context_u(js) * (CFL_ALPHABET_SIZE + 1);
            let row = &cfl_alpha_cdf[off..off + CFL_ALPHABET_SIZE + 1];
            let mut cost = [0i32; CFL_ALPHABET_SIZE];
            cost_tokens_from_cdf(&mut cost, row, None);
            out.0[js][0] = cost;
        }
        if cfl_sign_v(js) == 0 {
            out.0[js][1] = [0; CFL_ALPHABET_SIZE];
        } else {
            let off = cfl_context_v(js) * (CFL_ALPHABET_SIZE + 1);
            let row = &cfl_alpha_cdf[off..off + CFL_ALPHABET_SIZE + 1];
            let mut cost = [0i32; CFL_ALPHABET_SIZE];
            cost_tokens_from_cdf(&mut cost, row, None);
            out.0[js][1] = cost;
        }
        for u in 0..CFL_ALPHABET_SIZE {
            out.0[js][0][u] += sign_cost[js];
        }
    }
}

/// `UV_DC_PRED` (enums.h).
const UV_DC_PRED: usize = 0;

/// Bit-exact port of `intra_mode_info_cost_uv`
/// (av1/encoder/intra_mode_search_utils.h) for the **`palette_size[1] == 0`**
/// path: the UV mode-info signaling rate = `mode_cost` (the caller-selected
/// `intra_uv_mode_cost[cfl_allowed][y_mode][uv_mode]` entry) + the
/// no-uv-palette flag bit (when palette is allowed and the mode is
/// `UV_DC_PRED`; context = whether the Y palette is active) + the UV angle
/// delta for directional modes on angle-eligible block sizes. The
/// palette-USE branch (size/color/map rate) is out of scope — it belongs to
/// the UV palette search. `use_intrabc == 0` in the intra-frame UV RD context
/// (the C asserts at most one of {non-DC uv_mode, palette, intrabc}).
pub fn intra_mode_info_cost_uv(
    costs: &IntraModeCosts,
    mode_cost: i32,
    uv_mode: usize,
    bsize: usize,
    angle_delta_uv: i32,
    try_palette: bool,
    y_palette_active: bool,
) -> i32 {
    let mut total_rate = mode_cost;
    let use_palette = 0usize; // scope: palette_size[1] == 0
    assert!(usize::from(uv_mode != UV_DC_PRED) + use_palette <= 1);
    if try_palette && uv_mode == UV_DC_PRED {
        total_rate +=
            costs.palette_uv_mode_cost[usize::from(y_palette_active)][use_palette];
    }
    let intra_mode = aom_entropy::partition::get_uv_mode(uv_mode);
    if is_directional_mode(intra_mode) && use_angle_delta(bsize) {
        total_rate += costs.angle_delta_cost[(intra_mode - 1) as usize]
            [(angle_delta_uv + MAX_ANGLE_DELTA as i32) as usize];
    }
    total_rate
}

// ---------------------------------------------------------------------------
// tx-size signaling cost (ModeCosts.tx_size_cost + tx_search.h tx_size_cost)
// ---------------------------------------------------------------------------

/// `MAX_TX_CATS` (blockd.h) x `TX_SIZE_CONTEXTS` (enums.h) x
/// `MAX_TX_DEPTH + 1` — the tx-size depth signaling costs
/// (`mode_costs->tx_size_cost`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxSizeCosts(pub [[[i32; 3]; 3]; 4]);

impl TxSizeCosts {
    /// All-zero table, filled by [`fill_tx_size_costs`].
    pub fn zeroed() -> Self {
        TxSizeCosts([[[0; 3]; 3]; 4])
    }
}

/// The tx-size slice of `av1_fill_mode_rates` (rd.c 175-178):
/// `av1_cost_tokens_from_cdf(tx_size_cost[cat][ctx], tx_size_cdf[cat][ctx])`.
/// `tx_size_cdf` is flat `[MAX_TX_CATS=4][TX_SIZE_CONTEXTS=3][4]` (row padding
/// 4; category 0 has 2 symbols, the rest 3 — the CDF terminator decides).
pub fn fill_tx_size_costs(out: &mut TxSizeCosts, tx_size_cdf: &[u16]) {
    assert_eq!(tx_size_cdf.len(), 4 * 3 * 4);
    for cat in 0..4 {
        for ctx in 0..3 {
            let row = &tx_size_cdf[(cat * 3 + ctx) * 4..(cat * 3 + ctx) * 4 + 4];
            aom_txb::cost_tokens_from_cdf(&mut out.0[cat][ctx], row, None);
        }
    }
}

/// `block_signals_txsize` (blockd.h): every block above 4x4 codes its tx size.
#[inline]
pub fn block_signals_txsize(bsize: usize) -> bool {
    bsize > 0 // BLOCK_4X4
}

/// `tx_size_cost` (av1/encoder/tx_search.h): the tx-size signaling rate for a
/// block — `tx_size_cost[bsize_to_tx_size_cat(bsize)][ctx][tx_size_to_depth]`
/// under `TX_MODE_SELECT` on signaling blocks, else 0. `tx_size_ctx` is
/// `get_tx_size_context(xd)` (the neighbour facade, supplied by the caller —
/// same deferral as the aom-entropy tx-size writer).
pub fn tx_size_cost(
    costs: &TxSizeCosts,
    tx_mode_is_select: bool,
    bsize: usize,
    tx_size: usize,
    tx_size_ctx: usize,
) -> i32 {
    if !tx_mode_is_select || !block_signals_txsize(bsize) {
        return 0;
    }
    let cat = aom_entropy::partition::bsize_to_tx_size_cat(bsize) as usize;
    let depth = aom_entropy::partition::tx_size_to_depth(tx_size, bsize) as usize;
    costs.0[cat][tx_size_ctx][depth]
}
