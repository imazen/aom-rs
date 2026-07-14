//! Differential harness for the tx-type signaling cost path vs C libaom
//! v3.14.1: `fill_tx_type_costs` (the tx-type slice of `av1_fill_mode_rates`,
//! rd.c — transcription shim over the REAL exported `av1_cost_tokens_from_cdf`
//! and REAL `av1_ext_tx_inv`) and `get_tx_type_cost` (txb_rdopt.c static,
//! transcription shim over the REAL header-static set-derivation tables).

use aom_sys_ref as c;
use aom_txb::{
    fill_tx_type_costs, get_tx_type_cost, TxTypeCosts, EXT_TX_SETS_INTER, EXT_TX_SETS_INTRA,
    EXT_TX_SIZES, INTRA_MODES, TX_TYPES,
};

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
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo) as u64) as u32
    }
}

/// `av1_num_ext_tx_set[TxSetType]`.
const NUM_EXT_TX_SET: [usize; 6] = [1, 2, 5, 7, 12, 16];
/// `av1_ext_tx_set_idx_to_type[2][..]` (rd.c): cdf set index -> TxSetType.
const IDX_TO_TYPE: [[usize; 4]; 2] = [[0, 3, 2, 0], [0, 5, 4, 1]];

/// Valid `nsymbs`-symbol inverse-CDF row padded to `TX_TYPES + 1` entries:
/// strictly decreasing to the terminal 0 at `nsymbs - 1` (the walk in
/// `av1_cost_tokens_from_cdf` stops there), zeros beyond.
fn gen_cdf_row(rng: &mut Rng, nsymbs: usize) -> Vec<u16> {
    let mut row = vec![0u16; TX_TYPES + 1];
    let mut acc: u32 = 0;
    for e in row.iter_mut().take(nsymbs - 1) {
        acc += rng.range(1, (32000 / nsymbs as u32).max(2));
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    row[nsymbs - 1] = 0;
    row
}

/// Random full CDF arrays in the `FRAME_CONTEXT` flat layouts, each row valid
/// for its set's symbol count.
fn gen_cdfs(rng: &mut Rng) -> (Vec<u16>, Vec<u16>) {
    let mut intra = Vec::with_capacity(EXT_TX_SETS_INTRA * EXT_TX_SIZES * INTRA_MODES * (TX_TYPES + 1));
    for s in 0..EXT_TX_SETS_INTRA {
        let nsymbs = NUM_EXT_TX_SET[IDX_TO_TYPE[0][s]].max(2);
        for _ in 0..EXT_TX_SIZES * INTRA_MODES {
            intra.extend_from_slice(&gen_cdf_row(rng, nsymbs));
        }
    }
    let mut inter = Vec::with_capacity(EXT_TX_SETS_INTER * EXT_TX_SIZES * (TX_TYPES + 1));
    for s in 0..EXT_TX_SETS_INTER {
        let nsymbs = NUM_EXT_TX_SET[IDX_TO_TYPE[1][s]].max(2);
        for _ in 0..EXT_TX_SIZES {
            inter.extend_from_slice(&gen_cdf_row(rng, nsymbs));
        }
    }
    (intra, inter)
}

/// Flatten the Rust tables in the shim's layout for comparison.
fn flatten(costs: &TxTypeCosts) -> (Vec<i32>, Vec<i32>) {
    let mut intra = Vec::with_capacity(c::TX_TYPE_COSTS_INTRA_LEN);
    for s in &costs.intra {
        for i in s {
            for j in i {
                intra.extend_from_slice(j);
            }
        }
    }
    let mut inter = Vec::with_capacity(c::TX_TYPE_COSTS_INTER_LEN);
    for s in &costs.inter {
        for i in s {
            inter.extend_from_slice(i);
        }
    }
    (intra, inter)
}

/// Cost-table fill matches C over random CDFs (both sides start zeroed, so the
/// `use_*_ext_tx_for_txsize` gating must also agree exactly).
#[test]
fn fill_tx_type_costs_matches_c() {
    let mut rng = Rng(0x77c0_57f1_11ab_cde5);
    for trial in 0..2000 {
        let (intra_cdf, inter_cdf) = gen_cdfs(&mut rng);
        let (c_intra, c_inter) = c::ref_fill_tx_type_costs(&intra_cdf, &inter_cdf);
        let mut costs = TxTypeCosts::zeroed();
        fill_tx_type_costs(&mut costs, &intra_cdf, &inter_cdf);
        let (r_intra, r_inter) = flatten(&costs);
        assert_eq!(r_intra, c_intra, "intra tables trial={trial}");
        assert_eq!(r_inter, c_inter, "inter tables trial={trial}");
    }
}

/// Lookup matches C across the full argument grid, on multiple random cost
/// tables: every tx_size x tx_type x inter/intra x reduced x lossless x
/// filter-intra combination, planes 0..3, all intra dirs.
#[test]
fn get_tx_type_cost_matches_c() {
    let mut rng = Rng(0x6e7c_057c_0575_ca1e);
    let mut coverage_nonzero = 0u64;
    for _ in 0..8 {
        let (intra_cdf, inter_cdf) = gen_cdfs(&mut rng);
        let (c_intra, c_inter) = c::ref_fill_tx_type_costs(&intra_cdf, &inter_cdf);
        let mut costs = TxTypeCosts::zeroed();
        fill_tx_type_costs(&mut costs, &intra_cdf, &inter_cdf);

        for plane in 0..3usize {
            for tx_size in 0..19usize {
                for tx_type in 0..TX_TYPES {
                    for is_inter in [false, true] {
                        for reduced in [false, true] {
                            for lossless in [false, true] {
                                // !filter_intra: all 13 modes; filter_intra: all 5 fi modes.
                                for (use_fi, dir_n) in [(false, INTRA_MODES), (true, 5)] {
                                    for dir in 0..dir_n {
                                        let (fi_mode, mode) =
                                            if use_fi { (dir, 0) } else { (0, dir) };
                                        let want = c::ref_get_tx_type_cost(
                                            &c_intra,
                                            &c_inter,
                                            plane as i32,
                                            tx_size as i32,
                                            tx_type as i32,
                                            is_inter,
                                            reduced,
                                            lossless,
                                            use_fi,
                                            fi_mode as i32,
                                            mode as i32,
                                        );
                                        let got = get_tx_type_cost(
                                            &costs, plane, tx_size, tx_type, is_inter,
                                            reduced, lossless, use_fi, fi_mode, mode,
                                        );
                                        assert_eq!(
                                            got, want,
                                            "plane={plane} tx_size={tx_size} tx_type={tx_type} \
                                             is_inter={is_inter} reduced={reduced} \
                                             lossless={lossless} use_fi={use_fi} dir={dir}"
                                        );
                                        if got != 0 {
                                            coverage_nonzero += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Guard: the sweep must exercise real table lookups, not just the
    // zero-returning gates. The count is deterministic (every filled-and-used
    // entry costs >= 3): per trial, summing used tx_types over reachable
    // (tx_size, is_inter, reduced) cells whose (eset, sqr) the fill gating
    // populates, x 18 dirs x lossless=false = 4212. (Reduced-set lookups into
    // cells `use_*_ext_tx_for_txsize` leaves unfilled — intra eset 2 at sqr
    // 0/1, inter eset 3 at sqr 0 — read the zero-init value on BOTH sides and
    // count as zero here; that mirrors libaom, where fill and lookup gating
    // genuinely differ for reduced_tx_set.)
    assert_eq!(coverage_nonzero, 8 * 4212, "non-zero lookup count changed");
}
