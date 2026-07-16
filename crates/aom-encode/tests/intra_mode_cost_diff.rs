//! Differential harness for the intra mode-info signaling costs vs C libaom
//! v3.14.1: `fill_intra_mode_costs` (the intra slices of `av1_fill_mode_rates`,
//! rd.c — transcription shim over the REAL exported `av1_cost_tokens_from_cdf`
//! and REAL `av1_filter_intra_allowed_bsize`) and `intra_mode_info_cost_y`
//! (intra_mode_search_utils.h static inline, `palette_size[0]==0` path — shim
//! transcription over the REAL `av1_is_directional_mode` /
//! `av1_use_angle_delta` / `av1_filter_intra_allowed_bsize` gates, with the
//! C's exclusive-mode-flag assert LIVE).

use aom_encode::mode_costs::{
    BLOCK_SIZE_GROUPS, BLOCK_SIZES_ALL, DIRECTIONAL_MODES, FILTER_INTRA_MODES, INTRA_MODES,
    IntraModeCosts, KF_MODE_CONTEXTS, MAX_ANGLE_DELTA, PALETTE_BSIZE_CTXS, PALETTE_Y_MODE_CONTEXTS,
    UV_INTRA_MODES, fill_intra_mode_costs, intra_mode_info_cost_y,
};
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
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo) as u64) as u32
    }
}

/// Valid `nsymbs`-symbol inverse-CDF row padded to `padded` entries.
fn gen_cdf_row(rng: &mut Rng, nsymbs: usize, padded: usize) -> Vec<u16> {
    let mut row = vec![0u16; padded];
    let mut acc: u32 = 0;
    for e in row.iter_mut().take(nsymbs - 1) {
        acc += rng.range(1, (32000 / nsymbs as u32).max(2));
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    row[nsymbs - 1] = 0;
    row
}

/// `count` rows of `nsymbs`-symbol CDFs, each `nsymbs+1`-padded... except when
/// `padded` overrides (uv rows are 15 wide with 13 or 14 symbols).
fn gen_cdfs(rng: &mut Rng, count: usize, nsymbs: usize, padded: usize) -> Vec<u16> {
    let mut v = Vec::with_capacity(count * padded);
    for _ in 0..count {
        v.extend_from_slice(&gen_cdf_row(rng, nsymbs, padded));
    }
    v
}

struct CdfSet {
    kf_y: Vec<u16>,
    y_mode: Vec<u16>,
    uv: Vec<u16>,
    fi_mode: Vec<u16>,
    fi: Vec<u16>,
    pal_y_mode: Vec<u16>,
    angle: Vec<u16>,
    intrabc: Vec<u16>,
}

fn gen_all_cdfs(rng: &mut Rng) -> CdfSet {
    // uv: cfl_allowed=0 rows carry 13 symbols, cfl_allowed=1 rows carry 14.
    let mut uv = gen_cdfs(rng, INTRA_MODES, UV_INTRA_MODES - 1, UV_INTRA_MODES + 1);
    uv.extend_from_slice(&gen_cdfs(
        rng,
        INTRA_MODES,
        UV_INTRA_MODES,
        UV_INTRA_MODES + 1,
    ));
    CdfSet {
        kf_y: gen_cdfs(
            rng,
            KF_MODE_CONTEXTS * KF_MODE_CONTEXTS,
            INTRA_MODES,
            INTRA_MODES + 1,
        ),
        y_mode: gen_cdfs(rng, BLOCK_SIZE_GROUPS, INTRA_MODES, INTRA_MODES + 1),
        uv,
        fi_mode: gen_cdfs(rng, 1, FILTER_INTRA_MODES, FILTER_INTRA_MODES + 1),
        fi: gen_cdfs(rng, BLOCK_SIZES_ALL, 2, 3),
        pal_y_mode: gen_cdfs(rng, PALETTE_BSIZE_CTXS * PALETTE_Y_MODE_CONTEXTS, 2, 3),
        angle: gen_cdfs(
            rng,
            DIRECTIONAL_MODES,
            2 * MAX_ANGLE_DELTA + 1,
            2 * MAX_ANGLE_DELTA + 2,
        ),
        intrabc: gen_cdfs(rng, 1, 2, 3),
    }
}

fn fill_both(cdfs: &CdfSet, enable_fi: bool) -> (Box<IntraModeCosts>, c::RefIntraModeCosts) {
    let want = c::ref_fill_intra_mode_costs(
        &cdfs.kf_y,
        &cdfs.y_mode,
        &cdfs.uv,
        &cdfs.fi_mode,
        &cdfs.fi,
        &cdfs.pal_y_mode,
        &cdfs.angle,
        &cdfs.intrabc,
        enable_fi,
    );
    let mut costs = IntraModeCosts::zeroed();
    fill_intra_mode_costs(
        &mut costs,
        &cdfs.kf_y,
        &cdfs.y_mode,
        &cdfs.uv,
        &cdfs.fi_mode,
        &cdfs.fi,
        &cdfs.pal_y_mode,
        &cdfs.angle,
        &cdfs.intrabc,
        enable_fi,
    );
    (costs, want)
}

/// Cost-table fill matches C over random CDFs, both filter-intra gate values
/// (both sides zero-initialized: the eligible-bsize gating must agree exactly).
#[test]
fn fill_intra_mode_costs_matches_c() {
    let mut rng = Rng(0x1417_a30d_ec05_75f1);
    for trial in 0..1500 {
        let cdfs = gen_all_cdfs(&mut rng);
        let enable_fi = trial % 2 == 0;
        let (costs, want) = fill_both(&cdfs, enable_fi);

        let mut r_y = Vec::new();
        for i in &costs.y_mode_costs {
            for j in i {
                r_y.extend_from_slice(j);
            }
        }
        assert_eq!(r_y, want.y_mode, "y_mode_costs trial={trial}");
        let r_mb: Vec<i32> = costs.mbmode_cost.iter().flatten().copied().collect();
        assert_eq!(r_mb, want.mbmode, "mbmode trial={trial}");
        let mut r_uv = Vec::new();
        for i in &costs.intra_uv_mode_cost {
            for j in i {
                r_uv.extend_from_slice(j);
            }
        }
        assert_eq!(r_uv, want.uv, "uv trial={trial}");
        assert_eq!(
            costs.filter_intra_mode_cost.to_vec(),
            want.fi_mode,
            "fi_mode trial={trial}"
        );
        let r_fi: Vec<i32> = costs.filter_intra_cost.iter().flatten().copied().collect();
        assert_eq!(
            r_fi, want.fi,
            "filter_intra trial={trial} enable={enable_fi}"
        );
        let mut r_pal = Vec::new();
        for i in &costs.palette_y_mode_cost {
            for j in i {
                r_pal.extend_from_slice(j);
            }
        }
        assert_eq!(r_pal, want.pal_y_mode, "palette_y_mode trial={trial}");
        let r_angle: Vec<i32> = costs.angle_delta_cost.iter().flatten().copied().collect();
        assert_eq!(r_angle, want.angle, "angle_delta trial={trial}");
        assert_eq!(
            costs.intrabc_cost.to_vec(),
            want.intrabc,
            "intrabc trial={trial}"
        );
    }
}

/// `intra_mode_info_cost_y` matches C across the full valid argument grid
/// (exclusive mode flags per the C's live assert), on multiple CDF sets.
#[test]
fn intra_mode_info_cost_y_matches_c() {
    let mut rng = Rng(0xa4a1_e0de_17ac_0575);
    let mut nonbase = 0u64; // lookups where some gated term actually fired
    for _ in 0..6 {
        let cdfs = gen_all_cdfs(&mut rng);
        for enable_fi in [false, true] {
            let (costs, want_tables) = fill_both(&cdfs, enable_fi);
            for mode in 0..INTRA_MODES {
                for bsize in 0..BLOCK_SIZES_ALL {
                    // Exclusive flags: intrabc / filter-intra only with DC_PRED.
                    let mut flag_combos = vec![(false, false)];
                    if mode == 0 {
                        flag_combos.push((true, false));
                        flag_combos.push((false, true));
                    }
                    for (use_fi, use_intrabc) in flag_combos {
                        for angle_delta_y in -(MAX_ANGLE_DELTA as i32)..=(MAX_ANGLE_DELTA as i32) {
                            let mode_cost = rng.range(0, 8192) as i32;
                            let fi_mode = rng.range(0, FILTER_INTRA_MODES as u32) as usize;
                            let try_palette = rng.range(0, 2) == 1;
                            let pal_bctx = rng.range(0, PALETTE_BSIZE_CTXS as u32) as usize;
                            let pal_mctx = rng.range(0, PALETTE_Y_MODE_CONTEXTS as u32) as usize;
                            let allow_intrabc = use_intrabc || rng.range(0, 2) == 1;
                            let want = c::ref_intra_mode_info_cost_y(
                                &want_tables,
                                mode_cost,
                                mode as i32,
                                bsize as i32,
                                angle_delta_y,
                                use_fi,
                                fi_mode as i32,
                                use_intrabc,
                                try_palette,
                                pal_bctx as i32,
                                pal_mctx as i32,
                                enable_fi,
                                allow_intrabc
                            );
                            let got = intra_mode_info_cost_y(
                                &costs,
                                mode_cost,
                                mode,
                                bsize,
                                angle_delta_y,
                                use_fi,
                                fi_mode,
                                use_intrabc,
                                try_palette,
                                pal_bctx,
                                pal_mctx,
                                enable_fi,
                                allow_intrabc,
                                false,
                                0,
                            );
                            assert_eq!(
                                got, want,
                                "mode={mode} bsize={bsize} adelta={angle_delta_y} \
                                 use_fi={use_fi} intrabc={use_intrabc} try_pal={try_palette} \
                                 enable_fi={enable_fi} allow_intrabc={allow_intrabc}"
                            );
                            if got != mode_cost {
                                nonbase += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    // Guard: the sweep must exercise the gated terms (palette flag bit,
    // filter-intra, angle delta, intrabc), not just return mode_cost.
    assert!(nonbase > 20_000, "only {nonbase} lookups hit a gated term");
}
