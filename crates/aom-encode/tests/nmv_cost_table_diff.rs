//! INTER-ENCODE chunk 2 differential — `av1_build_nmv_cost_table`
//! (encodemv.c:294) vs the REAL exported C.
//!
//! Locks the port's [`aom_encode::intrabc_search::fill_nmv_costs`]
//! (`av1_build_nmv_cost_table` + `av1_build_nmv_component_cost_table`) against
//! the real builder across all three `MvSubpelPrecision` values (NONE / LOW /
//! HIGH) and a sweep of `nmv_context`s (the libaom default context + several
//! synthetic-but-valid CDF sets). Every produced cost is compared: the 4 joint
//! costs AND both full-length (`2*MV_MAX+1`) per-component magnitude tables.
//!
//! These are the real MV cost tables the inter motion search consumes
//! (`x->mv_costs` / `MV_COST_PARAMS`) — the subpel tree + full-pel search
//! currently take synthetic cost tables as input; this differential proves the
//! port builds the *real* ones byte-for-byte from the frame's nmv CDFs.

use aom_encode::intrabc_search::{
    fill_dv_costs, fill_nmv_costs, MV_MAX, MV_SUBPEL_HIGH, MV_SUBPEL_LOW, MV_SUBPEL_NONE,
};
use aom_dsp::entropy::default_cdfs::{DEFAULT_NMV_COMPS, DEFAULT_NMV_JOINTS};
use aom_sys_ref::ref_build_nmv_cost_table;

const MV_VALS: usize = (MV_MAX as usize) * 2 + 1;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// A symbol weight in `1..=64` (nonzero → no degenerate zero-prob symbols).
    fn weight(&mut self) -> u32 {
        1 + (self.next_u64() >> 40) as u32 % 64
    }
}

/// Build one valid AOM ICDF CDF of `n` symbols (`CDF_SIZE(n) = n+1` u16): `n-1`
/// strictly-derived cumulative ICDF values, then the 0 terminator at index
/// `n-1`, then a 0 count at index `n`. Matches the port's default-CDF shape
/// (e.g. `DEFAULT_NMV_JOINTS = [28672, 21504, 13440, 0, 0]`).
fn make_cdf(rng: &mut Rng, n: usize) -> Vec<u16> {
    let weights: Vec<u32> = (0..n).map(|_| rng.weight()).collect();
    let total: u32 = weights.iter().sum();
    let mut cdf = vec![0u16; n + 1];
    let mut cum = 0u32;
    for (i, w) in weights.iter().enumerate() {
        cum += w;
        // AOM_ICDF(cumulative-prob-through-symbol-i), scaled to 32768.
        let cp = ((cum as u64 * 32768) / total as u64).min(32768) as u32;
        cdf[i] = (32768 - cp) as u16;
    }
    // cdf[n-1] == 0 (cum == total) terminator; cdf[n] == 0 count.
    cdf
}

/// Assemble a random valid 69-u16 nmv_component blob in the port's packing:
/// sign 0..3, classes 3..15, class0 15..18, bits[10] 18..48,
/// class0_fp[2] 48..58, fp 58..63, class0_hp 63..66, hp 66..69.
fn make_comp(rng: &mut Rng) -> [u16; 69] {
    let mut b = [0u16; 69];
    let put = |b: &mut [u16; 69], off: usize, cdf: &[u16]| {
        b[off..off + cdf.len()].copy_from_slice(cdf);
    };
    put(&mut b, 0, &make_cdf(rng, 2)); // sign (3)
    put(&mut b, 3, &make_cdf(rng, 11)); // classes (12)
    put(&mut b, 15, &make_cdf(rng, 2)); // class0 (3)
    for i in 0..10 {
        put(&mut b, 18 + i * 3, &make_cdf(rng, 2)); // bits[i] (3)
    }
    for i in 0..2 {
        put(&mut b, 48 + i * 5, &make_cdf(rng, 4)); // class0_fp[i] (5)
    }
    put(&mut b, 58, &make_cdf(rng, 4)); // fp (5)
    put(&mut b, 63, &make_cdf(rng, 2)); // class0_hp (3)
    put(&mut b, 66, &make_cdf(rng, 2)); // hp (3)
    b
}

/// Compare the port's `fill_nmv_costs` output against the REAL builder for one
/// (joints, comp0, comp1, precision) tuple. Returns the port's two tables so the
/// caller can run anti-vacuity checks across precisions.
fn assert_matches(
    joints: &[u16; 5],
    comp0: &[u16; 69],
    comp1: &[u16; 69],
    precision: i32,
    tag: &str,
) -> (Vec<i32>, Vec<i32>) {
    let port = fill_nmv_costs(precision, joints, comp0, comp1);
    let (c_joint, c_cost0, c_cost1) = ref_build_nmv_cost_table(joints, comp0, comp1, precision);

    assert_eq!(
        port.joint_mv, c_joint,
        "{tag} prec={precision}: joint costs differ (port {:?} vs C {:?})",
        port.joint_mv, c_joint
    );
    assert_eq!(port.dv_costs[0].len(), MV_VALS);
    assert_eq!(port.dv_costs[1].len(), MV_VALS);
    // Compare full magnitude tables; report the first mismatch precisely.
    for (comp, (p, c)) in [
        (&port.dv_costs[0], &c_cost0),
        (&port.dv_costs[1], &c_cost1),
    ]
    .iter()
    .enumerate()
    {
        if let Some(idx) = (0..MV_VALS).find(|&i| p[i] != c[i]) {
            let v = idx as i32 - MV_MAX;
            panic!(
                "{tag} prec={precision}: comp{comp} mvcost[v={v}] differ (port {} vs C {})",
                p[idx], c[idx]
            );
        }
    }
    (port.dv_costs[0].clone(), port.dv_costs[1].clone())
}

/// The libaom DEFAULT nmv_context at every precision — the representative,
/// real-CDF lock. Also verifies precision actually changes the output (fp costs
/// appear at LOW, hp at HIGH), and that NONE reproduces `fill_dv_costs`.
#[test]
fn nmv_cost_table_default_context_matches_real_c() {
    let joints = &DEFAULT_NMV_JOINTS;
    let comp0 = &DEFAULT_NMV_COMPS[0];
    let comp1 = &DEFAULT_NMV_COMPS[1];

    let (none0, _none1) = assert_matches(joints, comp0, comp1, MV_SUBPEL_NONE, "default");
    let (low0, _low1) = assert_matches(joints, comp0, comp1, MV_SUBPEL_LOW, "default");
    let (high0, _high1) = assert_matches(joints, comp0, comp1, MV_SUBPEL_HIGH, "default");

    // Anti-vacuity: fractional (LOW) then high-precision (HIGH) bits genuinely
    // change the magnitude costs — the precision gates are exercised, not inert.
    assert_ne!(none0, low0, "LOW precision must add fractional-pel costs vs NONE");
    assert_ne!(low0, high0, "HIGH precision must add high-precision costs vs LOW");

    // Regression: fill_nmv_costs at NONE == the intrabc DV-cost builder.
    let dv = fill_dv_costs(joints, comp0, comp1);
    assert_eq!(dv.joint_mv, fill_nmv_costs(MV_SUBPEL_NONE, joints, comp0, comp1).joint_mv);
    assert_eq!(dv.dv_costs[0], none0, "NONE precision must equal fill_dv_costs");
}

/// A sweep of synthetic-but-valid nmv_contexts at every precision — exercises
/// the builder arithmetic across a range of CDF values, not just the default.
#[test]
fn nmv_cost_table_random_contexts_match_real_c() {
    let mut rng = Rng::new(0xC0FFEE_1234_5678);
    for iter in 0..24 {
        let joints = {
            let mut j = [0u16; 5];
            j.copy_from_slice(&make_cdf(&mut rng, 4));
            j
        };
        let comp0 = make_comp(&mut rng);
        let comp1 = make_comp(&mut rng);
        for &prec in &[MV_SUBPEL_NONE, MV_SUBPEL_LOW, MV_SUBPEL_HIGH] {
            assert_matches(&joints, &comp0, &comp1, prec, &format!("rand#{iter}"));
        }
    }
}
