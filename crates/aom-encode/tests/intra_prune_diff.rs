//! Differential: `prune_intra_y_mode` + `get_model_rd_index_for_pruning`
//! (intra_mode_search.c statics) vs their verbatim C transcriptions compiled
//! in rd_shim.c — the double-threshold model-RD prune driven as SEQUENCES
//! (the accumulators `best_model_rd` / `top_intra_model_rd` evolve across a
//! mode loop's candidates, so each step's decision depends on the whole
//! prefix). Lockstep per step: prune decision + both accumulators identical.

use aom_encode::intra_rd::{
    TOP_INTRA_MODEL_COUNT, get_model_rd_index_for_pruning, prune_intra_y_mode,
};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo) as u64) as i64
    }
}

#[test]
fn prune_intra_y_mode_matches_c_sequences() {
    c::ref_init();
    let mut rng = Rng(0x9a1e_50fa_11ad_ec0d);
    let mut pruned_steps = 0usize;
    let mut kept_steps = 0usize;
    let mut best_lowered = 0usize;

    for seq in 0..4000 {
        // max_model_cnt_allowed: 4 at speed 0; 2/3 at higher speeds.
        let max_cnt = [4usize, 4, 4, 3, 2][seq % 5];
        let idx = (rng.next() as usize) % max_cnt;
        let mut best_r = i64::MAX;
        let mut best_c = i64::MAX;
        let mut top_r = [i64::MAX; TOP_INTRA_MODEL_COUNT];
        let mut top_c = [i64::MAX; TOP_INTRA_MODEL_COUNT];

        // Magnitude regime per sequence: small SATDs, large SATDs, and a
        // near-tie regime that hammers the double-threshold boundaries
        // (values within +-2 of each other so 1.00 * top[idx] compares hit
        // equality frequently — C `>` must match exactly).
        let (lo, hi) = match seq % 3 {
            0 => (0i64, 1 << 12),
            1 => (1 << 20, 1 << 34),
            _ => (1000, 1016),
        };
        let steps = 3 + (rng.next() as usize) % 60;
        for step in 0..steps {
            // Occasional INT64_MAX candidate exercises both != INT64_MAX
            // guards.
            let this = if step % 13 == 12 {
                i64::MAX
            } else {
                rng.range(lo, hi)
            };
            let before_best = best_r;
            let got = prune_intra_y_mode(this, &mut best_r, &mut top_r, max_cnt, idx);
            let want = c::ref_prune_intra_y_mode(this, &mut best_c, &mut top_c, max_cnt, idx);
            assert_eq!(
                got, want,
                "decision seq={seq} step={step} this={this} max_cnt={max_cnt} idx={idx}",
            );
            assert_eq!(best_r, best_c, "best seq={seq} step={step}");
            assert_eq!(top_r, top_c, "top array seq={seq} step={step}");
            if got {
                pruned_steps += 1;
            } else {
                kept_steps += 1;
            }
            if best_r < before_best {
                best_lowered += 1;
            }
        }
    }
    assert!(pruned_steps > 10_000, "prunes: {pruned_steps}");
    assert!(kept_steps > 10_000, "keeps: {kept_steps}");
    assert!(best_lowered > 4_000, "best lowered: {best_lowered}");
}

#[test]
fn get_model_rd_index_for_pruning_matches_c() {
    c::ref_init();
    let mut rng = Rng(0x0d1c_ea5e_5eed_0714);
    let mut adapted = 0usize;
    for case in 0..20_000 {
        let cur_mode = (rng.next() as usize) % 13;
        let qindex = (rng.next() as i32).rem_euclid(256);
        let cnt = 1 + (rng.next() as i32).rem_euclid(4); // 1..=4
        let adapt = case % 2 == 1;
        let left = if rng.next().is_multiple_of(3) {
            None
        } else {
            Some((rng.next() as usize) % 13)
        };
        let above = if rng.next().is_multiple_of(3) {
            None
        } else {
            Some((rng.next() as usize) % 13)
        };
        let got = get_model_rd_index_for_pruning(cur_mode, qindex, cnt, adapt, left, above);
        let want = c::ref_get_model_rd_index_for_pruning(cur_mode, qindex, cnt, adapt, left, above);
        assert_eq!(
            got, want,
            "case={case} mode={cur_mode} q={qindex} cnt={cnt} adapt={adapt} \
             left={left:?} above={above:?}",
        );
        if adapt && got != cnt - 1 {
            adapted += 1;
        }
    }
    assert!(adapted > 2_000, "adaptive reductions: {adapted}");
}
