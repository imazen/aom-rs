//! Differential for the partition-RDO slice (aom_encode::partition):
//!
//! 1. The rd primitives `calculate_rd_cost` / `rd_cost_update` /
//!    `rd_stats_subtraction` vs the REAL rd.h static inlines
//!    (`ref_rd_cost_update` / `ref_rd_stats_subtraction`) — including the
//!    negative-rate `RDCOST_NEG_R` arm and every INT_MAX invalid arm.
//! 2. The NONE-vs-SPLIT recursion `rd_pick_partition_none_split` vs an
//!    independent transcription of `none_partition_search` +
//!    `split_partition_search` (partition_search.c:4399/4512) driving the
//!    REAL rd primitives, over deterministic synthetic leaves shared by both
//!    sides (the control-flow constrained sweep: budget threading via
//!    `best_remain` subtraction, running `sum_rdc` early-outs, out-of-frame
//!    child skips, the >=8x8 pt_cost handling, invalid-leaf propagation,
//!    strict-< updates, the un-penalized stored split rdcost). Asserts the
//!    winner tree shape, (rate, dist, rdcost), `found`, and the LEAF VISIT
//!    SEQUENCE (order + best_remain budgets — pins the exact traversal).

use aom_encode::partition::{
    PartRdStats, PartTree, calculate_rd_cost, rd_cost_update, rd_pick_partition_none_split,
    rd_stats_subtraction, split_subsize,
};
use aom_sys_ref as c;

mod common;
use common::*;

#[test]
fn rd_primitives_match_c() {
    c::ref_init();
    let mut rng = Rng(0x7d3a_11ce_0b5e_f00d);
    let mut neg_rate_arm = 0usize;
    let mut invalid_arm = 0usize;
    for i in 0..200_000 {
        let mult = rng.range(1, i32::MAX);
        let pick = |rng: &mut Rng, k: usize| -> (i32, i64, i64) {
            match k {
                0 => (i32::MAX, rng.next() as i64 & 0xFFFF, 0),
                1 => (rng.range(0, 1 << 24), i64::MAX, 0),
                2 => (
                    rng.range(0, 1 << 24),
                    rng.next() as i64 & 0xFFFF_FFFF,
                    i64::MAX,
                ),
                _ => (
                    rng.range(-(1 << 22), 1 << 24),
                    (rng.next() & 0x3FFF_FFFF) as i64,
                    (rng.next() & 0x3FFF_FFFF) as i64,
                ),
            }
        };
        let kind = (rng.next() % 8) as usize; // mostly valid
        let (rate, dist, rdc) = pick(&mut rng, kind.min(3));
        if rate < 0 {
            neg_rate_arm += 1;
        }
        if kind < 3 {
            invalid_arm += 1;
        }

        // rd_cost_update
        let mut s = PartRdStats {
            rate,
            dist,
            rdcost: rdc,
        };
        rd_cost_update(mult, &mut s);
        let (cr, cd, cc) = c::ref_rd_cost_update(mult, rate, dist, rdc);
        assert_eq!(
            (s.rate, s.dist, s.rdcost),
            (cr, cd, cc),
            "rd_cost_update i={i}"
        );

        // rd_stats_subtraction
        let kind2 = ((rng.next() % 8) as usize).min(3);
        let (r2, d2, c2) = pick(&mut rng, kind2);
        let left = PartRdStats {
            rate,
            dist,
            rdcost: rdc,
        };
        let right = PartRdStats {
            rate: r2,
            dist: d2,
            rdcost: c2,
        };
        let got = rd_stats_subtraction(mult, &left, &right);
        let want = c::ref_rd_stats_subtraction(mult, (rate, dist, rdc), (r2, d2, c2));
        assert_eq!(
            (got.rate, got.dist, got.rdcost),
            want,
            "rd_stats_subtraction i={i}"
        );
        if got.rate != i32::MAX && got.rate < 0 {
            neg_rate_arm += 1;
        }
    }
    assert!(neg_rate_arm > 1000, "negative-rate arm: {neg_rate_arm}");
    assert!(invalid_arm > 1000, "invalid arms: {invalid_arm}");
    // calculate_rd_cost negative arm direct pin.
    for _ in 0..10_000 {
        let mult = rng.range(1, i32::MAX);
        let rate = rng.range(-(1 << 24), 1 << 24);
        let dist = (rng.next() & 0x3FFF_FFFF) as i64;
        let mut s = PartRdStats {
            rate,
            dist,
            rdcost: 0,
        };
        rd_cost_update(mult, &mut s);
        assert_eq!(s.rdcost, calculate_rd_cost(mult, rate, dist));
    }
}

/// Deterministic synthetic leaf: a hash of (mi_row, mi_col, bsize) + the
/// case seed -> (rate, dist) with INT_MAX arms, shared by both sides. The
/// `best_remain` budget is recorded (visit log) but does not shape the
/// result (pick_sb_modes may use it only for early exits; the INT_MAX arm
/// models those).
fn synth_leaf(seed: u64, mi_row: i32, mi_col: i32, bsize: usize, scale: i32) -> PartRdStats {
    let mut h = seed
        ^ (mi_row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (mi_col as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
        ^ (bsize as u64).wrapping_mul(0x1656_67b1_9e37_79f9);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    if h.is_multiple_of(97) {
        // pick_sb_modes rate == INT_MAX -> rdcost = INT64_MAX
        // (partition_search.c:970).
        return PartRdStats::invalid();
    }
    // Leaf dist grows QUADRATICALLY with block area (scaled per case), so
    // small `scale` makes splitting win repeatedly (deep trees) and large
    // `scale` lets the noise + partition costs favour NONE.
    let area = (1u64 << (2 * (bsize as u32 / 3 + 2))) as i64;
    let rate = ((h >> 8) % 2048) as i32 + (area / 4) as i32;
    let dist = ((h >> 24) % 2048) as i64 + area * area * 4 / scale.max(1) as i64;
    PartRdStats {
        rate,
        dist,
        rdcost: 0,
    }
}

/// The C-side transcription of none_partition_search + split_partition_search
/// (NONE/SPLIT slice) over the REAL rd primitives, sharing the same leaves.
#[allow(clippy::too_many_arguments, unused_assignments)]
fn c_rd_pick_partition_none_split(
    visits: &mut Vec<(i32, i32, usize, i64)>,
    seed: u64,
    scale: i32,
    part_costs: &dyn Fn(usize) -> [i32; 4],
    rdmult: i32,
    mi_row: i32,
    mi_col: i32,
    mi_rows: i32,
    mi_cols: i32,
    bsize: usize,
    mut best: (i32, i64, i64),
) -> (Option<(bool, i32, i64, i64)>, bool) {
    // (is_split, rate, dist, rdcost) + found
    if best.2 < 0 {
        return (None, false);
    }
    let mi_step = ([1usize, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32][bsize] / 2) as i32;
    let has_rows = mi_row + mi_step < mi_rows;
    let has_cols = mi_col + mi_step < mi_cols;
    let none_allowed = has_rows && has_cols;
    let at_least_8x8 = bsize >= 3;
    let do_split = at_least_8x8;
    let costs = part_costs(bsize);

    let mut found = false;
    let mut best_res: Option<(bool, i32, i64, i64)> = None;

    if none_allowed {
        let pt_cost = if at_least_8x8 {
            if costs[0] < i32::MAX { costs[0] } else { 0 }
        } else {
            0
        };
        let (pr, pd, pc) = c::ref_rd_cost_update(rdmult, pt_cost, 0, 0);
        let best_remain = c::ref_rd_stats_subtraction(rdmult, best, (pr, pd, pc));
        visits.push((mi_row, mi_col, bsize, best_remain.2));
        let leaf = synth_leaf(seed, mi_row, mi_col, bsize, scale);
        let (mut lr, mut ld, mut lc) = if leaf.is_invalid() {
            (i32::MAX, i64::MAX, i64::MAX)
        } else {
            (
                leaf.rate,
                leaf.dist,
                c::ref_rdcost(rdmult, leaf.rate, leaf.dist),
            )
        };
        let upd = c::ref_rd_cost_update(rdmult, lr, ld, lc);
        (lr, ld, lc) = upd;
        if lr != i32::MAX {
            if at_least_8x8 {
                lr += pt_cost;
                lc = c::ref_rdcost(rdmult, lr, ld);
            }
            if lc < best.2 {
                best = (lr, ld, lc);
                found = true;
                best_res = Some((false, lr, ld, lc));
            }
        }
    }

    if do_split {
        let subsize = split_subsize(bsize);
        let mut sum = (costs[3], 0i64, c::ref_rdcost(rdmult, costs[3], 0));
        let mut idx = 0usize;
        while idx < 4 && sum.2 < best.2 {
            let x_idx = ((idx & 1) as i32) * mi_step;
            let y_idx = ((idx >> 1) as i32) * mi_step;
            if mi_row + y_idx >= mi_rows || mi_col + x_idx >= mi_cols {
                idx += 1;
                continue;
            }
            let best_remain = c::ref_rd_stats_subtraction(rdmult, best, sum);
            let (child, child_found) = c_rd_pick_partition_none_split(
                visits,
                seed,
                scale,
                part_costs,
                rdmult,
                mi_row + y_idx,
                mi_col + x_idx,
                mi_rows,
                mi_cols,
                subsize,
                best_remain,
            );
            if !child_found {
                sum = (i32::MAX, i64::MAX, i64::MAX);
                break;
            }
            let cbest = child.expect("found child has stats");
            sum.0 += cbest.1;
            sum.1 += cbest.2;
            let upd = c::ref_rd_cost_update(rdmult, sum.0, sum.1, sum.2);
            sum = upd;
            idx += 1;
        }
        let reached = idx == 4;
        if reached && sum.2 < best.2 {
            sum.2 = c::ref_rdcost(rdmult, sum.0, sum.1);
            if sum.2 < best.2 {
                best = sum;
                found = true;
                best_res = Some((true, sum.0, sum.1, sum.2));
            }
        }
    }

    if found {
        (best_res, true)
    } else {
        (None, false)
    }
}

#[test]
fn rd_pick_partition_none_split_matches_c_transcription() {
    c::ref_init();
    let mut rng = Rng(0x51ab_e77a_c3d0_9b26);
    let mut none_roots = 0usize;
    let mut split_roots = 0usize;
    let mut deep_splits = 0usize;
    let mut nofind = 0usize;
    let mut edge_cases = 0usize;
    let mut budget_break = 0usize;

    for case in 0..400u64 {
        // Roots: 8x8, 16x16, 32x32, 64x64; positions incl. near-frame-edge
        // (out-of-frame child skips + !none_allowed edge nodes).
        let bsize = [3usize, 6, 9, 12][(case % 4) as usize];
        let mi_dim = [1usize, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16][bsize] as i32;
        let (mi_rows, mi_cols, mi_row, mi_col) = match case % 5 {
            // interior
            0 | 1 => (512, 512, 8, 8),
            // right edge: block hangs over -> edge children skipped
            2 => (512, 8 + mi_dim / 2, 8, 8),
            // bottom edge
            3 => (8 + mi_dim / 2, 512, 8, 8),
            // corner
            _ => (8 + mi_dim / 2, 8 + mi_dim / 2, 8, 8),
        };
        if case % 5 >= 2 {
            edge_cases += 1;
        }
        let rdmult = rng.range(1, 1 << 22);
        let seed = rng.next();
        // scale steers leaf rate vs area: small scale -> big leaf rates ->
        // SPLIT competitive at small sizes; large -> NONE wins.
        let scale = [1, 4, 16, 64, 1024][(case % 5) as usize] * [1, 7][(case % 2) as usize];
        // Per-bsize partition costs (identical closure both sides).
        let cost_seed = rng.next();
        let part_costs = move |bs: usize| -> [i32; 4] {
            let mut h = cost_seed ^ (bs as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
            h ^= h >> 29;
            [
                (h % 1500) as i32,
                ((h >> 16) % 1500) as i32,
                ((h >> 32) % 1500) as i32,
                ((h >> 48) % 1500) as i32,
            ]
        };
        // Budgets: mostly generous; some tight (no-find + budget breaks).
        let best_in = match case % 7 {
            5 => 1 << 12,
            6 => -1, // invalid budget arm (rdcost < 0)
            _ => i64::MAX,
        };
        let best_stats = PartRdStats {
            rate: if best_in == i64::MAX { i32::MAX } else { 0 },
            dist: if best_in == i64::MAX { i64::MAX } else { 0 },
            rdcost: best_in,
        };

        // ---- Rust ----
        let mut visits_r: Vec<(i32, i32, usize, i64)> = Vec::new();
        let mut leaf = |mi_row: i32, mi_col: i32, bs: usize, best_remain: &PartRdStats| {
            visits_r.push((mi_row, mi_col, bs, best_remain.rdcost));
            let mut l = synth_leaf(seed, mi_row, mi_col, bs, scale);
            if !l.is_invalid() {
                l.rdcost = calculate_rd_cost(rdmult, l.rate, l.dist);
            } else {
                l.rdcost = i64::MAX;
            }
            l
        };
        // node_params: the partition-cost row + the geometry-derived
        // partition_none_allowed (init_partition_search_state_params
        // has_rows/cols) + do_square_split per node.
        let mut node_params_geo = |r: i32, ccol: i32, bs: usize| -> (Vec<i32>, bool, bool) {
            let step = ([1usize, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16][bs] / 2) as i32;
            let none_ok = (r + step < mi_rows) && (ccol + step < mi_cols);
            (part_costs(bs).to_vec(), none_ok, bs >= 3)
        };
        let (tree, best_out, found) = rd_pick_partition_none_split(
            &mut node_params_geo,
            &mut leaf,
            rdmult,
            mi_row,
            mi_col,
            mi_rows,
            mi_cols,
            bsize,
            best_stats,
        );

        // ---- C transcription ----
        let mut visits_c: Vec<(i32, i32, usize, i64)> = Vec::new();
        let (want, want_found) = c_rd_pick_partition_none_split(
            &mut visits_c,
            seed,
            scale,
            &part_costs,
            rdmult,
            mi_row,
            mi_col,
            mi_rows,
            mi_cols,
            bsize,
            (best_stats.rate, best_stats.dist, best_stats.rdcost),
        );

        let m = format!(
            "case={case} bsize={bsize} mi=({mi_row},{mi_col}) dims=({mi_rows},{mi_cols}) \
             rdmult={rdmult} scale={scale} best_in={best_in}"
        );
        assert_eq!(found, want_found, "found {m}");
        assert_eq!(visits_r, visits_c, "leaf visit sequence {m}");
        match (&tree, want) {
            (PartTree::NotFound, None) => {
                nofind += 1;
            }
            (PartTree::None(s), Some((false, r, d, cc))) => {
                assert_eq!((s.rate, s.dist, s.rdcost), (r, d, cc), "NONE stats {m}");
                assert_eq!(
                    (best_out.rate, best_out.dist, best_out.rdcost),
                    (r, d, cc),
                    "best_rdc {m}"
                );
                none_roots += 1;
            }
            (PartTree::Split(s, ch), Some((true, r, d, cc))) => {
                assert_eq!((s.rate, s.dist, s.rdcost), (r, d, cc), "SPLIT stats {m}");
                assert_eq!(
                    (best_out.rate, best_out.dist, best_out.rdcost),
                    (r, d, cc),
                    "best_rdc {m}"
                );
                split_roots += 1;
                if ch.iter().any(|t| matches!(t, PartTree::Split(..))) {
                    deep_splits += 1;
                }
            }
            (t, w) => panic!("tree/winner mismatch {m}: rust={t:?} c={w:?}"),
        }
        // Budget-break coverage: a case where the split child loop stopped
        // early is visible as fewer visits than the full tree would take.
        if found && matches!(tree, PartTree::None(_)) && bsize >= 6 {
            budget_break += 1; // NONE win at >=16x16 means split loop was
            // entered and lost or broke early
        }
    }
    assert!(none_roots >= 40, "NONE root winners: {none_roots}");
    assert!(split_roots >= 40, "SPLIT root winners: {split_roots}");
    assert!(deep_splits >= 10, "multi-level splits: {deep_splits}");
    assert!(nofind >= 20, "no-find (tight/invalid budgets): {nofind}");
    assert!(edge_cases >= 100, "edge geometries: {edge_cases}");
    assert!(
        budget_break >= 10,
        "split-loses-or-breaks cases: {budget_break}"
    );
}

/// Deterministic EXACT-TIE pin: at an 8x8 root, NONE's total (leaf + pt
/// cost) exactly equals SPLIT's total (split cost + 4 identical 4x4
/// leaves) in BOTH rate and dist. The C rejects the tying split through
/// LAYERED strict-`<` gates — with an rdmult where RDCOST is exactly
/// linear (512 here), the 4th child's own rd EXACTLY equals its
/// `best_remain` budget, so the child NONE gate rejects it first; were
/// that gate `<=`, the outer `reached && sum < best` gate rejects; were
/// that also `<=`, the post-recompute compare rejects. Any SINGLE `<=`
/// relaxation is therefore masked by the remaining strict layers (each
/// ported verbatim from the cited lines); this pin flips to SPLIT only if
/// a port relaxes them systematically — the regression it guards.
#[test]
fn none_split_exact_tie_keeps_first_winner() {
    c::ref_init();
    let rdmult = 512;
    // NONE: leaf(8x8) rate 100 dist 4000; pt_cost 40 -> rate 140, dist 4000.
    // SPLIT: cost 40 + 4 leaves rate 25 dist 1000 -> rate 140, dist 4000.
    let mut leaf = |_r: i32, _c: i32, bs: usize, _rem: &PartRdStats| -> PartRdStats {
        let (rate, dist) = if bs == 3 { (100, 4000) } else { (25, 1000) };
        PartRdStats {
            rate,
            dist,
            rdcost: calculate_rd_cost(rdmult, rate, dist),
        }
    };
    let mut node_params = |_r: i32, _c: i32, bs: usize| -> (Vec<i32>, bool, bool) {
        // do_square_split = bsize_at_least_8x8 (4x4 children are leaves).
        (vec![40, 0, 0, 40], true, bs >= 3)
    };
    let (tree, best, found) = rd_pick_partition_none_split(
        &mut node_params,
        &mut leaf,
        rdmult,
        8,
        8,
        512,
        512,
        3,
        PartRdStats {
            rate: i32::MAX,
            dist: i64::MAX,
            rdcost: i64::MAX,
        },
    );
    assert!(found);
    let expect = PartRdStats {
        rate: 140,
        dist: 4000,
        rdcost: calculate_rd_cost(rdmult, 140, 4000),
    };
    match tree {
        PartTree::None(s) => assert_eq!(s, expect, "NONE stats"),
        t => panic!("exact tie must keep the FIRST winner (PARTITION_NONE), got {t:?}"),
    }
    assert_eq!(
        (best.rate, best.dist, best.rdcost),
        (140, 4000, expect.rdcost)
    );
}
