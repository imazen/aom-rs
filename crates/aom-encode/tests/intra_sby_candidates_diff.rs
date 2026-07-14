//! Differential harness for the `av1_rd_pick_intra_sby_mode` candidate-loop
//! head (av1/encoder/intra_mode_search.c): `set_y_mode_and_delta_angle`
//! (oracle = the REAL EXPORTED C function), the static skip chain
//! (transcription over the REAL header statics), and
//! `prune_luma_odd_delta_angles_using_rd_cost` (pure transcription).

use aom_encode::intra_rd::{
    INTRA_MODES, IntraSbyGates, LUMA_MODE_COUNT, prune_luma_odd_delta_angles_using_rd_cost,
    set_y_mode_and_delta_angle,
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
}

#[test]
fn set_y_mode_and_delta_angle_matches_real_c() {
    for reorder in [false, true] {
        for mode_idx in 0..LUMA_MODE_COUNT {
            let (mode, delta) = set_y_mode_and_delta_angle(mode_idx, reorder);
            let (mode_c, delta_c) = c::ref_set_y_mode_and_delta_angle(mode_idx as i32, reorder);
            assert_eq!(
                (mode as i32, delta),
                (mode_c, delta_c),
                "mode_idx={mode_idx} reorder={reorder}",
            );
        }
    }
}

/// Every gate-flag combination that can flip a branch, x all 61 candidates x
/// a block-size sweep: the Rust visit decision equals the C chain.
#[test]
fn visit_gating_matches_c() {
    let mut rng = Rng(0x5eed_0fca_4d1d_a7e5);
    let bsizes: [usize; 8] = [0, 3, 6, 9, 12, 15, 16, 21]; // 4x4..128x128 + rect
    let mut visited = 0usize;
    let mut skipped = 0usize;
    for case in 0..3000 {
        let gates = IntraSbyGates {
            enable_diagonal_intra: rng.next() & 1 == 1,
            enable_directional_intra: rng.next() & 1 == 1,
            enable_smooth_intra: rng.next() & 1 == 1,
            enable_paeth_intra: rng.next() & 1 == 1,
            enable_angle_delta: rng.next() & 1 == 1,
            disable_smooth_intra: rng.next() & 1 == 1,
            prune_filter_intra_level: (rng.next() % 3) as i32,
            intra_y_mode_mask: if case % 3 == 0 {
                [0x1fff; 5]
            } else {
                [
                    (rng.next() & 0x1fff) as u16,
                    (rng.next() & 0x1fff) as u16,
                    (rng.next() & 0x1fff) as u16,
                    (rng.next() & 0x1fff) as u16,
                    (rng.next() & 0x1fff) as u16,
                ]
            },
            directional_mode_skip_mask: {
                let mut m = [false; INTRA_MODES];
                if case % 2 == 1 {
                    for f in m.iter_mut() {
                        *f = rng.next() & 3 == 0;
                    }
                }
                m
            },
            prune_luma_odd_delta_angles_in_intra: rng.next() & 1 == 1,
        };
        let bsize = bsizes[case % bsizes.len()];
        let skip_mask_u8: [u8; 13] = {
            let mut m = [0u8; 13];
            for (dst, &src) in m.iter_mut().zip(gates.directional_mode_skip_mask.iter()) {
                *dst = src as u8;
            }
            m
        };
        for mode_idx in 0..LUMA_MODE_COUNT {
            let (mode, delta) =
                set_y_mode_and_delta_angle(mode_idx, gates.prune_luma_odd_delta_angles_in_intra);
            let rust = gates.visits(mode, delta, bsize);
            let cref = c::ref_intra_sby_visits(
                mode as i32,
                delta,
                bsize as i32,
                gates.enable_diagonal_intra,
                gates.enable_directional_intra,
                gates.enable_smooth_intra,
                gates.enable_paeth_intra,
                gates.enable_angle_delta,
                gates.disable_smooth_intra,
                gates.prune_filter_intra_level,
                &gates.intra_y_mode_mask,
                &skip_mask_u8,
            );
            assert_eq!(
                rust, cref,
                "case={case} mode_idx={mode_idx} mode={mode} delta={delta} bsize={bsize}"
            );
            if rust {
                visited += 1;
            } else {
                skipped += 1;
            }
        }
    }
    // Non-vacuity: both outcomes must occur heavily.
    assert!(
        visited > 10_000 && skipped > 10_000,
        "visited={visited} skipped={skipped}"
    );
}

/// The speed-0 configuration visits ALL 61 candidates on angle-delta-capable
/// block sizes (no HOG skips), and exactly the 13 zero-delta modes on 4x4.
#[test]
fn speed0_visit_sequence_shape() {
    let gates = IntraSbyGates::speed0([false; INTRA_MODES]);
    let seq8 = gates.visit_sequence(3); // BLOCK_8X8
    assert_eq!(seq8.len(), LUMA_MODE_COUNT);
    // First 13 = intra_rd_search_mode_order at delta 0.
    let expect_head: [(usize, i32); 5] = [(0, 0), (2, 0), (1, 0), (9, 0), (12, 0)];
    assert_eq!(&seq8[..5], &expect_head);
    // Tail sweeps V..D67 x deltas -3..-1,1..3.
    assert_eq!(seq8[13], (1, -3));
    assert_eq!(seq8[18], (1, 3));
    assert_eq!(seq8[60], (8, 3));
    // 4x4: av1_use_angle_delta is false -> only the 13 zero-delta modes.
    let seq4 = gates.visit_sequence(0);
    assert_eq!(seq4.len(), INTRA_MODES);
    assert!(seq4.iter().all(|&(_, d)| d == 0));
}

#[test]
fn prune_odd_delta_matches_c() {
    let mut rng = Rng(0x0dd0_de17_a000_0001);
    for case in 0..40_000 {
        let mode = (rng.next() % 13) as usize;
        let delta = (rng.next() % 7) as i32 - 3;
        let best_rd = match case % 4 {
            0 => i64::MAX,
            1 => (rng.next() & 0xffff) as i64,
            _ => (rng.next() & 0x3fff_ffff) as i64,
        };
        let mut costs = [i64::MAX; 9];
        for c_ in costs.iter_mut() {
            if rng.next() & 1 == 1 {
                *c_ = (rng.next() & 0x7fff_ffff) as i64;
            }
        }
        let prune = rng.next() & 1 == 1;
        assert_eq!(
            prune_luma_odd_delta_angles_using_rd_cost(mode, delta, &costs, best_rd, prune),
            c::ref_prune_odd_delta(mode as i32, delta, &costs, best_rd, prune),
            "case={case} mode={mode} delta={delta} best_rd={best_rd} prune={prune}",
        );
    }
}
