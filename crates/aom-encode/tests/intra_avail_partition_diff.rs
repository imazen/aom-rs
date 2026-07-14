//! Differential harness for the `partition` input of `intra_avail` — the
//! `mbmi->partition` that `pick_sb_modes` (:887) / `av1_update_state` install
//! and `has_top_right`/`has_bottom_left` consume (reconintra.c:198/383).
//!
//! The C tables branch ONLY on `PARTITION_VERT_A`/`PARTITION_VERT_B`
//! (`get_has_tr_table`/`get_has_bl_table`, reconintra.c:182/367) — so for
//! the rect stage's PARTITION_HORZ/VERT leaves the availability must be
//! IDENTICAL to PARTITION_NONE. This sweep pins both facts against the REAL
//! C (`ref_intra_avail`): Rust == C at every partition in {NONE, HORZ,
//! VERT}, over the rect subsizes the HORZ/VERT stage evaluates (plus the
//! squares), sub-block offsets, both chroma samplings, and edge positions.
//!
//! (The AB chunk extends this to VERT_A/VERT_B, where the vert-alike order
//! table genuinely changes the result.)

use aom_entropy::partition::intra_avail;
use aom_sys_ref as c;

const MI_W: [i32; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_H: [i32; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];
const BW: [i32; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BH: [i32; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

#[test]
fn intra_avail_partition_input_matches_c() {
    let (mi_cols, mi_rows) = (32i32, 32i32);
    let sb = 12usize; // BLOCK_64X64
    // The HORZ/VERT subsizes of the 8..64 squares + the squares themselves.
    let bsizes: [usize; 12] = [2, 1, 5, 4, 8, 7, 11, 10, 3, 6, 9, 12];
    let modes = [0usize, 1, 3, 8, 12]; // DC, V, D45 (TR), D67 (TR), PAETH
    let mut checks = 0u64;
    let mut identity_checks = 0u64;
    for &bsize in &bsizes {
        for &(ss_x, ss_y) in &[(0i32, 0i32), (1, 1)] {
            let wpx = BW[bsize] >> ss_x;
            let hpx = BH[bsize] >> ss_y;
            let cmax = (MI_W[bsize] >> ss_x).max(1);
            let rmax = (MI_H[bsize] >> ss_y).max(1);
            for &mi_row in &[0i32, 8, 24] {
                for &mi_col in &[0i32, 8, 24] {
                    let up = mi_row > 0;
                    let left = mi_col > 0;
                    for &tx_size in &[0usize, 1] {
                        for &row_off in &[0i32, rmax / 2] {
                            for &col_off in &[0i32, cmax / 2] {
                                for &mode in &modes {
                                    let mut base: Option<(i32, i32, i32, i32)> = None;
                                    for partition in [0usize, 1, 2] {
                                        let g = intra_avail(
                                            sb, bsize, mi_row, mi_col, up, left, mi_cols, mi_rows,
                                            partition, tx_size, ss_x, ss_y, row_off, col_off, wpx,
                                            hpx, mi_cols, mi_rows, mode, 0, false,
                                        );
                                        let w = c::ref_intra_avail(
                                            sb, bsize, mi_row, mi_col, up, left, mi_cols, mi_rows,
                                            partition, tx_size, ss_x, ss_y, row_off, col_off, wpx,
                                            hpx, mi_cols, mi_rows, mode, 0, false,
                                        );
                                        assert_eq!(
                                            g, w,
                                            "avail bsize={bsize} part={partition} ss=({ss_x},{ss_y}) \
                                             mi=({mi_row},{mi_col}) tx={tx_size} off=({row_off},{col_off}) \
                                             mode={mode}"
                                        );
                                        checks += 1;
                                        // NONE/HORZ/VERT identity (the C
                                        // tables only branch on VERT_A/B).
                                        match base {
                                            None => base = Some(w),
                                            Some(b) => {
                                                assert_eq!(
                                                    w, b,
                                                    "partition {partition} != NONE at bsize={bsize}"
                                                );
                                                identity_checks += 1;
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
    }
    assert!(checks > 10_000, "sweep breadth: {checks}");
    assert!(
        identity_checks > 6_000,
        "identity breadth: {identity_checks}"
    );
}
