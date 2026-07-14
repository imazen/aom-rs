//! Differential harness for `has_top_right` / `has_bottom_left` (intra neighbour
//! availability) vs C libaom v3.14.1 (`ref_has_top_right` / `ref_has_bottom_left`,
//! a verbatim paste of the reconintra.c statics). Swept over both superblock
//! sizes, every block size that fits, both mixed-vertical and other partitions,
//! all plane subsamplings, block positions across a superblock, transform sizes,
//! transform-unit offsets, and availability flags — pinning the bitmap tables,
//! the table dispatch, and the branch logic.

use aom_entropy::partition::{has_bottom_left, has_top_right};
use aom_sys_ref as c;

const MI_SIZE_WIDE: [i32; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_SIZE_HIGH: [i32; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];
const BLOCK_SIZE_WIDE: [i32; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLOCK_SIZE_HIGH: [i32; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
// Block sizes whose has_*_vert_table is non-NULL (VERT_A/VERT_B partitions).
const VERT_OK: [usize; 10] = [1, 3, 4, 6, 7, 9, 10, 12, 13, 15];

fn fits(sb: usize, bsize: usize) -> bool {
    BLOCK_SIZE_WIDE[bsize] <= BLOCK_SIZE_WIDE[sb] && BLOCK_SIZE_HIGH[bsize] <= BLOCK_SIZE_HIGH[sb]
}

#[test]
fn has_top_right_bottom_left_match_c() {
    let mut checks = 0u64;
    // sb_size: BLOCK_64X64 (12), BLOCK_128X128 (15).
    for &sb in &[12usize, 15] {
        let sb_mi = MI_SIZE_HIGH[sb];
        for bsize in 0..22usize {
            if !fits(sb, bsize) {
                continue;
            }
            // Partitions: NONE (0), and VERT_A (6) when a vert table exists.
            let parts: &[usize] = if VERT_OK.contains(&bsize) {
                &[0usize, 6]
            } else {
                &[0usize]
            };
            for &partition in parts {
                for &(ssx, ssy) in &[(0i32, 0i32), (1, 1), (1, 0), (0, 1)] {
                    let pbw = (MI_SIZE_WIDE[bsize] >> ssx).max(1);
                    let pbh = (MI_SIZE_HIGH[bsize] >> ssy).max(1);
                    // Block positions spanning the superblock (covers all
                    // blk_row/col_in_sb after the >> shift).
                    let mi_vals: Vec<i32> = (0..sb_mi).collect();
                    for &mi_row in &mi_vals {
                        for &mi_col in &mi_vals {
                            for &txsz in &[0usize, 2] {
                                // TU offsets hitting both the "enough pixels" and
                                // the table branches.
                                for &co in &[0, pbw - 1, pbw] {
                                    for &ro in &[0, pbh - 1, pbh] {
                                        for &(a, b) in &[(true, true), (true, false), (false, true)]
                                        {
                                            let g = has_top_right(
                                                sb, bsize, mi_row, mi_col, a, b, partition, txsz,
                                                ro, co, ssx, ssy,
                                            );
                                            let w = c::ref_has_top_right(
                                                sb, bsize, mi_row, mi_col, a, b, partition, txsz,
                                                ro, co, ssx, ssy,
                                            );
                                            assert_eq!(g, w, "has_top_right sb={sb} bsize={bsize} mi=({mi_row},{mi_col}) avail=({a},{b}) part={partition} tx={txsz} off=({ro},{co}) ss=({ssx},{ssy})");

                                            let g2 = has_bottom_left(
                                                sb, bsize, mi_row, mi_col, a, b, partition, txsz,
                                                ro, co, ssx, ssy,
                                            );
                                            let w2 = c::ref_has_bottom_left(
                                                sb, bsize, mi_row, mi_col, a, b, partition, txsz,
                                                ro, co, ssx, ssy,
                                            );
                                            assert_eq!(g2, w2, "has_bottom_left sb={sb} bsize={bsize} mi=({mi_row},{mi_col}) avail=({a},{b}) part={partition} tx={txsz} off=({ro},{co}) ss=({ssx},{ssy})");
                                            checks += 2;
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
    assert!(checks > 100_000, "expected a broad sweep, got {checks}");
}
