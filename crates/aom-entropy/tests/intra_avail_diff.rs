//! Differential harness for `intra_avail` — the intra neighbour-availability
//! composition (`av1_predict_intra_block`'s n_top/n_topright/n_left/n_bottomleft
//! computation) vs C libaom v3.14.1 (`ref_intra_avail`, a verbatim transcription
//! that calls the same has_top_right/has_bottom_left). Swept over superblock size,
//! block size, plane subsampling, block position across a frame/tile, transform
//! size, TU offsets, intra mode, angle-delta, and filter-intra.

use aom_entropy::partition::intra_avail;
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

fn fits(sb: usize, bsize: usize) -> bool {
    BLOCK_SIZE_WIDE[bsize] <= BLOCK_SIZE_WIDE[sb] && BLOCK_SIZE_HIGH[bsize] <= BLOCK_SIZE_HIGH[sb]
}

#[test]
fn intra_avail_matches_c() {
    // 128x128-mi frame, single tile covering it.
    let (mi_cols, mi_rows) = (32i32, 32i32);
    let (tile_col_end, tile_row_end) = (mi_cols, mi_rows);
    let mut checks = 0u64;
    // Representative intra modes: DC, V, D45 (AR), D203 (BL), D67 (AR), SMOOTH, PAETH.
    let modes = [0usize, 1, 3, 7, 8, 9, 12];
    for &sb in &[12usize, 15] {
        for bsize in 0..22usize {
            if !fits(sb, bsize) {
                continue;
            }
            for &(ss_x, ss_y) in &[(0i32, 0i32), (1, 1), (1, 0), (0, 1)] {
                let wpx = BLOCK_SIZE_WIDE[bsize] >> ss_x;
                let hpx = BLOCK_SIZE_HIGH[bsize] >> ss_y;
                let cmax = (MI_SIZE_WIDE[bsize] >> ss_x).max(1);
                let rmax = (MI_SIZE_HIGH[bsize] >> ss_y).max(1);
                for &mi_row in &[0i32, 1, 8, 16, 31] {
                    for &mi_col in &[0i32, 1, 8, 16, 31] {
                        // Luma neighbour flags for a single tile at (0,0).
                        let up = mi_row > 0;
                        let left = mi_col > 0;
                        for &tx_size in &[0usize, 2] {
                            for &row_off in &[0i32, rmax / 2] {
                                for &col_off in &[0i32, cmax / 2] {
                                    for &mode in &modes {
                                        let is_dr = (1..=8).contains(&mode);
                                        let deltas: &[i32] = if is_dr { &[-9, 0, 9] } else { &[0] };
                                        for &ad in deltas {
                                            for &ufi in &[false, true] {
                                                let g = intra_avail(
                                                    sb,
                                                    bsize,
                                                    mi_row,
                                                    mi_col,
                                                    up,
                                                    left,
                                                    tile_col_end,
                                                    tile_row_end,
                                                    0,
                                                    tx_size,
                                                    ss_x,
                                                    ss_y,
                                                    row_off,
                                                    col_off,
                                                    wpx,
                                                    hpx,
                                                    mi_cols,
                                                    mi_rows,
                                                    mode,
                                                    ad,
                                                    ufi,
                                                );
                                                let w = c::ref_intra_avail(
                                                    sb,
                                                    bsize,
                                                    mi_row,
                                                    mi_col,
                                                    up,
                                                    left,
                                                    tile_col_end,
                                                    tile_row_end,
                                                    0,
                                                    tx_size,
                                                    ss_x,
                                                    ss_y,
                                                    row_off,
                                                    col_off,
                                                    wpx,
                                                    hpx,
                                                    mi_cols,
                                                    mi_rows,
                                                    mode,
                                                    ad,
                                                    ufi,
                                                );
                                                assert_eq!(g, w, "intra_avail sb={sb} bsize={bsize} mi=({mi_row},{mi_col}) ss=({ss_x},{ss_y}) tx={tx_size} off=({row_off},{col_off}) mode={mode} ad={ad} ufi={ufi}");
                                                checks += 1;
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
    assert!(checks > 50_000, "expected a broad sweep, got {checks}");
}
