//! aom-cdef — bit-exact CDEF (Constrained Directional Enhancement Filter)
//! kernels, port of libaom v3.14.1 `av1/common/cdef_block.c`. Both tracks.
//! Starts with `cdef_find_dir` (the 8x8 direction search).

/// Bit-exact port of `cdef_find_dir_c`. Operates on an 8x8 window of `img`
/// (row stride `stride`). Returns `(best_dir, var)`.
pub fn cdef_find_dir(img: &[u16], stride: usize, coeff_shift: i32) -> (i32, i32) {
    const DIV_TABLE: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];
    let mut cost = [0i32; 8];
    let mut partial = [[0i32; 15]; 8];

    for i in 0..8usize {
        for j in 0..8usize {
            // -128 to reduce the range of the squared partial sums.
            let x = ((img[i * stride + j] as i32) >> coeff_shift) - 128;
            let add = |p: &mut i32| *p = p.wrapping_add(x);
            add(&mut partial[0][i + j]);
            add(&mut partial[1][i + j / 2]);
            add(&mut partial[2][i]);
            add(&mut partial[3][3 + i - j / 2]);
            add(&mut partial[4][7 + i - j]);
            add(&mut partial[5][3 - i / 2 + j]);
            add(&mut partial[6][j]);
            add(&mut partial[7][i / 2 + j]);
        }
    }

    let sq = |a: i32| a.wrapping_mul(a);
    for i in 0..8 {
        cost[2] = cost[2].wrapping_add(sq(partial[2][i]));
        cost[6] = cost[6].wrapping_add(sq(partial[6][i]));
    }
    cost[2] = cost[2].wrapping_mul(DIV_TABLE[8]);
    cost[6] = cost[6].wrapping_mul(DIV_TABLE[8]);

    for i in 0..7 {
        cost[0] = cost[0]
            .wrapping_add(sq(partial[0][i]).wrapping_add(sq(partial[0][14 - i])).wrapping_mul(DIV_TABLE[i + 1]));
        cost[4] = cost[4]
            .wrapping_add(sq(partial[4][i]).wrapping_add(sq(partial[4][14 - i])).wrapping_mul(DIV_TABLE[i + 1]));
    }
    cost[0] = cost[0].wrapping_add(sq(partial[0][7]).wrapping_mul(DIV_TABLE[8]));
    cost[4] = cost[4].wrapping_add(sq(partial[4][7]).wrapping_mul(DIV_TABLE[8]));

    let mut i = 1;
    while i < 8 {
        for j in 0..5 {
            cost[i] = cost[i].wrapping_add(sq(partial[i][3 + j]));
        }
        cost[i] = cost[i].wrapping_mul(DIV_TABLE[8]);
        for j in 0..3 {
            cost[i] = cost[i].wrapping_add(
                sq(partial[i][j])
                    .wrapping_add(sq(partial[i][10 - j]))
                    .wrapping_mul(DIV_TABLE[2 * j + 2]),
            );
        }
        i += 2;
    }

    let mut best_cost = 0i32;
    let mut best_dir = 0usize;
    for i in 0..8 {
        if cost[i] > best_cost {
            best_cost = cost[i];
            best_dir = i;
        }
    }
    let var = (best_cost.wrapping_sub(cost[(best_dir + 4) & 7])) >> 10;
    (best_dir as i32, var)
}
