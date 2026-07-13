//! aom-cdef — bit-exact CDEF (Constrained Directional Enhancement Filter)
//! kernels, port of libaom v3.14.1 `av1/common/cdef_block.c`. Both tracks.
//! Starts with `cdef_find_dir` (the 8x8 direction search).


#![forbid(unsafe_code)]
pub const CDEF_BSTRIDE: usize = 144; // ALIGN_POWER_OF_TWO(128 + 16, 3)
pub const CDEF_VERY_LARGE: i32 = 0x4000;

const PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];
const SEC_TAPS: [i32; 2] = [2, 1];

// cdef_directions_padded[12][2] with CDEF_BSTRIDE=144; cdef_directions = &[2..].
#[rustfmt::skip]
static CDEF_DIRECTIONS_PADDED: [[i32; 2]; 12] = [
    [144, 288],   [144, 287],    // padding (dirs 6,7)
    [-143, -286], [1, -142],     // dir 0, 1
    [1, 2],       [1, 146],      // dir 2, 3
    [145, 290],   [144, 289],    // dir 4, 5
    [144, 288],   [144, 287],    // dir 6, 7
    [-143, -286], [1, -142],     // padding (dirs 0,1)
];

#[inline]
fn cdef_dir(dir: i32, k: usize) -> i32 {
    // cdef_directions[dir] == padded[dir + 2]
    CDEF_DIRECTIONS_PADDED[(dir + 2) as usize][k]
}

#[inline]
fn get_msb(n: u32) -> i32 {
    31 - n.leading_zeros() as i32
}

#[inline]
fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let shift = (damping - get_msb(threshold as u32)).max(0);
    let sign = if diff < 0 { -1 } else { 1 };
    let a = diff.abs();
    sign * (threshold - (a >> shift)).clamp(0, a)
}

/// Bit-exact port of `cdef_filter_block_internal` (8-bit). `in_buf`/`in_off`
/// give the block origin; stride is `CDEF_BSTRIDE`. Writes `dst`/`dstride`.
#[allow(clippy::too_many_arguments)]
pub fn cdef_filter_block(
    dst: &mut [u8], dstride: usize, in_buf: &[u16], in_off: usize,
    pri_strength: i32, sec_strength: i32, dir: i32, pri_damping: i32, sec_damping: i32,
    coeff_shift: i32, block_width: usize, block_height: usize,
    enable_primary: bool, enable_secondary: bool,
) {
    let clipping_required = enable_primary && enable_secondary;
    let s = CDEF_BSTRIDE as i32;
    let pri_taps = &PRI_TAPS[((pri_strength >> coeff_shift) & 1) as usize];
    let sec_taps = &SEC_TAPS;
    let g = |idx: i32| in_buf[(in_off as i32 + idx) as usize] as i32;
    for i in 0..block_height as i32 {
        for j in 0..block_width as i32 {
            let mut sum: i16 = 0;
            let x = g(i * s + j);
            let mut max = x;
            let mut min = x;
            for k in 0..2 {
                if enable_primary {
                    let p0 = g(i * s + j + cdef_dir(dir, k));
                    let p1 = g(i * s + j - cdef_dir(dir, k));
                    sum = sum.wrapping_add((pri_taps[k] * constrain(p0 - x, pri_strength, pri_damping)) as i16);
                    sum = sum.wrapping_add((pri_taps[k] * constrain(p1 - x, pri_strength, pri_damping)) as i16);
                    if clipping_required {
                        if p0 != CDEF_VERY_LARGE { max = max.max(p0); }
                        if p1 != CDEF_VERY_LARGE { max = max.max(p1); }
                        min = min.min(p0);
                        min = min.min(p1);
                    }
                }
                if enable_secondary {
                    let s0 = g(i * s + j + cdef_dir(dir + 2, k));
                    let s1 = g(i * s + j - cdef_dir(dir + 2, k));
                    let s2 = g(i * s + j + cdef_dir(dir - 2, k));
                    let s3 = g(i * s + j - cdef_dir(dir - 2, k));
                    if clipping_required {
                        if s0 != CDEF_VERY_LARGE { max = max.max(s0); }
                        if s1 != CDEF_VERY_LARGE { max = max.max(s1); }
                        if s2 != CDEF_VERY_LARGE { max = max.max(s2); }
                        if s3 != CDEF_VERY_LARGE { max = max.max(s3); }
                        min = min.min(s0);
                        min = min.min(s1);
                        min = min.min(s2);
                        min = min.min(s3);
                    }
                    sum = sum.wrapping_add((sec_taps[k] * constrain(s0 - x, sec_strength, sec_damping)) as i16);
                    sum = sum.wrapping_add((sec_taps[k] * constrain(s1 - x, sec_strength, sec_damping)) as i16);
                    sum = sum.wrapping_add((sec_taps[k] * constrain(s2 - x, sec_strength, sec_damping)) as i16);
                    sum = sum.wrapping_add((sec_taps[k] * constrain(s3 - x, sec_strength, sec_damping)) as i16);
                }
            }
            let sneg = (sum < 0) as i32;
            let mut y = (x as i16).wrapping_add(((8 + sum as i32 - sneg) >> 4) as i16) as i32;
            if clipping_required {
                y = y.clamp(min, max);
            }
            dst[(i as usize) * dstride + j as usize] = y as u8;
        }
    }
}

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
