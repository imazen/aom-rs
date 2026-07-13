//! Differential harness for the partition-symbol CDF primitives
//! (partition_cdf_length + the edge-block gather transforms) vs C libaom.

use aom_entropy::partition::{
    partition_cdf_length, partition_gather_horz_alike, partition_gather_vert_alike,
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
fn partition_cdf_length_matches_c() {
    // All 22 BLOCK_SIZE values (BLOCK_4X4=0 .. BLOCK_SIZES_ALL).
    for bsize in 0..22 {
        assert_eq!(
            partition_cdf_length(bsize as usize),
            c::ref_partition_cdf_length(bsize) as usize,
            "partition_cdf_length bsize={bsize}"
        );
    }
}

#[test]
fn partition_gather_matches_c() {
    let mut rng = Rng(0x9a27_c0de_a11a_0009);
    // A valid inverse-cumulative CDF over EXT_PARTITION_TYPES(10) symbols is a
    // strictly-decreasing sequence 32768 > c0 > c1 > ... > c9 = 0, stored as
    // [c0..c9, count]. Build one by drawing sorted breakpoints.
    for _ in 0..200_000 {
        // draw 9 distinct interior points in (0, 32768), sort descending
        let mut pts = [0i32; 9];
        for p in &mut pts {
            *p = 1 + (rng.next() % 32766) as i32; // [1, 32767]
        }
        pts.sort_unstable();
        pts.reverse(); // descending
        // cdf[0..10]: cdf[i] = pts[i] for i<9, cdf[9] = 0; ensure strictly decreasing
        let mut cdf = [0u16; 11];
        let mut prev = 32768i32;
        for i in 0..9 {
            // keep strictly below prev; if a duplicate collapsed ordering, nudge
            let v = pts[i].min(prev - 1).max(9 - i as i32);
            cdf[i] = v as u16;
            prev = v;
        }
        cdf[9] = 0;
        cdf[10] = 0; // count field, unused by gather

        for &bsize in &[3i32, 8, 12, 15] {
            // 8x8, 16x16, 64x64, 128x128
            assert_eq!(
                partition_gather_vert_alike(&cdf, bsize as usize),
                c::ref_partition_gather_vert(&cdf, bsize),
                "gather_vert bsize={bsize} cdf={cdf:?}"
            );
            assert_eq!(
                partition_gather_horz_alike(&cdf, bsize as usize),
                c::ref_partition_gather_horz(&cdf, bsize),
                "gather_horz bsize={bsize} cdf={cdf:?}"
            );
        }
    }
}
