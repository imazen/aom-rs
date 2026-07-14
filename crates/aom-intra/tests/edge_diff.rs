//! Differential harness for the intra edge-filter DSP vs C libaom:
//! intra_edge_filter_strength, av1_use_intra_edge_upsample,
//! av1_filter_intra_edge_c, av1_upsample_intra_edge_c.

use aom_intra::edge::{edge_filter_strength, filter_intra_edge, upsample_intra_edge, use_upsample};
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
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

#[test]
fn strength_and_upsample_decisions_exhaustive() {
    // block dims 4..64, delta -90..90, both filter types.
    for &bs0 in &[4, 8, 16, 32, 64] {
        for &bs1 in &[4, 8, 16, 32, 64] {
            for delta in -90..=90 {
                for ty in 0..2 {
                    assert_eq!(
                        edge_filter_strength(bs0, bs1, delta, ty),
                        c::ref_intra_edge_strength(bs0, bs1, delta, ty),
                        "strength {bs0}+{bs1} d={delta} ty={ty}"
                    );
                    assert_eq!(
                        use_upsample(bs0, bs1, delta, ty),
                        c::ref_use_intra_edge_upsample(bs0, bs1, delta, ty),
                        "upsample {bs0}+{bs1} d={delta} ty={ty}"
                    );
                }
            }
        }
    }
}

#[test]
fn filter_intra_edge_byte_identical() {
    let mut rng = Rng(0x_ed6e_f117_0000_1111);
    for sz in 2..=65usize {
        for strength in 0..=3 {
            for _ in 0..200 {
                let base: Vec<u8> = (0..sz).map(|_| rng.range(0, 256) as u8).collect();
                let mut a = base.clone();
                let mut b = base.clone();
                filter_intra_edge(&mut a, sz, strength);
                c::ref_filter_intra_edge(&mut b, 0, sz, strength);
                assert_eq!(a, b, "filter sz={sz} strength={strength}");
            }
        }
    }
}

#[test]
fn upsample_intra_edge_byte_identical() {
    let mut rng = Rng(0x_c057_9a1e_0000_2222);
    const OFF: usize = 4;
    for sz in 1..=16usize {
        for _ in 0..300 {
            // buffer: OFF pad bytes + (2*sz + a few) working region.
            let n = OFF + 2 * sz + 4;
            let base: Vec<u8> = (0..n).map(|_| rng.range(0, 256) as u8).collect();
            let mut a = base.clone();
            let mut b = base.clone();
            upsample_intra_edge(&mut a, OFF, sz);
            c::ref_upsample_intra_edge(&mut b, OFF, sz);
            assert_eq!(a, b, "upsample sz={sz}");
        }
    }
}

#[test]
fn highbd_filter_intra_edge_byte_identical() {
    let mut rng = Rng(0x_86bd_ed6e_0000_3333);
    for &bd in &[8u8, 10, 12] {
        let max = (1u32 << bd) - 1;
        for sz in 2..=65usize {
            for strength in 0..=3 {
                for _ in 0..120 {
                    let base: Vec<u16> =
                        (0..sz).map(|_| (rng.next() as u32 & max) as u16).collect();
                    let mut a = base.clone();
                    let mut b = base.clone();
                    aom_intra::edge::highbd_filter_intra_edge(&mut a, sz, strength);
                    c::ref_highbd_filter_intra_edge(&mut b, 0, sz, strength);
                    assert_eq!(a, b, "hbd filter bd={bd} sz={sz} s={strength}");
                }
            }
        }
    }
}

#[test]
fn highbd_upsample_intra_edge_byte_identical() {
    let mut rng = Rng(0x_86bd_c057_0000_4444);
    const OFF: usize = 4;
    for &bd in &[8u8, 10, 12] {
        let max = (1u32 << bd) - 1;
        for sz in 1..=16usize {
            for _ in 0..200 {
                let n = OFF + 2 * sz + 4;
                let base: Vec<u16> = (0..n).map(|_| (rng.next() as u32 & max) as u16).collect();
                let mut a = base.clone();
                let mut b = base.clone();
                aom_intra::edge::highbd_upsample_intra_edge(&mut a, OFF, sz, bd);
                c::ref_highbd_upsample_intra_edge(&mut b, OFF, sz, bd);
                assert_eq!(a, b, "hbd upsample bd={bd} sz={sz}");
            }
        }
    }
}
