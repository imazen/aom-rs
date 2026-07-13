//! Differential harness for highbd (10/12-bit) SAD and variance vs C libaom
//! v3.14.1 across all 22 block sizes and bit depths 8/10/12.
//!
//! These are the highbd counterparts of the speed-0 encoder motion-search /
//! RDO workhorses (`aom_highbd_sad*`, `aom_highbd_*_variance*`).

use aom_dist::{
    highbd_masked_sad, highbd_obmc_sad, highbd_sad, highbd_sad_avg, highbd_sub_pixel_variance,
    highbd_variance,
};
use aom_sys_ref as c;

const SIZES: [(usize, usize); 22] = [
    (4, 4), (4, 8), (4, 16), (8, 4), (8, 8), (8, 16), (8, 32), (16, 4), (16, 8), (16, 16), (16, 32),
    (16, 64), (32, 8), (32, 16), (32, 32), (32, 64), (64, 16), (64, 32), (64, 64), (64, 128),
    (128, 64), (128, 128),
];

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
    fn px(&mut self, mask: u16) -> u16 {
        (self.next() as u16) & mask
    }
}

fn plane(rng: &mut Rng, stride: usize, rows: usize, mask: u16) -> Vec<u16> {
    (0..stride * rows).map(|_| rng.px(mask)).collect()
}

#[test]
fn hbd_sad_variance_byte_identical() {
    let mut rng = Rng(0x_5add_1234_5678_9abc);
    for &bd in &[8u8, 10, 12] {
        let mask = ((1u32 << bd) - 1) as u16;
        for (idx, &(w, h)) in SIZES.iter().enumerate() {
            let a_stride = w + 8;
            let b_stride = w + 8;
            for _ in 0..1500 {
                let a = plane(&mut rng, a_stride, h + 2, mask);
                let b = plane(&mut rng, b_stride, h + 2, mask);

                // highbd SAD (bit-depth independent, but exercised per range)
                let got = highbd_sad(&a, a_stride, &b, b_stride, w, h);
                let want = c::ref_hbd_sad(idx, &a, a_stride, &b, b_stride);
                assert_eq!(got, want, "hbd_sad {w}x{h} bd={bd}");

                // highbd avg SAD (compound); 17 non-4-side sizes only
                if !matches!((w, h), (4, 4) | (4, 8) | (4, 16) | (8, 4) | (16, 4)) {
                    let sp: Vec<u16> = (0..w * h).map(|_| rng.px(mask)).collect();
                    let ga = highbd_sad_avg(&a, a_stride, &b, b_stride, &sp, w, h);
                    let wa = c::ref_hbd_sad_avg(idx, &a, a_stride, &b, b_stride, &sp);
                    assert_eq!(ga, wa, "hbd_sad_avg {w}x{h} bd={bd}");
                }

                // highbd masked SAD (all 22 sizes, both polarities)
                let sp2: Vec<u16> = (0..w * h).map(|_| rng.px(mask)).collect();
                let m_stride = w + 8;
                let msk: Vec<u8> = (0..m_stride * (h + 2)).map(|_| (rng.next() % 65) as u8).collect();
                for inv in [false, true] {
                    let gm = highbd_masked_sad(&a, a_stride, &b, b_stride, &sp2, &msk, m_stride, inv, w, h);
                    let wm = c::ref_hbd_masked_sad(idx, &a, a_stride, &b, b_stride, &sp2, &msk, m_stride, inv);
                    assert_eq!(gm, wm, "hbd_masked_sad {w}x{h} bd={bd} inv={inv}");
                }

                // highbd OBMC SAD
                let wsrc: Vec<i32> = (0..w * h).map(|_| (rng.next() % (4096 * 4096)) as i32).collect();
                let omask: Vec<i32> = (0..w * h).map(|_| (rng.next() % 4097) as i32).collect();
                let go = highbd_obmc_sad(&a, a_stride, &wsrc, &omask, w, h);
                let wo = c::ref_hbd_obmc_sad(idx, &a, a_stride, &wsrc, &omask);
                assert_eq!(go, wo, "hbd_obmc_sad {w}x{h} bd={bd}");

                // highbd variance (bd-normalised)
                let (gv, gs) = highbd_variance(&a, a_stride, &b, b_stride, w, h, bd);
                let (wv, ws) = c::ref_hbd_variance(idx, bd, &a, a_stride, &b, b_stride);
                assert_eq!((gv, gs), (wv, ws), "hbd_variance {w}x{h} bd={bd}");

                // highbd sub-pixel variance over all 8x8 subpel offsets
                let xo = (rng.next() % 8) as usize;
                let yo = (rng.next() % 8) as usize;
                let (gv2, gs2) = highbd_sub_pixel_variance(&a, a_stride, xo, yo, &b, b_stride, w, h, bd);
                let (wv2, ws2) = c::ref_hbd_subpel_var(idx, bd, &a, a_stride, xo, yo, &b, b_stride);
                assert_eq!((gv2, gs2), (wv2, ws2), "hbd_subpel_var {w}x{h} bd={bd} xo={xo} yo={yo}");
            }
        }
    }
}
