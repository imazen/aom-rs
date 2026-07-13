//! Differential harness for SAD / variance / sub-pixel variance vs C libaom
//! v3.14.1 across all 22 block sizes.

use aom_dist::{highbd_sse, masked_sad, obmc_sad, sad, sad_avg, sse, sub_pixel_variance, variance};
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
    fn u8(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
}

// Plane with padding so subpel first-pass can read one extra column/row.
fn plane(rng: &mut Rng, stride: usize, rows: usize) -> Vec<u8> {
    (0..stride * rows).map(|_| rng.u8()).collect()
}

#[test]
fn sad_variance_subpel_byte_identical() {
    let mut rng = Rng(0x_5ad0_1234_5678_9abc);
    for (idx, &(w, h)) in SIZES.iter().enumerate() {
        let a_stride = w + 8;
        let b_stride = w + 8;
        for _ in 0..3000 {
            // a needs (h+1) rows x (w+1) cols for subpel; give margin.
            let a = plane(&mut rng, a_stride, h + 2);
            let b = plane(&mut rng, b_stride, h + 2);

            // SAD
            let got = sad(&a, a_stride, &b, b_stride, w, h);
            let want = c::ref_sad(idx, &a, a_stride, &b, b_stride);
            assert_eq!(got, want, "sad {w}x{h}");

            // avg SAD (compound prediction): second_pred is contiguous w*h.
            // libaom only compiles avg-SAD for the 17 non-4-side sizes in this
            // config (compound isn't used for 4-wide/4-tall sub-blocks).
            if !matches!((w, h), (4, 4) | (4, 8) | (4, 16) | (8, 4) | (16, 4)) {
                let sp: Vec<u8> = (0..w * h).map(|_| rng.u8()).collect();
                let got_avg = sad_avg(&a, a_stride, &b, b_stride, &sp, w, h);
                let want_avg = c::ref_sad_avg(idx, &a, a_stride, &b, b_stride, &sp);
                assert_eq!(got_avg, want_avg, "sad_avg {w}x{h}");
            }

            // masked SAD (all 22 sizes): mask values 0..=64, both mask polarities.
            let sp2: Vec<u8> = (0..w * h).map(|_| rng.u8()).collect();
            let m_stride = w + 8;
            let msk: Vec<u8> = (0..m_stride * (h + 2)).map(|_| (rng.next() % 65) as u8).collect();
            for inv in [false, true] {
                let gm = masked_sad(&a, a_stride, &b, b_stride, &sp2, &msk, m_stride, inv, w, h);
                let wm = c::ref_masked_sad(idx, &a, a_stride, &b, b_stride, &sp2, &msk, m_stride, inv);
                assert_eq!(gm, wm, "masked_sad {w}x{h} inv={inv}");
            }

            // OBMC SAD: wsrc/mask contiguous i32 (mask in [0,4096], wsrc weighted).
            let wsrc: Vec<i32> = (0..w * h).map(|_| (rng.next() % (256 * 4096)) as i32).collect();
            let obmc_mask: Vec<i32> = (0..w * h).map(|_| (rng.next() % 4097) as i32).collect();
            let go = obmc_sad(&a, a_stride, &wsrc, &obmc_mask, w, h);
            let wo = c::ref_obmc_sad(idx, &a, a_stride, &wsrc, &obmc_mask);
            assert_eq!(go, wo, "obmc_sad {w}x{h}");

            // SSE (sum of squared errors, generic w×h) lowbd + highbd
            let gs = sse(&a, a_stride, &b, b_stride, w, h);
            let ws2 = c::ref_sse(&a, a_stride, &b, b_stride, w, h);
            assert_eq!(gs, ws2, "sse {w}x{h}");
            let ah: Vec<u16> = a.iter().map(|&v| v as u16 * 15).collect();
            let bh: Vec<u16> = b.iter().map(|&v| v as u16 * 15).collect();
            let ghs = highbd_sse(&ah, a_stride, &bh, b_stride, w, h);
            let whs = c::ref_hbd_sse(&ah, a_stride, &bh, b_stride, w, h);
            assert_eq!(ghs, whs, "highbd_sse {w}x{h}");

            // variance
            let (gv, gs) = variance(&a, a_stride, &b, b_stride, w, h);
            let (wv, ws) = c::ref_variance(idx, &a, a_stride, &b, b_stride);
            assert_eq!((gv, gs), (wv, ws), "variance {w}x{h}");

            // sub-pixel variance over all 8x8 subpel offsets
            let xo = (rng.next() % 8) as usize;
            let yo = (rng.next() % 8) as usize;
            let (gv2, gs2) = sub_pixel_variance(&a, a_stride, xo, yo, &b, b_stride, w, h);
            let (wv2, ws2) = c::ref_subpel_var(idx, &a, a_stride, xo, yo, &b, b_stride);
            assert_eq!((gv2, gs2), (wv2, ws2), "subpel_var {w}x{h} xo={xo} yo={yo}");
        }
    }
}
