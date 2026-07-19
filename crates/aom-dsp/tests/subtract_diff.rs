//! Differential harness for the residual generator (aom_subtract_block) vs C
//! libaom: diff = src - pred, row by row. Exercises strided buffers (independent
//! diff/src/pred row strides) as well as the contiguous case. Lowbd (u8) + highbd
//! (u16, bd 8/10/12 clamped magnitudes).

use aom_dsp::dist::{highbd_subtract_block, subtract_block};
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
    fn range(&mut self, hi: u32) -> u32 {
        (self.next() % hi as u64) as u32
    }
}

// (rows, cols) — transform block dimensions.
const DIMS: [(usize, usize); 10] =
    [(4, 4), (8, 8), (16, 16), (32, 32), (4, 8), (8, 4), (16, 8), (8, 16), (16, 4), (4, 16)];

#[test]
fn subtract_block_differential() {
    let mut rng = Rng(0x5067_c0de_9e37_79b9);
    for &(rows, cols) in &DIMS {
        for _ in 0..3000 {
            // Strided (pad) or contiguous.
            let pad = rng.range(4) as usize;
            let (ds, ss, ps) = (cols + pad, cols + rng.range(4) as usize, cols + rng.range(4) as usize);
            let src: Vec<u8> = (0..rows * ss).map(|_| rng.range(256) as u8).collect();
            let pred: Vec<u8> = (0..rows * ps).map(|_| rng.range(256) as u8).collect();
            let mut diff_r = vec![0i16; rows * ds];
            let mut diff_c = vec![0i16; rows * ds];
            subtract_block(rows, cols, &mut diff_r, ds, &src, ss, &pred, ps);
            c::ref_subtract_block(rows, cols, &mut diff_c, ds, &src, ss, &pred, ps);
            assert_eq!(diff_r, diff_c, "subtract rows={rows} cols={cols} strides=({ds},{ss},{ps})");
        }
    }
}

#[test]
fn highbd_subtract_block_differential() {
    let mut rng = Rng(0x5067_c057_0000_b111);
    for &(rows, cols) in &DIMS {
        for &bd in &[8u8, 10, 12] {
            let maxv = 1u32 << bd;
            for _ in 0..1500 {
                let (ds, ss, ps) =
                    (cols + rng.range(4) as usize, cols + rng.range(4) as usize, cols + rng.range(4) as usize);
                let src: Vec<u16> = (0..rows * ss).map(|_| rng.range(maxv) as u16).collect();
                let pred: Vec<u16> = (0..rows * ps).map(|_| rng.range(maxv) as u16).collect();
                let mut diff_r = vec![0i16; rows * ds];
                let mut diff_c = vec![0i16; rows * ds];
                highbd_subtract_block(rows, cols, &mut diff_r, ds, &src, ss, &pred, ps);
                c::ref_highbd_subtract_block(rows, cols, &mut diff_c, ds, &src, ss, &pred, ps);
                assert_eq!(diff_r, diff_c, "hbd subtract rows={rows} cols={cols} bd={bd}");
            }
        }
    }
}
