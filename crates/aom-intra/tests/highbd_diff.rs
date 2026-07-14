//! Differential harness for highbd (10/12-bit) intra predictors vs C libaom.
use aom_intra::{predict_highbd, AboveRef16};
use aom_sys_ref as c;

const SIZES: [(usize, usize); 19] = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (32, 64),
    (64, 32),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
    (16, 64),
    (64, 16),
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
}

#[test]
fn highbd_intra_byte_identical() {
    let mut rng = Rng(0x_11bd_9e37_79b9_1234);
    for &bd in &[10i32, 12] {
        let maxv = (1u32 << bd) - 1;
        for mode in 0..10usize {
            for (size_idx, &(bw, bh)) in SIZES.iter().enumerate() {
                for _ in 0..800 {
                    let above_tl: Vec<u16> = (0..bw + 1)
                        .map(|_| (rng.next() % (maxv as u64 + 1)) as u16)
                        .collect();
                    let left: Vec<u16> = (0..bh)
                        .map(|_| (rng.next() % (maxv as u64 + 1)) as u16)
                        .collect();
                    let mut got = vec![0u16; bw * bh];
                    predict_highbd(
                        mode,
                        &mut got,
                        bw,
                        bw,
                        bh,
                        &AboveRef16(&above_tl),
                        &left,
                        bd,
                    );
                    let want =
                        c::ref_highbd_intra_pred(mode, size_idx, bw, bh, &above_tl, &left, bd);
                    assert_eq!(got, want, "highbd intra mode={mode} {bw}x{bh} bd={bd}");
                }
            }
        }
    }
}
