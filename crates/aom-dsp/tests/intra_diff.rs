//! Differential harness for the non-directional intra predictors vs C libaom
//! v3.14.1: every (mode 0..10) x (block size 0..19) with random neighbours.

use aom_dsp::intra::{predict, AboveRef};
use aom_sys_ref as c;

// Must match src/lib.rs mode indices and the shim's size ordering.
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
    fn byte(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
}

#[test]
fn intra_predictors_byte_identical() {
    let mut rng = Rng(0x_1234_face_b00c_0007);
    for mode in 0..10usize {
        for (size_idx, &(bw, bh)) in SIZES.iter().enumerate() {
            for _ in 0..2000 {
                // above_tl[0] = top-left, then bw above samples.
                let above_tl: Vec<u8> = (0..bw + 1).map(|_| rng.byte()).collect();
                let left: Vec<u8> = (0..bh).map(|_| rng.byte()).collect();

                let mut got = vec![0u8; bw * bh];
                predict(mode, &mut got, bw, bw, bh, &AboveRef(&above_tl), &left);

                let want = c::ref_intra_pred(mode, size_idx, bw, bh, &above_tl, &left);
                assert_eq!(
                    got, want,
                    "intra divergence mode={mode} size={bw}x{bh}\nabove={above_tl:?}\nleft={left:?}"
                );
            }
        }
    }
}
