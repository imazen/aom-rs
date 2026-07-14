//! Differential harness for directional intra predictors z1/z2/z3 vs C libaom
//! v3.14.1, over valid angle-derived (dx, dy) and both upsample settings.

use aom_intra::dir::{get_dx, get_dy, z1, z2, z3, EdgeRef};
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

const PAD: usize = 8;

fn edges(rng: &mut Rng, bw: usize, bh: usize) -> Vec<u8> {
    let n = PAD + 2 * (bw + bh) + 16;
    (0..n).map(|_| (rng.next() & 0xff) as u8).collect()
}

#[test]
fn dr_predictors_byte_identical() {
    let mut rng = Rng(0xa11c_e0dd_1234);
    let mut checks = 0u64;
    for &(bw, bh) in SIZES.iter() {
        for up in 0..2i32 {
            // z1: angle in (0,90), dy = 1, dx > 0.
            for angle in 1..90 {
                let dx = get_dx(angle);
                if dx == 0 {
                    continue;
                }
                let above = edges(&mut rng, bw, bh);
                let left = edges(&mut rng, bw, bh);
                let mut got = vec![0u8; bw * bh];
                z1(&mut got, bw, bw, bh, &EdgeRef::new(&above, PAD), up, dx);
                let want = c::ref_dr_pred(1, bw, bh, &above, &left, PAD, up, 0, dx, 1);
                assert_eq!(got, want, "z1 divergence {bw}x{bh} up={up} angle={angle}");
                checks += 1;
            }
            // z3: angle in (180,270), dx = 1, dy > 0.
            for angle in 181..270 {
                let dy = get_dy(angle);
                if dy == 0 {
                    continue;
                }
                let above = edges(&mut rng, bw, bh);
                let left = edges(&mut rng, bw, bh);
                let mut got = vec![0u8; bw * bh];
                z3(&mut got, bw, bw, bh, &EdgeRef::new(&left, PAD), up, dy);
                let want = c::ref_dr_pred(3, bw, bh, &above, &left, PAD, 0, up, 1, dy);
                assert_eq!(got, want, "z3 divergence {bw}x{bh} up={up} angle={angle}");
                checks += 1;
            }
        }
        // z2: angle in (90,180), dx>0 && dy>0. Independent upsample per edge.
        for up_a in 0..2i32 {
            for up_l in 0..2i32 {
                for angle in 91..180 {
                    let dx = get_dx(angle);
                    let dy = get_dy(angle);
                    if dx == 0 || dy == 0 {
                        continue;
                    }
                    let above = edges(&mut rng, bw, bh);
                    let left = edges(&mut rng, bw, bh);
                    let mut got = vec![0u8; bw * bh];
                    z2(
                        &mut got,
                        bw,
                        bw,
                        bh,
                        &EdgeRef::new(&above, PAD),
                        &EdgeRef::new(&left, PAD),
                        up_a,
                        up_l,
                        dx,
                        dy,
                    );
                    let want = c::ref_dr_pred(2, bw, bh, &above, &left, PAD, up_a, up_l, dx, dy);
                    assert_eq!(
                        got, want,
                        "z2 divergence {bw}x{bh} up_a={up_a} up_l={up_l} angle={angle}"
                    );
                    checks += 1;
                }
            }
        }
    }
    assert!(checks > 3000, "expected many checks, got {checks}");
}
