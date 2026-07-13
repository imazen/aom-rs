//! Differential harness for the forward 2-D transform vs C libaom v3.14.1:
//! every supported (tx_type in 0..16) x (tx_size in 0..19) combination, output
//! compared byte-for-byte over the full coefficient buffer.
//!
//! Input is bounded to the 8-bit residual range [-255, 255] — exactly the
//! conformant range the encoder produces, where bit-identity is the real
//! contract and the C reference stays within defined behaviour.

use aom_sys_ref as c;
use aom_transform::txfm2d::{av1_fwd_txfm2d, fwd_txfm_valid};

const W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

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
    fn residual(&mut self) -> i16 {
        (self.next() % 511) as i16 - 255
    }
}

fn check(rng: &mut Rng, tx_size: usize, tx_type: usize, fill: impl Fn(&mut Rng) -> i16) {
    let (w, h) = (W[tx_size], H[tx_size]);
    let input: Vec<i16> = (0..w * h).map(|_| fill(rng)).collect();
    let mut got = vec![0i32; w * h];
    av1_fwd_txfm2d(&input, &mut got, w, tx_type, tx_size);
    let want = c::ref_fwd_txfm2d(tx_size, &input, w, tx_type);
    assert_eq!(
        got, want,
        "fwd_txfm2d divergence: tx_size={tx_size} ({w}x{h}) tx_type={tx_type}\ninput={input:?}"
    );
}

#[test]
fn txfm2d_edge_cases() {
    let mut rng = Rng(7);
    for tx_size in 0..19 {
        for tx_type in 0..16 {
            if !fwd_txfm_valid(tx_type, tx_size) {
                continue;
            }
            check(&mut rng, tx_size, tx_type, |_| 0); // all zero
            check(&mut rng, tx_size, tx_type, |_| 255); // saturated
            check(&mut rng, tx_size, tx_type, |_| -255);
        }
    }
}

#[test]
fn txfm2d_differential_fuzz() {
    let mut rng = Rng(0x_0bad_f00d_1337_c0de);
    let mut combos = 0;
    for tx_size in 0..19 {
        for tx_type in 0..16 {
            if !fwd_txfm_valid(tx_type, tx_size) {
                continue;
            }
            combos += 1;
            for _ in 0..2000 {
                check(&mut rng, tx_size, tx_type, |r| r.residual());
            }
        }
    }
    assert!(combos >= 100, "expected many valid combos, got {combos}");
}
