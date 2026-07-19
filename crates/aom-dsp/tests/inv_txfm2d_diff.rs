//! Differential harness for the inverse 2-D transform + reconstruction vs C
//! libaom v3.14.1: every supported (tx_type x tx_size) x bd in {8,10,12}.
//! Both sides get an identical randomized destination buffer; the reconstructed
//! pixel planes must be byte-identical.

use aom_sys_ref as c;
use aom_dsp::transform::inv_txfm2d::{av1_inv_txfm2d_add, inv_input_len, inv_txfm_valid};

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
    fn coeff(&mut self) -> i32 {
        // dequantized coefficient range; row clamp_buf keeps C defined regardless
        (self.next() % (1 << 17)) as i32 - (1 << 16)
    }
    fn pixel(&mut self, bd: i32) -> u16 {
        (self.next() % (1u64 << bd)) as u16
    }
}

fn check(rng: &mut Rng, tx_size: usize, tx_type: usize, bd: i32) {
    let (w, h) = (W[tx_size], H[tx_size]);
    let input: Vec<i32> = (0..inv_input_len(tx_size)).map(|_| rng.coeff()).collect();
    let dest0: Vec<u16> = (0..w * h).map(|_| rng.pixel(bd)).collect();

    let mut got = dest0.clone();
    av1_inv_txfm2d_add(&input, &mut got, w, tx_type, tx_size, bd);

    let mut want = dest0.clone();
    c::ref_inv_txfm2d_add(tx_size, &input, &mut want, w, tx_type, bd);

    assert_eq!(
        got, want,
        "inv_txfm2d_add divergence: tx_size={tx_size} ({w}x{h}) tx_type={tx_type} bd={bd}\ninput={input:?}"
    );
}

#[test]
fn inv_txfm2d_edge_cases() {
    let mut rng = Rng(3);
    for tx_size in 0..19 {
        for tx_type in 0..16 {
            if !inv_txfm_valid(tx_type, tx_size) {
                continue;
            }
            for bd in [8, 10, 12] {
                // all-zero coeffs (dest must pass through unchanged)
                let (w, h) = (W[tx_size], H[tx_size]);
                let input = vec![0i32; inv_input_len(tx_size)];
                let dest0: Vec<u16> = (0..w * h).map(|_| rng.pixel(bd)).collect();
                let mut got = dest0.clone();
                av1_inv_txfm2d_add(&input, &mut got, w, tx_type, tx_size, bd);
                let mut want = dest0.clone();
                c::ref_inv_txfm2d_add(tx_size, &input, &mut want, w, tx_type, bd);
                assert_eq!(got, want, "zero-coeff divergence tx_size={tx_size} bd={bd}");
            }
        }
    }
}

#[test]
fn inv_txfm2d_differential_fuzz() {
    let mut rng = Rng(0x_c0ffee_5eed_1234);
    for tx_size in 0..19 {
        for tx_type in 0..16 {
            if !inv_txfm_valid(tx_type, tx_size) {
                continue;
            }
            for bd in [8, 10, 12] {
                for _ in 0..700 {
                    check(&mut rng, tx_size, tx_type, bd);
                }
            }
        }
    }
}

/// The lossless 4x4 Walsh–Hadamard inverse-add (`av1_highbd_iwht4x4_add`) vs the
/// REAL exported C kernels (`av1_highbd_iwht4x4_16_add_c` / `_1_add_c`). Both eob
/// arms (full 16-point and DC-only), all three bit depths, and strided
/// destinations (the decoder feeds a plane stride > 4). Inputs span the full
/// dequant clamp range `±(1<<(7+bd))` so the whole butterfly is exercised.
#[test]
fn highbd_iwht4x4_add_matches_c() {
    use aom_dsp::transform::inv_txfm2d::av1_highbd_iwht4x4_add;
    let mut rng = Rng(0x1D7_4A17);
    let (mut full_cases, mut dc_cases) = (0usize, 0usize);
    for bd in [8i32, 10, 12] {
        let bound = 1i64 << (7 + bd); // dequant_txb clamp bound
        let span = (2 * bound) as u64;
        for stride in [4usize, 7, 16, 33] {
            for _ in 0..3000 {
                let full = rng.next() & 1 == 0;
                let eob = if full { 2 + (rng.next() % 15) as usize } else { 1 };
                let mut input = [0i32; 16];
                if full {
                    for v in input.iter_mut() {
                        *v = ((rng.next() % span) as i64 - bound) as i32;
                    }
                    full_cases += 1;
                } else {
                    input[0] = ((rng.next() % span) as i64 - bound) as i32;
                    dc_cases += 1;
                }
                let dest0: Vec<u16> = (0..4 * stride).map(|_| rng.pixel(bd)).collect();

                let mut got = dest0.clone();
                av1_highbd_iwht4x4_add(&input, &mut got, stride, eob, bd);

                let mut want = dest0.clone();
                c::ref_highbd_iwht4x4_add(&input, &mut want, stride, eob, bd);

                assert_eq!(
                    got, want,
                    "WHT divergence: bd={bd} stride={stride} eob={eob} input={input:?}"
                );
            }
        }
    }
    // Anti-vacuous: both dispatch arms must actually be exercised.
    assert!(
        full_cases > 1000 && dc_cases > 1000,
        "WHT diff must exercise both eob>1 ({full_cases}) and eob<=1 ({dc_cases}) arms"
    );
}
