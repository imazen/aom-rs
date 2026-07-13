//! Differential harness: aom-transform `av1_fdct4` vs C libaom v3.14.1 oracle.
//!
//! Contract: byte-identical output over the conformant input range for every
//! supported `cos_bit`. Inputs are bounded to +/- 2^17 so the C reference stays
//! within defined behaviour (its inner 32-bit `w0*in0` must not truly overflow;
//! that only happens for non-conformant inputs, where "bit-identical" would be
//! comparing against C undefined behaviour rather than a real divergence).

use aom_sys_ref::ref_fdct4;
use aom_transform::av1_fdct4;

/// Deterministic xorshift64* — reproducible, no external crates.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Signed value in [-(1<<bits), (1<<bits)].
    fn bounded(&mut self, bits: u32) -> i32 {
        let range = (1i64 << (bits + 1)) + 1;
        ((self.next_u64() as i64).rem_euclid(range) - (1i64 << bits)) as i32
    }
}

const STAGE_RANGE: [i8; 8] = [32; 8]; // consulted only by the disabled range checker

fn check(input: &[i32; 4], cos_bit: i8) {
    let mut got = [0i32; 4];
    av1_fdct4(input, &mut got, cos_bit as i32, &STAGE_RANGE);
    let want = ref_fdct4(input, cos_bit, &STAGE_RANGE);
    assert_eq!(
        got, want,
        "fdct4 divergence: input={input:?} cos_bit={cos_bit} rust={got:?} c={want:?}"
    );
}

#[test]
fn fdct4_edge_cases() {
    let b = 1i32 << 17;
    let cases: &[[i32; 4]] = &[
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [b, b, b, b],
        [-b, -b, -b, -b],
        [b, -b, b, -b],
        [-b, b, -b, b],
        [1, -1, 1, -1],
        [b, 0, -b, 0],
    ];
    for cos_bit in 10..=13i8 {
        for c in cases {
            check(c, cos_bit);
        }
    }
}

#[test]
fn fdct4_differential_fuzz() {
    let mut rng = Rng(0x_dead_beef_cafe_f00d);
    let iters = 500_000;
    for _ in 0..iters {
        let input = [
            rng.bounded(17),
            rng.bounded(17),
            rng.bounded(17),
            rng.bounded(17),
        ];
        for cos_bit in 10..=13i8 {
            check(&input, cos_bit);
        }
    }
}
