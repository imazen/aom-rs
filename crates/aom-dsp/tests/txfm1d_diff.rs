//! Differential harness for the full forward 1-D transform family vs the C
//! libaom v3.14.1 oracle: fdct{4,8,16,32,64}, fadst{4,8,16}, fidentity{4,8,16,32}.
//!
//! Byte-identical output is required for every `cos_bit` in 10..=13 over the
//! conformant input range. Per-size magnitude bounds keep the C reference within
//! defined behaviour (its inner 32-bit `w0*in0` must not truly overflow — that
//! only occurs for non-conformant inputs, where equivalence would be measured
//! against C undefined behaviour rather than a real divergence).

use aom_sys_ref as c;
use aom_dsp::transform as r;

type RFn = fn(&[i32], &mut [i32], i32, &[i8]);
type CFn = unsafe extern "C" fn(*const i32, *mut i32, i8, *const i8);

struct Case {
    name: &'static str,
    size: usize,
    bits: u32, // input magnitude bound = 1<<bits
    rf: RFn,
    cf: CFn,
}

fn cases() -> Vec<Case> {
    vec![
        Case { name: "fdct4",  size: 4,  bits: 14, rf: r::av1_fdct4,  cf: c::av1_fdct4 },
        Case { name: "fdct8",  size: 8,  bits: 13, rf: r::av1_fdct8,  cf: c::av1_fdct8 },
        Case { name: "fdct16", size: 16, bits: 12, rf: r::av1_fdct16, cf: c::av1_fdct16 },
        Case { name: "fdct32", size: 32, bits: 11, rf: r::av1_fdct32, cf: c::av1_fdct32 },
        Case { name: "fdct64", size: 64, bits: 10, rf: r::av1_fdct64, cf: c::av1_fdct64 },
        Case { name: "fadst4",  size: 4,  bits: 14, rf: r::av1_fadst4,  cf: c::av1_fadst4 },
        Case { name: "fadst8",  size: 8,  bits: 13, rf: r::av1_fadst8,  cf: c::av1_fadst8 },
        Case { name: "fadst16", size: 16, bits: 12, rf: r::av1_fadst16, cf: c::av1_fadst16 },
        Case { name: "fidentity4",  size: 4,  bits: 20, rf: r::av1_fidentity4,  cf: c::av1_fidentity4_c },
        Case { name: "fidentity8",  size: 8,  bits: 20, rf: r::av1_fidentity8,  cf: c::av1_fidentity8_c },
        Case { name: "fidentity16", size: 16, bits: 20, rf: r::av1_fidentity16, cf: c::av1_fidentity16_c },
        Case { name: "fidentity32", size: 32, bits: 20, rf: r::av1_fidentity32, cf: c::av1_fidentity32_c },
    ]
}

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
    fn bounded(&mut self, bits: u32) -> i32 {
        let range = (1i64 << (bits + 1)) + 1;
        ((self.next() as i64).rem_euclid(range) - (1i64 << bits)) as i32
    }
}

const SR: [i8; 16] = [24; 16]; // ignored by the disabled range checker/asserts

fn run_case(case: &Case, rng: &mut Rng, iters: usize) {
    let mut got = vec![0i32; case.size];
    for _ in 0..iters {
        let input: Vec<i32> = (0..case.size).map(|_| rng.bounded(case.bits)).collect();
        for cos_bit in 10..=13i32 {
            (case.rf)(&input, &mut got, cos_bit, &SR);
            let want = c::ref_txfm1d(case.cf, &input, cos_bit as i8, &SR);
            assert_eq!(
                got, want,
                "{} divergence: cos_bit={cos_bit} input={input:?}",
                case.name
            );
        }
    }
}

#[test]
fn txfm1d_all_edge_cases() {
    let mut rng = Rng(1);
    for case in cases() {
        // all-zero (exercises fadst early-out) and single-impulse
        for input in [vec![0i32; case.size], {
            let mut v = vec![0i32; case.size];
            v[0] = 1 << case.bits;
            v
        }] {
            let mut got = vec![0i32; case.size];
            for cos_bit in 10..=13i32 {
                (case.rf)(&input, &mut got, cos_bit, &SR);
                let want = c::ref_txfm1d(case.cf, &input, cos_bit as i8, &SR);
                assert_eq!(got, want, "{} edge divergence input={input:?}", case.name);
            }
        }
        run_case(&case, &mut rng, 0);
    }
}

#[test]
fn txfm1d_differential_fuzz() {
    let mut rng = Rng(0x_1234_5678_9abc_def0);
    for case in cases() {
        run_case(&case, &mut rng, 100_000);
    }
}
