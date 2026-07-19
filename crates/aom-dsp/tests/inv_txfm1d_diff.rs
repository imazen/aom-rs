//! Differential harness for the inverse 1-D transform family vs C libaom
//! v3.14.1: idct{4,8,16,32,64}, iadst{4,8,16}, iidentity{4,8,16,32}.
//!
//! Unlike the forward path, the inverse kernels apply a live `clamp_value` per
//! stage driven by `stage_range`. The same `stage_range` is fed to both sides,
//! so equality validates the arithmetic + clamp logic together. `stage_range`
//! and input bounds are chosen so the C `half_btf` inner 32-bit products stay
//! within defined behaviour (clamp keeps operands <= 2^16, cospi <= 2^13).

use aom_sys_ref as c;
use aom_dsp::transform as r;

type RFn = fn(&[i32], &mut [i32], i32, &[i8]);
type CFn = unsafe extern "C" fn(*const i32, *mut i32, i8, *const i8);

struct Case {
    name: &'static str,
    size: usize,
    rf: RFn,
    cf: CFn,
}

fn cases() -> Vec<Case> {
    vec![
        Case { name: "idct4",  size: 4,  rf: r::av1_idct4,  cf: c::av1_idct4 },
        Case { name: "idct8",  size: 8,  rf: r::av1_idct8,  cf: c::av1_idct8 },
        Case { name: "idct16", size: 16, rf: r::av1_idct16, cf: c::av1_idct16 },
        Case { name: "idct32", size: 32, rf: r::av1_idct32, cf: c::av1_idct32 },
        Case { name: "idct64", size: 64, rf: r::av1_idct64, cf: c::av1_idct64 },
        Case { name: "iadst4",  size: 4,  rf: r::av1_iadst4,  cf: c::av1_iadst4 },
        Case { name: "iadst8",  size: 8,  rf: r::av1_iadst8,  cf: c::av1_iadst8 },
        Case { name: "iadst16", size: 16, rf: r::av1_iadst16, cf: c::av1_iadst16 },
        Case { name: "iidentity4",  size: 4,  rf: r::av1_iidentity4,  cf: c::av1_iidentity4_c },
        Case { name: "iidentity8",  size: 8,  rf: r::av1_iidentity8,  cf: c::av1_iidentity8_c },
        Case { name: "iidentity16", size: 16, rf: r::av1_iidentity16, cf: c::av1_iidentity16_c },
        Case { name: "iidentity32", size: 32, rf: r::av1_iidentity32, cf: c::av1_iidentity32_c },
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

// clamp at 2^16, so half_btf operands stay <= 2^16 and cospi(2^13)*op < 2^30.
const SR: [i8; 16] = [17; 16];
const IN_BITS: u32 = 13;

fn c_call(cf: CFn, input: &[i32], cos_bit: i8) -> Vec<i32> {
    let mut out = vec![0i32; input.len()];
    unsafe { cf(input.as_ptr(), out.as_mut_ptr(), cos_bit, SR.as_ptr()) }
    out
}

#[test]
fn inv_txfm1d_edge_cases() {
    for case in cases() {
        for input in [vec![0i32; case.size], {
            let mut v = vec![0i32; case.size];
            v[0] = 1 << IN_BITS;
            v
        }] {
            let mut got = vec![0i32; case.size];
            for cos_bit in 10..=13i32 {
                (case.rf)(&input, &mut got, cos_bit, &SR);
                let want = c_call(case.cf, &input, cos_bit as i8);
                assert_eq!(got, want, "{} edge divergence input={input:?}", case.name);
            }
        }
    }
}

#[test]
fn inv_txfm1d_differential_fuzz() {
    let mut rng = Rng(0x_feed_face_5eed_1e55);
    for case in cases() {
        let mut got = vec![0i32; case.size];
        for _ in 0..100_000 {
            let input: Vec<i32> = (0..case.size).map(|_| rng.bounded(IN_BITS)).collect();
            for cos_bit in 10..=13i32 {
                (case.rf)(&input, &mut got, cos_bit, &SR);
                let want = c_call(case.cf, &input, cos_bit as i8);
                assert_eq!(
                    got, want,
                    "{} divergence: cos_bit={cos_bit} input={input:?}",
                    case.name
                );
            }
        }
    }
}
