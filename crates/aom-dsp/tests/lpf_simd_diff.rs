//! SIMD-vs-scalar differential for the highbd deblock loop filter (Gate-3
//! parity rule 1: bit-identical, no slip), at every archmage token
//! permutation, over the full highbd domain (bd 8/10/12).
//!
//! The dispatching entry (`highbd::horizontal`/`vertical`) runs the tier's
//! SIMD; `highbd::horizontal_scalar`/`vertical_scalar` is the fixed,
//! never-dispatched scalar reference. This complements `hbd_lpf_diff.rs`
//! (dispatch-vs-REAL-C) with per-tier fallback coverage: the dispatching entry
//! must match the scalar core at EVERY token tier, not only the top one.

use aom_dsp::loopfilter::highbd;
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

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
    fn upto(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

const PITCH: usize = 32;
const ROWS: usize = 32;
const CENTER: usize = 12 * PITCH + 12;

#[test]
fn hbd_lpf_simd_bit_identical_to_scalar_at_every_tier() {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "x86-64 CI must have AVX2 for the SIMD differential to be non-vacuous"
        );
    }
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        let mut rng = Rng(0x_1eaf_5117_c0de_0f11);
        for &bd in &[8i32, 10, 12] {
            let maxv = (1u32 << bd) - 1;
            for &dir in b"hv" {
                for &width in &[4u32, 6, 8, 14] {
                    for _ in 0..3000 {
                        // Sometimes near-flat to hit flat/flat2 branches.
                        let base = rng.upto(maxv + 1);
                        let amp = 1 + rng.upto(1 << (bd - 4));
                        let strat = rng.upto(3);
                        let buf: Vec<u16> = (0..PITCH * ROWS)
                            .map(|_| {
                                if strat == 0 {
                                    rng.upto(maxv + 1) as u16
                                } else {
                                    (base as i32 + rng.upto(2 * amp + 1) as i32 - amp as i32)
                                        .clamp(0, maxv as i32)
                                        as u16
                                }
                            })
                            .collect();
                        let bl = if rng.upto(2) == 0 {
                            rng.upto(256) as u8
                        } else {
                            (16 + rng.upto(200)) as u8
                        };
                        let li = if rng.upto(2) == 0 {
                            rng.upto(256) as u8
                        } else {
                            (1 + rng.upto(64)) as u8
                        };
                        let th = rng.upto(256) as u8;

                        let mut got = buf.clone();
                        let mut want = buf.clone();
                        if dir == b'h' {
                            highbd::horizontal(width, &mut got, CENTER, PITCH, bl, li, th, bd);
                            highbd::horizontal_scalar(width, &mut want, CENTER, PITCH, bl, li, th, bd);
                        } else {
                            highbd::vertical(width, &mut got, CENTER, PITCH, bl, li, th, bd);
                            highbd::vertical_scalar(width, &mut want, CENTER, PITCH, bl, li, th, bd);
                        }
                        assert_eq!(
                            got, want,
                            "[{tier}] dir={} width={width} bd={bd} bl={bl} li={li} th={th}",
                            dir as char
                        );
                    }
                }
            }
        }
    });
    eprintln!("hbd lpf SIMD parity: {report}");
    assert!(report.permutations_run >= 2);
}
