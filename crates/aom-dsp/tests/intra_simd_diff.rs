//! SIMD-vs-scalar differential for the non-directional highbd intra predictors
//! (Gate-3 parity rule 1: bit-identical, no slip), at every archmage token
//! permutation, over the full highbd domain (bd 8/10/12), every mode, and every
//! block size — tight AND padded output stride.
//!
//! The dispatching entry `aom_dsp::intra::predict_highbd` runs the tier's SIMD (for
//! SMOOTH / SMOOTH_V / SMOOTH_H / PAETH; memset/memcpy for DC / V / H);
//! `aom_dsp::intra::predict_highbd_scalar` is the fixed, never-dispatched scalar
//! transcription. The dispatching entry must match the scalar core byte-for-byte
//! at EVERY token tier — which also validates the DC/V/H slice-op rewrite
//! against the original per-element scalar loops.

use aom_dsp::intra::{predict_highbd, predict_highbd_scalar, AboveRef16};
use archmage::testing::{for_each_token_permutation, CompileTimePolicy};

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

const SIZES: [usize; 5] = [4, 8, 16, 32, 64];
// DC, DC_TOP, DC_LEFT, DC_128, V, H, PAETH, SMOOTH, SMOOTH_V, SMOOTH_H
const MODES: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

#[test]
fn intra_highbd_simd_bit_identical_to_scalar_at_every_tier() {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "x86-64 CI must have AVX2 for the SIMD differential to be non-vacuous"
        );
    }
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |_tier| {
        let mut rng = Rng(0xa0e1_47ee_5117_c0de);
        for &bd in &[8i32, 10, 12] {
            let maxv = (1u32 << bd) - 1;
            for &bw in &SIZES {
                for &bh in &SIZES {
                    // Reference edges: above buffer holds the top-left corner at
                    // [0] then bw samples; left holds bh samples.
                    for &stride in &[bw, bw + 7] {
                        for _ in 0..40 {
                            let above: Vec<u16> =
                                (0..1 + bw).map(|_| rng.upto(maxv + 1) as u16).collect();
                            let left: Vec<u16> =
                                (0..bh).map(|_| rng.upto(maxv + 1) as u16).collect();
                            let aref = AboveRef16(&above);
                            for &mode in &MODES {
                                let mut got = vec![0u16; bh * stride];
                                let mut want = vec![0u16; bh * stride];
                                predict_highbd(mode, &mut got, stride, bw, bh, &aref, &left, bd);
                                predict_highbd_scalar(
                                    mode, &mut want, stride, bw, bh, &aref, &left, bd,
                                );
                                assert_eq!(
                                    got, want,
                                    "mode={mode} bw={bw} bh={bh} stride={stride} bd={bd}: \
                                     SIMD dispatch != scalar core"
                                );
                            }
                        }
                    }
                }
            }
        }
    });
    eprintln!("intra highbd SIMD parity: {report}");
    assert!(report.permutations_run >= 2);
}
