//! SIMD-vs-scalar differential for `av1_quantize_fp_no_qmatrix_dispatch`
//! (Gate-3 parity rule 1: integer SIMD MUST be bit-identical to the scalar
//! port — qcoeff, dqcoeff, AND eob — at every dispatch tier).
//!
//! The scalar port is itself C-differentially validated
//! (`quantize_fp_diff.rs`), so SIMD == scalar here transitively pins SIMD ==
//! C. Every case runs under `archmage::testing::for_each_token_permutation`,
//! which re-executes the dispatch with each SIMD tier disabled down to
//! scalar-only — proving the incant fallback chain AND the vector kernel.
//!
//! Domain: FULL adversarial i32 coefficients (including i32::MIN/MAX) and
//! full-range i16 tables (including zero/negative values that
//! `av1_build_quantizer` never produces) — the kernel's bit-exactness
//! argument (see `src/simd.rs`) holds on the whole domain, so the test
//! asserts it on the whole domain. Scan orders are random permutations plus
//! identity (real `av1_scan_orders` rows are permutations of the same shape).

use aom_dsp::quant::av1_quantize_fp_no_qmatrix;
use aom_dsp::quant::simd::av1_quantize_fp_no_qmatrix_dispatch;
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
    fn i16_any(&mut self) -> i16 {
        self.next() as i16
    }
    fn pos_i16(&mut self, lo: i32, hi: i32) -> i16 {
        (lo + (self.next() % (hi - lo) as u64) as i32) as i16
    }
}

/// Random permutation of 0..n (a valid scan order) + its inverse.
fn perm_pair(rng: &mut Rng, n: usize) -> (Vec<i16>, Vec<i16>) {
    let mut v: Vec<i16> = (0..n as i16).collect();
    for i in (1..n).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    let mut inv = vec![0i16; n];
    for (i, &rc) in v.iter().enumerate() {
        inv[rc as usize] = i as i16;
    }
    (v, inv)
}

#[allow(clippy::too_many_arguments)]
fn assert_case(
    label: &str,
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    iscan: &[i16],
    coeff: &[i32],
) {
    let n = coeff.len();
    let mut q_ref = vec![0i32; n];
    let mut dq_ref = vec![0i32; n];
    let eob_ref = av1_quantize_fp_no_qmatrix(
        quant, dequant, round, log_scale, scan, coeff, &mut q_ref, &mut dq_ref,
    );
    let mut q_got = vec![0i32; n];
    let mut dq_got = vec![0i32; n];
    let eob_got = av1_quantize_fp_no_qmatrix_dispatch(
        quant, dequant, round, log_scale, scan, iscan, coeff, &mut q_got, &mut dq_got,
    );
    assert_eq!(eob_got, eob_ref, "{label}: eob");
    assert_eq!(q_got, q_ref, "{label}: qcoeff\ncoeff={coeff:?}");
    assert_eq!(dq_got, dq_ref, "{label}: dqcoeff\ncoeff={coeff:?}");
}

#[test]
fn quantize_fp_simd_bit_identical_to_scalar_at_every_tier() {
    // Anti-vacuous: on x86-64/aarch64 CI the SIMD tier must actually be
    // available, or every permutation would be scalar==scalar.
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "x86-64 CI must have AVX2 for the SIMD differential to be non-vacuous"
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        assert!(archmage::NeonToken::summon().is_some());
    }
    let sizes = [16usize, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        let mut rng = Rng(0x_9e37_79b9_7f4a_7c15);
        for &n in &sizes {
            let (scan, iscan) = perm_pair(&mut rng, n);
            let identity: Vec<i16> = (0..n as i16).collect();
            for ls in 0..3i32 {
                // Production-shaped tables (positive, av1_build_quantizer-like).
                for rep in 0..6 {
                    let quant = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
                    let dequant = [rng.pos_i16(1, 8000), rng.pos_i16(1, 8000)];
                    let round = [rng.pos_i16(0, 2000), rng.pos_i16(0, 2000)];
                    let coeff: Vec<i32> = (0..n)
                        .map(|_| (rng.next() % (1 << 19)) as i32 - (1 << 18))
                        .collect();
                    assert_case(
                        &format!("[{tier}] prod n={n} ls={ls} rep={rep}"),
                        &quant,
                        &dequant,
                        &round,
                        ls,
                        &scan,
                        &iscan,
                        &coeff,
                    );
                }
                // Adversarial: full-range i16 tables + full-range i32 coeffs.
                for rep in 0..6 {
                    let quant = [rng.i16_any(), rng.i16_any()];
                    let dequant = [rng.i16_any(), rng.i16_any()];
                    let round = [rng.i16_any(), rng.i16_any()];
                    let coeff: Vec<i32> = (0..n).map(|_| rng.next() as i32).collect();
                    assert_case(
                        &format!("[{tier}] adv n={n} ls={ls} rep={rep}"),
                        &quant,
                        &dequant,
                        &round,
                        ls,
                        &scan,
                        &iscan,
                        &coeff,
                    );
                }
                // Edge cases on the identity scan: all-zero, extreme lanes,
                // spikes at the first/last scan position, threshold straddles.
                let quant = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
                let dequant = [rng.pos_i16(1, 8000), rng.pos_i16(1, 8000)];
                let round = [rng.pos_i16(0, 2000), rng.pos_i16(0, 2000)];
                let mut edge = vec![0i32; n];
                assert_case(
                    &format!("[{tier}] zeros n={n} ls={ls}"),
                    &quant,
                    &dequant,
                    &round,
                    ls,
                    &identity,
                    &identity,
                    &edge,
                );
                edge[0] = i32::MIN;
                edge[1] = i32::MAX;
                edge[2] = -32768;
                edge[3] = 32767;
                edge[n - 1] = i32::MAX;
                edge[n - 2] = i32::MIN;
                edge[n / 2] = (dequant[1] as i32) << 3;
                edge[n / 2 + 1] = ((dequant[1] as i32) << 3) - 1;
                assert_case(
                    &format!("[{tier}] extremes n={n} ls={ls}"),
                    &quant,
                    &dequant,
                    &round,
                    ls,
                    &identity,
                    &identity,
                    &edge,
                );
            }
        }
    });
    eprintln!("quantize_fp SIMD parity: {report}");
    assert!(
        report.permutations_run >= 2,
        "expected at least all-enabled + all-disabled tiers to run"
    );
}
