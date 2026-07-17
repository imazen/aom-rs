//! SIMD-vs-scalar differential for `txb_init_levels` (Gate-3 parity rule 1:
//! bit-identical, no slip), at every archmage token permutation, on the FULL
//! i32 domain (adversarial values incl. i32::MIN/MAX — the kernel's
//! exactness argument covers the whole domain, so the test asserts it there).
//!
//! The C pin is the pre-existing `txb_diff.rs`, which drives the DISPATCHING
//! `txb_init_levels` against the REAL `av1_txb_init_levels` including the
//! exact write footprint.

use aom_txb::{TX_PAD_2D, txb_init_levels, txb_init_levels_scalar};
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
}

#[test]
fn txb_init_levels_simd_bit_identical_to_scalar_at_every_tier() {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "x86-64 CI must have AVX2 for the SIMD differential to be non-vacuous"
        );
    }
    // All (adjusted) txb geometries: widths/heights 4..32.
    let dims: &[(usize, usize)] = &[
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8),
        (8, 16),
        (16, 8),
        (16, 16),
        (16, 32),
        (32, 16),
        (32, 32),
        (4, 16),
        (16, 4),
        (8, 32),
        (32, 8),
    ];
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        let mut rng = Rng(0x_7b17_1234_5678_9abc);
        for &(w, h) in dims {
            for case in 0..12 {
                let mut coeff: Vec<i32> = (0..w * h).map(|_| rng.next() as i32).collect();
                if case == 0 {
                    coeff.fill(0);
                }
                if case == 1 {
                    coeff[0] = i32::MIN;
                    coeff[1] = i32::MAX;
                    coeff[w * h - 1] = i32::MIN;
                    coeff[w * h / 2] = -128;
                    coeff[w * h / 2 + 1] = 127;
                }
                if case == 2 {
                    for (i, c) in coeff.iter_mut().enumerate() {
                        *c = (i as i32 % 300) - 150; // realistic small levels
                    }
                }
                // Prefill both level buffers with a sentinel to also pin the
                // exact write FOOTPRINT (bytes outside it must stay 0xEE).
                let mut got = vec![0xEEu8; TX_PAD_2D];
                let mut want = vec![0xEEu8; TX_PAD_2D];
                txb_init_levels(&coeff, w, h, &mut got);
                txb_init_levels_scalar(&coeff, w, h, &mut want);
                assert_eq!(got, want, "[{tier}] {w}x{h} case {case}");
            }
        }
    });
    eprintln!("txb_init_levels SIMD parity: {report}");
    assert!(report.permutations_run >= 2);
}
