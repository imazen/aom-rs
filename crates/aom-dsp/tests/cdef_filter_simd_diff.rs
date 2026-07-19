//! SIMD-vs-scalar differential for the `cdef_filter_block_16` width-8 row
//! kernel AND the width-4 two-row kernel (Gate-3 parity rule 1:
//! bit-identical, no slip), at every archmage token permutation.
//!
//! The scalar reference is `cdef_filter_block_16_scalar` (the transcribed
//! core, never SIMD-routed); the C pin is the pre-existing
//! `cdef_filter_diff.rs::cdef_filter16_byte_identical`, which drives the
//! DISPATCHING `cdef_filter_block_16` against the REAL C kernels over 240k
//! cases — this test adds the per-tier fallback coverage.
//!
//! Domain = the structural CDEF domain (module docs in `src/simd.rs`):
//! bd 8/10/12 pixel values + `CDEF_VERY_LARGE` border fill, frame-walk
//! strength/damping ranges, all 8 directions, both tap parities, all four
//! primary/secondary enable combos, heights 4 and 8.

use aom_dsp::cdef::{CDEF_BSTRIDE, CDEF_VERY_LARGE, cdef_filter_block_16, cdef_filter_block_16_scalar};
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

#[test]
fn cdef_filter16_w8_simd_bit_identical_to_scalar_at_every_tier() {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "x86-64 CI must have AVX2 for the SIMD differential to be non-vacuous"
        );
    }
    let rows = 16;
    let buf_len = rows * CDEF_BSTRIDE;
    let in_off = 4 * CDEF_BSTRIDE + 8;
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        let mut rng = Rng(0x_cdef_51b0_77e5_9a01);
        for variant in 0..4i32 {
            let (en_pri, en_sec) = (variant == 0 || variant == 1, variant == 0 || variant == 2);
            for it in 0..800 {
                let bw = if rng.upto(2) == 0 { 4 } else { 8 };
                let bh = if rng.upto(2) == 0 { 4 } else { 8 };
                let cshift = [0i32, 2, 4][(it % 3) as usize];
                let maxv = (1u32 << (8 + cshift)) - 1;
                let mut inbuf = vec![0u16; buf_len];
                for v in inbuf.iter_mut() {
                    *v = if rng.upto(20) == 0 {
                        CDEF_VERY_LARGE as u16
                    } else {
                        rng.upto(maxv + 1) as u16
                    };
                }
                // Boundary flavours: all-border rows above, max-pixel rows.
                if it % 7 == 0 {
                    for v in inbuf[..2 * CDEF_BSTRIDE].iter_mut() {
                        *v = CDEF_VERY_LARGE as u16;
                    }
                }
                if it % 11 == 0 {
                    for v in inbuf[buf_len - 2 * CDEF_BSTRIDE..].iter_mut() {
                        *v = maxv as u16;
                    }
                }
                let pri = (rng.upto(16) as i32) << cshift;
                let sec = (rng.upto(5) as i32) << cshift;
                let dir = rng.upto(8) as i32;
                let prid = (3 + rng.upto(4)) as i32 + cshift;
                let secd = (3 + rng.upto(4)) as i32 + cshift;

                let mut got = vec![0u16; bw * bh];
                cdef_filter_block_16(
                    &mut got, 0, bw, &inbuf, in_off, pri, sec, dir, prid, secd, cshift, bw, bh,
                    en_pri, en_sec,
                );
                let mut want = vec![0u16; bw * bh];
                cdef_filter_block_16_scalar(
                    &mut want, 0, bw, &inbuf, in_off, pri, sec, dir, prid, secd, cshift, bw, bh,
                    en_pri, en_sec,
                );
                assert_eq!(
                    got, want,
                    "[{tier}] v{variant} bw={bw} bh={bh} dir={dir} pri={pri} sec={sec} \
                     cshift={cshift} prid={prid} secd={secd}"
                );
            }
        }
    });
    eprintln!("cdef_filter16 w8 SIMD parity: {report}");
    assert!(report.permutations_run >= 2);
}
