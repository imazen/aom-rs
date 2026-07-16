//! KB-5 diagnostic: does the port's coded-lossless BLOCK path round-trip? A
//! coded-lossless TX_4X4 block must satisfy `IWHT(dequant(quant(WHT(r)))) == r`
//! exactly (the encode is lossless), so reconstructing the port's own
//! `xform_quant` output must recover `pred + r` byte for byte. If it doesn't, the
//! quantizer at qindex 0 is dropping/altering coefficients that lossless must
//! preserve.

use aom_encode::{QuantKind, QuantParams, xform_quant};
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_transform::inv_txfm2d::av1_highbd_iwht4x4_add;

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
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i32
    }
}

#[test]
fn lossless_block_roundtrips_through_quant() {
    let bd = 8u8;
    let mut quants = Quants::zeroed();
    let mut deq = Dequants::zeroed();
    // qindex 0, zero deltas -> the coded-lossless quantizer rows.
    av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
    let rows = set_q_index(&quants, &deq, 0, 0);

    // Coded-lossless uses NO trellis -> B quant (USE_B_QUANT_NO_TRELLIS).
    let qp_b = QuantParams::from_plane_rows(&rows, QuantKind::B, bd, true);
    // Also probe FP for comparison.
    let qp_fp = QuantParams::from_plane_rows(&rows, QuantKind::Fp, bd, true);

    let mut rng = Rng(0xabcd_1234_5678_9012);
    let mut fails_b = 0u64;
    let mut fails_fp = 0u64;
    let mut first_fail_b: Option<String> = None;
    for _ in 0..200000 {
        // Real encode: pred in [0,255], source in [0,255], residual = src - pred
        // spanning the FULL [-255, 255] range (not the narrow [-128,127] band).
        let pred = rng.range(0, 255);
        let mut r = [0i16; 16];
        for e in r.iter_mut() {
            let src = rng.range(0, 255);
            *e = (src - pred) as i16;
        }
        let mut discard: Option<String> = None;
        for (kind_name, kind, qp, fails, first) in [
            ("B", QuantKind::B, &qp_b, &mut fails_b, &mut first_fail_b),
            ("FP", QuantKind::Fp, &qp_fp, &mut fails_fp, &mut discard),
        ] {
            let res = xform_quant(&r, 0, 0, kind, qp, false);
            // DECODER-faithful reconstruction: the coder only writes coeffs at
            // scan positions [0, eob); the decoder zeroes the rest. Mirror that.
            // libaom `default_scan_4x4` (DCT_DCT 4x4), matching aom_txb::scan(0,0).
            const DEFAULT_SCAN_4X4: [usize; 16] =
                [0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15];
            let mut dq_dec = vec![0i32; 16];
            for &rc in DEFAULT_SCAN_4X4.iter().take(res.eob as usize) {
                dq_dec[rc] = res.dqcoeff[rc];
            }
            let mut recon = vec![pred as u16; 16];
            av1_highbd_iwht4x4_add(&dq_dec, &mut recon, 4, res.eob as usize, i32::from(bd));
            let ok = (0..16).all(|i| recon[i] == (pred + i32::from(r[i])) as u16);
            if !ok {
                *fails += 1;
                if first.is_none() {
                    let want: Vec<u16> = (0..16).map(|i| (pred + i32::from(r[i])) as u16).collect();
                    *first = Some(format!(
                        "kind={kind_name} eob={} r={:?}\n    coeff={:?}\n    qcoeff={:?}\n    dqcoeff={:?}\n    recon={:?}\n    want ={:?}",
                        res.eob, r, res.coeff, res.qcoeff, res.dqcoeff, recon, want
                    ));
                }
            }
        }
    }
    eprintln!("B fails={fails_b}/20000  FP fails={fails_fp}/20000");
    if let Some(f) = &first_fail_b {
        eprintln!("first B failure:\n    {f}");
    }
    assert_eq!(fails_b, 0, "coded-lossless B-quant block must round-trip exactly");
}
