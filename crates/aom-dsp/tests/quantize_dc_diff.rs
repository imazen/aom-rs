//! Differential harness for the DC-only quantizers (`AV1_XFORM_QUANT_DC`) vs C
//! libaom: av1_quantize_dc / av1_highbd_quantize_dc quantize coefficient 0 only
//! (zeroing the rest). The C oracles reach the static quantize_dc /
//! highbd_quantize_dc through the real facades. Flat + QM, log_scale 0/1/2.

use aom_dsp::quant::{av1_highbd_quantize_dc, av1_quantize_dc};
use aom_sys_ref as c;

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
    fn i16r(&mut self, lo: i32, hi: i32) -> i16 {
        (lo + (self.next() % (hi - lo) as u64) as i32) as i16
    }
    fn qm(&mut self) -> u8 {
        1 + (self.next() % 255) as u8
    }
}

#[test]
fn quantize_dc_differential() {
    let mut rng = Rng(0x_dc00_c0de_9e37_79b9);
    for &n in &[16usize, 64, 256, 1024] {
        for log_scale in 0..=2 {
            for _ in 0..3000 {
                // Lowbd magnitudes.
                let coeff: Vec<i32> = (0..n).map(|_| (rng.next() % (1 << 19)) as i32 - (1 << 18)).collect();
                let round = [rng.i16r(1, 2000), rng.i16r(1, 2000)];
                let quant = rng.i16r(1, 32767);
                let dequant = rng.i16r(1, 8000);
                let qm: Vec<u8> = (0..n).map(|_| rng.qm()).collect();
                let iqm: Vec<u8> = (0..n).map(|_| rng.qm()).collect();
                let use_qm = rng.next() & 1 == 1;
                let (q, iq) = if use_qm { (Some(&qm[..]), Some(&iqm[..])) } else { (None, None) };

                let mut qc = vec![0i32; n];
                let mut dqc = vec![0i32; n];
                let eob = av1_quantize_dc(&round, quant, dequant, log_scale, q, iq, &coeff, &mut qc, &mut dqc);
                let (qw, dqw, ew) = c::ref_quantize_dc(log_scale, &coeff, &round, quant, dequant, q, iq);
                assert_eq!(eob, ew, "dc eob n={n} ls={log_scale} qm={use_qm}");
                assert_eq!(qc, qw, "dc qcoeff n={n} ls={log_scale} qm={use_qm}");
                assert_eq!(dqc, dqw, "dc dqcoeff n={n} ls={log_scale} qm={use_qm}");
            }
        }
    }
}

#[test]
fn highbd_quantize_dc_differential() {
    let mut rng = Rng(0x_dc00_c057_0000_b111);
    for &n in &[16usize, 64, 256, 1024] {
        for log_scale in 0..=2 {
            for _ in 0..3000 {
                // Highbd (12-bit) magnitudes.
                let coeff: Vec<i32> = (0..n).map(|_| (rng.next() % (1 << 23)) as i32 - (1 << 22)).collect();
                let round = [rng.i16r(1, 2000), rng.i16r(1, 2000)];
                let quant = rng.i16r(1, 32767);
                let dequant = rng.i16r(1, 8000);
                let qm: Vec<u8> = (0..n).map(|_| rng.qm()).collect();
                let iqm: Vec<u8> = (0..n).map(|_| rng.qm()).collect();
                let use_qm = rng.next() & 1 == 1;
                let (q, iq) = if use_qm { (Some(&qm[..]), Some(&iqm[..])) } else { (None, None) };

                let mut qc = vec![0i32; n];
                let mut dqc = vec![0i32; n];
                let eob = av1_highbd_quantize_dc(&round, quant, dequant, log_scale, q, iq, &coeff, &mut qc, &mut dqc);
                let (qw, dqw, ew) = c::ref_highbd_quantize_dc(log_scale, &coeff, &round, quant, dequant, q, iq);
                assert_eq!(eob, ew, "hbd dc eob n={n} ls={log_scale} qm={use_qm}");
                assert_eq!(qc, qw, "hbd dc qcoeff n={n} ls={log_scale} qm={use_qm}");
                assert_eq!(dqc, dqw, "hbd dc dqcoeff n={n} ls={log_scale} qm={use_qm}");
            }
        }
    }
}
