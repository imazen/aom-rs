//! Roundtrip harness for `read_coeffs_txb_full` — the complete per-txb decode
//! (txb_skip + luma tx_type + coefficient payload), inverse of the byte-identical-
//! to-C `write_coeffs_txb_full`. A clean roundtrip pins it to libaom's decode.

use aom_entropy::dec::OdEcDec;
use aom_entropy::enc::OdEcEnc;
use aom_txb::{
    ext_tx_derive, read_coeffs_txb_full, scan, txb_high, txb_wide, write_coeffs_txb_full,
    CDF_ARENA_LEN,
};

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
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo) as u64) as u32
    }
}

const REGIONS: [(usize, usize, usize); 13] = [
    (0, 5 * 13, 2), (195, 4, 5), (219, 4, 6), (247, 4, 7), (279, 4, 8), (315, 4, 9),
    (355, 4, 10), (399, 4, 11), (447, 5 * 2 * 9, 2), (717, 5 * 2 * 4, 3), (877, 5 * 2 * 42, 4),
    (2977, 5 * 2 * 21, 4), (4027, 2 * 3, 2),
];

fn gen_arena(rng: &mut Rng) -> Vec<u16> {
    let mut a = vec![0u16; CDF_ARENA_LEN];
    for &(off, count, n) in &REGIONS {
        for slot in 0..count {
            let base = off + slot * (n + 1);
            let mut acc: u32 = 0;
            for e in a[base..base + n - 1].iter_mut() {
                acc += rng.range(1, (32000 / n as u32).max(2));
                *e = (32768u32.saturating_sub(acc)).max(1) as u16;
            }
            a[base + n - 1] = 0;
            a[base + n] = 0;
        }
    }
    a
}

fn gen_cdf(rng: &mut Rng, n: usize) -> Vec<u16> {
    let mut cdf = vec![0u16; 17.max(n + 1)];
    let mut acc = 0u32;
    for e in cdf.iter_mut().take(n.saturating_sub(1)) {
        acc += rng.range(1, (32000 / n as u32).max(2));
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    cdf
}

fn gen_coeffs(rng: &mut Rng, scan: &[i16], area: usize) -> (Vec<i32>, usize) {
    let mut coeff = vec![0i32; area];
    let eob = rng.range(1, area as u32 + 1) as usize;
    let nz = |rng: &mut Rng| -> i32 {
        let mag = match rng.range(0, 10) {
            0..=4 => rng.range(1, 3) as i32,
            5..=7 => rng.range(1, 20) as i32,
            _ => rng.range(1, 3000) as i32,
        };
        if rng.next() & 1 == 1 { -mag } else { mag }
    };
    for i in 0..eob {
        let pos = scan[i] as usize;
        if i == eob - 1 || rng.range(0, 10) >= 4 {
            coeff[pos] = nz(rng);
        }
    }
    (coeff, eob)
}

#[test]
fn read_coeffs_txb_full_roundtrips_write() {
    let mut rng = Rng(0x0f01_1c0d_ede0_9e37);
    let mut tx_type_read = 0usize;
    for tx_size in 0..19usize {
        let area = txb_wide(tx_size) * txb_high(tx_size);
        for &is_inter in &[false, true] {
            for &reduced in &[false, true] {
                for tx_type in 0..16usize {
                    let use_fi = tx_type % 3 == 0;
                    let fi_mode = tx_type % 5;
                    let mode = [0usize, 1, 2, 6, 12][tx_type % 5];
                    let d = ext_tx_derive(tx_size, is_inter, reduced, tx_type, use_fi, fi_mode, mode);
                    for &plane_type in &[0usize, 1] {
                        for &signal_gate in &[true, false] {
                            let writes = plane_type == 0 && signal_gate && d.num > 1;
                            if writes && d.used == 0 {
                                continue;
                            }
                            // luma non-written tx_type must be DCT_DCT (inferred).
                            if plane_type == 0 && !writes && tx_type != 0 {
                                continue;
                            }
                            for &upd in &[true, false] {
                                let sc = scan(tx_size, tx_type);
                                let (coeff, eob) = gen_coeffs(&mut rng, sc, area);
                                let txb_skip_ctx = rng.range(0, 13) as usize;
                                let dc_sign_ctx = rng.range(0, 3) as usize;
                                let arena0 = gen_arena(&mut rng);
                                let extcdf0 = gen_cdf(&mut rng, d.num.max(2) as usize);

                                let mut enc = OdEcEnc::new();
                                let mut ae = arena0.clone();
                                let mut ee = extcdf0.clone();
                                write_coeffs_txb_full(
                                    &mut enc, &mut ae, &mut ee, &coeff, eob, tx_size, tx_type,
                                    plane_type, txb_skip_ctx, dc_sign_ctx, upd, is_inter, reduced,
                                    use_fi, fi_mode, mode, signal_gate,
                                );
                                let bytes = enc.done().to_vec();

                                let mut dec = OdEcDec::new(&bytes);
                                let mut ad = arena0.clone();
                                let mut ed = extcdf0.clone();
                                let mut tcoeff = vec![0i32; area];
                                let (eob_d, tt_d) = read_coeffs_txb_full(
                                    &mut dec, &mut ad, &mut ed, &mut tcoeff, tx_size, plane_type,
                                    txb_skip_ctx, dc_sign_ctx, upd, is_inter, reduced, signal_gate,
                                    tx_type,
                                );
                                let m = format!("ts={tx_size} tt={tx_type} inter={is_inter} red={reduced} pl={plane_type} gate={signal_gate} upd={upd} eob={eob}");
                                assert_eq!(eob_d, eob, "eob {m}");
                                assert_eq!(tcoeff, coeff, "coeffs {m}");
                                let want_tt = if plane_type == 0 { if writes { tx_type } else { 0 } } else { tx_type };
                                assert_eq!(tt_d, want_tt, "tx_type {m}");
                                assert_eq!(ae, ad, "coeff cdf {m}");
                                assert_eq!(ee, ed, "ext_tx cdf {m}");
                                if writes {
                                    tx_type_read += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(tx_type_read > 0, "tx_type read path never fired");
}
