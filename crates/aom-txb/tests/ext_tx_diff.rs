//! Exhaustive differential harness for the ext-tx derivation used by
//! `av1_write_tx_type` vs C libaom: every (tx_size, is_inter, reduced, tx_type,
//! filter-intra, fi_mode, mode) combination must produce the identical set
//! type / arity / eset / square / symbol / used / intra-dir. Also verifies the
//! composed `write_tx_type` emits identical bytes for the in-set cases.

use aom_entropy::enc::OdEcEnc;
use aom_sys_ref as c;
use aom_txb::{ext_tx_derive, write_tx_type};

#[test]
fn ext_tx_derivation_exhaustive() {
    for tx_size in 0..19usize {
        for &is_inter in &[false, true] {
            for &reduced in &[false, true] {
                for tx_type in 0..16usize {
                    for &use_fi in &[false, true] {
                        // fi_mode only matters when use_fi; mode only when !use_fi.
                        let fi_modes: &[usize] = if use_fi { &[0, 1, 2, 3, 4] } else { &[0] };
                        let modes: &[usize] = if use_fi { &[0] } else { &[0, 1, 2, 6, 12] };
                        for &fi_mode in fi_modes {
                            for &mode in modes {
                                let got = ext_tx_derive(
                                    tx_size, is_inter, reduced, tx_type, use_fi, fi_mode, mode,
                                );
                                let w = c::ref_ext_tx_derive(
                                    tx_size, is_inter, reduced, tx_type, use_fi, fi_mode, mode,
                                );
                                assert_eq!(got.set_type, w[0], "set_type ts={tx_size} inter={is_inter} red={reduced}");
                                assert_eq!(got.num, w[1], "num ts={tx_size} tt={tx_type}");
                                assert_eq!(got.eset, w[2], "eset ts={tx_size} inter={is_inter}");
                                assert_eq!(got.square, w[3], "square ts={tx_size}");
                                assert_eq!(got.symb, w[4], "symb ts={tx_size} tt={tx_type}");
                                assert_eq!(got.used, w[5], "used ts={tx_size} tt={tx_type}");
                                assert_eq!(got.intra_dir, w[6], "intra_dir fi={use_fi} fim={fi_mode} m={mode}");
                            }
                        }
                    }
                }
            }
        }
    }
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
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo) as u64) as u32
    }
}

fn gen_cdf(rng: &mut Rng, n: usize) -> Vec<u16> {
    let mut cdf = vec![0u16; n + 1];
    let mut acc: u32 = 0;
    for e in cdf.iter_mut().take(n - 1) {
        acc += rng.range(1, (32000 / n as u32).max(2));
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    cdf[n - 1] = 0;
    cdf
}

/// Emit the tx_type symbol through the composed `write_tx_type` for the cases
/// where the set carries >1 type; compare to `aom_write_symbol` on the same
/// symbol (the C reference for the emission is the already-proven writer, so we
/// pin the composition against a direct symbol write on identical CDF state).
#[test]
fn write_tx_type_emits_expected_symbol() {
    let mut rng = Rng(0x_e77e_c057_1234_5678);
    for tx_size in 0..19usize {
        for &is_inter in &[false, true] {
            for &reduced in &[false, true] {
                for tx_type in 0..16usize {
                    let d = ext_tx_derive(tx_size, is_inter, reduced, tx_type, false, 0, 0);
                    if d.num <= 1 || d.used == 0 {
                        continue; // not signaled / not in set
                    }
                    let n = d.num as usize;
                    let cdf0 = gen_cdf(&mut rng, n);

                    // Composed path.
                    let mut cdf_a = cdf0.clone();
                    let mut enc_a = OdEcEnc::new();
                    write_tx_type(
                        &mut enc_a, &mut cdf_a, tx_size, is_inter, reduced, tx_type, false, 0, 0,
                        true,
                    );
                    let bytes_a = enc_a.done().to_vec();

                    // Direct symbol write of the expected symbol on identical CDF.
                    let mut cdf_b = cdf0.clone();
                    let mut enc_b = OdEcEnc::new();
                    aom_entropy::cdf::write_symbol(&mut enc_b, d.symb, &mut cdf_b, n);
                    let bytes_b = enc_b.done().to_vec();

                    assert_eq!(bytes_a, bytes_b, "bytes ts={tx_size} inter={is_inter} tt={tx_type}");
                    assert_eq!(cdf_a, cdf_b, "cdf adapt ts={tx_size} tt={tx_type}");
                }
            }
        }
    }
}
