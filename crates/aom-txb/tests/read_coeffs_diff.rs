//! Roundtrip harness for `read_coeffs_txb`: the decoder must invert
//! `write_coeffs_txb` exactly. Since `write_coeffs_txb` is byte-identical to C
//! libaom (`write_coeffs_txb_byte_identical`), a clean roundtrip — recovered
//! coefficients, `eob`, and (with adaptation on) the lockstep CDF arena — pins
//! `read_coeffs_txb` to libaom's `av1_read_coeffs_txb`.

use aom_entropy::dec::OdEcDec;
use aom_entropy::enc::OdEcEnc;
use aom_txb::{read_coeffs_txb, scan, txb_high, txb_wide, write_coeffs_txb, CDF_ARENA_LEN};

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
    (0, 5 * 13, 2),
    (195, 4, 5),
    (219, 4, 6),
    (247, 4, 7),
    (279, 4, 8),
    (315, 4, 9),
    (355, 4, 10),
    (399, 4, 11),
    (447, 5 * 2 * 9, 2),
    (717, 5 * 2 * 4, 3),
    (877, 5 * 2 * 42, 4),
    (2977, 5 * 2 * 21, 4),
    (4027, 2 * 3, 2),
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
    #[allow(clippy::needless_range_loop)]
    for i in 0..eob {
        let pos = scan[i] as usize;
        // eob position always nonzero; ~60% of interior positions nonzero.
        // `||` short-circuits so the range() draw is skipped at eob-1, exactly
        // as the separate-branch form would.
        if i == eob - 1 || rng.range(0, 10) >= 4 {
            coeff[pos] = nz(rng);
        }
    }
    (coeff, eob)
}

#[test]
fn read_coeffs_txb_roundtrips_write() {
    let mut rng = Rng(0x_00c0_ffee_dec0_de01);
    const TX_TYPES: [usize; 7] = [0, 3, 9, 10, 14, 11, 15];

    for tx_size in 0..19usize {
        let area = txb_wide(tx_size) * txb_high(tx_size);
        for &tx_type in &TX_TYPES {
            for &plane_type in &[0usize, 1] {
                for &upd in &[true, false] {
                    for _ in 0..80 {
                        let sc = scan(tx_size, tx_type);
                        let (coeff, eob) = gen_coeffs(&mut rng, sc, area);
                        let txb_skip_ctx = rng.range(0, 13) as usize;
                        let dc_sign_ctx = rng.range(0, 3) as usize;

                        let arena0 = gen_arena(&mut rng);
                        let mut arena_e = arena0.clone();
                        let mut arena_d = arena0.clone();

                        let mut enc = OdEcEnc::new();
                        write_coeffs_txb(
                            &mut enc, &mut arena_e, &coeff, eob, tx_size, tx_type,
                            plane_type, txb_skip_ctx, dc_sign_ctx, upd,
                        );
                        let bytes = enc.done().to_vec();

                        let mut dec = OdEcDec::new(&bytes);
                        let mut tcoeff = vec![0i32; area];
                        let eob_d = read_coeffs_txb(
                            &mut dec, &mut arena_d, &mut tcoeff, tx_size, tx_type,
                            plane_type, txb_skip_ctx, dc_sign_ctx, upd,
                        );

                        assert_eq!(
                            eob_d, eob,
                            "eob tx_size={tx_size} tx_type={tx_type} plane={plane_type} upd={upd}"
                        );
                        assert_eq!(
                            tcoeff, coeff,
                            "coeffs tx_size={tx_size} tx_type={tx_type} plane={plane_type} \
                             upd={upd} eob={eob}"
                        );
                        assert_eq!(
                            arena_d, arena_e,
                            "cdf adaptation diverged tx_size={tx_size} tx_type={tx_type} upd={upd}"
                        );
                    }
                }
            }
        }
    }
}
