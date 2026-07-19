//! Roundtrip harness for `read_coding_block_plane` — the per-plane transform-block
//! coefficient decode loop. A matching write loop (write_coeffs_txb_full per txb +
//! entropy-context threading) encodes given quantized coeffs; read_coding_block_plane
//! reads them back. Since write_coeffs_txb_full is byte-identical to C and the entropy-
//! context threading mirrors av1_decode_coding_block, a clean roundtrip pins the loop.

use aom_encode::{TxTypeContext, read_coding_block_plane};
use aom_dsp::entropy::dec::OdEcDec;
use aom_dsp::entropy::enc::OdEcEnc;
use aom_dsp::txb::{
    CDF_ARENA_LEN, get_txb_ctx, scan, txb_entropy_context, txb_high, txb_wide,
    write_coeffs_txb_full,
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

const BLK_W: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLK_H: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
const TXU_W: [usize; 19] = [1, 2, 4, 8, 16, 1, 2, 2, 4, 4, 8, 8, 16, 1, 4, 2, 8, 4, 16];
const TXU_H: [usize; 19] = [1, 2, 4, 8, 16, 2, 1, 4, 2, 8, 4, 16, 8, 4, 1, 8, 2, 16, 4];

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
fn gen_coeffs(rng: &mut Rng, sc: &[i16], area: usize) -> (Vec<i32>, usize) {
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
        let pos = sc[i] as usize;
        if i == eob - 1 || rng.range(0, 10) >= 4 {
            coeff[pos] = nz(rng);
        }
    }
    (coeff, eob)
}

#[test]
fn read_coding_block_plane_roundtrips() {
    let mut rng = Rng(0x0f01_b10c_ede0_9e37);
    // (plane_bsize, tx_size) pairs where the tx tiles the plane evenly (multi-txb).
    let cases = [
        (3usize, 0usize),
        (6, 1),
        (9, 2),
        (12, 3),
        (6, 0),
        (9, 1),
        (12, 2),
    ];
    let ttx = TxTypeContext {
        is_inter: false,
        reduced: false,
        use_filter_intra: false,
        fi_mode: 0,
        mode: 0,
        signal_gate: false,
    };
    for &(pbsize, tx_size) in &cases {
        let area = txb_wide(tx_size) * txb_high(tx_size);
        let uw = BLK_W[pbsize] >> 2;
        let uh = BLK_H[pbsize] >> 2;
        let txw = TXU_W[tx_size];
        let txh = TXU_H[tx_size];
        for &plane in &[0usize, 1] {
            for &upd in &[true, false] {
                for _ in 0..2000 {
                    let arena0 = gen_arena(&mut rng);
                    let mut ext0 = vec![0u16; 17];
                    // write loop (mirrors read_coding_block_plane), tx_type = DCT_DCT.
                    let mut enc = OdEcEnc::new();
                    let mut aw = arena0.clone();
                    let mut ew = ext0.clone();
                    let mut above = vec![0i8; uw];
                    let mut left = vec![0i8; uh];
                    let mut want_coeffs: Vec<Vec<i32>> = Vec::new();
                    let mut blk_row = 0;
                    while blk_row < uh {
                        let mut blk_col = 0;
                        while blk_col < uw {
                            let (ts, ds) = get_txb_ctx(
                                pbsize,
                                tx_size,
                                plane,
                                &above[blk_col..],
                                &left[blk_row..],
                            );
                            let sc = scan(tx_size, 0);
                            let (coeff, eob) = gen_coeffs(&mut rng, sc, area);
                            write_coeffs_txb_full(
                                &mut enc,
                                &mut aw,
                                &mut ew,
                                &coeff,
                                eob,
                                tx_size,
                                0,
                                plane,
                                ts as usize,
                                ds as usize,
                                upd,
                                false,
                                false,
                                false,
                                0,
                                0,
                                false,
                            );
                            let cul = txb_entropy_context(&coeff, tx_size, 0, eob) as i8;
                            above[blk_col..blk_col + txw].fill(cul);
                            left[blk_row..blk_row + txh].fill(cul);
                            want_coeffs.push(coeff);
                            blk_col += txw;
                        }
                        blk_row += txh;
                    }
                    let bytes = enc.done().to_vec();
                    let mut dec = OdEcDec::new(&bytes);
                    let mut ar = arena0.clone();
                    ext0.clear();
                    ext0.resize(17, 0);
                    let mut er = ew.clone(); // any; unused (signal_gate false)
                    er.clear();
                    er.resize(17, 0);
                    let (gc, gctx) = read_coding_block_plane(
                        &mut dec, &mut ar, &mut er, pbsize, tx_size, 0, plane, upd, &ttx,
                    );
                    assert_eq!(
                        gc, want_coeffs,
                        "coeffs pbsize={pbsize} tx={tx_size} plane={plane}"
                    );
                    assert_eq!(gctx.above, above, "above ctx");
                    assert_eq!(gctx.left, left, "left ctx");
                    if upd {
                        assert_eq!(
                            ar, aw,
                            "arena adapt pbsize={pbsize} tx={tx_size} plane={plane}"
                        );
                    }
                }
            }
        }
    }
}
