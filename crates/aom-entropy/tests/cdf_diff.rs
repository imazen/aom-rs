//! Differential harness for CDF adaptation + adaptive symbol coding vs C.

use aom_entropy::{read_symbol, update_cdf, write_symbol, OdEcDec, OdEcEnc};
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
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo) as u64) as u32
    }
}

/// cdf array of length nsymbs+1: inverse cdf (icdf[nsymbs-1]==0) + count(=0).
fn gen_cdf(rng: &mut Rng, nsymbs: usize) -> Vec<u16> {
    let mut cdf = vec![0u16; nsymbs + 1];
    let mut acc: u32 = 0;
    for e in cdf.iter_mut().take(nsymbs - 1) {
        acc += rng.range(2, 2002);
        *e = (32768 - acc) as u16;
    }
    cdf[nsymbs - 1] = 0;
    cdf[nsymbs] = 0; // count
    cdf
}

#[test]
fn update_cdf_matches_c() {
    let mut rng = Rng(0x_abcd_0001_1111_2222);
    for _ in 0..5000 {
        let nsymbs = rng.range(2, 17) as usize;
        let mut cdf = gen_cdf(&mut rng, nsymbs);
        // Stateful: apply a sequence of updates, checking each step vs C.
        for _ in 0..200 {
            let val = rng.range(0, nsymbs as u32) as i32;
            let want = c::ref_update_cdf(&cdf, val, nsymbs);
            update_cdf(&mut cdf, val, nsymbs);
            assert_eq!(cdf, want, "update_cdf divergence nsymbs={nsymbs} val={val}");
        }
    }
}

#[test]
fn adaptive_symbol_coding_matches_c() {
    let mut rng = Rng(0x_1357_9bdf_2468_ace0);
    for _ in 0..10_000 {
        let nsymbs = rng.range(2, 17) as usize;
        let cdf_init = gen_cdf(&mut rng, nsymbs);
        let n = rng.range(1, 300) as usize;
        let syms: Vec<i32> = (0..n).map(|_| rng.range(0, nsymbs as u32) as i32).collect();

        // Rust adaptive encode (write_symbol loop, shared adapting context).
        let mut enc = OdEcEnc::new();
        let mut cdf = cdf_init.clone();
        for &s in &syms {
            write_symbol(&mut enc, s, &mut cdf, nsymbs);
        }
        let rbuf = enc.done().to_vec();

        // Must be byte-identical to the C adaptive encoder.
        let cbuf = c::ref_adapt_encode(&syms, &cdf_init, nsymbs);
        assert_eq!(
            rbuf, cbuf,
            "adaptive encode byte divergence nsymbs={nsymbs} n={n}"
        );

        // Rust adaptive decode recovers the symbols.
        let mut dec = OdEcDec::new(&cbuf);
        let mut cdf_d = cdf_init.clone();
        let dec_syms: Vec<i32> = (0..n)
            .map(|_| read_symbol(&mut dec, &mut cdf_d, nsymbs))
            .collect();
        assert_eq!(dec_syms, syms, "adaptive decode != original symbols");

        // C decoder agrees.
        let c_syms = c::ref_adapt_decode(&cbuf, n, &cdf_init, nsymbs);
        assert_eq!(dec_syms, c_syms, "rust vs C adaptive decode divergence");
    }
}
