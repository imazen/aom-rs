//! Differential harness for the Daala range coder vs C libaom v3.14.1.
//!
//! For random op sequences: (1) Rust-encode == C-encode byte-for-byte;
//! (2) decoding the C bytes with the Rust decoder recovers the original
//! symbols; (3) the C decoder agrees. Also a pure-Rust encode→decode round-trip.

use aom_entropy::{OdEcDec, OdEcEnc};
use aom_sys_ref::{self as c, EcOp};

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

/// Build a valid inverse CDF (strictly decreasing, ends at 0).
fn gen_icdf(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
    let mut icdf = vec![0u16; nsyms];
    let mut acc: u32 = 0;
    for e in icdf.iter_mut().take(nsyms - 1) {
        acc += rng.range(2, 2002);
        *e = (32768 - acc) as u16; // strictly decreasing, in (0, 32768)
    }
    icdf[nsyms - 1] = 0;
    icdf
}

fn gen_ops(rng: &mut Rng, n: usize) -> (Vec<EcOp>, Vec<i32>) {
    let mut ops = Vec::with_capacity(n);
    let mut syms = Vec::with_capacity(n);
    for _ in 0..n {
        if rng.next() & 1 == 0 {
            let val = (rng.next() & 1) as i32;
            let f = rng.range(1, 32768);
            ops.push(EcOp::Bool { val, f });
            syms.push(val);
        } else {
            let nsyms = rng.range(2, 17) as usize;
            let icdf = gen_icdf(rng, nsyms);
            let s = rng.range(0, nsyms as u32) as i32;
            ops.push(EcOp::Cdf { s, icdf });
            syms.push(s);
        }
    }
    (ops, syms)
}

fn rust_encode(ops: &[EcOp]) -> Vec<u8> {
    let mut enc = OdEcEnc::new();
    for op in ops {
        match op {
            EcOp::Bool { val, f } => enc.encode_bool_q15(*val, *f),
            EcOp::Cdf { s, icdf } => enc.encode_cdf_q15(*s, icdf, icdf.len() as i32),
        }
    }
    enc.done().to_vec()
}

fn rust_decode(buf: &[u8], ops: &[EcOp]) -> Vec<i32> {
    let mut dec = OdEcDec::new(buf);
    ops.iter()
        .map(|op| match op {
            EcOp::Bool { f, .. } => dec.decode_bool_q15(*f),
            EcOp::Cdf { icdf, .. } => dec.decode_cdf_q15(icdf, icdf.len() as i32),
        })
        .collect()
}

#[test]
fn entropy_encode_byte_identical() {
    let mut rng = Rng(0x_5eed_1234_abcd_0001);
    for _ in 0..20_000 {
        let n = rng.range(1, 400) as usize;
        let (ops, _syms) = gen_ops(&mut rng, n);
        let r = rust_encode(&ops);
        let cc = c::ref_ec_encode(&ops);
        assert_eq!(r, cc, "encoder byte divergence at n={n}");
    }
}

#[test]
fn entropy_decode_matches_and_roundtrips() {
    let mut rng = Rng(0x_d00d_5678_ef01_0002);
    for _ in 0..20_000 {
        let n = rng.range(1, 400) as usize;
        let (ops, syms) = gen_ops(&mut rng, n);
        let cbuf = c::ref_ec_encode(&ops);

        // Rust decoder recovers the original symbols from the C bytes.
        let dec_r = rust_decode(&cbuf, &ops);
        assert_eq!(dec_r, syms, "rust decode of C bytes != original symbols");

        // C decoder agrees with Rust decoder.
        let dec_c = c::ref_ec_decode(&cbuf, &ops);
        assert_eq!(dec_r, dec_c, "rust vs C decoder divergence");

        // Pure-Rust round trip.
        let rbuf = rust_encode(&ops);
        let dec_rr = rust_decode(&rbuf, &ops);
        assert_eq!(dec_rr, syms, "pure-rust encode/decode round trip failed");
    }
}
