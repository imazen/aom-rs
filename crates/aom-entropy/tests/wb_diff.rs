//! Differential harness for the byte-aligned bit writer (aom_write_bit_buffer)
//! vs C libaom: a random sequence of write_literal / write_unsigned_literal /
//! write_inv_signed_literal ops must produce byte-identical output (and the same
//! bytes_written rounding).

use aom_entropy::wb::WriteBitBuffer;
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

#[test]
fn wb_write_sequence_identical() {
    let mut rng = Rng(0x0b17_c0de_a11a_0009);
    for _ in 0..40_000 {
        let nops = rng.range(1, 40) as usize;
        let mut data = Vec::with_capacity(nops);
        let mut bits = Vec::with_capacity(nops);
        let mut kind = Vec::with_capacity(nops);
        let mut wb = WriteBitBuffer::new();
        for _ in 0..nops {
            let k = rng.range(0, 4) as i32; // 3 = add_trailing_bits
            // signed / inv-signed literals use bits <= 31; unsigned <= 32.
            let b = if k == 1 { rng.range(1, 33) } else { rng.range(1, 32) };
            // data must fit the field so both sides interpret it identically.
            let mask: u32 = if b >= 32 { u32::MAX } else { (1u32 << b) - 1 };
            let d = (rng.next() as u32) & mask;
            data.push(d);
            bits.push(b as i32);
            kind.push(k);
            match k {
                1 => wb.write_unsigned_literal(d, b),
                2 => wb.write_inv_signed_literal(d as i32, b),
                3 => wb.add_trailing_bits(),
                _ => wb.write_literal(d as i32, b),
            }
        }
        let got = wb.bytes().to_vec();
        let want = c::ref_wb_apply(&data, &bits, &kind);
        assert_eq!(got, want, "wb sequence nops={nops}");
        assert_eq!(wb.bytes_written(), want.len(), "bytes_written nops={nops}");
    }
}

#[test]
fn signed_subexpfin_matches_c() {
    let mut rng = Rng(0x5f00_c0de_a11a_0009);
    // Exercise the GM parameter ranges: n = GM_ALPHA_MAX+1 (4097) and the
    // translation n = (1<<trans_bits)+1 for trans_bits in {9,12}; k = SUBEXPFIN_K = 3.
    for &n in &[4097i32, (1 << 12) + 1, (1 << 9) + 1] {
        for _ in 0..300_000 {
            let r = rng.range(0, (2 * n - 1) as u32) as i32 - (n - 1);
            let v = rng.range(0, (2 * n - 1) as u32) as i32 - (n - 1);
            let mut wb = WriteBitBuffer::new();
            wb.write_signed_primitive_refsubexpfin(n as u16, 3, r as i16, v as i16);
            let got = wb.bytes().to_vec();
            let want = c::ref_wb_signed_subexpfin(n, 3, r, v);
            assert_eq!(got, want, "signed_subexpfin n={n} k=3 ref={r} v={v}");
        }
    }
}

#[test]
fn uvlc_matches_c() {
    let mut rng = Rng(0x00c1_c0de_a11a_0009);
    // small values densely, plus the full 32-bit range (excluding u32::MAX).
    for v in 0u32..2000 {
        let mut wb = WriteBitBuffer::new();
        wb.write_uvlc(v);
        assert_eq!(wb.bytes(), &c::ref_wb_uvlc(v)[..], "uvlc {v}");
    }
    for _ in 0..500_000 {
        let v = (rng.next() as u32) & 0xffff_fffe; // avoid u32::MAX
        let mut wb = WriteBitBuffer::new();
        wb.write_uvlc(v);
        assert_eq!(wb.bytes(), &c::ref_wb_uvlc(v)[..], "uvlc {v}");
    }
}
