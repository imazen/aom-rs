//! Differential harness for the LEB128 varint codec vs C libaom: size_in_bytes,
//! encode (incl. failure on the 32-bit cap / insufficient space), and decode
//! (incl. failure on overrun / over-cap) must all match exactly.

use aom_entropy::leb128::{uleb_decode, uleb_encode, uleb_size_in_bytes};
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
}

#[test]
fn uleb_size_and_encode_match_c() {
    let mut rng = Rng(0x1eb1_2800_9e37_79b9);
    for _ in 0..300_000 {
        // Mix in-range (<=u32::MAX) and out-of-range (>u32::MAX) values.
        let value = if rng.next().is_multiple_of(4) {
            rng.next()
        } else {
            rng.next() % (1u64 << 33)
        };
        assert_eq!(
            uleb_size_in_bytes(value),
            c::ref_uleb_size_in_bytes(value),
            "size {value}"
        );
        let available = (rng.next() % 10) as usize; // sometimes too small
        assert_eq!(
            uleb_encode(value, available),
            c::ref_uleb_encode(value, available),
            "encode {value} avail={available}"
        );
    }
}

#[test]
fn uleb_decode_matches_c() {
    let mut rng = Rng(0x1eb1_2800_c057_0b11);
    for _ in 0..300_000 {
        // Random byte streams (some valid LEB128, some overrunning / over-cap).
        let n = (rng.next() % 10) as usize;
        let buffer: Vec<u8> = (0..n).map(|_| (rng.next() % 256) as u8).collect();
        assert_eq!(
            uleb_decode(&buffer),
            c::ref_uleb_decode(&buffer),
            "decode {buffer:?}"
        );
    }
}
