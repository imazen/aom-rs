//! Differential harness for cdef_filter_8_{0,1,2,3} vs C libaom v3.14.1.
use aom_cdef::{cdef_filter_block, CDEF_BSTRIDE, CDEF_VERY_LARGE};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { let mut x=self.0; x^=x>>12; x^=x<<25; x^=x>>27; self.0=x; x.wrapping_mul(0x2545_F491_4F6C_DD1D) }
    fn upto(&mut self, n: u32) -> u32 { (self.next() % n as u64) as u32 }
}

#[test]
fn cdef_filter8_byte_identical() {
    let mut rng = Rng(0x_cdef_f117_e12a_3344);
    let rows = 8 + 8; // >= 2 border rows around 8-high block
    let buf_len = rows * CDEF_BSTRIDE;
    // block origin: 4 rows + 8 cols border (>= VBORDER=2 / dir reach)
    let in_off = 4 * CDEF_BSTRIDE + 8;
    for variant in 0..4i32 {
        for _ in 0..80_000 {
            let bw = if rng.upto(2) == 0 { 4 } else { 8 };
            let bh = if rng.upto(2) == 0 { 4 } else { 8 };
            let mut inbuf = vec![0u16; buf_len];
            for v in inbuf.iter_mut() {
                *v = if rng.upto(20) == 0 { CDEF_VERY_LARGE as u16 } else { rng.upto(256) as u16 };
            }
            let pri = rng.upto(16) as i32;
            let sec = rng.upto(5) as i32;
            let dir = rng.upto(8) as i32;
            let prid = (3 + rng.upto(4)) as i32;
            let secd = (3 + rng.upto(4)) as i32;
            let cshift = 0;

            let mut got = vec![0u8; bw * bh];
            cdef_filter_block(&mut got, bw, &inbuf, in_off, pri, sec, dir, prid, secd, cshift,
                              bw, bh, variant == 0 || variant == 1, variant == 0 || variant == 2);
            let want = c::ref_cdef_filter8(variant, &inbuf, in_off, pri, sec, dir, prid, secd, cshift, bw, bh);
            assert_eq!(got, want, "cdef_filter_8_{variant} bw={bw} bh={bh} dir={dir} pri={pri} sec={sec}");
        }
    }
}
