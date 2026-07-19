//! Differential harness for `cdef_find_dir` vs C libaom v3.14.1.

use aom_dsp::cdef::cdef_find_dir;
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
fn cdef_find_dir_matches_c() {
    let mut rng = Rng(0x_cdef_1234_5678_9abc);
    let stride = 8 + 4;
    // Test 8/10/12-bit (coeff_shift 0/2/4) with matching pixel ranges.
    for &(coeff_shift, maxv) in &[(0i32, 255u64), (2, 1023), (4, 4095)] {
        for _ in 0..200_000 {
            let img: Vec<u16> = (0..stride * 8)
                .map(|_| (rng.next() % (maxv + 1)) as u16)
                .collect();
            let got = cdef_find_dir(&img, stride, coeff_shift);
            let want = c::ref_cdef_find_dir(&img, stride, coeff_shift);
            assert_eq!(
                got, want,
                "cdef_find_dir divergence coeff_shift={coeff_shift}"
            );
        }
    }
}
