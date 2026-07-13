//! Differential harness for av1_convolve_{x,y}_sr (EIGHTTAP_REGULAR) vs C.

use aom_convolve::{convolve_x_sr, convolve_y_sr};
use aom_sys_ref as c;

const BORDER: usize = 8;
const SIZES: [(usize, usize); 10] = [
    (8, 8), (8, 16), (16, 8), (16, 16), (16, 32), (32, 16), (32, 32), (64, 64), (8, 32), (64, 16),
];

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
    fn u8(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
}

#[test]
fn convolve_x_y_sr_byte_identical() {
    let mut rng = Rng(0x_c07f_0123_4567_89ab);
    for &(w, h) in SIZES.iter() {
        let stride = w + 2 * BORDER;
        let rows = h + 2 * BORDER;
        let src_off = BORDER * stride + BORDER;
        for _ in 0..4000 {
            let src: Vec<u8> = (0..stride * rows).map(|_| rng.u8()).collect();
            let subpel = (rng.next() % 16) as usize;

            let mut gx = vec![0u8; w * h];
            convolve_x_sr(&src, src_off, stride, &mut gx, w, w, h, subpel);
            let wx = c::ref_convolve_x_sr(&src, src_off, stride, w, h, subpel);
            assert_eq!(gx, wx, "convolve_x_sr {w}x{h} subpel={subpel}");

            let mut gy = vec![0u8; w * h];
            convolve_y_sr(&src, src_off, stride, &mut gy, w, w, h, subpel);
            let wy = c::ref_convolve_y_sr(&src, src_off, stride, w, h, subpel);
            assert_eq!(gy, wy, "convolve_y_sr {w}x{h} subpel={subpel}");
        }
    }
}
