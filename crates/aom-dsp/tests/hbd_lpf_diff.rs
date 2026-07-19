//! Differential harness for highbd (10/12-bit) deblocking loop filter vs C.
use aom_dsp::loopfilter::highbd;
use aom_sys_ref as c;

const PITCH: usize = 32;
const ROWS: usize = 32;
const CENTER: usize = 12 * PITCH + 12;

struct Rng(u64);
impl Rng { fn next(&mut self)->u64{let mut x=self.0;x^=x>>12;x^=x<<25;x^=x>>27;self.0=x;x.wrapping_mul(0x2545_F491_4F6C_DD1D)} fn upto(&mut self,n:u32)->u32{(self.next()%n as u64)as u32} }

#[test]
fn hbd_loopfilter_byte_identical() {
    let mut rng = Rng(0x_a1b2_c3d4_e5f6_0789);
    for &bd in &[10i32, 12] {
        let maxv = (1u32 << bd) - 1;
        for &dir in b"hv" {
            for &width in &[4u32, 6, 8, 14] {
                for _ in 0..15_000 {
                    // near-flat sometimes to hit flat/flat2 branches
                    let base = rng.upto(maxv + 1);
                    let amp = 1 + rng.upto(1 << (bd - 4));
                    let strat = rng.upto(3);
                    let buf: Vec<u16> = (0..PITCH*ROWS).map(|_| {
                        if strat == 0 { rng.upto(maxv+1) as u16 }
                        else { (base as i32 + rng.upto(2*amp+1) as i32 - amp as i32).clamp(0, maxv as i32) as u16 }
                    }).collect();
                    let bl = if rng.upto(2)==0 { rng.upto(256) as u8 } else { (16 + rng.upto(200)) as u8 };
                    let li = if rng.upto(2)==0 { rng.upto(256) as u8 } else { (1 + rng.upto(64)) as u8 };
                    let th = rng.upto(256) as u8;

                    let mut got = buf.clone();
                    let mut want = buf.clone();
                    if dir == b'h' { highbd::horizontal(width, &mut got, CENTER, PITCH, bl, li, th, bd); }
                    else { highbd::vertical(width, &mut got, CENTER, PITCH, bl, li, th, bd); }
                    c::ref_hbd_lpf(dir, width, &mut want, CENTER, PITCH, bl, li, th, bd);
                    assert_eq!(got, want, "hbd lpf dir={} width={width} bd={bd} bl={bl} li={li} th={th}", dir as char);
                }
            }
        }
    }
}
