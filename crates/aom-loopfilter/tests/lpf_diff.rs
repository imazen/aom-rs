//! Differential harness for the deblocking loop filter vs C libaom v3.14.1:
//! horizontal + vertical, widths {4,6,8,14}, over pixel windows chosen to
//! exercise the mask/flat/flat2 branches.

use aom_loopfilter::{horizontal, vertical};
use aom_sys_ref as c;

const PITCH: usize = 32;
const ROWS: usize = 32;
const CENTER: usize = 12 * PITCH + 12; // row 12, col 12 — >=7 margin all sides

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
    fn upto(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

fn gen_buf(rng: &mut Rng, strategy: u32) -> Vec<u8> {
    match strategy {
        0 => (0..PITCH * ROWS).map(|_| rng.u8()).collect(), // fully random
        _ => {
            // near-flat: base + small noise (triggers flat/flat2 paths)
            let base = rng.u8() as i32;
            let amp = 1 + rng.upto(8) as i32;
            (0..PITCH * ROWS)
                .map(|_| (base + rng.upto((2 * amp + 1) as u32) as i32 - amp).clamp(0, 255) as u8)
                .collect()
        }
    }
}

#[test]
fn loopfilter_byte_identical() {
    let mut rng = Rng(0x_10fd_ead0_1234_5678);
    let mut checks = 0u64;
    for &dir in b"hv" {
        for &width in &[4u32, 6, 8, 14] {
            for _ in 0..30_000 {
                let strategy = rng.upto(3);
                let base = gen_buf(&mut rng, strategy);
                // thresholds: vary to trigger/skip filtering.
                let blimit = if rng.upto(2) == 0 { rng.u8() } else { (16 + rng.upto(200)) as u8 };
                let limit = if rng.upto(2) == 0 { rng.u8() } else { (1 + rng.upto(64)) as u8 };
                let thresh = rng.u8();

                let mut got = base.clone();
                let mut want = base.clone();
                if dir == b'h' {
                    horizontal(width, &mut got, CENTER, PITCH, blimit, limit, thresh);
                } else {
                    vertical(width, &mut got, CENTER, PITCH, blimit, limit, thresh);
                }
                c::ref_lpf(dir, width, &mut want, CENTER, PITCH, blimit, limit, thresh);
                assert_eq!(
                    got, want,
                    "lpf divergence dir={} width={width} blimit={blimit} limit={limit} thresh={thresh}",
                    dir as char
                );
                checks += 1;
            }
        }
    }
    assert!(checks > 100_000, "{checks}");
}
