//! Differential test for the `--deltaq-mode=2` (`DELTA_Q_PERCEPTUAL`, wavelet
//! AC energy) dwt kernel vs the REAL exported libaom C function
//! `av1_haar_ac_sad_mxn_uint8_input` (dwt.c:135 — a pure-C RTCD entry, so the
//! reference is exactly what real aomenc runs). This pins the 5/3 dyadic
//! wavelet + AC-SAD chain (`av1_fdwt8x8_uint8_input` + `haar_ac_sad`) before it
//! can perturb a mode-2 e2e stream. The rate-model tail
//! (`av1_compute_q_from_energy_level_deltaq_mode`) needs a `cpi` and is gated
//! e2e vs `aomenc --deltaq-mode=2` instead.

use aom_encode::allintra_vis;
use aom_sys_ref as c;

/// A tiny deterministic LCG so the sweep is reproducible without a dep.
struct Lcg(u64);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

/// Build a `w*h` u8 image with the given `stride` under a pattern, returning
/// the tightly-referenced u8 buffer and a u16 copy (the port's pixel type).
fn make_image(w: usize, h: usize, stride: usize, pattern: usize, rng: &mut Lcg) -> (Vec<u8>, Vec<u16>) {
    assert!(stride >= w);
    let mut u8buf = vec![0u8; stride * h];
    for r in 0..h {
        for col in 0..w {
            let v = match pattern {
                0 => rng.next_u8(),                                  // random
                1 => 0,                                              // flat black
                2 => 255,                                            // flat white
                3 => ((r + col) & 1) as u8 * 255,                    // checkerboard
                4 => ((r * 8 + col * 8) & 0xff) as u8,                // gradient
                5 => if col < w / 2 { 0 } else { 255 },              // vertical edge
                6 => if r < h / 2 { 20 } else { 235 },               // horizontal edge
                _ => (r as u8).wrapping_mul(31).wrapping_add(col as u8).wrapping_mul(17),
            };
            u8buf[r * stride + col] = v;
        }
    }
    let u16buf: Vec<u16> = u8buf.iter().map(|&b| u16::from(b)).collect();
    (u8buf, u16buf)
}

#[test]
fn haar_ac_sad_mxn_matches_c() {
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let mut checked = 0usize;
    let mut mismatches = 0usize;
    // Grid sizes covering the 64x64 SB (8x8 grid) and the sub-blocks the deeper
    // wavelet levels operate on, plus non-square shapes.
    let grids = [(1, 1), (2, 2), (4, 4), (8, 8), (1, 8), (8, 1), (3, 5), (5, 3), (2, 4), (4, 2)];
    for &(rows, cols) in &grids {
        let w = cols * 8;
        let h = rows * 8;
        for &pad in &[0usize, 3, 16] {
            let stride = w + pad;
            for pattern in 0..8usize {
                let iters = if pattern == 0 { 8 } else { 1 };
                for _ in 0..iters {
                    let (u8buf, u16buf) = make_image(w, h, stride, pattern, &mut rng);
                    let cref = c::ref_av1_haar_ac_sad_mxn_uint8_input(
                        &u8buf,
                        stride as i32,
                        rows as i32,
                        cols as i32,
                    );
                    let port =
                        allintra_vis::haar_ac_sad_mxn_for_test(&u16buf, 0, stride, 8, rows, cols);
                    checked += 1;
                    if port != cref {
                        mismatches += 1;
                        if mismatches <= 20 {
                            eprintln!(
                                "MISMATCH grid={rows}x{cols} stride={stride} pattern={pattern}: port={port} c={cref}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "av1_haar_ac_sad_mxn_uint8_input diverged on {mismatches}/{checked} cases"
    );
    eprintln!("haar_ac_sad_mxn vs C: {checked}/{checked} byte-exact");
}
