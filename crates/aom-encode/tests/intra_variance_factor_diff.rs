//! Differential: `intra_rd_variance_factor` (+ `compute_avg_log_variance` +
//! `av1_calc_normalized_variance`) vs the verbatim C transcription compiled in
//! rd_shim.c over the REAL 4x4 variance kernels + libm log1p. Bit-level f64
//! equality of the factor (`to_bits`), plus the evolving per-SB source-var
//! cache (var + log_var bits) — the cache is shared state across a mode
//! loop's candidates and across a superblock's coding blocks, so each case
//! runs SEQUENCES of calls (fresh cache -> warm cache -> var-only-seeded
//! cache) in lockstep.

use aom_encode::intra_rd::{Block4x4VarInfo, VarFactorInputs, intra_rd_variance_factor};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

const MI_W: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_H: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

/// Fill a `bw x bh` window with one content class: flat / noisy / gradient /
/// mixed (flat rows + noisy rows — drives per-4x4 var asymmetry).
#[allow(clippy::too_many_arguments)]
fn fill_class(
    rng: &mut Rng,
    plane: &mut [u16],
    off: usize,
    stride: usize,
    bw: usize,
    bh: usize,
    class: usize,
    bd: u8,
) {
    let maxv = (1u32 << bd) - 1;
    let base = (rng.next() % u64::from(maxv)) as i64;
    for r in 0..bh {
        for cx in 0..bw {
            let v: i64 = match class {
                0 => base,                                        // flat
                1 => (rng.next() % (u64::from(maxv) + 1)) as i64, // noisy
                2 => base + (cx as i64 * 2) + (r as i64),         // gradient
                _ => {
                    if (r / 4) % 2 == 0 {
                        base
                    } else {
                        (rng.next() % (u64::from(maxv) + 1)) as i64
                    }
                }
            };
            plane[off + r * stride + cx] = v.clamp(0, i64::from(maxv)) as u16;
        }
    }
}

#[test]
fn intra_rd_variance_factor_matches_c() {
    c::ref_init();
    let mut rng = Rng(0xfac7_0a1d_2026_0714);
    const STRIDE: usize = 256;
    // (bsize, sb_size, mi_row, mi_col): block placed inside its superblock.
    let cases: [(usize, usize, i32, i32); 6] = [
        (3, 12, 8, 8),    // 8x8 in 64x64 sb
        (6, 12, 20, 24),  // 16x16
        (9, 12, 8, 8),    // 32x32
        (12, 12, 16, 16), // 64x64 (mi_in_sb = 0)
        (4, 12, 9, 10),   // 8x16 rect
        (9, 15, 40, 8),   // 32x32 in 128x128 sb
    ];
    let mut scaled_up_src = 0usize; // src >= recon arm fired
    let mut scaled_up_recon = 0usize; // recon > src arm fired
    let mut clamped = 0usize;
    let mut unity = 0usize;
    let mut edge_cases = 0usize;
    let mut cache_reused = 0usize;

    for (ci, &(bsize, sb_size, mi_row, mi_col)) in cases.iter().enumerate() {
        let (bw, bh) = (4 * MI_W[bsize], 4 * MI_H[bsize]);
        for iter in 0..24 {
            let bd: u8 = match iter % 3 {
                0 => 8,
                1 => 10,
                _ => 12,
            };
            // Content classes chosen to hit both factor arms: (src, recon).
            let (src_class, rec_class) = match iter % 6 {
                0 => (1, 0), // noisy src, flat recon -> src arm
                1 => (0, 1), // flat src, noisy recon -> recon arm
                2 => (1, 1),
                3 => (0, 0),
                4 => (3, 2),
                _ => (2, 3),
            };
            // Frame-edge overhang (1/8-pel negative edges) on some iters —
            // the clipped-walk variance edge cases.
            let (right_edge, bottom_edge) = if iter % 5 == 4 && bw >= 8 && bh >= 8 {
                edge_cases += 1;
                (-(8 * (bw as i32 / 2 / 4 * 4)), -(8 * 4))
            } else {
                (1 << 12, 1 << 12)
            };

            let mut src = vec![0u16; STRIDE * 160];
            let mut recon = vec![0u16; STRIDE * 160];
            for v in src.iter_mut() {
                *v = (rng.next() % (1 << bd)) as u16;
            }
            for v in recon.iter_mut() {
                *v = (rng.next() % (1 << bd)) as u16;
            }
            let src_off = 16 * STRIDE + 32;
            let ref_off = 24 * STRIDE + 48;
            fill_class(&mut rng, &mut src, src_off, STRIDE, bw, bh, src_class, bd);
            fill_class(&mut rng, &mut recon, ref_off, STRIDE, bw, bh, rec_class, bd);

            let n_mi = MI_W[sb_size] * MI_H[sb_size];
            let mut cache_rust = Block4x4VarInfo::sb_cache(sb_size);
            let mut cvar: Vec<i32> = vec![-1; n_mi];
            let mut clog: Vec<f64> = vec![-1.0; n_mi];
            // Var-only-seeded regime: pre-seed SOME entries with a valid var
            // but log_var = -1 (the C's "var cached, log not yet" branch).
            if iter % 4 == 3 {
                for k in 0..n_mi {
                    if k % 3 == 0 {
                        let seeded = rng.range(0, 1 << 16);
                        cache_rust[k] = Block4x4VarInfo {
                            var: seeded,
                            log_var: -1.0,
                        };
                        cvar[k] = seeded;
                    }
                }
            }

            let p = VarFactorInputs {
                src: &src,
                src_off,
                src_stride: STRIDE,
                recon: &recon,
                ref_off,
                ref_stride: STRIDE,
                bsize,
                sb_size,
                mi_row,
                mi_col,
                mb_to_right_edge: right_edge,
                mb_to_bottom_edge: bottom_edge,
                bd,
            };

            // Sequence: two calls — the second reuses the now-warm cache
            // (mode-loop shape: every candidate shares the SB cache).
            for call in 0..2 {
                let speed = if iter % 7 == 6 { 4 } else { 0 }; // speed>=4: threshold <= 0
                let got = intra_rd_variance_factor(speed, &p, &mut cache_rust);
                let want = c::ref_intra_rd_variance_factor(
                    speed,
                    &src,
                    src_off,
                    STRIDE,
                    &recon,
                    ref_off,
                    STRIDE,
                    bsize,
                    sb_size,
                    mi_row,
                    mi_col,
                    right_edge,
                    bottom_edge,
                    bd,
                    &mut cvar,
                    &mut clog,
                );
                let m = format!(
                    "ci={ci} bsize={bsize} sb={sb_size} iter={iter} call={call} bd={bd} \
                     classes=({src_class},{rec_class}) edges=({right_edge},{bottom_edge})",
                );
                assert_eq!(got.to_bits(), want.to_bits(), "factor {got} vs {want} {m}");
                for k in 0..n_mi {
                    assert_eq!(cache_rust[k].var, cvar[k], "cache var k={k} {m}");
                    assert_eq!(
                        cache_rust[k].log_var.to_bits(),
                        clog[k].to_bits(),
                        "cache log_var k={k} {m}",
                    );
                }
                if call == 1 {
                    cache_reused += 1;
                }
                if got == 1.0 {
                    unity += 1;
                } else if got == 3.0 {
                    clamped += 1;
                } else if src_class == 1 || src_class == 3 {
                    scaled_up_src += 1;
                } else {
                    scaled_up_recon += 1;
                }
            }
        }
    }
    assert!(scaled_up_src > 8, "src>=recon factor arm: {scaled_up_src}");
    assert!(
        scaled_up_recon > 8,
        "recon>src factor arm: {scaled_up_recon}"
    );
    assert!(clamped > 4, "3.0 clamp: {clamped}");
    assert!(unity > 30, "unity factor: {unity}");
    assert!(edge_cases > 10, "frame-edge clipped walks: {edge_cases}");
    assert!(cache_reused > 100, "warm-cache calls: {cache_reused}");
}
