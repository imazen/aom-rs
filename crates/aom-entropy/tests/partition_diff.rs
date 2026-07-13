//! Differential harness for the partition-symbol CDF primitives
//! (partition_cdf_length + the edge-block gather transforms) vs C libaom.

use aom_entropy::partition::{
    partition_cdf_length, partition_gather_horz_alike, partition_gather_vert_alike,
};
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
fn partition_cdf_length_matches_c() {
    // All 22 BLOCK_SIZE values (BLOCK_4X4=0 .. BLOCK_SIZES_ALL).
    for bsize in 0..22 {
        assert_eq!(
            partition_cdf_length(bsize as usize),
            c::ref_partition_cdf_length(bsize) as usize,
            "partition_cdf_length bsize={bsize}"
        );
    }
}

#[test]
fn partition_gather_matches_c() {
    let mut rng = Rng(0x9a27_c0de_a11a_0009);
    // A valid inverse-cumulative CDF over EXT_PARTITION_TYPES(10) symbols is a
    // strictly-decreasing sequence 32768 > c0 > c1 > ... > c9 = 0, stored as
    // [c0..c9, count]. Build one by drawing sorted breakpoints.
    for _ in 0..200_000 {
        // draw 9 distinct interior points in (0, 32768), sort descending
        let mut pts = [0i32; 9];
        for p in &mut pts {
            *p = 1 + (rng.next() % 32766) as i32; // [1, 32767]
        }
        pts.sort_unstable();
        pts.reverse(); // descending
        // cdf[0..10]: cdf[i] = pts[i] for i<9, cdf[9] = 0; ensure strictly decreasing
        let mut cdf = [0u16; 11];
        let mut prev = 32768i32;
        for i in 0..9 {
            // keep strictly below prev; if a duplicate collapsed ordering, nudge
            let v = pts[i].min(prev - 1).max(9 - i as i32);
            cdf[i] = v as u16;
            prev = v;
        }
        cdf[9] = 0;
        cdf[10] = 0; // count field, unused by gather

        for &bsize in &[3i32, 8, 12, 15] {
            // 8x8, 16x16, 64x64, 128x128
            assert_eq!(
                partition_gather_vert_alike(&cdf, bsize as usize),
                c::ref_partition_gather_vert(&cdf, bsize),
                "gather_vert bsize={bsize} cdf={cdf:?}"
            );
            assert_eq!(
                partition_gather_horz_alike(&cdf, bsize as usize),
                c::ref_partition_gather_horz(&cdf, bsize),
                "gather_horz bsize={bsize} cdf={cdf:?}"
            );
        }
    }
}

#[test]
fn partition_plane_context_matches_c() {
    use aom_entropy::partition::partition_plane_context;
    let mut rng = Rng(0x9c2e_c0de_a11a_0009);
    // square partition points: 8x8=3, 16x16=6, 32x32=9, 64x64=12, 128x128=15
    let squares = [3i32, 6, 9, 12, 15];
    for _ in 0..300_000 {
        let mut above = [0i8; 64];
        let mut left = [0i8; 64];
        for a in &mut above {
            *a = (rng.next() & 0xff) as i8;
        }
        for l in &mut left {
            *l = (rng.next() & 0xff) as i8;
        }
        let bsize = squares[(rng.next() % 5) as usize];
        let mi_col = (rng.next() % 64) as i32;
        let mi_row = (rng.next() % 64) as i32;
        let got = partition_plane_context(&above, &left, mi_row as usize, mi_col as usize, bsize as usize);
        let want = c::ref_partition_plane_context(&above, &left, mi_row, mi_col, bsize);
        assert_eq!(got, want, "partition_plane_context bsize={bsize} r={mi_row} c={mi_col}");
    }
}

#[test]
fn write_partition_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{partition_cdf_length, write_partition};
    let mut rng = Rng(0x9a17_c0de_a11a_0009);
    let squares = [3i32, 6, 9, 12, 15]; // 8x8, 16x16, 32x32, 64x64, 128x128
    for _ in 0..200_000 {
        let bsize = squares[(rng.next() % 5) as usize];
        let ns = partition_cdf_length(bsize as usize);
        // Build a valid ns-symbol inverse-cumulative CDF: cdf[0..ns-2] strictly
        // descending in (0, 32768), cdf[ns-1]=0, cdf[ns]=count=0.
        let mut vals = [0i32; 9];
        for v in vals.iter_mut().take(ns - 1) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut cdf = [0u16; 11];
        let mut prev = 32768i32;
        for i in 0..ns - 1 {
            let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32);
            cdf[i] = v as u16;
            prev = v;
        }
        cdf[ns - 1] = 0;
        cdf[ns] = 0; // count

        let cdf_len = ns;
        let p = (rng.next() % cdf_len as u64) as i32;
        // 8x8 never reaches a frame-edge partition (asserted bsize > 8x8); keep valid.
        let (has_rows, has_cols) = if bsize == 3 {
            (true, true)
        } else {
            (rng.next().is_multiple_of(2), rng.next().is_multiple_of(2))
        };

        let mut my_cdf = cdf;
        let mut enc = OdEcEnc::new();
        write_partition(&mut enc, &mut my_cdf, cdf_len, p, has_rows, has_cols, bsize as usize);
        let got_bytes = enc.done().to_vec();

        let (want_bytes, want_cdf) = c::ref_write_partition(&cdf, cdf_len as i32, p, has_rows, has_cols, bsize);
        assert_eq!(got_bytes, want_bytes, "write_partition bytes bsize={bsize} p={p} r={has_rows} c={has_cols}");
        // compare the adapted CDF (only the has_rows&&has_cols path adapts)
        assert_eq!(
            &my_cdf[..cdf_len + 1],
            &want_cdf[..cdf_len + 1],
            "write_partition cdf bsize={bsize} p={p} r={has_rows} c={has_cols}"
        );
    }
}

#[test]
fn skip_txfm_context_matches_c() {
    use aom_entropy::partition::skip_txfm_context;
    let mut rng = Rng(0x54e6_c0de_a11a_0009);
    for _ in 0..50_000 {
        let ap = rng.next().is_multiple_of(2);
        let lp = rng.next().is_multiple_of(2);
        let as_ = rng.next().is_multiple_of(2) as i32;
        let ls = rng.next().is_multiple_of(2) as i32;
        // the real fn resolves absent neighbours to 0
        let above = if ap { as_ } else { 0 };
        let left = if lp { ls } else { 0 };
        assert_eq!(
            skip_txfm_context(above, left),
            c::ref_skip_txfm_context(ap, as_, lp, ls),
            "skip_txfm_context ap={ap} lp={lp}"
        );
    }
}

#[test]
fn write_skip_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_skip;
    let mut rng = Rng(0x54ab_c0de_a11a_0009);
    for _ in 0..200_000 {
        // valid 2-symbol CDF: cdf[0] in (0,32768), cdf[1]=0, cdf[2]=count=0
        let c0 = 1 + (rng.next() % 32766) as u16;
        let cdf = [c0, 0u16, 0u16];
        let seg_skip = rng.next().is_multiple_of(3);
        let skip_txfm = (rng.next() % 2) as i32;

        let mut my_cdf = cdf;
        let mut enc = OdEcEnc::new();
        let r = write_skip(&mut enc, &mut my_cdf, seg_skip, skip_txfm);
        let got = enc.done().to_vec();

        let (want, want_cdf) = c::ref_write_skip(&cdf, seg_skip, skip_txfm);
        assert_eq!(got, want, "write_skip bytes seg={seg_skip} s={skip_txfm}");
        assert_eq!(my_cdf, want_cdf, "write_skip cdf seg={seg_skip} s={skip_txfm}");
        let want_ret = if seg_skip { 1 } else { skip_txfm };
        assert_eq!(r, want_ret, "write_skip return seg={seg_skip} s={skip_txfm}");
    }
}

#[test]
fn write_delta_qindex_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_delta_qindex;
    let mut rng = Rng(0xd17a_c0de_a11a_0009);
    for _ in 0..200_000 {
        // valid 4-symbol CDF (DELTA_Q_PROBS+1): cdf[0..2] descending, cdf[3]=0, cdf[4]=count=0
        let mut vals = [0i32; 3];
        for v in &mut vals {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals.sort_unstable();
        vals.reverse();
        let mut cdf = [0u16; 5];
        let mut prev = 32768i32;
        for i in 0..3 {
            let v = vals[i].min(prev - 1).max((3 - i) as i32);
            cdf[i] = v as u16;
            prev = v;
        }
        cdf[3] = 0;
        cdf[4] = 0;
        // delta in [-255, 255] exercises smallval + exp-Golomb remainder + sign
        let delta_qindex = (rng.next() % 511) as i32 - 255;

        let mut my_cdf = cdf;
        let mut enc = OdEcEnc::new();
        write_delta_qindex(&mut enc, &mut my_cdf, delta_qindex);
        let got = enc.done().to_vec();

        let (want, want_cdf) = c::ref_write_delta_qindex(&cdf, delta_qindex);
        assert_eq!(got, want, "write_delta_qindex bytes dq={delta_qindex}");
        assert_eq!(my_cdf, want_cdf, "write_delta_qindex cdf dq={delta_qindex}");
    }
}

#[test]
fn write_delta_lflevel_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_delta_lflevel;
    let mut rng = Rng(0xd11f_c0de_a11a_0009);
    for _ in 0..200_000 {
        let mut vals = [0i32; 3];
        for v in &mut vals {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals.sort_unstable();
        vals.reverse();
        let mut cdf = [0u16; 5];
        let mut prev = 32768i32;
        for i in 0..3 {
            let v = vals[i].min(prev - 1).max((3 - i) as i32);
            cdf[i] = v as u16;
            prev = v;
        }
        cdf[3] = 0;
        cdf[4] = 0;
        let delta_lflevel = (rng.next() % 511) as i32 - 255;
        let mut my_cdf = cdf;
        let mut enc = OdEcEnc::new();
        write_delta_lflevel(&mut enc, &mut my_cdf, delta_lflevel);
        let got = enc.done().to_vec();
        let (want, want_cdf) = c::ref_write_delta_lflevel(&cdf, delta_lflevel);
        assert_eq!(got, want, "write_delta_lflevel bytes d={delta_lflevel}");
        assert_eq!(my_cdf, want_cdf, "write_delta_lflevel cdf d={delta_lflevel}");
    }
}

#[test]
fn write_cfl_alphas_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_cfl_alphas;
    let mut rng = Rng(0xcf1a_c0de_a11a_0009);
    // build a valid ns-symbol inverse-cumulative CDF into cdf[0..ns], count at cdf[ns]
    let mk = |rng: &mut Rng, ns: usize, cdf: &mut [u16]| {
        let mut vals = [0i32; 16];
        for v in vals.iter_mut().take(ns - 1) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 {
            let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32);
            cdf[i] = v as u16;
            prev = v;
        }
        cdf[ns - 1] = 0;
        cdf[ns] = 0;
    };
    for _ in 0..200_000 {
        let mut sign_cdf = [0u16; 9];
        mk(&mut rng, 8, &mut sign_cdf);
        let mut alpha_flat = [0u16; 102];
        let mut alpha = [[0u16; 17]; 6];
        for ctx in 0..6 {
            let mut c = [0u16; 17];
            mk(&mut rng, 16, &mut c);
            alpha[ctx] = c;
            alpha_flat[ctx * 17..ctx * 17 + 17].copy_from_slice(&c);
        }
        let idx = (rng.next() % 256) as i32;
        let joint_sign = (rng.next() % 8) as i32;

        let mut my_sign = sign_cdf;
        let mut my_alpha = alpha;
        let mut enc = OdEcEnc::new();
        write_cfl_alphas(&mut enc, &mut my_sign, &mut my_alpha, idx, joint_sign);
        let got = enc.done().to_vec();

        let (want, want_sign, want_alpha) = c::ref_write_cfl_alphas(&sign_cdf, &alpha_flat, idx, joint_sign);
        assert_eq!(got, want, "cfl bytes idx={idx} js={joint_sign}");
        assert_eq!(my_sign, want_sign, "cfl sign cdf idx={idx} js={joint_sign}");
        let my_alpha_flat: Vec<u16> = my_alpha.iter().flatten().copied().collect();
        assert_eq!(&my_alpha_flat[..], &want_alpha[..], "cfl alpha cdf idx={idx} js={joint_sign}");
    }
}

#[test]
fn get_y_mode_ctx_matches_c() {
    use aom_entropy::partition::get_y_mode_ctx;
    for ap in [false, true] {
        for lp in [false, true] {
            for am in 0..13 {
                for lm in 0..13 {
                    let above = if ap { Some(am) } else { None };
                    let left = if lp { Some(lm) } else { None };
                    assert_eq!(
                        get_y_mode_ctx(above, left),
                        c::ref_get_y_mode_ctx(ap, am, lp, lm),
                        "y_mode_ctx ap={ap} am={am} lp={lp} lm={lm}"
                    );
                }
            }
        }
    }
}

#[test]
fn write_intra_y_mode_kf_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_intra_y_mode_kf;
    let mut rng = Rng(0x14a5_c0de_a11a_0009);
    for _ in 0..200_000 {
        // valid 13-symbol CDF (14 entries incl count)
        let mut vals = [0i32; 13];
        for v in vals.iter_mut().take(12) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..12].sort_unstable();
        vals[..12].reverse();
        let mut cdf = [0u16; 14];
        let mut prev = 32768i32;
        for i in 0..12 {
            let v = vals[i].min(prev - 1).max((12 - i) as i32);
            cdf[i] = v as u16;
            prev = v;
        }
        cdf[12] = 0;
        cdf[13] = 0;
        let mode = (rng.next() % 13) as i32;
        let mut my_cdf = cdf;
        let mut enc = OdEcEnc::new();
        write_intra_y_mode_kf(&mut enc, &mut my_cdf, mode);
        let got = enc.done().to_vec();
        let (want, want_cdf) = c::ref_write_intra_y_mode_kf(&cdf, mode);
        assert_eq!(got, want, "intra_y_kf bytes mode={mode}");
        assert_eq!(my_cdf, want_cdf, "intra_y_kf cdf mode={mode}");
    }
}
