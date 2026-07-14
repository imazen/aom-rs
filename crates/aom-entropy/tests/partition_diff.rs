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

#[test]
fn size_group_lookup_matches_c() {
    use aom_entropy::partition::y_mode_size_group;
    for bsize in 0..22 {
        assert_eq!(y_mode_size_group(bsize as usize), c::ref_size_group_lookup(bsize) as usize, "size_group {bsize}");
    }
}

#[test]
fn write_intra_uv_mode_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_intra_uv_mode;
    let mut rng = Rng(0x00e5_c0de_a11a_0009);
    for _ in 0..200_000 {
        let cfl_allowed = rng.next().is_multiple_of(2);
        let ns = if cfl_allowed { 14 } else { 13 };
        // valid ns-symbol CDF in a 15-entry buffer (UV_INTRA_MODES+1)
        let mut vals = [0i32; 14];
        for v in vals.iter_mut().take(ns - 1) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut cdf = [0u16; 15];
        let mut prev = 32768i32;
        for i in 0..ns - 1 {
            let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32);
            cdf[i] = v as u16;
            prev = v;
        }
        cdf[ns - 1] = 0;
        cdf[ns] = 0;
        let uv_mode = (rng.next() % ns as u64) as i32;
        let mut my_cdf = cdf;
        let mut enc = OdEcEnc::new();
        write_intra_uv_mode(&mut enc, &mut my_cdf, uv_mode, cfl_allowed);
        let got = enc.done().to_vec();
        let (want, want_cdf) = c::ref_write_intra_uv_mode(&cdf, uv_mode, cfl_allowed);
        assert_eq!(got, want, "uv_mode bytes cfl={cfl_allowed} m={uv_mode}");
        assert_eq!(&my_cdf[..ns + 1], &want_cdf[..ns + 1], "uv_mode cdf cfl={cfl_allowed} m={uv_mode}");
    }
}

#[test]
fn write_inter_mode_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_inter_mode;
    let mut rng = Rng(0x14e6_c0de_a11a_0009);
    let bin_cdf = |rng: &mut Rng| -> [u16; 3] { [1 + (rng.next() % 32766) as u16, 0, 0] };
    for _ in 0..300_000 {
        let mut newmv = [[0u16; 3]; 6];
        let mut zeromv = [[0u16; 3]; 2];
        let mut refmv = [[0u16; 3]; 6];
        for c in &mut newmv {
            *c = bin_cdf(&mut rng);
        }
        for c in &mut zeromv {
            *c = bin_cdf(&mut rng);
        }
        for c in &mut refmv {
            *c = bin_cdf(&mut rng);
        }
        // valid mode_ctx: newmv_ctx in [0,5], zeromv_ctx in [0,1], refmv_ctx in [0,5]
        let newmv_ctx = (rng.next() % 6) as i32;
        let zeromv_ctx = (rng.next() % 2) as i32;
        let refmv_ctx = (rng.next() % 6) as i32;
        let mode_ctx = newmv_ctx | (zeromv_ctx << 3) | (refmv_ctx << 4);
        let mode = [13i32, 14, 15, 16][(rng.next() % 4) as usize]; // NEAREST/NEAR/GLOBAL/NEW MV

        let mut my_nm = newmv;
        let mut my_zm = zeromv;
        let mut my_rm = refmv;
        let mut enc = OdEcEnc::new();
        write_inter_mode(&mut enc, &mut my_nm, &mut my_zm, &mut my_rm, mode, mode_ctx);
        let got = enc.done().to_vec();

        let nf: [u16; 18] = std::array::from_fn(|i| newmv[i / 3][i % 3]);
        let zf: [u16; 6] = std::array::from_fn(|i| zeromv[i / 3][i % 3]);
        let rf: [u16; 18] = std::array::from_fn(|i| refmv[i / 3][i % 3]);
        let (want, onm, ozm, orm) = c::ref_write_inter_mode(&nf, &zf, &rf, mode, mode_ctx);
        assert_eq!(got, want, "inter_mode bytes mode={mode} ctx={mode_ctx}");
        let my_nf: [u16; 18] = std::array::from_fn(|i| my_nm[i / 3][i % 3]);
        let my_zf: [u16; 6] = std::array::from_fn(|i| my_zm[i / 3][i % 3]);
        let my_rf: [u16; 18] = std::array::from_fn(|i| my_rm[i / 3][i % 3]);
        assert_eq!(my_nf, onm, "inter_mode newmv cdf");
        assert_eq!(my_zf, ozm, "inter_mode zeromv cdf");
        assert_eq!(my_rf, orm, "inter_mode refmv cdf");
    }
}

#[test]
fn write_drl_idx_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_drl_idx;
    let mut rng = Rng(0xd21c_c0de_a11a_0009);
    let bin_cdf = |rng: &mut Rng| -> [u16; 3] { [1 + (rng.next() % 32766) as u16, 0, 0] };
    for _ in 0..300_000 {
        let mut drl = [[0u16; 3]; 3];
        for c in &mut drl {
            *c = bin_cdf(&mut rng);
        }
        // weights straddle REF_CAT_LEVEL=640
        let mut weight = [0u16; 4];
        for w in &mut weight {
            *w = (rng.next() % 1400) as u16;
        }
        // modes that write DRL: NEWMV=16, NEW_NEWMV=24, NEARMV=14, NEAR_NEARMV=18, NEAR_NEWMV=21, NEW_NEARMV=22; plus some that skip
        let mode = [16i32, 24, 14, 18, 21, 22, 15, 13][(rng.next() % 8) as usize];
        let ref_mv_idx = (rng.next() % 3) as i32;
        let ref_mv_count = (rng.next() % 5) as i32;

        let mut my_drl = drl;
        let mut enc = OdEcEnc::new();
        write_drl_idx(&mut enc, &mut my_drl, mode, ref_mv_idx, ref_mv_count, &weight);
        let got = enc.done().to_vec();

        let df: [u16; 9] = std::array::from_fn(|i| drl[i / 3][i % 3]);
        let (want, odf) = c::ref_write_drl_idx(&df, mode, ref_mv_idx, ref_mv_count, &weight);
        assert_eq!(got, want, "drl bytes mode={mode} idx={ref_mv_idx} cnt={ref_mv_count}");
        let my_df: [u16; 9] = std::array::from_fn(|i| my_drl[i / 3][i % 3]);
        assert_eq!(my_df, odf, "drl cdf mode={mode} idx={ref_mv_idx} cnt={ref_mv_count}");
    }
}

#[test]
fn mv_class_joint_math_matches_c() {
    use aom_entropy::partition::{get_mv_class, get_mv_joint};
    let mut rng = Rng(0x0acc_c0de_a11a_0009);
    // joint: rows/cols across zero + nonzero (int16 range)
    for _ in 0..200_000 {
        let row = (rng.next() % 65536) as i32 - 32768;
        let col = (rng.next() % 65536) as i32 - 32768;
        assert_eq!(get_mv_joint(row, col), c::ref_get_mv_joint(row, col), "mv_joint r={row} c={col}");
    }
    // class: z = |diff|-1 >= 0; valid MV range keeps class <= MV_CLASS_10 (z <= 16383)
    for z in 0..16384 {
        assert_eq!(get_mv_class(z), c::ref_get_mv_class(z), "mv_class z={z}");
    }
}

#[test]
fn encode_mv_component_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::encode_mv_component;
    let mut rng = Rng(0x0ace_c0de_a11a_0009);
    // fill cdf[off..off+ns] as a valid ns-symbol CDF (count at [off+ns-1..? no: ns-sym => ns entries + count)
    let mk = |rng: &mut Rng, ns: usize, out: &mut [u16]| {
        let mut vals = [0i32; 11];
        for v in vals.iter_mut().take(ns - 1) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 {
            let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32);
            out[i] = v as u16;
            prev = v;
        }
        out[ns - 1] = 0;
        out[ns] = 0;
    };
    for _ in 0..300_000 {
        let mut cdf = [0u16; 69];
        mk(&mut rng, 2, &mut cdf[0..3]); // sign
        mk(&mut rng, 11, &mut cdf[3..15]); // classes
        mk(&mut rng, 2, &mut cdf[15..18]); // class0
        for i in 0..10 {
            let off = 18 + i * 3;
            mk(&mut rng, 2, &mut cdf[off..off + 3]);
        }
        for i in 0..2 {
            let off = 48 + i * 5;
            mk(&mut rng, 4, &mut cdf[off..off + 5]);
        }
        mk(&mut rng, 4, &mut cdf[58..63]); // fp
        mk(&mut rng, 2, &mut cdf[63..66]); // class0_hp
        mk(&mut rng, 2, &mut cdf[66..69]); // hp

        // comp != 0, |comp| <= 16384 so class <= MV_CLASS_10
        let mag = 1 + (rng.next() % 16384) as i32;
        let comp = if rng.next().is_multiple_of(2) { mag } else { -mag };
        let precision = [-1i32, 0, 1][(rng.next() % 3) as usize];

        let mut my_cdf = cdf;
        let mut enc = OdEcEnc::new();
        encode_mv_component(&mut enc, &mut my_cdf, comp, precision);
        let got = enc.done().to_vec();
        let (want, want_cdf) = c::ref_encode_mv_component(&cdf, comp, precision);
        assert_eq!(got, want, "mv_comp bytes comp={comp} prec={precision}");
        assert_eq!(my_cdf, want_cdf, "mv_comp cdf comp={comp} prec={precision}");
    }
}

#[test]
fn encode_mv_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::encode_mv;
    let mut rng = Rng(0x0acf_c0de_a11a_0009);
    let mk = |rng: &mut Rng, ns: usize, out: &mut [u16]| {
        let mut vals = [0i32; 11];
        for v in vals.iter_mut().take(ns - 1) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 {
            let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32);
            out[i] = v as u16;
            prev = v;
        }
        out[ns - 1] = 0;
        out[ns] = 0;
    };
    let mk_comp = |rng: &mut Rng| -> [u16; 69] {
        let mut cdf = [0u16; 69];
        mk(rng, 2, &mut cdf[0..3]);
        mk(rng, 11, &mut cdf[3..15]);
        mk(rng, 2, &mut cdf[15..18]);
        for i in 0..10 {
            let off = 18 + i * 3;
            mk(rng, 2, &mut cdf[off..off + 3]);
        }
        for i in 0..2 {
            let off = 48 + i * 5;
            mk(rng, 4, &mut cdf[off..off + 5]);
        }
        mk(rng, 4, &mut cdf[58..63]);
        mk(rng, 2, &mut cdf[63..66]);
        mk(rng, 2, &mut cdf[66..69]);
        cdf
    };
    for _ in 0..300_000 {
        let mut joints = [0u16; 5];
        mk(&mut rng, 4, &mut joints);
        let comp0 = mk_comp(&mut rng);
        let comp1 = mk_comp(&mut rng);
        // diff not both-zero (assert j != ZERO); components in valid class range
        let dr = (rng.next() % 32769) as i32 - 16384;
        let dc = (rng.next() % 32769) as i32 - 16384;
        let (dr, dc) = if dr == 0 && dc == 0 { (1, 0) } else { (dr, dc) };
        let usehp = [-1i32, 0, 1][(rng.next() % 3) as usize];

        let mut my_j = joints;
        let mut my_c0 = comp0;
        let mut my_c1 = comp1;
        let mut enc = OdEcEnc::new();
        encode_mv(&mut enc, &mut my_j, &mut my_c0, &mut my_c1, dr, dc, usehp);
        let got = enc.done().to_vec();

        let (want, oj, o0, o1) = c::ref_encode_mv(&joints, &comp0, &comp1, dr, dc, usehp);
        assert_eq!(got, want, "encode_mv bytes dr={dr} dc={dc} hp={usehp}");
        assert_eq!(my_j, oj, "encode_mv joints cdf");
        assert_eq!(my_c0, o0, "encode_mv comp0 cdf");
        assert_eq!(my_c1, o1, "encode_mv comp1 cdf");
    }
}

#[test]
fn write_angle_delta_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_angle_delta;
    let mut rng = Rng(0xa06e_c0de_a11a_0009);
    for _ in 0..200_000 {
        // valid 7-symbol CDF in an 8-entry buffer
        let mut vals = [0i32; 7];
        for v in vals.iter_mut().take(6) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..6].sort_unstable();
        vals[..6].reverse();
        let mut cdf = [0u16; 8];
        let mut prev = 32768i32;
        for i in 0..6 {
            let v = vals[i].min(prev - 1).max((6 - i) as i32);
            cdf[i] = v as u16;
            prev = v;
        }
        cdf[6] = 0;
        cdf[7] = 0;
        let angle_delta = (rng.next() % 7) as i32 - 3; // [-3, 3]
        let mut my_cdf = cdf;
        let mut enc = OdEcEnc::new();
        write_angle_delta(&mut enc, &mut my_cdf, angle_delta);
        let got = enc.done().to_vec();
        let (want, want_cdf) = c::ref_write_angle_delta(&cdf, angle_delta);
        assert_eq!(got, want, "angle_delta bytes ad={angle_delta}");
        assert_eq!(my_cdf, want_cdf, "angle_delta cdf ad={angle_delta}");
    }
}

#[test]
fn tx_size_depth_cat_matches_c() {
    use aom_entropy::partition::{bsize_to_max_depth, bsize_to_tx_size_cat};
    for bsize in 0..22 {
        assert_eq!(bsize_to_max_depth(bsize as usize), c::ref_bsize_to_max_depth(bsize), "max_depth {bsize}");
        // cat only meaningful for bsize > BLOCK_4X4
        if bsize > 0 {
            assert_eq!(bsize_to_tx_size_cat(bsize as usize), c::ref_bsize_to_tx_size_cat(bsize), "cat {bsize}");
        }
    }
}

#[test]
fn write_selected_tx_size_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_selected_tx_size;
    let mut rng = Rng(0x7a51_c0de_a11a_0009);
    for _ in 0..200_000 {
        // MAX_TX_DEPTH=2 => max_depths in {1,2}, cdf has max_depths+1 symbols (<=3), 4-entry buf
        let max_depths = 1 + (rng.next() % 2) as usize;
        let ns = max_depths + 1;
        let mut vals = [0i32; 3];
        for v in vals.iter_mut().take(ns - 1) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut cdf = [0u16; 4];
        let mut prev = 32768i32;
        for i in 0..ns - 1 {
            let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32);
            cdf[i] = v as u16;
            prev = v;
        }
        cdf[ns - 1] = 0;
        cdf[ns] = 0;
        let bsize = (rng.next() % 22) as i32;
        let depth = (rng.next() % ns as u64) as i32;
        let mut my_cdf = cdf;
        let mut enc = OdEcEnc::new();
        write_selected_tx_size(&mut enc, &mut my_cdf, bsize as usize, depth, max_depths);
        let got = enc.done().to_vec();
        let (want, want_cdf) = c::ref_write_selected_tx_size(&cdf, bsize, depth, max_depths as i32);
        assert_eq!(got, want, "tx_size bytes bsize={bsize} depth={depth} md={max_depths}");
        assert_eq!(&my_cdf[..ns + 1], &want_cdf[..ns + 1], "tx_size cdf bsize={bsize} depth={depth}");
    }
}

#[test]
fn write_filter_intra_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_filter_intra_mode_info;
    let mut rng = Rng(0xf114_c0de_a11a_0009);
    let mk = |rng: &mut Rng, ns: usize, out: &mut [u16]| {
        let mut vals = [0i32; 5];
        for v in vals.iter_mut().take(ns - 1) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 {
            let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32);
            out[i] = v as u16;
            prev = v;
        }
        out[ns - 1] = 0;
        out[ns] = 0;
    };
    for _ in 0..200_000 {
        let mut use_cdf = [0u16; 3];
        mk(&mut rng, 2, &mut use_cdf);
        let mut mode_cdf = [0u16; 6];
        mk(&mut rng, 5, &mut mode_cdf);
        let allowed = rng.next().is_multiple_of(2);
        let use_fi = (rng.next() % 2) as i32;
        let mode = (rng.next() % 5) as i32;

        let mut mu = use_cdf;
        let mut mm = mode_cdf;
        let mut enc = OdEcEnc::new();
        write_filter_intra_mode_info(&mut enc, &mut mu, &mut mm, allowed, use_fi, mode);
        let got = enc.done().to_vec();
        let (want, ou, om) = c::ref_write_filter_intra(&use_cdf, &mode_cdf, allowed, use_fi, mode);
        assert_eq!(got, want, "filter_intra bytes allowed={allowed} use={use_fi} mode={mode}");
        assert_eq!(mu, ou, "filter_intra use cdf");
        assert_eq!(mm, om, "filter_intra mode cdf");
    }
}

#[test]
fn write_inter_compound_mode_and_is_inter_match_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{write_inter_compound_mode, write_is_inter};
    let mut rng = Rng(0x1c0e_c0de_a11a_0009);
    let mk = |rng: &mut Rng, ns: usize, out: &mut [u16]| {
        let mut vals = [0i32; 8];
        for v in vals.iter_mut().take(ns - 1) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 {
            let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32);
            out[i] = v as u16;
            prev = v;
        }
        out[ns - 1] = 0;
        out[ns] = 0;
    };
    for _ in 0..200_000 {
        // inter_compound_mode: 8-symbol, mode in [17,24]
        let mut cdf = [0u16; 9];
        mk(&mut rng, 8, &mut cdf);
        let mode = 17 + (rng.next() % 8) as i32;
        let mut mc = cdf;
        let mut enc = OdEcEnc::new();
        write_inter_compound_mode(&mut enc, &mut mc, mode);
        let got = enc.done().to_vec();
        let (want, oc) = c::ref_write_inter_compound_mode(&cdf, mode);
        assert_eq!(got, want, "inter_compound bytes mode={mode}");
        assert_eq!(mc, oc, "inter_compound cdf mode={mode}");

        // is_inter: 2-symbol with seg gates
        let mut icdf = [0u16; 3];
        mk(&mut rng, 2, &mut icdf);
        let seg_ref = rng.next().is_multiple_of(3);
        let seg_gmv = rng.next().is_multiple_of(3);
        let is_inter = (rng.next() % 2) as i32;
        let mut mi = icdf;
        let mut enc = OdEcEnc::new();
        write_is_inter(&mut enc, &mut mi, seg_ref, seg_gmv, is_inter);
        let got = enc.done().to_vec();
        let (want, oi) = c::ref_write_is_inter(&icdf, seg_ref, seg_gmv, is_inter);
        assert_eq!(got, want, "is_inter bytes sr={seg_ref} sg={seg_gmv} ii={is_inter}");
        assert_eq!(mi, oi, "is_inter cdf");
    }
}

#[test]
fn write_motion_mode_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_motion_mode;
    let mut rng = Rng(0x3070_c0de_a11a_0009);
    let mk = |rng: &mut Rng, ns: usize, out: &mut [u16]| {
        let mut vals = [0i32; 3];
        for v in vals.iter_mut().take(ns - 1) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 {
            let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32);
            out[i] = v as u16;
            prev = v;
        }
        out[ns - 1] = 0;
        out[ns] = 0;
    };
    for _ in 0..200_000 {
        let mut obmc = [0u16; 3];
        mk(&mut rng, 2, &mut obmc);
        let mut mm_cdf = [0u16; 4];
        mk(&mut rng, 3, &mut mm_cdf);
        let last_allowed = (rng.next() % 3) as i32; // 0/1/2
        // motion_mode <= last_allowed
        let mm = (rng.next() % (last_allowed as u64 + 1)) as i32;
        let mut mo = obmc;
        let mut mmc = mm_cdf;
        let mut enc = OdEcEnc::new();
        write_motion_mode(&mut enc, &mut mo, &mut mmc, last_allowed, mm);
        let got = enc.done().to_vec();
        let (want, oo, om) = c::ref_write_motion_mode(&obmc, &mm_cdf, last_allowed, mm);
        assert_eq!(got, want, "motion_mode bytes la={last_allowed} mm={mm}");
        assert_eq!(mo, oo, "motion_mode obmc cdf");
        assert_eq!(mmc, om, "motion_mode mm cdf");
    }
}

#[test]
fn write_mb_interp_filter_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_mb_interp_filter;
    let mut rng = Rng(0x14f0_c0de_a11a_0009);
    let mk = |rng: &mut Rng, out: &mut [u16; 4]| {
        // 3-symbol CDF
        let mut vals = [1 + (rng.next() % 32766) as i32, 1 + (rng.next() % 32766) as i32];
        vals.sort_unstable();
        vals.reverse();
        out[0] = vals[0].max(2) as u16;
        out[1] = vals[1].min(out[0] as i32 - 1).max(1) as u16;
        out[2] = 0;
        out[3] = 0;
    };
    for _ in 0..200_000 {
        let mut cdf0 = [0u16; 4];
        let mut cdf1 = [0u16; 4];
        mk(&mut rng, &mut cdf0);
        mk(&mut rng, &mut cdf1);
        let interp_needed = rng.next().is_multiple_of(2);
        let is_switchable = rng.next().is_multiple_of(2);
        let enable_dual = rng.next().is_multiple_of(2);
        let f0 = (rng.next() % 3) as i32;
        let f1 = (rng.next() % 3) as i32;
        let mut m0 = cdf0;
        let mut m1 = cdf1;
        let mut enc = OdEcEnc::new();
        write_mb_interp_filter(&mut enc, &mut m0, &mut m1, interp_needed, is_switchable, enable_dual, f0, f1);
        let got = enc.done().to_vec();
        let (want, o0, o1) = c::ref_write_mb_interp_filter(&cdf0, &cdf1, interp_needed, is_switchable, enable_dual, f0, f1);
        assert_eq!(got, want, "interp bytes n={interp_needed} sw={is_switchable} dual={enable_dual}");
        assert_eq!(m0, o0, "interp cdf0");
        assert_eq!(m1, o1, "interp cdf1");
    }
}

#[test]
fn get_intra_inter_context_matches_c() {
    use aom_entropy::partition::get_intra_inter_context;
    for ha in [false, true] {
        for ai in [false, true] {
            for hl in [false, true] {
                for li in [false, true] {
                    assert_eq!(
                        get_intra_inter_context(ha, ai, hl, li),
                        c::ref_get_intra_inter_context(ha, ai, hl, li),
                        "intra_inter_ctx ha={ha} ai={ai} hl={hl} li={li}"
                    );
                }
            }
        }
    }
}

#[test]
fn get_reference_mode_context_matches_c() {
    use aom_entropy::partition::get_reference_mode_context;
    let mut rng = Rng(0x2ef0_c0de_a11a_0009);
    for _ in 0..300_000 {
        let ha = rng.next().is_multiple_of(2);
        let hl = rng.next().is_multiple_of(2);
        // ref_frame[0] in 0..8, ref_frame[1] in -1..8 (NONE=-1 or a ref)
        let a_r0 = (rng.next() % 8) as i32;
        let a_r1 = (rng.next() % 9) as i32 - 1;
        let l_r0 = (rng.next() % 8) as i32;
        let l_r1 = (rng.next() % 9) as i32 - 1;
        let a_ibc = rng.next().is_multiple_of(3);
        let l_ibc = rng.next().is_multiple_of(3);
        assert_eq!(
            get_reference_mode_context(ha, a_r0, a_r1, a_ibc, hl, l_r0, l_r1, l_ibc),
            c::ref_get_reference_mode_context(ha, a_r0, a_r1, a_ibc, hl, l_r0, l_r1, l_ibc),
            "ref_mode_ctx ha={ha} a=({a_r0},{a_r1},{a_ibc}) hl={hl} l=({l_r0},{l_r1},{l_ibc})"
        );
    }
}

#[test]
fn get_comp_reference_type_context_matches_c() {
    use aom_entropy::partition::get_comp_reference_type_context;
    let mut rng = Rng(0xc0e0_c0de_a11a_0009);
    for _ in 0..400_000 {
        let ha = rng.next().is_multiple_of(2);
        let hl = rng.next().is_multiple_of(2);
        let a_r0 = (rng.next() % 8) as i32;
        let a_r1 = (rng.next() % 9) as i32 - 1;
        let l_r0 = (rng.next() % 8) as i32;
        let l_r1 = (rng.next() % 9) as i32 - 1;
        let a_ibc = rng.next().is_multiple_of(4);
        let l_ibc = rng.next().is_multiple_of(4);
        assert_eq!(
            get_comp_reference_type_context(ha, a_r0, a_r1, a_ibc, hl, l_r0, l_r1, l_ibc),
            c::ref_get_comp_reference_type_context(ha, a_r0, a_r1, a_ibc, hl, l_r0, l_r1, l_ibc),
            "comp_ref_type ha={ha} a=({a_r0},{a_r1},{a_ibc}) hl={hl} l=({l_r0},{l_r1},{l_ibc})"
        );
    }
}

#[test]
fn single_ref_p1_context_matches_c() {
    use aom_entropy::partition::single_ref_p1_context;
    let mut rng = Rng(0x51e1_c0de_a11a_0009);
    for _ in 0..200_000 {
        let mut counts = [0u8; 8];
        for c in &mut counts {
            *c = (rng.next() % 3) as u8; // neighbor ref counts are small (0..2)
        }
        assert_eq!(single_ref_p1_context(&counts), c::ref_single_ref_p1_context(&counts), "single_ref_p1 {counts:?}");
    }
}

#[test]
fn single_ref_count_contexts_match_c() {
    use aom_entropy::partition::*;
    let mut rng = Rng(0x51e2_c0de_a11a_0009);
    for _ in 0..200_000 {
        let mut rc = [0u8; 8];
        for c in &mut rc {
            *c = (rng.next() % 3) as u8;
        }
        assert_eq!(pred_ctx_brfarf2_or_arf(&rc), c::ref_single_ref_p2_context(&rc), "p2 {rc:?}");
        assert_eq!(pred_ctx_ll2_or_l3gld(&rc), c::ref_single_ref_p3_context(&rc), "p3 {rc:?}");
        assert_eq!(pred_ctx_last_or_last2(&rc), c::ref_single_ref_p4_context(&rc), "p4 {rc:?}");
        assert_eq!(pred_ctx_last3_or_gld(&rc), c::ref_single_ref_p5_context(&rc), "p5 {rc:?}");
        assert_eq!(pred_ctx_brf_or_arf2(&rc), c::ref_single_ref_p6_context(&rc), "p6 {rc:?}");
    }
}

#[test]
fn uni_comp_ref_contexts_match_c() {
    use aom_entropy::partition::{pred_ctx_last2_or_l3gld, pred_ctx_last3_or_gld, single_ref_p1_context};
    let mut rng = Rng(0x0c17_c0de_a11a_0009);
    for _ in 0..200_000 {
        let mut rc = [0u8; 8];
        for c in &mut rc {
            *c = (rng.next() % 3) as u8;
        }
        // identities: uni_comp_ref_p == single_ref_p1 (fwd/bwd); uni_comp_ref_p2 == last3_or_gld
        assert_eq!(single_ref_p1_context(&rc), c::ref_uni_comp_ref_p_context(&rc), "ucr_p {rc:?}");
        assert_eq!(pred_ctx_last2_or_l3gld(&rc), c::ref_uni_comp_ref_p1_context(&rc), "ucr_p1 {rc:?}");
        assert_eq!(pred_ctx_last3_or_gld(&rc), c::ref_uni_comp_ref_p2_context(&rc), "ucr_p2 {rc:?}");
    }
}

#[test]
fn write_ref_frames_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_ref_frames;
    let mut rng = Rng(0x2ef1_c0de_a11a_0009);
    for _ in 0..400_000 {
        // 16 valid 2-symbol CDFs
        let mut cdfs = [[0u16; 3]; 16];
        for c in &mut cdfs {
            c[0] = 1 + (rng.next() % 32766) as u16;
            c[1] = 0;
            c[2] = 0;
        }
        let seg_ref = rng.next().is_multiple_of(5);
        let seg_skipgmv = rng.next().is_multiple_of(5);
        let rmode_select = rng.next().is_multiple_of(2);
        let comp_allowed = rng.next().is_multiple_of(2);
        let is_compound = rng.next().is_multiple_of(2);
        let comp_ref_type = (rng.next() % 2) as i32;
        let ref0 = (rng.next() % 8) as i32; // 0..7
        let ref1 = (rng.next() % 8) as i32;

        let mut my = cdfs;
        let mut enc = OdEcEnc::new();
        write_ref_frames(&mut enc, &mut my, seg_ref, seg_skipgmv, rmode_select, comp_allowed, is_compound, comp_ref_type, ref0, ref1);
        let got = enc.done().to_vec();

        let flat: [u16; 48] = std::array::from_fn(|i| cdfs[i / 3][i % 3]);
        let (want, oc) = c::ref_write_ref_frames(&flat, seg_ref, seg_skipgmv, rmode_select, comp_allowed, is_compound, comp_ref_type, ref0, ref1);
        assert_eq!(got, want, "ref_frames bytes comp={is_compound} crt={comp_ref_type} r=({ref0},{ref1})");
        let my_flat: [u16; 48] = std::array::from_fn(|i| my[i / 3][i % 3]);
        assert_eq!(my_flat, oc, "ref_frames cdfs comp={is_compound} r=({ref0},{ref1})");
    }
}

#[test]
fn neg_interleave_and_segment_id_match_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{neg_interleave, write_segment_id};
    let mut rng = Rng(0x5e61_c0de_a11a_0009);
    // neg_interleave: x < max, over the segment range
    for _ in 0..300_000 {
        let max = 1 + (rng.next() % 8) as i32; // [1,8]
        let x = (rng.next() % max as u64) as i32;
        let r = (rng.next() % max as u64) as i32;
        assert_eq!(neg_interleave(x, r, max), c::ref_neg_interleave(x, r, max), "neg_interleave x={x} r={r} max={max}");
    }
    // write_segment_id: 8-symbol CDF
    let mk = |rng: &mut Rng, out: &mut [u16; 9]| {
        let mut vals = [0i32; 8];
        for v in vals.iter_mut().take(7) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..7].sort_unstable();
        vals[..7].reverse();
        let mut prev = 32768i32;
        for i in 0..7 {
            let v = vals[i].min(prev - 1).max((7 - i) as i32);
            out[i] = v as u16;
            prev = v;
        }
        out[7] = 0;
        out[8] = 0;
    };
    for _ in 0..200_000 {
        let mut cdf = [0u16; 9];
        mk(&mut rng, &mut cdf);
        let seg_enabled = rng.next().is_multiple_of(2);
        let update_map = rng.next().is_multiple_of(2);
        let skip_txfm = rng.next().is_multiple_of(3);
        // last_active_segid+1 = max in [1,8]; segment_id/pred < max
        let last = (rng.next() % 8) as i32; // 0..7 -> max 1..8
        let max = last + 1;
        let segment_id = (rng.next() % max as u64) as i32;
        let pred = (rng.next() % max as u64) as i32;
        let mut mc = cdf;
        let mut enc = OdEcEnc::new();
        write_segment_id(&mut enc, &mut mc, seg_enabled, update_map, skip_txfm, segment_id, pred, last);
        let got = enc.done().to_vec();
        let (want, oc) = c::ref_write_segment_id(&cdf, seg_enabled, update_map, skip_txfm, segment_id, pred, last);
        assert_eq!(got, want, "seg_id bytes en={seg_enabled} um={update_map} skip={skip_txfm} s={segment_id} p={pred} last={last}");
        assert_eq!(mc, oc, "seg_id cdf");
    }
}

#[test]
fn write_intrabc_info_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_intrabc_info;
    let mut rng = Rng(0x1bcc_c0de_a11a_0009);
    let mk = |rng: &mut Rng, ns: usize, out: &mut [u16]| {
        let mut vals = [0i32; 11];
        for v in vals.iter_mut().take(ns - 1) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 {
            let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32);
            out[i] = v as u16;
            prev = v;
        }
        out[ns - 1] = 0;
        out[ns] = 0;
    };
    let mk_comp = |rng: &mut Rng| -> [u16; 69] {
        let mut c = [0u16; 69];
        mk(rng, 2, &mut c[0..3]);
        mk(rng, 11, &mut c[3..15]);
        mk(rng, 2, &mut c[15..18]);
        for i in 0..10 { let o = 18 + i * 3; mk(rng, 2, &mut c[o..o + 3]); }
        for i in 0..2 { let o = 48 + i * 5; mk(rng, 4, &mut c[o..o + 5]); }
        mk(rng, 4, &mut c[58..63]); mk(rng, 2, &mut c[63..66]); mk(rng, 2, &mut c[66..69]);
        c
    };
    for _ in 0..200_000 {
        let mut ibc = [0u16; 3];
        mk(&mut rng, 2, &mut ibc);
        let mut joints = [0u16; 5];
        mk(&mut rng, 4, &mut joints);
        let comp0 = mk_comp(&mut rng);
        let comp1 = mk_comp(&mut rng);
        let use_intrabc = (rng.next() % 2) as i32;
        // DV diffs (integer, multiples of 8) in valid class range
        let dr = ((rng.next() % 4097) as i32 - 2048) * 8;
        let dc = ((rng.next() % 4097) as i32 - 2048) * 8;

        let mut mib = ibc; let mut mj = joints; let mut m0 = comp0; let mut m1 = comp1;
        let mut enc = OdEcEnc::new();
        write_intrabc_info(&mut enc, &mut mib, &mut mj, &mut m0, &mut m1, use_intrabc, dr, dc);
        let got = enc.done().to_vec();
        let (want, oib, oj, o0, o1) = c::ref_write_intrabc_info(&ibc, &joints, &comp0, &comp1, use_intrabc, dr, dc);
        assert_eq!(got, want, "intrabc bytes use={use_intrabc} d=({dr},{dc})");
        assert_eq!(mib, oib, "intrabc ibc cdf");
        assert_eq!(mj, oj, "intrabc joints cdf");
        assert_eq!(m0, o0, "intrabc comp0");
        assert_eq!(m1, o1, "intrabc comp1");
    }
}

#[test]
fn write_skip_mode_and_context_match_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{skip_mode_context, write_skip_mode};
    let mut rng = Rng(0x5309_c0de_a11a_0009);
    // context
    for ha in [false, true] {
        for a in 0..2 {
            for hl in [false, true] {
                for l in 0..2 {
                    let above = if ha { a } else { 0 };
                    let left = if hl { l } else { 0 };
                    assert_eq!(skip_mode_context(above, left), c::ref_get_skip_mode_context(ha, a, hl, l), "skip_mode_ctx");
                }
            }
        }
    }
    // write
    for _ in 0..200_000 {
        let c0 = 1 + (rng.next() % 32766) as u16;
        let cdf = [c0, 0u16, 0u16];
        let frame_flag = rng.next().is_multiple_of(2);
        let seg_skip = rng.next().is_multiple_of(3);
        let comp_allowed = rng.next().is_multiple_of(2);
        let seg_ref_gmv = rng.next().is_multiple_of(3);
        let sm = (rng.next() % 2) as i32;
        let mut mc = cdf;
        let mut enc = OdEcEnc::new();
        write_skip_mode(&mut enc, &mut mc, frame_flag, seg_skip, comp_allowed, seg_ref_gmv, sm);
        let got = enc.done().to_vec();
        let (want, oc) = c::ref_write_skip_mode(&cdf, frame_flag, seg_skip, comp_allowed, seg_ref_gmv, sm);
        assert_eq!(got, want, "skip_mode bytes ff={frame_flag} ss={seg_skip} ca={comp_allowed} srg={seg_ref_gmv} sm={sm}");
        assert_eq!(mc, oc, "skip_mode cdf");
    }
}

#[test]
fn txfm_partition_context_matches_c() {
    use aom_entropy::partition::txfm_partition_context;
    // Neighbour txfm-context values are stored tx widths/heights (pixels) or 0.
    let nbr_vals: [u8; 7] = [0, 4, 8, 16, 32, 64, 128];
    for tx_size in 0..19usize {
        for bsize in 0..22usize {
            // C asserts max_tx_size >= TX_8X8 when tx_size > TX_4X4; only BLOCK_4X4
            // (index 0, max dim 4) violates that. Skip that single illegal combo.
            if tx_size > 0 && bsize == 0 {
                continue;
            }
            for &above in &nbr_vals {
                for &left in &nbr_vals {
                    let got = txfm_partition_context(above, left, bsize, tx_size) as i32;
                    let want = c::ref_txfm_partition_context(above, left, bsize as i32, tx_size as i32);
                    assert_eq!(got, want, "ctx above={above} left={left} bsize={bsize} tx={tx_size}");
                }
            }
        }
    }
}

#[test]
fn txfm_partition_update_matches_c() {
    use aom_entropy::partition::txfm_partition_update;
    // MAX_MIB_SIZE (128/4) = 32 context slots per direction.
    for tx_size in 0..19usize {
        for txb_size in 0..19usize {
            // Distinct sentinel fills so an over/under-write is caught.
            let mut a_rs = [0xAAu8; 32];
            let mut l_rs = [0x55u8; 32];
            let mut a_c = [0xAAu8; 32];
            let mut l_c = [0x55u8; 32];
            txfm_partition_update(&mut a_rs, &mut l_rs, tx_size, txb_size);
            c::ref_txfm_partition_update(&mut a_c, &mut l_c, tx_size as i32, txb_size as i32);
            assert_eq!(a_rs, a_c, "above tx={tx_size} txb={txb_size}");
            assert_eq!(l_rs, l_c, "left tx={tx_size} txb={txb_size}");
        }
    }
}

#[test]
fn write_tx_size_vartx_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_tx_size_vartx;
    // max_txsize_rect_lookup[BLOCK_SIZES_ALL] — the block's top var-tx size.
    const MAX_TX_RECT: [usize; 22] =
        [0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18];
    // Inter block sizes >= 8x8 (var-tx applies): squares, rects, and 128s.
    let bsizes: [usize; 13] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 19];
    let nbr: [u8; 6] = [0, 4, 8, 16, 32, 64];
    let mut rng = Rng(0x7a12_b0de_5a1e_0001);
    for &bsize in &bsizes {
        let top = MAX_TX_RECT[bsize];
        for _ in 0..6000 {
            // Any inter_tx_size values are valid — the recursion always terminates.
            let mut its = [0u8; 16];
            for v in its.iter_mut() {
                *v = (rng.next() % 19) as u8;
            }
            let its_usize: [usize; 16] = core::array::from_fn(|i| its[i] as usize);
            let mut above = [0u8; 32];
            let mut left = [0u8; 32];
            for i in 0..32 {
                above[i] = nbr[(rng.next() % 6) as usize];
                left[i] = nbr[(rng.next() % 6) as usize];
            }
            // Frame-edge clip in whole tx units (each -32 in 1/8-pel = -1 tx unit).
            let re = -((rng.next() % 4) as i32) * 32;
            let be = -((rng.next() % 4) as i32) * 32;
            // Random starting txfm_partition_cdf (21 ctxs, 2-symbol [prob,0,count]).
            let mut cdf = [[0u16; 3]; 21];
            let mut cflat = [0u16; 63];
            for c in 0..21 {
                let p = 1 + (rng.next() % 32766) as u16;
                cdf[c] = [p, 0, 0];
                cflat[c * 3] = p;
            }
            let mut enc = OdEcEnc::new();
            let mut a_rs = above;
            let mut l_rs = left;
            let mut cdf_rs = cdf;
            write_tx_size_vartx(
                &mut enc, &mut cdf_rs, bsize, &its_usize, re, be, &mut a_rs, &mut l_rs, top, 0, 0, 0,
            );
            let got = enc.done().to_vec();
            let (want, ao, lo, co) =
                c::ref_write_tx_size_vartx(bsize as i32, top as i32, &its, re, be, &above, &left, &cflat);
            assert_eq!(got, want, "bytes bsize={bsize} top={top} re={re} be={be} its={its:?}");
            assert_eq!(a_rs, ao, "above bsize={bsize} its={its:?}");
            assert_eq!(l_rs, lo, "left bsize={bsize} its={its:?}");
            let co_nested: [[u16; 3]; 21] = core::array::from_fn(|c| [co[c * 3], co[c * 3 + 1], co[c * 3 + 2]]);
            assert_eq!(cdf_rs, co_nested, "cdf bsize={bsize} its={its:?}");
        }
    }
}

#[test]
fn palette_contexts_and_flags_match_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{palette_bsize_ctx, palette_mode_ctx, write_palette_mode_info_flags};
    // bsize_ctx over all block sizes.
    for bsize in 0..22usize {
        assert_eq!(palette_bsize_ctx(bsize), c::ref_get_palette_bsize_ctx(bsize as i32), "bsize_ctx {bsize}");
    }
    // mode_ctx over neighbour presence x palette sizes.
    for ha in [false, true] {
        for a in [0, 1, 5, 8] {
            for hl in [false, true] {
                for l in [0, 1, 5, 8] {
                    assert_eq!(
                        palette_mode_ctx(ha, a, hl, l),
                        c::ref_get_palette_mode_ctx(ha, a, hl, l),
                        "mode_ctx ha={ha} a={a} hl={hl} l={l}"
                    );
                }
            }
        }
    }
    // flag/size symbols.
    let mut rng = Rng(0x9a1e_77e0_c0de_0002);
    for _ in 0..200_000 {
        let mode_dc = rng.next().is_multiple_of(2);
        let uv_dc = rng.next().is_multiple_of(2);
        // n in {0} (no palette) or 2..=8 (PALETTE_MIN_SIZE..PALETTE_MAX_SIZE).
        let n_y = if rng.next().is_multiple_of(3) { 0 } else { 2 + (rng.next() % 7) as i32 };
        let n_uv = if rng.next().is_multiple_of(3) { 0 } else { 2 + (rng.next() % 7) as i32 };
        let p2 = 1 + (rng.next() % 32766) as u16;
        let ym = [p2, 0, 0];
        let um = [1 + (rng.next() % 32766) as u16, 0, 0];
        // 7-symbol size CDFs (8 entries: 7 cumulative + count), strictly decreasing.
        let mk7 = |rng: &mut Rng| {
            let mut c = [0u16; 8];
            let mut prev = 32768i32;
            for e in c.iter_mut().take(7) {
                let span = (prev - 7).max(1);
                let v = prev - 1 - (rng.next() % span.max(1) as u64) as i32;
                *e = v.max(1) as u16;
                prev = v.max(1);
            }
            c[6] = 0; // last cumulative is AOM_ICDF top == 0
            c
        };
        let ys = mk7(&mut rng);
        let us = mk7(&mut rng);
        let mut enc = OdEcEnc::new();
        let (mut rym, mut rys, mut rum, mut rus) = (ym, ys, um, us);
        write_palette_mode_info_flags(&mut enc, mode_dc, n_y, &mut rym, &mut rys, uv_dc, n_uv, &mut rum, &mut rus);
        let got = enc.done().to_vec();
        let (want, oym, oys, oum, ous) =
            c::ref_write_palette_flags_sizes(mode_dc, n_y, &ym, &ys, uv_dc, n_uv, &um, &us);
        assert_eq!(got, want, "bytes mode_dc={mode_dc} n_y={n_y} uv_dc={uv_dc} n_uv={n_uv}");
        assert_eq!(rym, oym, "y_mode_cdf");
        assert_eq!(rys, oys, "y_size_cdf");
        assert_eq!(rum, oum, "uv_mode_cdf");
        assert_eq!(rus, ous, "uv_size_cdf");
    }
}

#[test]
fn delta_encode_palette_colors_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::delta_encode_palette_colors;
    let mut rng = Rng(0xde17_a000_c0de_0003);
    for &bit_depth in &[8i32, 10, 12] {
        let maxv = 1i32 << bit_depth;
        for min_val in [0i32, 1] {
            for num in 1..=8usize {
                let step_max = (maxv / (num as i32 + 1)).max(1);
                for _ in 0..40_000 {
                    // Build an ascending colour list with deltas >= min_val, all < 2^bd.
                    let mut colors = vec![0i32; num];
                    let mut cur = (rng.next() % step_max as u64) as i32;
                    for c in colors.iter_mut() {
                        *c = cur;
                        cur += min_val + (rng.next() % step_max as u64) as i32;
                    }
                    let mut enc = OdEcEnc::new();
                    delta_encode_palette_colors(&mut enc, &colors, bit_depth, min_val);
                    let got = enc.done().to_vec();
                    let want = c::ref_delta_encode_palette_colors(&colors, bit_depth, min_val);
                    assert_eq!(got, want, "bd={bit_depth} min_val={min_val} num={num} colors={colors:?}");
                }
            }
        }
    }
}

#[test]
fn write_palette_colors_v_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_palette_colors_v;
    let mut rng = Rng(0x0c01_0f5f_c0de_0004);
    for &bit_depth in &[8i32, 10, 12] {
        let maxv = 1u64 << bit_depth;
        for n in 1..=8usize {
            for _ in 0..60_000 {
                // V colours are unsorted; any values in [0, 2^bd).
                let colors: Vec<u16> = (0..n).map(|_| (rng.next() % maxv) as u16).collect();
                let mut enc = OdEcEnc::new();
                write_palette_colors_v(&mut enc, &colors, bit_depth);
                let got = enc.done().to_vec();
                let want = c::ref_write_palette_colors_v(&colors, bit_depth);
                assert_eq!(got, want, "bd={bit_depth} n={n} colors={colors:?}");
            }
        }
    }
}

#[test]
fn get_palette_cache_matches_c() {
    use aom_entropy::partition::get_palette_cache;
    let mut rng = Rng(0xca6e_0000_c0de_0005);
    // Build a full 3*8 palette_colors array with `n` sorted colours at plane offset.
    let mk = |rng: &mut Rng, plane: usize, n: usize, bd: i32| -> ([u16; 24], i32) {
        let maxv = 1u64 << bd;
        let mut v: Vec<u16> = (0..n).map(|_| (rng.next() % maxv) as u16).collect();
        v.sort_unstable();
        let mut arr = [0u16; 24];
        for (k, &c) in v.iter().enumerate() {
            arr[plane * 8 + k] = c;
        }
        (arr, n as i32)
    };
    for plane in 0..2usize {
        for _ in 0..120_000 {
            let bd = [8i32, 10, 12][(rng.next() % 3) as usize];
            let ha = rng.next().is_multiple_of(2);
            let hl = rng.next().is_multiple_of(2);
            let an = (rng.next() % 9) as usize; // 0..8
            let ln = (rng.next() % 9) as usize;
            let (a_colors, a_n) = mk(&mut rng, plane, an, bd);
            let (l_colors, l_n) = mk(&mut rng, plane, ln, bd);
            // row = -mb_to_top_edge>>3; sweep boundary + interior.
            let mte = -((rng.next() % 20) as i32) * 32;
            let mut cache = [0u16; 16];
            let n = get_palette_cache(&mut cache, plane, mte, ha, &a_colors, a_n, hl, &l_colors, l_n);
            // C facade takes both sizes; only plane's is used.
            let (a_s0, a_s1) = if plane == 0 { (a_n, 0) } else { (0, a_n) };
            let (l_s0, l_s1) = if plane == 0 { (l_n, 0) } else { (0, l_n) };
            let (want, wn) = c::ref_get_palette_cache(plane as i32, mte, ha, &a_colors, a_s0, a_s1, hl, &l_colors, l_s0, l_s1);
            assert_eq!(n as i32, wn, "n plane={plane} an={an} ln={ln} mte={mte}");
            assert_eq!(&cache[..n], &want[..], "cache plane={plane} an={an} ln={ln} mte={mte}");
        }
    }
}

#[test]
fn index_color_cache_matches_c() {
    use aom_entropy::partition::index_color_cache;
    let mut rng = Rng(0x1de0_0000_c0de_0006);
    for _ in 0..300_000 {
        let bd = [8i32, 10, 12][(rng.next() % 3) as usize];
        let maxv = 1u64 << bd;
        let n_cache = (rng.next() % 17) as usize; // 0..16
        let n_colors = 1 + (rng.next() % 8) as usize; // 1..8
        // cache is sorted+deduped in practice; make it sorted (dups allowed — the fn
        // just does membership tests so any cache is a valid differential input).
        let mut cache: Vec<u16> = (0..n_cache).map(|_| (rng.next() % maxv) as u16).collect();
        cache.sort_unstable();
        // colours: some drawn from the cache to force matches, some random.
        let colors: Vec<u16> = (0..n_colors)
            .map(|_| {
                if n_cache > 0 && rng.next().is_multiple_of(2) {
                    cache[(rng.next() as usize) % n_cache]
                } else {
                    (rng.next() % maxv) as u16
                }
            })
            .collect();
        let (found, out, n_out) = index_color_cache(&cache, &colors);
        let (wfound, wout, wn) = c::ref_index_color_cache(&cache, &colors);
        assert_eq!(n_out as i32, wn, "n_out cache={cache:?} colors={colors:?}");
        assert_eq!(out, wout, "out_colors cache={cache:?} colors={colors:?}");
        assert_eq!(found, wfound[..n_cache], "found cache={cache:?} colors={colors:?}");
    }
}

#[test]
fn write_palette_mode_info_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_palette_mode_info;
    let mut rng = Rng(0xfa11_e77e_c0de_0007);
    // Strictly-increasing colour list of length `k` in [0, 2^bd) at plane offset.
    let fill = |rng: &mut Rng, arr: &mut [u16; 24], plane: usize, k: usize, bd: i32| {
        let maxv = 1i32 << bd;
        let step = (maxv / (k as i32 + 2)).max(1);
        let mut cur = (rng.next() % step as u64) as i32;
        for j in 0..k {
            arr[plane * 8 + j] = cur as u16;
            cur += 1 + (rng.next() % step as u64) as i32;
        }
    };
    for _ in 0..200_000 {
        let bd = [8i32, 10, 12][(rng.next() % 3) as usize];
        let maxv = 1u64 << bd;
        let mode_dc = rng.next().is_multiple_of(2);
        let uv_dc = rng.next().is_multiple_of(2);
        let n_y = if rng.next().is_multiple_of(3) { 0 } else { 2 + (rng.next() % 7) as usize };
        let n_uv = if rng.next().is_multiple_of(3) { 0 } else { 2 + (rng.next() % 7) as usize };
        // Block palette: Y (sorted) @0, U (sorted) @8, V (unsorted) @16.
        let mut pc = [0u16; 24];
        fill(&mut rng, &mut pc, 0, n_y, bd);
        fill(&mut rng, &mut pc, 1, n_uv, bd);
        for j in 0..n_uv {
            pc[16 + j] = (rng.next() % maxv) as u16;
        }
        let psize = [n_y as u8, n_uv as u8];
        // Neighbours (sorted Y@0, U@8) — chance overlap with block drives cache hits.
        let mk_nb = |rng: &mut Rng| -> ([u16; 24], [i32; 2]) {
            let ay = (rng.next() % 9) as usize;
            let au = (rng.next() % 9) as usize;
            let mut a = [0u16; 24];
            fill(rng, &mut a, 0, ay, bd);
            fill(rng, &mut a, 1, au, bd);
            (a, [ay as i32, au as i32])
        };
        let ha = rng.next().is_multiple_of(2);
        let hl = rng.next().is_multiple_of(2);
        let (a_colors, a_size) = mk_nb(&mut rng);
        let (l_colors, l_size) = mk_nb(&mut rng);
        let mte = -((rng.next() % 20) as i32) * 32;
        // Pre-selected CDFs.
        let ym = [1 + (rng.next() % 32766) as u16, 0, 0];
        let um = [1 + (rng.next() % 32766) as u16, 0, 0];
        let mk7 = |rng: &mut Rng| {
            let mut c = [0u16; 8];
            let mut prev = 32768i32;
            for e in c.iter_mut().take(6) {
                let v = (prev - 1 - (rng.next() % 200) as i32).max(1);
                *e = v as u16;
                prev = v;
            }
            c
        };
        let ys = mk7(&mut rng);
        let us = mk7(&mut rng);

        let mut enc = OdEcEnc::new();
        let (mut rym, mut rys, mut rum, mut rus) = (ym, ys, um, us);
        write_palette_mode_info(
            &mut enc, mode_dc, uv_dc, bd, [n_y as i32, n_uv as i32], &pc,
            &mut rym, &mut rys, &mut rum, &mut rus, mte, ha, &a_colors, a_size, hl, &l_colors, l_size,
        );
        let got = enc.done().to_vec();
        let (want, oym, oys, oum, ous) = c::ref_write_palette_mode_info(
            mode_dc, uv_dc, bd, &psize, &pc, mte, ha, &a_colors, &a_size, hl, &l_colors, &l_size,
            &ym, &ys, &um, &us,
        );
        assert_eq!(got, want, "bytes bd={bd} mode_dc={mode_dc} uv_dc={uv_dc} n_y={n_y} n_uv={n_uv} pc={pc:?}");
        assert_eq!(rym, oym, "y_mode_cdf");
        assert_eq!(rys, oys, "y_size_cdf");
        assert_eq!(rum, oum, "uv_mode_cdf");
        assert_eq!(rus, ous, "uv_size_cdf");
    }
}

#[test]
fn write_interintra_info_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_interintra_info;
    let mut rng = Rng(0x11a_c0de_0000_0008);
    // Monotone-decreasing CDF of nsyms (nsyms cumulative incl. trailing 0, +count).
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 300) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c // c[nsyms-1] = 0 (top), c[nsyms] = 0 (count)
    }
    for _ in 0..300_000 {
        let interintra = (rng.next() % 2) as i32;
        let interintra_mode = (rng.next() % 4) as i32;
        let wedge_used = rng.next().is_multiple_of(2);
        let use_wedge = (rng.next() % 2) as i32;
        let wedge_index = (rng.next() % 16) as i32;
        let ii: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let iim: [u16; 5] = mk(&mut rng, 4).try_into().unwrap();
        let wii: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let wix: [u16; 17] = mk(&mut rng, 16).try_into().unwrap();

        let mut enc = OdEcEnc::new();
        let (mut rii, mut riim, mut rwii, mut rwix) = (ii, iim, wii, wix);
        write_interintra_info(
            &mut enc, true, interintra, &mut rii, interintra_mode, &mut riim, wedge_used,
            use_wedge, &mut rwii, wedge_index, &mut rwix,
        );
        let got = enc.done().to_vec();
        let (want, oii, oiim, owii, owix) = c::ref_write_interintra_info(
            interintra, &ii, interintra_mode, &iim, wedge_used, use_wedge, &wii, wedge_index, &wix,
        );
        assert_eq!(got, want, "bytes ii={interintra} mode={interintra_mode} wu={wedge_used} uw={use_wedge} wi={wedge_index}");
        assert_eq!(rii, oii, "ii_cdf");
        assert_eq!(riim, oiim, "ii_mode_cdf");
        assert_eq!(rwii, owii, "wedge_ii_cdf");
        assert_eq!(rwix, owix, "wedge_idx_cdf");
    }
}

#[test]
fn get_comp_group_idx_context_matches_c() {
    use aom_entropy::partition::get_comp_group_idx_context;
    // ref_frame[1] <= 0 => single ref (NONE=-1 / INTRA=0); >0 => compound.
    let rf1s = [-1i32, 0, 1, 4, 7];
    let rf0s = [1i32, 4, 6, 7]; // incl ALTREF=7
    for ha in [false, true] {
        for &a_rf0 in &rf0s {
            for &a_rf1 in &rf1s {
                for a_cgi in 0..2 {
                    for hl in [false, true] {
                        for &l_rf0 in &rf0s {
                            for &l_rf1 in &rf1s {
                                for l_cgi in 0..2 {
                                    let got = get_comp_group_idx_context(ha, a_rf0, a_rf1, a_cgi, hl, l_rf0, l_rf1, l_cgi);
                                    let want = c::ref_get_comp_group_idx_context(ha, a_rf0, a_rf1, a_cgi, hl, l_rf0, l_rf1, l_cgi);
                                    assert_eq!(got, want, "ha={ha} a=({a_rf0},{a_rf1},{a_cgi}) hl={hl} l=({l_rf0},{l_rf1},{l_cgi})");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn write_compound_type_info_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_compound_type_info;
    let mut rng = Rng(0xc02b_0000_c0de_0009);
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 300) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c
    }
    for _ in 0..400_000 {
        let masked = rng.next().is_multiple_of(2);
        let cgi = (rng.next() % 2) as i32;
        let dist_wtd = rng.next().is_multiple_of(2);
        let compound_idx = (rng.next() % 2) as i32;
        let wedge_used = rng.next().is_multiple_of(2);
        let comp_type = 2 + (rng.next() % 2) as i32; // COMPOUND_WEDGE=2 / DIFFWTD=3
        let wedge_index = (rng.next() % 16) as i32;
        let wedge_sign = (rng.next() % 2) as i32;
        let mask_type = (rng.next() % 2) as i32;
        let cgi_cdf: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let cidx_cdf: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let ct_cdf: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let wix_cdf: [u16; 17] = mk(&mut rng, 16).try_into().unwrap();

        let mut enc = OdEcEnc::new();
        let (mut rcgi, mut rcidx, mut rct, mut rwix) = (cgi_cdf, cidx_cdf, ct_cdf, wix_cdf);
        write_compound_type_info(
            &mut enc, masked, cgi, &mut rcgi, dist_wtd, compound_idx, &mut rcidx, wedge_used,
            comp_type, &mut rct, wedge_index, &mut rwix, wedge_sign, mask_type,
        );
        let got = enc.done().to_vec();
        let (want, ocgi, ocidx, oct, owix) = c::ref_write_compound_type_info(
            masked, cgi, &cgi_cdf, dist_wtd, compound_idx, &cidx_cdf, wedge_used, comp_type,
            &ct_cdf, wedge_index, &wix_cdf, wedge_sign, mask_type,
        );
        assert_eq!(got, want, "bytes masked={masked} cgi={cgi} dw={dist_wtd} ci={compound_idx} wu={wedge_used} ct={comp_type}");
        assert_eq!(rcgi, ocgi, "cgi_cdf");
        assert_eq!(rcidx, ocidx, "cidx_cdf");
        assert_eq!(rct, oct, "ctype_cdf");
        assert_eq!(rwix, owix, "wedge_idx_cdf");
    }
}

#[test]
fn get_relative_dist_matches_c() {
    use aom_entropy::partition::get_relative_dist;
    for enable in [false, true] {
        for bm1 in 0..8i32 {
            let bits = bm1 + 1;
            let n = 1i32 << bits;
            // enable==false ignores a,b; a,b must be in [0, 2^bits) when enabled (C assert).
            let hi = if enable { n } else { 1 };
            for a in 0..hi {
                for b in 0..hi {
                    assert_eq!(
                        get_relative_dist(enable, bm1, a, b),
                        c::ref_get_relative_dist(enable, bm1, a, b),
                        "enable={enable} bm1={bm1} a={a} b={b}"
                    );
                }
            }
        }
    }
}

#[test]
fn get_comp_index_context_matches_c() {
    use aom_entropy::partition::get_comp_index_context;
    let mut rng = Rng(0x0de7_0000_c0de_000a);
    for _ in 0..400_000 {
        let enable = rng.next().is_multiple_of(2);
        let bm1 = (rng.next() % 8) as i32;
        let bits = bm1 + 1;
        let n = 1u64 << bits;
        let cur = (rng.next() % n) as i32;
        let fwd = (rng.next() % n) as i32;
        let bck = (rng.next() % n) as i32;
        let ha = rng.next().is_multiple_of(2);
        let a_has2 = rng.next().is_multiple_of(2);
        let a_cidx = (rng.next() % 2) as i32;
        let a_rf0 = 1 + (rng.next() % 7) as i32; // 1..7 (incl ALTREF=7)
        let hl = rng.next().is_multiple_of(2);
        let l_has2 = rng.next().is_multiple_of(2);
        let l_cidx = (rng.next() % 2) as i32;
        let l_rf0 = 1 + (rng.next() % 7) as i32;
        let got = get_comp_index_context(enable, bm1, cur, fwd, bck, ha, a_has2, a_cidx, a_rf0, hl, l_has2, l_cidx, l_rf0);
        let want = c::ref_get_comp_index_context(enable, bm1, cur, fwd, bck, ha, a_has2, a_cidx, a_rf0, hl, l_has2, l_cidx, l_rf0);
        assert_eq!(got, want, "enable={enable} bm1={bm1} cur={cur} fwd={fwd} bck={bck}");
    }
}

#[test]
fn intra_prediction_mode_gates_match_c() {
    use aom_entropy::partition::{
        allow_palette, get_uv_mode, is_cfl_allowed, is_directional_mode, use_angle_delta,
    };
    // Pure bsize/mode gates — exhaustive.
    for bsize in 0..22usize {
        assert_eq!(use_angle_delta(bsize), c::ref_use_angle_delta(bsize as i32), "use_angle_delta {bsize}");
        for allow_sct in [false, true] {
            assert_eq!(allow_palette(allow_sct, bsize), c::ref_allow_palette(allow_sct, bsize as i32), "allow_palette {allow_sct} {bsize}");
        }
    }
    for mode in 0..13i32 {
        assert_eq!(is_directional_mode(mode), c::ref_is_directional_mode(mode), "is_directional_mode {mode}");
    }
    for uv_mode in 0..14usize {
        assert_eq!(get_uv_mode(uv_mode), c::ref_get_uv_mode(uv_mode as i32), "get_uv_mode {uv_mode}");
    }
    // is_cfl_allowed: bsize x lossless x subsampling x seg_id.
    for bsize in 0..22usize {
        for lossless in [false, true] {
            for ssx in 0..2usize {
                for ssy in 0..2usize {
                    for seg_id in [0i32, 3, 7] {
                        assert_eq!(
                            is_cfl_allowed(bsize, lossless, ssx, ssy),
                            c::ref_is_cfl_allowed(bsize as i32, seg_id, lossless, ssx as i32, ssy as i32),
                            "is_cfl_allowed bsize={bsize} lossless={lossless} ssx={ssx} ssy={ssy}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn write_intra_y_and_angle_delta_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_intra_y_and_angle_delta;
    let mut rng = Rng(0x1a7a_c0de_0000_000b);
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 300) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c
    }
    for _ in 0..300_000 {
        let mode = (rng.next() % 13) as i32; // INTRA_MODES
        let bsize = (rng.next() % 22) as usize;
        let angle_delta_y = (rng.next() % 7) as i32 - 3; // [-3,3]
        let yc: [u16; 14] = mk(&mut rng, 13).try_into().unwrap();
        let ac: [u16; 8] = mk(&mut rng, 7).try_into().unwrap();
        let mut enc = OdEcEnc::new();
        let (mut ryc, mut rac) = (yc, ac);
        write_intra_y_and_angle_delta(&mut enc, &mut ryc, mode, bsize, angle_delta_y, &mut rac);
        let got = enc.done().to_vec();
        let (want, oyc, oac) = c::ref_write_intra_y_and_angle(mode, bsize as i32, &yc, angle_delta_y, &ac);
        assert_eq!(got, want, "bytes mode={mode} bsize={bsize} ad={angle_delta_y}");
        assert_eq!(ryc, oyc, "y_cdf");
        assert_eq!(rac, oac, "y_angle_cdf");
    }
}

#[test]
fn write_intra_uv_and_angle_delta_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_intra_uv_and_angle_delta;
    let mut rng = Rng(0x1a_c0de_0000_000c);
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 300) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c
    }
    for _ in 0..300_000 {
        let monochrome = rng.next().is_multiple_of(5); // mostly false (1 in 5)
        let is_chroma_ref = !rng.next().is_multiple_of(5); // mostly true
        let cfl_allowed = rng.next().is_multiple_of(2);
        let n = if cfl_allowed { 14 } else { 13 };
        let uv_mode = (rng.next() % n as u64) as i32;
        let bsize = (rng.next() % 22) as usize;
        let cfl_idx = (rng.next() % 256) as i32;
        let cfl_joint_sign = (rng.next() % 8) as i32;
        let angle_delta_uv = (rng.next() % 7) as i32 - 3;
        let uc: [u16; 15] = mk(&mut rng, 14).try_into().unwrap();
        let sc: [u16; 9] = mk(&mut rng, 8).try_into().unwrap();
        let mut alpha_nested = [[0u16; 17]; 6];
        let mut alpha_flat = [0u16; 102];
        for ctx in 0..6 {
            let row = mk(&mut rng, 16);
            for j in 0..17 {
                alpha_nested[ctx][j] = row[j];
                alpha_flat[ctx * 17 + j] = row[j];
            }
        }
        let uac: [u16; 8] = mk(&mut rng, 7).try_into().unwrap();

        let mut enc = OdEcEnc::new();
        let (mut ruc, mut rsc, mut rac, mut ruac) = (uc, sc, alpha_nested, uac);
        write_intra_uv_and_angle_delta(
            &mut enc, monochrome, is_chroma_ref, uv_mode, cfl_allowed, bsize, cfl_idx,
            cfl_joint_sign, angle_delta_uv, &mut ruc, &mut rsc, &mut rac, &mut ruac,
        );
        let got = enc.done().to_vec();
        let (want, ouc, osc, oac, ouac) = c::ref_write_intra_uv_and_angle(
            monochrome, is_chroma_ref, uv_mode, cfl_allowed, bsize as i32, cfl_idx, cfl_joint_sign,
            angle_delta_uv, &uc, &sc, &alpha_flat, &uac,
        );
        assert_eq!(got, want, "bytes mono={monochrome} cr={is_chroma_ref} uv={uv_mode} cfl={cfl_allowed} bsize={bsize}");
        assert_eq!(ruc, ouc, "uv_mode_cdf");
        assert_eq!(rsc, osc, "cfl_sign_cdf");
        let rac_flat: [u16; 102] = core::array::from_fn(|i| rac[i / 17][i % 17]);
        assert_eq!(rac_flat, oac, "cfl_alpha_cdf");
        assert_eq!(ruac, ouac, "uv_angle_cdf");
    }
}

#[test]
fn write_intra_prediction_modes_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_intra_prediction_modes;
    use aom_sys_ref::IntraPredModesRef;
    let mut rng = Rng(0x1a_9de5_c0de_000d);
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 300) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c
    }
    // strictly-increasing colours of length k at plane offset (valid for delta coder).
    let fill = |rng: &mut Rng, arr: &mut [u16; 24], plane: usize, k: usize, bd: i32| {
        let maxv = 1i32 << bd;
        let step = (maxv / (k as i32 + 2)).max(1);
        let mut cur = (rng.next() % step as u64) as i32;
        for j in 0..k {
            arr[plane * 8 + j] = cur as u16;
            cur += 1 + (rng.next() % step as u64) as i32;
        }
    };
    for _ in 0..200_000 {
        let bd = [8i32, 10, 12][(rng.next() % 3) as usize];
        let maxv = 1u64 << bd;
        let mode = (rng.next() % 13) as i32;
        let bsize = (rng.next() % 22) as usize;
        let angle_delta_y = (rng.next() % 7) as i32 - 3;
        let monochrome = rng.next().is_multiple_of(5);
        let is_chroma_ref = !rng.next().is_multiple_of(5);
        let cfl_allowed = rng.next().is_multiple_of(2);
        let n_uvmode = if cfl_allowed { 14 } else { 13 };
        let uv_mode = (rng.next() % n_uvmode as u64) as i32;
        let cfl_idx = (rng.next() % 256) as i32;
        let cfl_joint_sign = (rng.next() % 8) as i32;
        let angle_delta_uv = (rng.next() % 7) as i32 - 3;
        let allow_palette = rng.next().is_multiple_of(2);
        let n_y = 2 + (rng.next() % 7) as usize;
        let n_uv = 2 + (rng.next() % 7) as usize;
        let mut pc = [0u16; 24];
        fill(&mut rng, &mut pc, 0, n_y, bd);
        fill(&mut rng, &mut pc, 1, n_uv, bd);
        for j in 0..n_uv {
            pc[16 + j] = (rng.next() % maxv) as u16;
        }
        let palette_size = [n_y as u8, n_uv as u8];
        let mk_nb = |rng: &mut Rng| -> ([u16; 24], [i32; 2]) {
            let ay = (rng.next() % 9) as usize;
            let au = (rng.next() % 9) as usize;
            let mut a = [0u16; 24];
            fill(rng, &mut a, 0, ay, bd);
            fill(rng, &mut a, 1, au, bd);
            (a, [ay as i32, au as i32])
        };
        let ha = rng.next().is_multiple_of(2);
        let hl = rng.next().is_multiple_of(2);
        let (a_colors, a_size) = mk_nb(&mut rng);
        let (l_colors, l_size) = mk_nb(&mut rng);
        let mte = -((rng.next() % 20) as i32) * 32;
        let filter_allowed = rng.next().is_multiple_of(2);
        let use_filter_intra = (rng.next() % 2) as i32;
        let filter_intra_mode = (rng.next() % 5) as i32;

        // CDFs.
        let yc: [u16; 14] = mk(&mut rng, 13).try_into().unwrap();
        let yac: [u16; 8] = mk(&mut rng, 7).try_into().unwrap();
        let uc: [u16; 15] = mk(&mut rng, 14).try_into().unwrap();
        let sc: [u16; 9] = mk(&mut rng, 8).try_into().unwrap();
        let mut alpha_n = [[0u16; 17]; 6];
        let mut alpha_f = [0u16; 102];
        for ctx in 0..6 {
            let row = mk(&mut rng, 16);
            for j in 0..17 {
                alpha_n[ctx][j] = row[j];
                alpha_f[ctx * 17 + j] = row[j];
            }
        }
        let uac: [u16; 8] = mk(&mut rng, 7).try_into().unwrap();
        let pym: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let pys: [u16; 8] = mk(&mut rng, 7).try_into().unwrap();
        let pum: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let pus: [u16; 8] = mk(&mut rng, 7).try_into().unwrap();
        let fiu: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let fim: [u16; 6] = mk(&mut rng, 5).try_into().unwrap();

        // Rust.
        let mut enc = OdEcEnc::new();
        let (mut ryc, mut ryac, mut ruc, mut rsc, mut ran, mut ruac) = (yc, yac, uc, sc, alpha_n, uac);
        let (mut rpym, mut rpys, mut rpum, mut rpus, mut rfiu, mut rfim) = (pym, pys, pum, pus, fiu, fim);
        write_intra_prediction_modes(
            &mut enc, mode, bsize, &mut ryc, angle_delta_y, &mut ryac, monochrome, is_chroma_ref,
            uv_mode, cfl_allowed, cfl_idx, cfl_joint_sign, angle_delta_uv, &mut ruc, &mut rsc,
            &mut ran, &mut ruac, allow_palette, bd, [n_y as i32, n_uv as i32], &pc, mte, ha,
            &a_colors, a_size, hl, &l_colors, l_size, &mut rpym, &mut rpys, &mut rpum, &mut rpus,
            filter_allowed, use_filter_intra, filter_intra_mode, &mut rfiu, &mut rfim,
        );
        let got = enc.done().to_vec();

        // C.
        let inp = IntraPredModesRef {
            mode, bsize: bsize as i32, y_cdf: &yc, angle_delta_y, y_angle_cdf: &yac,
            monochrome, is_chroma_ref, uv_mode, cfl_allowed, cfl_idx, cfl_joint_sign, angle_delta_uv,
            uv_mode_cdf: &uc, cfl_sign_cdf: &sc, cfl_alpha_cdf: &alpha_f, uv_angle_cdf: &uac,
            allow_palette, bit_depth: bd, palette_size: &palette_size, palette_colors: &pc,
            mb_to_top_edge: mte, ha, a_colors: &a_colors, a_size: &a_size, hl, l_colors: &l_colors,
            l_size: &l_size, pal_y_mode_cdf: &pym, pal_y_size_cdf: &pys, pal_uv_mode_cdf: &pum,
            pal_uv_size_cdf: &pus, filter_allowed, use_filter_intra, filter_intra_mode,
            fi_use_cdf: &fiu, fi_mode_cdf: &fim,
        };
        let (want, o_all) = c::ref_write_intra_pred_modes(&inp);
        assert_eq!(got, want, "bytes mode={mode} bsize={bsize} mono={monochrome} cr={is_chroma_ref} uv={uv_mode} pal={allow_palette} fi={filter_allowed}");

        // Compare all adapted CDFs (packed order matches the shim).
        let mut all = Vec::with_capacity(187);
        all.extend_from_slice(&ryc);
        all.extend_from_slice(&ryac);
        all.extend_from_slice(&ruc);
        all.extend_from_slice(&rsc);
        for row in &ran { all.extend_from_slice(row); }
        all.extend_from_slice(&ruac);
        all.extend_from_slice(&rpym);
        all.extend_from_slice(&rpys);
        all.extend_from_slice(&rpum);
        all.extend_from_slice(&rpus);
        all.extend_from_slice(&rfiu);
        all.extend_from_slice(&rfim);
        assert_eq!(all.as_slice(), &o_all[..], "adapted CDFs mode={mode} uv={uv_mode} pal={allow_palette}");
    }
}

#[test]
fn write_delta_q_params_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_delta_q_params_sb;
    let mut rng = Rng(0xde17_a0de_0000_000e);
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c
    }
    for _ in 0..200_000 {
        let dq_present = !rng.next().is_multiple_of(4);
        let dlf_present = rng.next().is_multiple_of(2);
        let dlf_multi = rng.next().is_multiple_of(2);
        let num_planes = if rng.next().is_multiple_of(2) { 3 } else { 1 };
        let bsize = (rng.next() % 22) as usize;
        let sb_size = if rng.next().is_multiple_of(2) { 12 } else { 15 }; // 64x64 / 128x128
        let skip = (rng.next() % 2) as i32;
        let sbul = rng.next().is_multiple_of(2);
        let cur_qindex = 1 + (rng.next() % 255) as i32;
        let cur_base = (rng.next() % 256) as i32;
        let dq_res = [1i32, 2, 4][(rng.next() % 3) as usize];
        let mut mbmi_dlf = [0i32; 4];
        let mut xd_dlf = [0i32; 4];
        for k in 0..4 {
            mbmi_dlf[k] = (rng.next() % 129) as i32 - 64;
            xd_dlf[k] = (rng.next() % 129) as i32 - 64;
        }
        let mbmi_dlf_base = (rng.next() % 129) as i32 - 64;
        let xd_dlf_base = (rng.next() % 129) as i32 - 64;
        let dlf_res = [1i32, 2, 4][(rng.next() % 3) as usize];
        let dq_cdf: [u16; 5] = mk(&mut rng, 4).try_into().unwrap();
        let mut dlmc_n = [[0u16; 5]; 4];
        let mut dlmc_f = [0u16; 20];
        for id in 0..4 {
            let row = mk(&mut rng, 4);
            for j in 0..5 {
                dlmc_n[id][j] = row[j];
                dlmc_f[id * 5 + j] = row[j];
            }
        }
        let dlf_cdf: [u16; 5] = mk(&mut rng, 4).try_into().unwrap();

        let mut enc = OdEcEnc::new();
        let (mut rdqc, mut rdlmc, mut rdlc) = (dq_cdf, dlmc_n, dlf_cdf);
        let mut r_base = cur_base;
        let mut r_xd_dlf = xd_dlf;
        let mut r_xd_dlf_base = xd_dlf_base;
        write_delta_q_params_sb(
            &mut enc, dq_present, dlf_present, dlf_multi, num_planes, bsize, sb_size, skip, sbul,
            cur_qindex, &mut r_base, dq_res, &mbmi_dlf, &mut r_xd_dlf, mbmi_dlf_base,
            &mut r_xd_dlf_base, dlf_res, &mut rdqc, &mut rdlmc, &mut rdlc,
        );
        let got = enc.done().to_vec();
        let (want, odqc, odlmc, odlc, ob, oxd, oxdb) = c::ref_write_delta_q_params_sb(
            dq_present, dlf_present, dlf_multi, num_planes, bsize as i32, sb_size as i32, skip, sbul,
            cur_qindex, cur_base, dq_res, &mbmi_dlf, &xd_dlf, mbmi_dlf_base, xd_dlf_base, dlf_res,
            &dq_cdf, &dlmc_f, &dlf_cdf,
        );
        assert_eq!(got, want, "bytes dq={dq_present} dlf={dlf_present} multi={dlf_multi} np={num_planes} bsize={bsize} sb={sb_size} skip={skip} sbul={sbul}");
        assert_eq!(rdqc, odqc, "dq_cdf");
        let rdlmc_f: [u16; 20] = core::array::from_fn(|i| rdlmc[i / 5][i % 5]);
        assert_eq!(rdlmc_f, odlmc, "dlf_multi_cdf");
        assert_eq!(rdlc, odlc, "dlf_cdf");
        assert_eq!(r_base, ob, "base_qindex");
        assert_eq!(r_xd_dlf, oxd, "xd_delta_lf");
        assert_eq!(r_xd_dlf_base, oxdb, "xd_delta_lf_from_base");
    }
}

#[test]
fn write_cdef_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_cdef;
    let mut rng = Rng(0xcde_f000_c0de_000f);
    for _ in 0..300_000 {
        let coded_lossless = rng.next().is_multiple_of(6);
        let allow_intrabc = rng.next().is_multiple_of(6);
        // mi_row/col within a couple of SBs; mib_size 16 (64) or 32 (128).
        let mib_size = if rng.next().is_multiple_of(2) { 16 } else { 32 };
        let sb_size = if mib_size == 32 { 15usize } else { 12 }; // 128x128 / 64x64
        let mi_row = (rng.next() % 64) as i32;
        let mi_col = (rng.next() % 64) as i32;
        let skip = (rng.next() % 2) as i32;
        let mut trans = [0i32; 4];
        for t in trans.iter_mut() {
            *t = (rng.next() % 2) as i32;
        }
        let cdef_bits = (rng.next() % 4) as i32; // 0..3
        let cdef_strength = if cdef_bits == 0 { 0 } else { (rng.next() % (1u64 << cdef_bits)) as i32 };

        let mut enc = OdEcEnc::new();
        let mut r_trans = [trans[0] != 0, trans[1] != 0, trans[2] != 0, trans[3] != 0];
        write_cdef(&mut enc, coded_lossless, allow_intrabc, mi_row, mi_col, mib_size, sb_size, skip, &mut r_trans, cdef_bits as u32, cdef_strength);
        let got = enc.done().to_vec();
        let (want, otrans) = c::ref_write_cdef(coded_lossless, allow_intrabc, mi_row, mi_col, mib_size, sb_size as i32, skip, &trans, cdef_bits, cdef_strength);
        assert_eq!(got, want, "bytes cl={coded_lossless} ib={allow_intrabc} r={mi_row} c={mi_col} mib={mib_size} skip={skip} bits={cdef_bits}");
        let r_trans_i: [i32; 4] = core::array::from_fn(|i| r_trans[i] as i32);
        assert_eq!(r_trans_i, otrans, "cdef_transmitted r={mi_row} c={mi_col}");
    }
}

#[test]
fn write_mb_modes_kf_prefix_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_mb_modes_kf_prefix;
    use aom_sys_ref::KfPrefixRef;
    let mut rng = Rng(0x11b_0de5_c0de_0010);
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c
    }
    for _ in 0..200_000 {
        let segid_preskip = rng.next().is_multiple_of(2);
        let seg_enabled = !rng.next().is_multiple_of(3);
        let update_map = !rng.next().is_multiple_of(3);
        let last_active_segid = (rng.next() % 8) as i32;
        let segment_id = (rng.next() % (last_active_segid as u64 + 1)) as i32;
        let seg_pred = (rng.next() % (last_active_segid as u64 + 1)) as i32;
        let seg_skip_active = rng.next().is_multiple_of(4);
        let skip_txfm = (rng.next() % 2) as i32;
        let coded_lossless = rng.next().is_multiple_of(6);
        let allow_intrabc = rng.next().is_multiple_of(6);
        let mib_size = if rng.next().is_multiple_of(2) { 16 } else { 32 };
        let sb_size = if mib_size == 32 { 15 } else { 12 };
        let mi_row = (rng.next() % 64) as i32;
        let mi_col = (rng.next() % 64) as i32;
        let mut cdef_trans = [0i32; 4];
        for t in cdef_trans.iter_mut() {
            *t = (rng.next() % 2) as i32;
        }
        let cdef_bits = (rng.next() % 4) as i32;
        let cdef_strength = if cdef_bits == 0 { 0 } else { (rng.next() % (1u64 << cdef_bits)) as i32 };
        let dq_present = !rng.next().is_multiple_of(3);
        let dlf_present = rng.next().is_multiple_of(2);
        let dlf_multi = rng.next().is_multiple_of(2);
        let num_planes = if rng.next().is_multiple_of(2) { 3 } else { 1 };
        let bsize = (rng.next() % 22) as i32;
        let cur_qindex = 1 + (rng.next() % 255) as i32;
        let cur_base = (rng.next() % 256) as i32;
        let dq_res = [1i32, 2, 4][(rng.next() % 3) as usize];
        let mut mbmi_dlf = [0i32; 4];
        let mut xd_dlf = [0i32; 4];
        for k in 0..4 {
            mbmi_dlf[k] = (rng.next() % 129) as i32 - 64;
            xd_dlf[k] = (rng.next() % 129) as i32 - 64;
        }
        let mbmi_dlf_base = (rng.next() % 129) as i32 - 64;
        let xd_dlf_base = (rng.next() % 129) as i32 - 64;
        let dlf_res = [1i32, 2, 4][(rng.next() % 3) as usize];
        let seg_cdf: [u16; 9] = mk(&mut rng, 8).try_into().unwrap();
        let skip_cdf: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let dq_cdf: [u16; 5] = mk(&mut rng, 4).try_into().unwrap();
        let mut dlmc_n = [[0u16; 5]; 4];
        let mut dlmc_f = [0u16; 20];
        for id in 0..4 {
            let row = mk(&mut rng, 4);
            for j in 0..5 {
                dlmc_n[id][j] = row[j];
                dlmc_f[id * 5 + j] = row[j];
            }
        }
        let dlf_cdf: [u16; 5] = mk(&mut rng, 4).try_into().unwrap();

        let mut enc = OdEcEnc::new();
        let (mut rseg, mut rskc, mut rdqc, mut rdlmc, mut rdlc) = (seg_cdf, skip_cdf, dq_cdf, dlmc_n, dlf_cdf);
        let mut r_ctr = [cdef_trans[0] != 0, cdef_trans[1] != 0, cdef_trans[2] != 0, cdef_trans[3] != 0];
        let mut r_base = cur_base;
        let mut r_xd = xd_dlf;
        let mut r_xdb = xd_dlf_base;
        let skip = write_mb_modes_kf_prefix(
            &mut enc, segid_preskip, seg_enabled, update_map, segment_id, seg_pred, last_active_segid,
            &mut rseg, seg_skip_active, skip_txfm, &mut rskc, coded_lossless, allow_intrabc, mi_row,
            mi_col, mib_size, sb_size as usize, &mut r_ctr, cdef_bits as u32, cdef_strength, dq_present,
            dlf_present, dlf_multi, num_planes, bsize as usize, cur_qindex, &mut r_base, dq_res,
            &mbmi_dlf, &mut r_xd, mbmi_dlf_base, &mut r_xdb, dlf_res, &mut rdqc, &mut rdlmc, &mut rdlc,
        );
        let got = enc.done().to_vec();

        let inp = KfPrefixRef {
            segid_preskip, seg_enabled, update_map, segment_id, seg_pred, last_active_segid,
            seg_cdf: &seg_cdf, seg_skip_active, skip_txfm, skip_cdf: &skip_cdf, coded_lossless,
            allow_intrabc, mi_row, mi_col, mib_size, sb_size, cdef_trans: &cdef_trans, cdef_bits,
            cdef_strength, dq_present, dlf_present, dlf_multi, num_planes, bsize, cur_qindex,
            cur_base_qindex: cur_base, dq_res, mbmi_dlf: &mbmi_dlf, xd_dlf: &xd_dlf, mbmi_dlf_base,
            xd_dlf_base, dlf_res, dq_cdf: &dq_cdf, dlf_multi_cdf: &dlmc_f, dlf_cdf: &dlf_cdf,
        };
        let o = c::ref_write_mb_modes_kf_prefix(&inp);
        assert_eq!(got, o.bytes, "bytes preskip={segid_preskip} seg={seg_enabled} um={update_map} ssa={seg_skip_active} skip={skip_txfm} dq={dq_present}");
        assert_eq!(skip, o.skip, "skip return");
        assert_eq!(rseg, o.seg_cdf, "seg_cdf");
        assert_eq!(rskc, o.skip_cdf, "skip_cdf");
        let r_ctr_i: [i32; 4] = core::array::from_fn(|i| r_ctr[i] as i32);
        assert_eq!(r_ctr_i, o.cdef_trans, "cdef_trans");
        assert_eq!(rdqc, o.dq_cdf, "dq_cdf");
        let rdlmc_f: [u16; 20] = core::array::from_fn(|i| rdlmc[i / 5][i % 5]);
        assert_eq!(rdlmc_f, o.dlf_multi_cdf, "dlf_multi_cdf");
        assert_eq!(rdlc, o.dlf_cdf, "dlf_cdf");
        assert_eq!(r_base, o.base_qindex, "base_qindex");
        assert_eq!(r_xd, o.xd_dlf, "xd_dlf");
        assert_eq!(r_xdb, o.xd_dlf_base, "xd_dlf_base");
    }
}

#[test]
fn write_kf_tail_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_kf_tail;
    use aom_sys_ref::IntraPredModesRef;
    let mut rng = Rng(0x11b_7a11_c0de_0011);
    // intrabc/MV-style CDF (sorted descending).
    let mk_ibc = |rng: &mut Rng, ns: usize, out: &mut [u16]| {
        let mut vals = [0i32; 11];
        for v in vals.iter_mut().take(ns - 1) {
            *v = 1 + (rng.next() % 32766) as i32;
        }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 {
            let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32);
            out[i] = v as u16;
            prev = v;
        }
        out[ns - 1] = 0;
        out[ns] = 0;
    };
    let mk_comp = |rng: &mut Rng| -> [u16; 69] {
        let mut c = [0u16; 69];
        mk_ibc(rng, 2, &mut c[0..3]);
        mk_ibc(rng, 11, &mut c[3..15]);
        mk_ibc(rng, 2, &mut c[15..18]);
        for i in 0..10 { let o = 18 + i * 3; mk_ibc(rng, 2, &mut c[o..o + 3]); }
        for i in 0..2 { let o = 48 + i * 5; mk_ibc(rng, 4, &mut c[o..o + 5]); }
        mk_ibc(rng, 4, &mut c[58..63]); mk_ibc(rng, 2, &mut c[63..66]); mk_ibc(rng, 2, &mut c[66..69]);
        c
    };
    // simple decreasing CDF for intra symbols.
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 300) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c
    }
    let fill = |rng: &mut Rng, arr: &mut [u16; 24], plane: usize, k: usize, bd: i32| {
        let maxv = 1i32 << bd;
        let step = (maxv / (k as i32 + 2)).max(1);
        let mut cur = (rng.next() % step as u64) as i32;
        for j in 0..k {
            arr[plane * 8 + j] = cur as u16;
            cur += 1 + (rng.next() % step as u64) as i32;
        }
    };
    for _ in 0..120_000 {
        // intrabc state
        let allow_intrabc = !rng.next().is_multiple_of(3);
        let use_intrabc = (rng.next() % 2) as i32;
        let mut ibc = [0u16; 3];
        mk_ibc(&mut rng, 2, &mut ibc);
        let mut joints = [0u16; 5];
        mk_ibc(&mut rng, 4, &mut joints);
        let comp0 = mk_comp(&mut rng);
        let comp1 = mk_comp(&mut rng);
        let dr = ((rng.next() % 4097) as i32 - 2048) * 8;
        let dc = ((rng.next() % 4097) as i32 - 2048) * 8;
        // intra state
        let bd = [8i32, 10, 12][(rng.next() % 3) as usize];
        let maxv = 1u64 << bd;
        let mode = (rng.next() % 13) as i32;
        let bsize = (rng.next() % 22) as usize;
        let angle_delta_y = (rng.next() % 7) as i32 - 3;
        let monochrome = rng.next().is_multiple_of(5);
        let is_chroma_ref = !rng.next().is_multiple_of(5);
        let cfl_allowed = rng.next().is_multiple_of(2);
        let n_uvmode = if cfl_allowed { 14 } else { 13 };
        let uv_mode = (rng.next() % n_uvmode as u64) as i32;
        let cfl_idx = (rng.next() % 256) as i32;
        let cfl_joint_sign = (rng.next() % 8) as i32;
        let angle_delta_uv = (rng.next() % 7) as i32 - 3;
        let allow_palette = rng.next().is_multiple_of(2);
        let n_y = 2 + (rng.next() % 7) as usize;
        let n_uv = 2 + (rng.next() % 7) as usize;
        let mut pc = [0u16; 24];
        fill(&mut rng, &mut pc, 0, n_y, bd);
        fill(&mut rng, &mut pc, 1, n_uv, bd);
        for j in 0..n_uv { pc[16 + j] = (rng.next() % maxv) as u16; }
        let palette_size = [n_y as u8, n_uv as u8];
        let mk_nb = |rng: &mut Rng| -> ([u16; 24], [i32; 2]) {
            let ay = (rng.next() % 9) as usize;
            let au = (rng.next() % 9) as usize;
            let mut a = [0u16; 24];
            fill(rng, &mut a, 0, ay, bd);
            fill(rng, &mut a, 1, au, bd);
            (a, [ay as i32, au as i32])
        };
        let ha = rng.next().is_multiple_of(2);
        let hl = rng.next().is_multiple_of(2);
        let (a_colors, a_size) = mk_nb(&mut rng);
        let (l_colors, l_size) = mk_nb(&mut rng);
        let mte = -((rng.next() % 20) as i32) * 32;
        let filter_allowed = rng.next().is_multiple_of(2);
        let use_filter_intra = (rng.next() % 2) as i32;
        let filter_intra_mode = (rng.next() % 5) as i32;
        let yc: [u16; 14] = mk(&mut rng, 13).try_into().unwrap();
        let yac: [u16; 8] = mk(&mut rng, 7).try_into().unwrap();
        let uc: [u16; 15] = mk(&mut rng, 14).try_into().unwrap();
        let sc: [u16; 9] = mk(&mut rng, 8).try_into().unwrap();
        let mut alpha_n = [[0u16; 17]; 6];
        let mut alpha_f = [0u16; 102];
        for ctx in 0..6 {
            let row = mk(&mut rng, 16);
            for j in 0..17 { alpha_n[ctx][j] = row[j]; alpha_f[ctx * 17 + j] = row[j]; }
        }
        let uac: [u16; 8] = mk(&mut rng, 7).try_into().unwrap();
        let pym: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let pys: [u16; 8] = mk(&mut rng, 7).try_into().unwrap();
        let pum: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let pus: [u16; 8] = mk(&mut rng, 7).try_into().unwrap();
        let fiu: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let fim: [u16; 6] = mk(&mut rng, 5).try_into().unwrap();

        // Rust
        let mut enc = OdEcEnc::new();
        let (mut rib, mut rjo, mut rc0, mut rc1) = (ibc, joints, comp0, comp1);
        let (mut ryc, mut ryac, mut ruc, mut rsc, mut ran, mut ruac) = (yc, yac, uc, sc, alpha_n, uac);
        let (mut rpym, mut rpys, mut rpum, mut rpus, mut rfiu, mut rfim) = (pym, pys, pum, pus, fiu, fim);
        write_kf_tail(
            &mut enc, allow_intrabc, &mut rib, &mut rjo, &mut rc0, &mut rc1, use_intrabc, dr, dc,
            mode, bsize, &mut ryc, angle_delta_y, &mut ryac, monochrome, is_chroma_ref, uv_mode,
            cfl_allowed, cfl_idx, cfl_joint_sign, angle_delta_uv, &mut ruc, &mut rsc, &mut ran,
            &mut ruac, allow_palette, bd, [n_y as i32, n_uv as i32], &pc, mte, ha, &a_colors, a_size,
            hl, &l_colors, l_size, &mut rpym, &mut rpys, &mut rpum, &mut rpus, filter_allowed,
            use_filter_intra, filter_intra_mode, &mut rfiu, &mut rfim,
        );
        let got = enc.done().to_vec();

        let intra = IntraPredModesRef {
            mode, bsize: bsize as i32, y_cdf: &yc, angle_delta_y, y_angle_cdf: &yac, monochrome,
            is_chroma_ref, uv_mode, cfl_allowed, cfl_idx, cfl_joint_sign, angle_delta_uv,
            uv_mode_cdf: &uc, cfl_sign_cdf: &sc, cfl_alpha_cdf: &alpha_f, uv_angle_cdf: &uac,
            allow_palette, bit_depth: bd, palette_size: &palette_size, palette_colors: &pc,
            mb_to_top_edge: mte, ha, a_colors: &a_colors, a_size: &a_size, hl, l_colors: &l_colors,
            l_size: &l_size, pal_y_mode_cdf: &pym, pal_y_size_cdf: &pys, pal_uv_mode_cdf: &pum,
            pal_uv_size_cdf: &pus, filter_allowed, use_filter_intra, filter_intra_mode,
            fi_use_cdf: &fiu, fi_mode_cdf: &fim,
        };
        let (want, oib, ojo, oc0, oc1, o_all) = c::ref_write_kf_tail(
            allow_intrabc, &ibc, &joints, &comp0, &comp1, use_intrabc != 0, dr, dc, &intra,
        );
        assert_eq!(got, want, "bytes aib={allow_intrabc} uib={use_intrabc} mode={mode} bsize={bsize} pal={allow_palette}");
        assert_eq!(rib, oib, "intrabc_cdf");
        assert_eq!(rjo, ojo, "joints");
        assert_eq!(rc0, oc0, "comp0");
        assert_eq!(rc1, oc1, "comp1");
        let mut all = Vec::with_capacity(187);
        all.extend_from_slice(&ryc); all.extend_from_slice(&ryac); all.extend_from_slice(&ruc);
        all.extend_from_slice(&rsc);
        for row in &ran { all.extend_from_slice(row); }
        all.extend_from_slice(&ruac); all.extend_from_slice(&rpym); all.extend_from_slice(&rpys);
        all.extend_from_slice(&rpum); all.extend_from_slice(&rpus); all.extend_from_slice(&rfiu);
        all.extend_from_slice(&rfim);
        assert_eq!(all.as_slice(), &o_all[..], "intra CDFs");
    }
}

#[test]
fn write_inter_segment_id_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{get_pred_context_seg_id, write_inter_segment_id};
    // context (exhaustive)
    for ha in [false, true] {
        for a in 0..2 {
            for hl in [false, true] {
                for l in 0..2 {
                    assert_eq!(get_pred_context_seg_id(ha, a, hl, l), c::ref_get_pred_context_seg_id(ha, a, hl, l), "seg_id_pred_ctx");
                }
            }
        }
    }
    let mut rng = Rng(0x1e5_e610_c0de_0012);
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c
    }
    for _ in 0..300_000 {
        let update_map = !rng.next().is_multiple_of(4);
        let preskip = rng.next().is_multiple_of(2);
        let segid_preskip = rng.next().is_multiple_of(2);
        let skip = rng.next().is_multiple_of(2);
        let temporal_update = rng.next().is_multiple_of(2);
        let seg_id_predicted = (rng.next() % 2) as i32;
        let seg_enabled = update_map; // enabled whenever the map updates (realistic)
        let last_active_segid = (rng.next() % 8) as i32;
        let segment_id = (rng.next() % (last_active_segid as u64 + 1)) as i32;
        let seg_pred = (rng.next() % (last_active_segid as u64 + 1)) as i32;
        let pred_cdf: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let seg_cdf: [u16; 9] = mk(&mut rng, 8).try_into().unwrap();

        let mut enc = OdEcEnc::new();
        let (mut rpc, mut rsc) = (pred_cdf, seg_cdf);
        write_inter_segment_id(&mut enc, update_map, preskip, segid_preskip, skip, temporal_update, seg_id_predicted, &mut rpc, &mut rsc, seg_enabled, segment_id, seg_pred, last_active_segid);
        let got = enc.done().to_vec();
        let (want, opc, osc) = c::ref_write_inter_segment_id(update_map, preskip, segid_preskip, skip, temporal_update, seg_id_predicted, &pred_cdf, &seg_cdf, seg_enabled, segment_id, seg_pred, last_active_segid);
        assert_eq!(got, want, "bytes um={update_map} pre={preskip} sps={segid_preskip} skip={skip} tu={temporal_update} sip={seg_id_predicted}");
        assert_eq!(rpc, opc, "pred_cdf");
        assert_eq!(rsc, osc, "seg_cdf");
    }
}

#[test]
fn write_inter_prefix_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_inter_prefix;
    use aom_sys_ref::InterPrefixRef;
    let mut rng = Rng(0x1e_7a5e_c0de_0013);
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c
    }
    for _ in 0..200_000 {
        let update_map = !rng.next().is_multiple_of(3);
        let segid_preskip = rng.next().is_multiple_of(2);
        let temporal_update = rng.next().is_multiple_of(2);
        let seg_id_predicted = (rng.next() % 2) as i32;
        let seg_enabled = update_map;
        let last_active_segid = (rng.next() % 8) as i32;
        let segment_id = (rng.next() % (last_active_segid as u64 + 1)) as i32;
        let seg_pred = (rng.next() % (last_active_segid as u64 + 1)) as i32;
        let frame_skip_mode_flag = rng.next().is_multiple_of(2);
        let sm_seg_skip = rng.next().is_multiple_of(4);
        let sm_comp_allowed = !rng.next().is_multiple_of(3);
        let sm_seg_ref_gmv = rng.next().is_multiple_of(4);
        let skip_mode = (rng.next() % 2) as i32;
        let skip_seg_active = rng.next().is_multiple_of(4);
        let skip_txfm = (rng.next() % 2) as i32;
        let coded_lossless = rng.next().is_multiple_of(6);
        let allow_intrabc = rng.next().is_multiple_of(6);
        let mib_size = if rng.next().is_multiple_of(2) { 16 } else { 32 };
        let sb_size = if mib_size == 32 { 15 } else { 12 };
        let mi_row = (rng.next() % 64) as i32;
        let mi_col = (rng.next() % 64) as i32;
        let mut cdef_trans = [0i32; 4];
        for t in cdef_trans.iter_mut() { *t = (rng.next() % 2) as i32; }
        let cdef_bits = (rng.next() % 4) as i32;
        let cdef_strength = if cdef_bits == 0 { 0 } else { (rng.next() % (1u64 << cdef_bits)) as i32 };
        let dq_present = !rng.next().is_multiple_of(3);
        let dlf_present = rng.next().is_multiple_of(2);
        let dlf_multi = rng.next().is_multiple_of(2);
        let num_planes = if rng.next().is_multiple_of(2) { 3 } else { 1 };
        let bsize = (rng.next() % 22) as i32;
        let cur_qindex = 1 + (rng.next() % 255) as i32;
        let cur_base = (rng.next() % 256) as i32;
        let dq_res = [1i32, 2, 4][(rng.next() % 3) as usize];
        let mut mbmi_dlf = [0i32; 4];
        let mut xd_dlf = [0i32; 4];
        for k in 0..4 { mbmi_dlf[k] = (rng.next() % 129) as i32 - 64; xd_dlf[k] = (rng.next() % 129) as i32 - 64; }
        let mbmi_dlf_base = (rng.next() % 129) as i32 - 64;
        let xd_dlf_base = (rng.next() % 129) as i32 - 64;
        let dlf_res = [1i32, 2, 4][(rng.next() % 3) as usize];
        let seg_ref_frame_active = rng.next().is_multiple_of(3);
        let seg_globalmv_active = rng.next().is_multiple_of(3);
        let is_inter = (rng.next() % 2) as i32;
        let pred_cdf: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let seg_cdf: [u16; 9] = mk(&mut rng, 8).try_into().unwrap();
        let skip_mode_cdf: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let skip_cdf: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let dq_cdf: [u16; 5] = mk(&mut rng, 4).try_into().unwrap();
        let mut dlmc_n = [[0u16; 5]; 4];
        let mut dlmc_f = [0u16; 20];
        for id in 0..4 { let row = mk(&mut rng, 4); for j in 0..5 { dlmc_n[id][j] = row[j]; dlmc_f[id * 5 + j] = row[j]; } }
        let dlf_cdf: [u16; 5] = mk(&mut rng, 4).try_into().unwrap();
        let ii_cdf: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();

        let mut enc = OdEcEnc::new();
        let (mut rpc, mut rsc, mut rsmc, mut rskc, mut rdqc, mut rdlmc, mut rdlc, mut riic) =
            (pred_cdf, seg_cdf, skip_mode_cdf, skip_cdf, dq_cdf, dlmc_n, dlf_cdf, ii_cdf);
        let mut r_ctr = [cdef_trans[0] != 0, cdef_trans[1] != 0, cdef_trans[2] != 0, cdef_trans[3] != 0];
        let mut r_base = cur_base;
        let mut r_xd = xd_dlf;
        let mut r_xdb = xd_dlf_base;
        let (skip, sm) = write_inter_prefix(
            &mut enc, update_map, segid_preskip, temporal_update, seg_id_predicted, &mut rpc, &mut rsc,
            seg_enabled, segment_id, seg_pred, last_active_segid, &mut rsmc, frame_skip_mode_flag,
            sm_seg_skip, sm_comp_allowed, sm_seg_ref_gmv, skip_mode, &mut rskc, skip_seg_active,
            skip_txfm, coded_lossless, allow_intrabc, mi_row, mi_col, mib_size, sb_size as usize,
            &mut r_ctr, cdef_bits as u32, cdef_strength, dq_present, dlf_present, dlf_multi, num_planes,
            bsize as usize, cur_qindex, &mut r_base, dq_res, &mbmi_dlf, &mut r_xd, mbmi_dlf_base,
            &mut r_xdb, dlf_res, &mut rdqc, &mut rdlmc, &mut rdlc, &mut riic, seg_ref_frame_active,
            seg_globalmv_active, is_inter,
        );
        let got = enc.done().to_vec();

        let inp = InterPrefixRef {
            update_map, segid_preskip, temporal_update, seg_id_predicted, pred_cdf: &pred_cdf,
            seg_cdf: &seg_cdf, seg_enabled, segment_id, seg_pred, last_active_segid,
            skip_mode_cdf: &skip_mode_cdf, frame_skip_mode_flag, sm_seg_skip, sm_comp_allowed,
            sm_seg_ref_gmv, skip_mode, skip_cdf: &skip_cdf, skip_seg_active, skip_txfm, coded_lossless,
            allow_intrabc, mi_row, mi_col, mib_size, sb_size, cdef_trans: &cdef_trans, cdef_bits,
            cdef_strength, dq_present, dlf_present, dlf_multi, num_planes, bsize, cur_qindex,
            cur_base_qindex: cur_base, dq_res, mbmi_dlf: &mbmi_dlf, xd_dlf: &xd_dlf, mbmi_dlf_base,
            xd_dlf_base, dlf_res, dq_cdf: &dq_cdf, dlf_multi_cdf: &dlmc_f, dlf_cdf: &dlf_cdf,
            intra_inter_cdf: &ii_cdf, seg_ref_frame_active, seg_globalmv_active, is_inter,
        };
        let o = c::ref_write_inter_prefix(&inp);
        assert_eq!(got, o.bytes, "bytes um={update_map} sm={skip_mode} skip_seg={skip_seg_active} dq={dq_present} ii={is_inter}");
        assert_eq!(skip, o.skip, "skip"); assert_eq!(sm, o.skip_mode, "skip_mode");
        assert_eq!(rpc, o.pred_cdf, "pred_cdf"); assert_eq!(rsc, o.seg_cdf, "seg_cdf");
        assert_eq!(rsmc, o.skip_mode_cdf, "skip_mode_cdf"); assert_eq!(rskc, o.skip_cdf, "skip_cdf");
        let r_ctr_i: [i32; 4] = core::array::from_fn(|i| r_ctr[i] as i32);
        assert_eq!(r_ctr_i, o.cdef_trans, "cdef_trans"); assert_eq!(rdqc, o.dq_cdf, "dq_cdf");
        let rdlmc_f: [u16; 20] = core::array::from_fn(|i| rdlmc[i / 5][i % 5]);
        assert_eq!(rdlmc_f, o.dlf_multi_cdf, "dlf_multi_cdf"); assert_eq!(rdlc, o.dlf_cdf, "dlf_cdf");
        assert_eq!(r_base, o.base_qindex, "base"); assert_eq!(r_xd, o.xd_dlf, "xd_dlf");
        assert_eq!(r_xdb, o.xd_dlf_base, "xd_dlf_base"); assert_eq!(riic, o.intra_inter_cdf, "intra_inter_cdf");
    }
}

#[test]
fn inter_mode_gates_and_ctx_match_c() {
    use aom_entropy::partition::{
        have_nearmv_in_inter_mode, is_inter_compound_mode, is_inter_singleref_mode,
        mode_context_analyzer,
    };
    for mode in 0..25i32 {
        assert_eq!(is_inter_compound_mode(mode), c::ref_is_inter_compound_mode(mode), "compound {mode}");
        assert_eq!(is_inter_singleref_mode(mode), c::ref_is_inter_singleref_mode(mode), "singleref {mode}");
        assert_eq!(have_nearmv_in_inter_mode(mode), c::ref_have_nearmv_in_inter_mode(mode), "nearmv {mode}");
    }
    // mode_context_analyzer: single-ref (rf1<=0) + compound (rf1 in 1..7).
    // Compound path indexes compound_mode_ctx_map[refmv>>1][..], so keep refmv nibble < 6.
    let mut rng = Rng(0x1de_c72a_c0de_0014);
    for _ in 0..300_000 {
        let rf0 = 1 + (rng.next() % 7) as i32; // LAST..ALTREF
        let compound = rng.next().is_multiple_of(2);
        let rf1 = if compound { 1 + (rng.next() % 7) as i32 } else { -((rng.next() % 2) as i32) }; // -1/0 single
        // mode_context: low 3 bits (newmv), bits 4..8 (refmv, keep <6 for the compound path).
        let newmv = (rng.next() % 8) as i32; // bits 0..2
        let refmv = (rng.next() % 6) as i32; // bits 4..7 (kept < 6 so refmv>>1 < 3)
        let dc = (rng.next() % 2) as i32; // bit 3 (GLOBALMV region, don't-care)
        let mc_val = newmv | (dc << 3) | (refmv << 4);
        let got = mode_context_analyzer(mc_val, rf1 > 0);
        let want = c::ref_mode_context_analyzer(rf0, rf1, mc_val);
        assert_eq!(got, want, "rf0={rf0} rf1={rf1} mc_val={mc_val}");
    }
}

#[test]
fn write_inter_block_mvs_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_inter_block_mvs;
    let mut rng = Rng(0x1eb1_2c0d_e001_5015);
    let mk_ibc = |rng: &mut Rng, ns: usize, out: &mut [u16]| {
        let mut vals = [0i32; 11];
        for v in vals.iter_mut().take(ns - 1) { *v = 1 + (rng.next() % 32766) as i32; }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 { let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32); out[i] = v as u16; prev = v; }
        out[ns - 1] = 0; out[ns] = 0;
    };
    let mk_comp = |rng: &mut Rng| -> [u16; 69] {
        let mut c = [0u16; 69];
        mk_ibc(rng, 2, &mut c[0..3]); mk_ibc(rng, 11, &mut c[3..15]); mk_ibc(rng, 2, &mut c[15..18]);
        for i in 0..10 { let o = 18 + i * 3; mk_ibc(rng, 2, &mut c[o..o + 3]); }
        for i in 0..2 { let o = 48 + i * 5; mk_ibc(rng, 4, &mut c[o..o + 5]); }
        mk_ibc(rng, 4, &mut c[58..63]); mk_ibc(rng, 2, &mut c[63..66]); mk_ibc(rng, 2, &mut c[66..69]);
        c
    };
    // Modes that code MVs (+ some that don't, to exercise the no-op paths).
    let modes = [16i32, 24, 19, 21, 20, 22, 13, 14, 15, 17, 18, 23];
    for _ in 0..200_000 {
        let mode = modes[(rng.next() % modes.len() as u64) as usize];
        let is_compound = rng.next().is_multiple_of(2);
        let usehp = (rng.next() % 2) as i32; // 0/1 (or -1 for NONE)
        let usehp = if rng.next().is_multiple_of(5) { -1 } else { usehp };
        // Non-zero MV diffs (multiples of 1, in valid class range |diff| <= 16384).
        let nz = |rng: &mut Rng| -> (i32, i32) {
            loop {
                let r = (rng.next() % 32769) as i32 - 16384;
                let c = (rng.next() % 32769) as i32 - 16384;
                if r != 0 || c != 0 { return (r, c); }
            }
        };
        let (r0, c0d) = nz(&mut rng);
        let (r1, c1d) = nz(&mut rng);
        let mut joints = [0u16; 5];
        mk_ibc(&mut rng, 4, &mut joints);
        let comp0 = mk_comp(&mut rng);
        let comp1 = mk_comp(&mut rng);

        let mut enc = OdEcEnc::new();
        let (mut rjo, mut rc0, mut rc1) = (joints, comp0, comp1);
        write_inter_block_mvs(&mut enc, mode, is_compound, [r0, r1], [c0d, c1d], usehp, &mut rjo, &mut rc0, &mut rc1);
        let got = enc.done().to_vec();
        let (want, ojo, oc0, oc1) = c::ref_write_inter_block_mvs(mode, is_compound, r0, c0d, r1, c1d, usehp, &joints, &comp0, &comp1);
        assert_eq!(got, want, "bytes mode={mode} comp={is_compound} usehp={usehp}");
        assert_eq!(rjo, ojo, "joints"); assert_eq!(rc0, oc0, "comp0"); assert_eq!(rc1, oc1, "comp1");
    }
}

#[test]
fn write_inter_mode_drl_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_inter_mode_drl;
    let mut rng = Rng(0x1e_0dd1_c0de_0016u64);
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c
    }
    // 2-symbol nested CDF table [n][3].
    let mk2n = |rng: &mut Rng, n: usize, flat: &mut [u16], nested: &mut [[u16; 3]]| {
        for i in 0..n {
            let row = mk(rng, 2);
            for j in 0..3 { flat[i * 3 + j] = row[j]; nested[i][j] = row[j]; }
        }
    };
    let modes = [16i32, 24, 19, 21, 20, 22, 13, 14, 15, 17, 18, 23];
    for _ in 0..200_000 {
        let seg_skip = rng.next().is_multiple_of(4);
        let mode = modes[(rng.next() % modes.len() as u64) as usize];
        // Valid mode_ctx: newmv_ctx (&7) and refmv_ctx (>>4 &15) index 6-entry tables, so
        // keep both < 6 (NEWMV/REFMV_MODE_CONTEXTS); zeromv is bit 3 (2-entry, always ok).
        let newmv_ctx = (rng.next() % 6) as i32;
        let zeromv_bit = (rng.next() % 2) as i32;
        let refmv_ctx = (rng.next() % 6) as i32;
        let mode_ctx = newmv_ctx | (zeromv_bit << 3) | (refmv_ctx << 4);
        let ref_mv_count = (rng.next() % 8) as i32;
        let ref_mv_idx = (rng.next() % 3) as i32;
        let mut weight = [0u16; 8];
        for w in weight.iter_mut() { *w = (rng.next() % 1281) as u16; } // spans REF_CAT_LEVEL=640
        let icm: [u16; 9] = mk(&mut rng, 8).try_into().unwrap();
        let mut nm_f = [0u16; 18]; let mut nm_n = [[0u16; 3]; 6];
        mk2n(&mut rng, 6, &mut nm_f, &mut nm_n);
        let mut zm_f = [0u16; 6]; let mut zm_n = [[0u16; 3]; 2];
        mk2n(&mut rng, 2, &mut zm_f, &mut zm_n);
        let mut rm_f = [0u16; 18]; let mut rm_n = [[0u16; 3]; 6];
        mk2n(&mut rng, 6, &mut rm_f, &mut rm_n);
        let mut drl_f = [0u16; 9]; let mut drl_n = [[0u16; 3]; 3];
        mk2n(&mut rng, 3, &mut drl_f, &mut drl_n);

        let mut enc = OdEcEnc::new();
        let (mut ricm, mut rnm, mut rzm, mut rrm, mut rdrl) = (icm, nm_n, zm_n, rm_n, drl_n);
        write_inter_mode_drl(&mut enc, seg_skip, mode, mode_ctx, &mut ricm, &mut rnm, &mut rzm, &mut rrm, &mut rdrl, ref_mv_idx, ref_mv_count, &weight);
        let got = enc.done().to_vec();
        let (want, oicm, onm, ozm, orm, odrl) = c::ref_write_inter_mode_drl(seg_skip, mode, mode_ctx, &icm, &nm_f, &zm_f, &rm_f, &drl_f, ref_mv_idx, ref_mv_count, &weight);
        assert_eq!(got, want, "bytes ss={seg_skip} mode={mode} ctx={mode_ctx} idx={ref_mv_idx} cnt={ref_mv_count}");
        assert_eq!(ricm, oicm, "icm");
        let rnm_f: [u16; 18] = core::array::from_fn(|i| rnm[i / 3][i % 3]);
        assert_eq!(rnm_f, onm, "newmv");
        let rzm_f: [u16; 6] = core::array::from_fn(|i| rzm[i / 3][i % 3]);
        assert_eq!(rzm_f, ozm, "zeromv");
        let rrm_f: [u16; 18] = core::array::from_fn(|i| rrm[i / 3][i % 3]);
        assert_eq!(rrm_f, orm, "refmv");
        let rdrl_f: [u16; 9] = core::array::from_fn(|i| rdrl[i / 3][i % 3]);
        assert_eq!(rdrl_f, odrl, "drl");
    }
}

#[test]
fn write_inter_mode_tail_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_inter_mode_tail;
    use aom_sys_ref::InterTailRef;
    let mut rng = Rng(0x1e_7a11_c0de_0017u64);
    fn mk(rng: &mut Rng, nsyms: usize) -> Vec<u16> {
        let mut c = vec![0u16; nsyms + 1];
        let mut prev = 32768i32;
        for e in c.iter_mut().take(nsyms - 1) {
            let v = (prev - 1 - (rng.next() % 300) as i32).max(1);
            *e = v as u16;
            prev = v;
        }
        c
    }
    for _ in 0..200_000 {
        let interintra_allowed = rng.next().is_multiple_of(2);
        let interintra = (rng.next() % 2) as i32;
        let ii_mode = (rng.next() % 4) as i32; // INTERINTRA_MODES
        let wedge_used_ii = rng.next().is_multiple_of(2);
        let use_wedge_ii = (rng.next() % 2) as i32;
        let ii_wedge_index = (rng.next() % 16) as i32;
        let motion_mode_present = rng.next().is_multiple_of(2);
        let last_motion_mode_allowed = (rng.next() % 3) as i32;
        let motion_mode = (rng.next() % 3) as i32; // MOTION_MODES
        let has_second_ref = rng.next().is_multiple_of(2);
        let masked_used = rng.next().is_multiple_of(2);
        let comp_group_idx = (rng.next() % 2) as i32;
        let dist_wtd = rng.next().is_multiple_of(2);
        let compound_idx = (rng.next() % 2) as i32;
        let wedge_used_ct = rng.next().is_multiple_of(2);
        let comp_type = 2 + (rng.next() % 2) as i32; // WEDGE/DIFFWTD
        let ct_wedge_index = (rng.next() % 16) as i32;
        let wedge_sign = (rng.next() % 2) as i32;
        let mask_type = (rng.next() % 2) as i32;
        let interp_needed = rng.next().is_multiple_of(2);
        let is_switchable = rng.next().is_multiple_of(2);
        let enable_dual = rng.next().is_multiple_of(2);
        let f0 = (rng.next() % 3) as i32;
        let f1 = (rng.next() % 3) as i32;
        let ii: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let iim: [u16; 5] = mk(&mut rng, 4).try_into().unwrap();
        let wii: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let wix: [u16; 17] = mk(&mut rng, 16).try_into().unwrap();
        let obmc: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let mm: [u16; 4] = mk(&mut rng, 3).try_into().unwrap();
        let cgi: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let cidx: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let ct: [u16; 3] = mk(&mut rng, 2).try_into().unwrap();
        let ic0: [u16; 4] = mk(&mut rng, 3).try_into().unwrap();
        let ic1: [u16; 4] = mk(&mut rng, 3).try_into().unwrap();

        let mut enc = OdEcEnc::new();
        let (mut rii, mut riim, mut rwii, mut rwix, mut robmc, mut rmm) = (ii, iim, wii, wix, obmc, mm);
        let (mut rcgi, mut rcidx, mut rct, mut ric0, mut ric1) = (cgi, cidx, ct, ic0, ic1);
        write_inter_mode_tail(
            &mut enc, interintra_allowed, interintra, &mut rii, ii_mode, &mut riim, wedge_used_ii,
            use_wedge_ii, &mut rwii, ii_wedge_index, &mut rwix, motion_mode_present, &mut robmc,
            &mut rmm, last_motion_mode_allowed, motion_mode, has_second_ref, masked_used,
            comp_group_idx, &mut rcgi, dist_wtd, compound_idx, &mut rcidx, wedge_used_ct, comp_type,
            &mut rct, ct_wedge_index, wedge_sign, mask_type, interp_needed, is_switchable,
            enable_dual, f0, f1, &mut ric0, &mut ric1,
        );
        let got = enc.done().to_vec();

        let inp = InterTailRef {
            interintra_allowed, interintra, ii_cdf: &ii, ii_mode, ii_mode_cdf: &iim, wedge_used_ii,
            use_wedge_ii, wedge_ii_cdf: &wii, ii_wedge_index, wedge_idx_cdf: &wix, motion_mode_present,
            obmc_cdf: &obmc, mm_cdf: &mm, last_motion_mode_allowed, motion_mode, has_second_ref,
            masked_used, comp_group_idx, cgi_cdf: &cgi, dist_wtd, compound_idx, cidx_cdf: &cidx,
            wedge_used_ct, comp_type, ctype_cdf: &ct, ct_wedge_index, wedge_sign, mask_type,
            interp_needed, is_switchable, enable_dual, f0, f1, interp_cdf0: &ic0, interp_cdf1: &ic1,
        };
        let (want, o_all) = c::ref_write_inter_mode_tail(&inp);
        assert_eq!(got, want, "bytes iia={interintra_allowed} ii={interintra} mmp={motion_mode_present} h2r={has_second_ref}");
        let mut all = Vec::with_capacity(52);
        all.extend_from_slice(&rii); all.extend_from_slice(&riim); all.extend_from_slice(&rwii);
        all.extend_from_slice(&rwix); all.extend_from_slice(&robmc); all.extend_from_slice(&rmm);
        all.extend_from_slice(&rcgi); all.extend_from_slice(&rcidx); all.extend_from_slice(&rct);
        all.extend_from_slice(&ric0); all.extend_from_slice(&ric1);
        assert_eq!(all.as_slice(), &o_all[..], "adapted CDFs");
    }
}

#[test]
fn collect_neighbors_ref_counts_matches_c() {
    use aom_entropy::partition::collect_neighbors_ref_counts;
    // A neighbour is: absent, intra (rf0=0,intrabc=0), intrabc (intrabc=1,rf0=0),
    // single-ref (rf0 1..7, rf1=-1), or compound (rf0 1..7, rf1 1..7).
    // Enumerate representative neighbour states.
    let states: [(bool, bool, i32, i32); 6] = [
        (false, false, 0, -1),  // absent
        (true, false, 0, -1),   // intra (not inter)
        (true, true, 0, -1),    // intrabc
        (true, false, 3, -1),   // single-ref LAST3
        (true, false, 1, 7),    // compound LAST + ALTREF
        (true, false, 4, 5),    // compound GOLDEN + BWDREF
    ];
    for &(ha, aib, arf0, arf1) in &states {
        for &(hl, lib, lrf0, lrf1) in &states {
            let got = collect_neighbors_ref_counts(ha, aib, arf0, arf1, hl, lib, lrf0, lrf1);
            let want = c::ref_collect_neighbors_ref_counts(ha, aib, arf0, arf1, hl, lib, lrf0, lrf1);
            assert_eq!(got, want, "a=({ha},{aib},{arf0},{arf1}) l=({hl},{lib},{lrf0},{lrf1})");
        }
    }
}

#[test]
fn get_partition_subsize_matches_c() {
    use aom_entropy::partition::get_partition_subsize;
    for bsize in 0..22usize {
        for partition in 0..10i32 {
            assert_eq!(get_partition_subsize(bsize, partition), c::ref_get_partition_subsize(bsize as i32, partition), "bsize={bsize} part={partition}");
        }
        // PARTITION_INVALID
        assert_eq!(get_partition_subsize(bsize, 255), c::ref_get_partition_subsize(bsize as i32, 255), "bsize={bsize} part=INVALID");
    }
}

#[test]
fn update_ext_partition_context_matches_c() {
    use aom_entropy::partition::{get_partition_subsize, update_ext_partition_context};
    let mut rng = Rng(0x1e_c07e_c0de_0018u64);
    // square bsizes >= BLOCK_8X8 and their width in mi units.
    let sizes = [(3usize, 2i32), (6, 4), (9, 8), (12, 16), (15, 32)];
    for _ in 0..200_000 {
        let (bsize, mi_sz) = sizes[(rng.next() % 5) as usize];
        let partition = (rng.next() % 10) as i32;
        let subsize = get_partition_subsize(bsize, partition);
        if subsize == 255 {
            continue; // partition illegal for this size; write_modes_sb never emits it
        }
        // Aligned position so every stamp stays within above[64] / left[32].
        let rows = 32 / mi_sz;
        let cols = (64 / mi_sz).min(32 / mi_sz.max(1)); // keep col stamps < 32 too for the extended cases
        let mi_row = ((rng.next() % rows.max(1) as u64) as i32) * mi_sz;
        let mi_col = ((rng.next() % cols.max(1) as u64) as i32) * mi_sz;
        let mut above_in = [0i8; 64];
        let mut left_in = [0i8; 32];
        for a in above_in.iter_mut() {
            *a = (rng.next() % 32) as i8;
        }
        for l in left_in.iter_mut() {
            *l = (rng.next() % 32) as i8;
        }
        let mut a_rs = above_in;
        let mut l_rs = left_in;
        update_ext_partition_context(&mut a_rs, &mut l_rs, mi_row, mi_col, subsize as usize, bsize, partition);
        let (a_c, l_c) = c::ref_update_ext_partition_context(mi_row, mi_col, subsize, bsize as i32, partition, &above_in, &left_in);
        assert_eq!(a_rs, a_c, "above bsize={bsize} part={partition} r={mi_row} c={mi_col}");
        assert_eq!(l_rs, l_c, "left bsize={bsize} part={partition} r={mi_row} c={mi_col}");
    }
}

#[test]
fn write_partition_node_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::write_partition_node;
    let mut rng = Rng(0x1e_9d05_c0de_0019u64);
    // partition CDF length per square bsize.
    let cdf_len = |bsize: usize| -> i32 {
        if bsize == 3 { 4 } else if bsize == 15 { 8 } else { 10 }
    };
    let mk = |rng: &mut Rng, n: usize, out: &mut [u16]| {
        let mut prev = 32768i32;
        for e in out.iter_mut().take(n - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max((n) as i32);
            *e = v as u16;
            prev = v;
        }
        out[n - 1] = 0;
        out[n] = 0;
    };
    let sizes = [3usize, 6, 9, 12, 15];
    for _ in 0..200_000 {
        let bsize = sizes[(rng.next() % 5) as usize];
        // hbs in mi units; mi_row=mi_col=0 keeps every stamp within above[64]/left[32].
        let hbs = [1i32, 2, 4, 8, 16][sizes.iter().position(|&s| s == bsize).unwrap()];
        // scenario: 0 full, 1 no-rows, 2 no-cols, 3 none (edges need bsize>8X8).
        let scenario = if bsize == 3 { 0 } else { (rng.next() % 4) as i32 };
        let (mi_rows, mi_cols, partition) = match scenario {
            1 => (hbs, hbs + 1, if rng.next().is_multiple_of(2) { 1 } else { 3 }), // !hr,hc: HORZ/SPLIT
            2 => (hbs + 1, hbs, if rng.next().is_multiple_of(2) { 2 } else { 3 }), // hr,!hc: VERT/SPLIT
            3 => (hbs, hbs, 3),                                                    // !hr,!hc: SPLIT
            _ => (hbs + 1, hbs + 1, (rng.next() % cdf_len(bsize) as u64) as i32),  // full
        };
        let mut above_in = [0i8; 64];
        let mut left_in = [0i8; 32];
        for a in above_in.iter_mut() { *a = (rng.next() % 32) as i8; }
        for l in left_in.iter_mut() { *l = (rng.next() % 32) as i8; }
        let mut arena_n = [[0u16; 11]; 20];
        let mut arena_f = [0u16; 220];
        for c in 0..20 {
            // ctx encodes bsl = c/4: bsl 0 (8X8) -> 4 symbols, bsl 4 (128X128) -> 8, else 10.
            // The CDF at each ctx must be sized to the cdf_len it is used with, or
            // update_cdf reads a cumulative as its count.
            let bsl = c / 4;
            let ns = if bsl == 0 { 4 } else if bsl == 4 { 8 } else { 10 };
            let mut row = [0u16; 11];
            mk(&mut rng, ns, &mut row);
            arena_n[c] = row;
            for j in 0..11 { arena_f[c * 11 + j] = row[j]; }
        }
        let mut enc = OdEcEnc::new();
        let (mut a_rs, mut l_rs, mut ar_rs) = (above_in, left_in, arena_n);
        write_partition_node(&mut enc, &mut a_rs, &mut l_rs, 0, 0, bsize, partition, mi_rows, mi_cols, &mut ar_rs);
        let got = enc.done().to_vec();
        let (want, a_c, l_c, ar_c) = c::ref_write_partition_node(&above_in, &left_in, 0, 0, bsize as i32, partition, mi_rows, mi_cols, &arena_f);
        assert_eq!(got, want, "bytes bsize={bsize} scen={scenario} part={partition} rows={mi_rows} cols={mi_cols}");
        assert_eq!(a_rs, a_c, "above bsize={bsize} scen={scenario}");
        assert_eq!(l_rs, l_c, "left bsize={bsize} scen={scenario}");
        let ar_rs_f: [u16; 220] = core::array::from_fn(|i| ar_rs[i / 11][i % 11]);
        assert_eq!(ar_rs_f, ar_c, "arena bsize={bsize} scen={scenario}");
    }
}

#[test]
fn write_modes_sb_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{get_partition_subsize, write_modes_sb};
    let mut rng = Rng(0x1e_9d05_c0de_001au64);
    // Generate a fully-in-frame partition tree (pre-order) for a square bsize.
    fn gen(rng: &mut Rng, bsize: usize, out: &mut Vec<i8>) {
        let cdf_len = if bsize == 3 { 4 } else if bsize == 15 { 8 } else { 10 };
        let p = (rng.next() % cdf_len as u64) as i32;
        out.push(p as i8);
        if p == 3 && bsize > 3 {
            let sub = get_partition_subsize(bsize, 3) as usize;
            for _ in 0..4 {
                gen(rng, sub, out);
            }
        }
    }
    let mk = |rng: &mut Rng, n: usize, out: &mut [u16]| {
        let mut prev = 32768i32;
        for e in out.iter_mut().take(n - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max(n as i32);
            *e = v as u16;
            prev = v;
        }
        out[n - 1] = 0;
        out[n] = 0;
    };
    for _ in 0..60_000 {
        let mut tree = Vec::new();
        gen(&mut rng, 12, &mut tree); // start at BLOCK_64X64
        let tree_i8: Vec<i8> = tree.clone();
        let mut above_in = [0i8; 64];
        let mut left_in = [0i8; 32];
        for a in above_in.iter_mut() {
            *a = (rng.next() % 32) as i8;
        }
        for l in left_in.iter_mut() {
            *l = (rng.next() % 32) as i8;
        }
        let mut arena_n = [[0u16; 11]; 20];
        let mut arena_f = [0u16; 220];
        for c in 0..20 {
            let bsl = c / 4;
            let ns = if bsl == 0 { 4 } else if bsl == 4 { 8 } else { 10 };
            let mut row = [0u16; 11];
            mk(&mut rng, ns, &mut row);
            arena_n[c] = row;
            for j in 0..11 {
                arena_f[c * 11 + j] = row[j];
            }
        }
        let mut enc = OdEcEnc::new();
        let (mut a_rs, mut l_rs, mut ar_rs) = (above_in, left_in, arena_n);
        let consumed = write_modes_sb(&mut enc, &mut a_rs, &mut l_rs, &mut ar_rs, &tree_i8, 0, 0, 12);
        let got = enc.done().to_vec();
        let (want, a_c, l_c, ar_c, consumed_c) =
            c::ref_write_modes_sb(&above_in, &left_in, 0, 0, 12, &tree_i8, &arena_f);
        assert_eq!(got, want, "bytes tree_len={}", tree.len());
        assert_eq!(consumed as i32, consumed_c, "tree consumed");
        assert_eq!(a_rs, a_c, "above");
        assert_eq!(l_rs, l_c, "left");
        let ar_rs_f: [u16; 220] = core::array::from_fn(|i| ar_rs[i / 11][i % 11]);
        assert_eq!(ar_rs_f, ar_c, "arena");
    }
}

#[test]
fn write_modes_tile_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{get_partition_subsize, write_modes_tile};
    let mut rng = Rng(0x1e_7115_c0de_001bu64);
    fn gen(rng: &mut Rng, bsize: usize, out: &mut Vec<i8>) {
        let cdf_len = if bsize == 3 { 4 } else if bsize == 15 { 8 } else { 10 };
        let p = (rng.next() % cdf_len as u64) as i32;
        out.push(p as i8);
        if p == 3 && bsize > 3 {
            let sub = get_partition_subsize(bsize, 3) as usize;
            for _ in 0..4 {
                gen(rng, sub, out);
            }
        }
    }
    let mk = |rng: &mut Rng, n: usize, out: &mut [u16]| {
        let mut prev = 32768i32;
        for e in out.iter_mut().take(n - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max(n as i32);
            *e = v as u16;
            prev = v;
        }
        out[n - 1] = 0;
        out[n] = 0;
    };
    for _ in 0..30_000 {
        let n_sb_rows = 1 + (rng.next() % 3) as i32;
        let n_sb_cols = 1 + (rng.next() % 3) as i32;
        // one tree per SB (row-major), concatenated.
        let mut tree = Vec::new();
        for _ in 0..(n_sb_rows * n_sb_cols) {
            gen(&mut rng, 12, &mut tree); // BLOCK_64X64 SBs
        }
        let tree_i8: Vec<i8> = tree.clone();
        let mut arena_n = [[0u16; 11]; 20];
        let mut arena_f = [0u16; 220];
        for c in 0..20 {
            let bsl = c / 4;
            let ns = if bsl == 0 { 4 } else if bsl == 4 { 8 } else { 10 };
            let mut row = [0u16; 11];
            mk(&mut rng, ns, &mut row);
            arena_n[c] = row;
            for j in 0..11 {
                arena_f[c * 11 + j] = row[j];
            }
        }
        let mut enc = OdEcEnc::new();
        let mut above_rs = [0i8; 128];
        let mut ar_rs = arena_n;
        let consumed = write_modes_tile(&mut enc, &mut above_rs, &mut ar_rs, &tree_i8, n_sb_rows, n_sb_cols, 16, 12);
        let got = enc.done().to_vec();
        let (want, a_c, ar_c, consumed_c) =
            c::ref_write_modes_tile(n_sb_rows, n_sb_cols, 16, 12, &tree_i8, &arena_f);
        assert_eq!(got, want, "bytes {}x{} sbs, tree_len={}", n_sb_rows, n_sb_cols, tree.len());
        assert_eq!(consumed as i32, consumed_c, "consumed");
        assert_eq!(above_rs, a_c, "above");
        let ar_rs_f: [u16; 220] = core::array::from_fn(|i| ar_rs[i / 11][i % 11]);
        assert_eq!(ar_rs_f, ar_c, "arena");
    }
}

#[test]
fn write_inter_txfm_size_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{get_vartx_max_txsize_luma, write_inter_txfm_size};
    // inter block sizes >= 8x8 (var-tx applies).
    let bsizes: [usize; 13] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 19];
    let nbr: [u8; 6] = [0, 4, 8, 16, 32, 64];
    let mut rng = Rng(0x1e_7f5c_c0de_001cu64);
    for &bsize in &bsizes {
        let max_tx = get_vartx_max_txsize_luma(bsize);
        for _ in 0..6000 {
            let mut its = [0u8; 16];
            for v in its.iter_mut() {
                *v = (rng.next() % 19) as u8;
            }
            let its_usize: [usize; 16] = core::array::from_fn(|i| its[i] as usize);
            let mut above = [0u8; 32];
            let mut left = [0u8; 32];
            for i in 0..32 {
                above[i] = nbr[(rng.next() % 6) as usize];
                left[i] = nbr[(rng.next() % 6) as usize];
            }
            let re = -((rng.next() % 4) as i32) * 32;
            let be = -((rng.next() % 4) as i32) * 32;
            let mut cdf = [[0u16; 3]; 21];
            let mut cflat = [0u16; 63];
            for c in 0..21 {
                let p = 1 + (rng.next() % 32766) as u16;
                cdf[c] = [p, 0, 0];
                cflat[c * 3] = p;
            }
            let mut enc = OdEcEnc::new();
            let mut a_rs = above;
            let mut l_rs = left;
            let mut cdf_rs = cdf;
            write_inter_txfm_size(&mut enc, &mut cdf_rs, bsize, &its_usize, re, be, &mut a_rs, &mut l_rs, max_tx);
            let got = enc.done().to_vec();
            let (want, ao, lo, co) =
                c::ref_write_inter_txfm_size(bsize as i32, max_tx as i32, &its, re, be, &above, &left, &cflat);
            assert_eq!(got, want, "bytes bsize={bsize} max_tx={max_tx} re={re} be={be}");
            assert_eq!(a_rs, ao, "above bsize={bsize}");
            assert_eq!(l_rs, lo, "left bsize={bsize}");
            let co_nested: [[u16; 3]; 21] = core::array::from_fn(|c| [co[c * 3], co[c * 3 + 1], co[c * 3 + 2]]);
            assert_eq!(cdf_rs, co_nested, "cdf bsize={bsize}");
        }
    }
}

#[test]
fn pack_map_tokens_matches_c() {
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::pack_map_tokens;
    let mut rng = Rng(0x1e_9a70_c0de_001du64);
    // n-symbol CDF (count at [n]); the map CDF for palette size n uses n symbols.
    let mk = |rng: &mut Rng, n: usize, out: &mut [u16; 9]| {
        let mut prev = 32768i32;
        for e in out.iter_mut().take(n - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max(n as i32);
            *e = v as u16;
            prev = v;
        }
        out[n - 1] = 0;
        out[n] = 0;
    };
    for _ in 0..300_000 {
        let n = 2 + (rng.next() % 7) as i32; // palette size 2..8
        let num = 1 + (rng.next() % 40) as usize; // number of color indices (>=1)
        let tokens: Vec<i32> = (0..num).map(|_| (rng.next() % n as u64) as i32).collect();
        let tokens_u8: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
        let color_ctxs: Vec<usize> = (0..num).map(|_| (rng.next() % 5) as usize).collect();
        let color_ctxs_u8: Vec<u8> = color_ctxs.iter().map(|&c| c as u8).collect();
        let mut map_n = [[0u16; 9]; 5];
        let mut map_f = [0u16; 45];
        for ctx in 0..5 {
            let mut row = [0u16; 9];
            mk(&mut rng, n as usize, &mut row);
            map_n[ctx] = row;
            for j in 0..9 {
                map_f[ctx * 9 + j] = row[j];
            }
        }
        let mut enc = OdEcEnc::new();
        let mut map_rs = map_n;
        pack_map_tokens(&mut enc, n, &tokens, &color_ctxs, &mut map_rs);
        let got = enc.done().to_vec();
        let (want, mco) = c::ref_pack_map_tokens(n, &tokens_u8, &color_ctxs_u8, &map_f);
        assert_eq!(got, want, "bytes n={n} num={num}");
        let map_rs_f: [u16; 45] = core::array::from_fn(|i| map_rs[i / 9][i % 9]);
        assert_eq!(map_rs_f, mco, "map_cdf n={n} num={num}");
    }
}

#[test]
fn read_partition_roundtrips_write_partition() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{read_partition, write_partition};
    let mut rng = Rng(0x1e_de00_c0de_001eu64);
    let mk = |rng: &mut Rng, n: usize, out: &mut [u16]| {
        let mut prev = 32768i32;
        for e in out.iter_mut().take(n - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max(n as i32);
            *e = v as u16;
            prev = v;
        }
        out[n - 1] = 0;
        out[n] = 0;
    };
    // square bsizes >= 8x8 and their cdf_len.
    let sizes: [(usize, usize); 5] = [(3, 4), (6, 10), (9, 10), (12, 10), (15, 8)];
    for _ in 0..300_000 {
        let (bsize, cdf_len) = sizes[(rng.next() % 5) as usize];
        // scenario: 0 full, 1 !rows, 2 !cols, 3 neither (edges need bsize>8X8).
        let scenario = if bsize == 3 { 0 } else { (rng.next() % 4) as i32 };
        let (has_rows, has_cols, p) = match scenario {
            1 => (false, true, if rng.next().is_multiple_of(2) { 1 } else { 3 }), // HORZ/SPLIT
            2 => (true, false, if rng.next().is_multiple_of(2) { 2 } else { 3 }), // VERT/SPLIT
            3 => (false, false, 3),                                              // SPLIT
            _ => (true, true, (rng.next() % cdf_len as u64) as i32),             // full
        };
        let mut cdf0 = [0u16; 11];
        mk(&mut rng, cdf_len, &mut cdf0);

        // encode
        let mut enc = OdEcEnc::new();
        let mut cdf_e = cdf0;
        write_partition(&mut enc, &mut cdf_e, cdf_len, p, has_rows, has_cols, bsize);
        let bytes = enc.done().to_vec();

        // decode from the same initial CDF
        let mut dec = OdEcDec::new(&bytes);
        let mut cdf_d = cdf0;
        let p_dec = read_partition(&mut dec, &mut cdf_d, cdf_len, has_rows, has_cols, bsize);

        assert_eq!(p_dec, p, "partition roundtrip bsize={bsize} scen={scenario} p={p}");
        assert_eq!(cdf_e, cdf_d, "adapted CDF roundtrip bsize={bsize} scen={scenario}");
    }
}

#[test]
fn decode_mode_info_symbols_roundtrip() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        read_angle_delta, read_inter_compound_mode, read_intra_uv_mode, read_intra_y_mode,
        read_skip, write_angle_delta, write_inter_compound_mode, write_intra_uv_mode,
        write_intra_y_mode_kf, write_skip,
    };
    let mut rng = Rng(0x1e_de5c_c0de_001fu64);
    let mk = |rng: &mut Rng, n: usize, out: &mut [u16]| {
        let mut prev = 32768i32;
        for e in out.iter_mut().take(n - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max(n as i32);
            *e = v as u16;
            prev = v;
        }
        out[n - 1] = 0;
        out[n] = 0;
    };
    for _ in 0..100_000 {
        // skip
        {
            let ssa = rng.next().is_multiple_of(3);
            let skip = (rng.next() % 2) as i32;
            let mut c = [0u16; 3];
            mk(&mut rng, 2, &mut c);
            let mut enc = OdEcEnc::new();
            let mut ce = c;
            let r = write_skip(&mut enc, &mut ce, ssa, skip);
            let bytes = enc.done().to_vec();
            let mut dec = OdEcDec::new(&bytes);
            let mut cd = c;
            let d = read_skip(&mut dec, &mut cd, ssa);
            assert_eq!(d, r, "skip roundtrip ssa={ssa} skip={skip}");
            assert_eq!(ce, cd, "skip cdf");
        }
        // intra Y mode (13 symbols)
        {
            let mode = (rng.next() % 13) as i32;
            let mut c = [0u16; 14];
            mk(&mut rng, 13, &mut c);
            let mut enc = OdEcEnc::new();
            let mut ce = c;
            write_intra_y_mode_kf(&mut enc, &mut ce, mode);
            let bytes = enc.done().to_vec();
            let mut dec = OdEcDec::new(&bytes);
            let mut cd = c;
            let d = read_intra_y_mode(&mut dec, &mut cd);
            assert_eq!(d, mode, "y_mode roundtrip");
            assert_eq!(ce, cd, "y_mode cdf");
        }
        // intra UV mode (13/14 symbols)
        {
            let cfl = rng.next().is_multiple_of(2);
            let n = if cfl { 14 } else { 13 };
            let uv = (rng.next() % n as u64) as i32;
            let mut c = [0u16; 15];
            mk(&mut rng, n, &mut c);
            let mut enc = OdEcEnc::new();
            let mut ce = c;
            write_intra_uv_mode(&mut enc, &mut ce, uv, cfl);
            let bytes = enc.done().to_vec();
            let mut dec = OdEcDec::new(&bytes);
            let mut cd = c;
            let d = read_intra_uv_mode(&mut dec, &mut cd, cfl);
            assert_eq!(d, uv, "uv_mode roundtrip cfl={cfl}");
            assert_eq!(ce, cd, "uv_mode cdf");
        }
        // inter compound mode (8 symbols, offset NEAREST_NEARESTMV=17)
        {
            let mode = 17 + (rng.next() % 8) as i32;
            let mut c = [0u16; 9];
            mk(&mut rng, 8, &mut c);
            let mut enc = OdEcEnc::new();
            let mut ce = c;
            write_inter_compound_mode(&mut enc, &mut ce, mode);
            let bytes = enc.done().to_vec();
            let mut dec = OdEcDec::new(&bytes);
            let mut cd = c;
            let d = read_inter_compound_mode(&mut dec, &mut cd);
            assert_eq!(d, mode, "compound_mode roundtrip");
            assert_eq!(ce, cd, "compound_mode cdf");
        }
        // angle delta (7 symbols, offset -3)
        {
            let ad = (rng.next() % 7) as i32 - 3;
            let mut c = [0u16; 8];
            mk(&mut rng, 7, &mut c);
            let mut enc = OdEcEnc::new();
            let mut ce = c;
            write_angle_delta(&mut enc, &mut ce, ad);
            let bytes = enc.done().to_vec();
            let mut dec = OdEcDec::new(&bytes);
            let mut cd = c;
            let d = read_angle_delta(&mut dec, &mut cd);
            assert_eq!(d, ad, "angle_delta roundtrip");
            assert_eq!(ce, cd, "angle_delta cdf");
        }
    }
}

#[test]
fn read_inter_mode_roundtrips() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{read_inter_mode, write_inter_mode};
    let mut rng = Rng(0x1e_1de1_c0de_0020u64);
    let mk2 = |rng: &mut Rng, n: usize, flat: &mut [[u16; 3]]| {
        for row in flat.iter_mut().take(n) {
            let p = 1 + (rng.next() % 32766) as u16;
            *row = [p, 0, 0];
        }
    };
    let modes = [13i32, 14, 15, 16]; // NEARESTMV/NEARMV/GLOBALMV/NEWMV
    for _ in 0..300_000 {
        let mode = modes[(rng.next() % 4) as usize];
        let newmv_ctx = (rng.next() % 6) as i32;
        let zeromv_bit = (rng.next() % 2) as i32;
        let refmv_ctx = (rng.next() % 6) as i32;
        let mode_ctx = newmv_ctx | (zeromv_bit << 3) | (refmv_ctx << 4);
        let mut nm = [[0u16; 3]; 6];
        let mut zm = [[0u16; 3]; 2];
        let mut rm = [[0u16; 3]; 6];
        mk2(&mut rng, 6, &mut nm);
        mk2(&mut rng, 2, &mut zm);
        mk2(&mut rng, 6, &mut rm);
        let mut enc = OdEcEnc::new();
        let (mut nme, mut zme, mut rme) = (nm, zm, rm);
        write_inter_mode(&mut enc, &mut nme, &mut zme, &mut rme, mode, mode_ctx);
        let bytes = enc.done().to_vec();
        let mut dec = OdEcDec::new(&bytes);
        let (mut nmd, mut zmd, mut rmd) = (nm, zm, rm);
        let d = read_inter_mode(&mut dec, &mut nmd, &mut zmd, &mut rmd, mode_ctx);
        assert_eq!(d, mode, "inter_mode roundtrip mode={mode} ctx={mode_ctx}");
        assert_eq!(nme, nmd, "newmv cdf");
        assert_eq!(zme, zmd, "zeromv cdf");
        assert_eq!(rme, rmd, "refmv cdf");
    }
}

#[test]
fn read_mv_roundtrips_encode_mv() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{encode_mv, read_mv};
    let mut rng = Rng(0x1e_11de_c0de_0021u64);
    let mk_ibc = |rng: &mut Rng, ns: usize, out: &mut [u16]| {
        let mut vals = [0i32; 11];
        for v in vals.iter_mut().take(ns - 1) { *v = 1 + (rng.next() % 32766) as i32; }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 { let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32); out[i] = v as u16; prev = v; }
        out[ns - 1] = 0; out[ns] = 0;
    };
    let mk_comp = |rng: &mut Rng| -> [u16; 69] {
        let mut c = [0u16; 69];
        mk_ibc(rng, 2, &mut c[0..3]); mk_ibc(rng, 11, &mut c[3..15]); mk_ibc(rng, 2, &mut c[15..18]);
        for i in 0..10 { let o = 18 + i * 3; mk_ibc(rng, 2, &mut c[o..o + 3]); }
        for i in 0..2 { let o = 48 + i * 5; mk_ibc(rng, 4, &mut c[o..o + 5]); }
        mk_ibc(rng, 4, &mut c[58..63]); mk_ibc(rng, 2, &mut c[63..66]); mk_ibc(rng, 2, &mut c[66..69]);
        c
    };
    for _ in 0..300_000 {
        // full precision so both fr + hp are coded -> exact roundtrip. diffs in class range.
        let usehp = 1;
        let dr = (rng.next() % 32769) as i32 - 16384;
        let dc = (rng.next() % 32769) as i32 - 16384;
        let mut joints = [0u16; 5];
        mk_ibc(&mut rng, 4, &mut joints);
        let comp0 = mk_comp(&mut rng);
        let comp1 = mk_comp(&mut rng);
        let mut enc = OdEcEnc::new();
        let (mut je, mut c0e, mut c1e) = (joints, comp0, comp1);
        encode_mv(&mut enc, &mut je, &mut c0e, &mut c1e, dr, dc, usehp);
        let bytes = enc.done().to_vec();
        let mut dec = OdEcDec::new(&bytes);
        let (mut jd, mut c0d, mut c1d) = (joints, comp0, comp1);
        let (rr, rc) = read_mv(&mut dec, &mut jd, &mut c0d, &mut c1d, usehp);
        assert_eq!((rr, rc), (dr, dc), "mv roundtrip dr={dr} dc={dc}");
        assert_eq!(je, jd, "joints cdf");
        assert_eq!(c0e, c0d, "comp0 cdf");
        assert_eq!(c1e, c1d, "comp1 cdf");
    }
}

#[test]
fn read_drl_idx_roundtrips_write_drl_idx() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{read_drl_idx, write_drl_idx};
    let mut rng = Rng(0x1ed1_2c0d_e003_3000u64);
    // reachable ref_mv_idx given (mode, count): NEWMV/NEW_NEWMV walk idx 0..2,
    // have-nearmv modes walk idx 1..3; all values 0..=max are reachable.
    let max_reach = |mode: i32, count: i32| -> i32 {
        let new_mv = mode == 16 || mode == 24;
        let nearmv = mode == 14 || mode == 18 || mode == 21 || mode == 22;
        if new_mv {
            if count <= 1 { 0 } else if count == 2 { 1 } else { 2 }
        } else if nearmv {
            if count <= 2 { 0 } else if count == 3 { 1 } else { 2 }
        } else {
            0
        }
    };
    let modes = [16i32, 24, 14, 18, 21, 22, 15, 13, 17]; // new/near + GLOBALMV/NEARESTMV/NEAREST_NEARESTMV (no-drl)
    let mk_ibc = |rng: &mut Rng, out: &mut [u16; 3]| {
        // 2-symbol cdf: cdf[0] in [1,32767], cdf[1]=0, cdf[2]=count(0)
        out[0] = 1 + (rng.next() % 32766) as u16;
        out[1] = 0;
        out[2] = 0;
    };
    for _ in 0..300_000 {
        let mode = modes[(rng.next() % modes.len() as u64) as usize];
        let count = 1 + (rng.next() % 4) as i32; // 1..=4
        let max = max_reach(mode, count);
        let ref_mv_idx = (rng.next() % (max as u64 + 1)) as i32;
        // weight[0..4] spanning REF_CAT_LEVEL=640 so ctx varies across {0,1,2}
        let weight: [u16; 4] = [
            (rng.next() % 1280) as u16,
            (rng.next() % 1280) as u16,
            (rng.next() % 1280) as u16,
            (rng.next() % 1280) as u16,
        ];
        let mut cdf = [[0u16; 3]; 3];
        for c in cdf.iter_mut() {
            mk_ibc(&mut rng, c);
        }
        let mut enc = OdEcEnc::new();
        let mut ce = cdf;
        write_drl_idx(&mut enc, &mut ce, mode, ref_mv_idx, count, &weight);
        let bytes = enc.done().to_vec();
        let mut dec = OdEcDec::new(&bytes);
        let mut cd = cdf;
        let got = read_drl_idx(&mut dec, &mut cd, mode, count, &weight);
        assert_eq!(got, ref_mv_idx, "mode={mode} count={count} idx={ref_mv_idx}");
        assert_eq!(ce, cd, "drl cdf mode={mode} count={count} idx={ref_mv_idx}");
    }
}

#[test]
fn read_ref_frames_roundtrips_write_ref_frames() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{read_ref_frames, write_ref_frames};
    let mut rng = Rng(0x1ea5_5c0d_e004_1000u64);
    let mk_ibc2 = |rng: &mut Rng, c: &mut [u16; 3]| {
        c[0] = 1 + (rng.next() % 32766) as u16;
        c[1] = 0;
        c[2] = 0;
    };
    // seg-active: encoder codes nothing; decoder returns the no-op sentinel, cdfs untouched.
    {
        let mut cdf = [[0u16; 3]; 16];
        for c in cdf.iter_mut() { mk_ibc2(&mut rng, c); }
        let mut enc = OdEcEnc::new();
        let mut ce = cdf;
        write_ref_frames(&mut enc, &mut ce, true, false, true, true, false, -1, 3, -1);
        let bytes = enc.done().to_vec();
        let mut dec = OdEcDec::new(&bytes);
        let mut cd = cdf;
        let got = read_ref_frames(&mut dec, &mut cd, true, false, true, true);
        assert_eq!(got, (false, -1, -1, -1), "seg-active no-op");
        assert_eq!(ce, cd, "seg-active cdfs untouched");
    }
    for _ in 0..300_000 {
        let select = rng.next() & 1 == 1;
        let comp_allowed = rng.next() & 1 == 1;
        let can_compound = select && comp_allowed;
        let is_compound = can_compound && (rng.next() & 1 == 1);
        let (comp_ref_type, ref0, ref1) = if !is_compound {
            (-1, 1 + (rng.next() % 7) as i32, -1) // single: LAST..ALTREF
        } else if rng.next() & 1 == 0 {
            let uni = [(5i32, 7i32), (1, 2), (1, 3), (1, 4)];
            let (r0, r1) = uni[(rng.next() % 4) as usize];
            (0, r0, r1) // UNIDIR reachable pairs
        } else {
            let r0 = [1i32, 2, 3, 4][(rng.next() % 4) as usize];
            let r1 = [5i32, 6, 7][(rng.next() % 3) as usize];
            (1, r0, r1) // BIDIR: fwd x bwd
        };
        let mut cdf = [[0u16; 3]; 16];
        for c in cdf.iter_mut() { mk_ibc2(&mut rng, c); }
        let mut enc = OdEcEnc::new();
        let mut ce = cdf;
        write_ref_frames(&mut enc, &mut ce, false, false, select, comp_allowed, is_compound, comp_ref_type, ref0, ref1);
        let bytes = enc.done().to_vec();
        let mut dec = OdEcDec::new(&bytes);
        let mut cd = cdf;
        let (gc, gcrt, gr0, gr1) = read_ref_frames(&mut dec, &mut cd, false, false, select, comp_allowed);
        assert_eq!(gc, is_compound, "is_compound sel={select} allow={comp_allowed}");
        if is_compound {
            assert_eq!((gcrt, gr0, gr1), (comp_ref_type, ref0, ref1), "compound refs crt={comp_ref_type} r=({ref0},{ref1})");
        } else {
            assert_eq!(gr0, ref0, "single ref0={ref0}");
        }
        assert_eq!(ce, cd, "ref cdfs comp={is_compound} r=({ref0},{ref1})");
    }
}

#[test]
fn read_selected_tx_size_roundtrips() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{read_selected_tx_size, write_selected_tx_size};
    let mut rng = Rng(0x1e_7451_c0de_0050u64);
    let mk = |rng: &mut Rng, ns: usize, out: &mut [u16]| {
        let mut vals = [0i32; 4];
        for v in vals.iter_mut().take(ns - 1) { *v = 1 + (rng.next() % 32766) as i32; }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 { let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32); out[i] = v as u16; prev = v; }
        out[ns - 1] = 0; out[ns] = 0;
    };
    for _ in 0..200_000 {
        let max_depths = (rng.next() % 2 + 1) as usize; // MAX_TX_DEPTH 1..=2
        let ns = max_depths + 1;
        let bsize = (rng.next() % 4) as usize; // 0 => no signal
        let depth = if bsize > 0 { (rng.next() % ns as u64) as i32 } else { 0 };
        let mut cdf = vec![0u16; ns + 1];
        mk(&mut rng, ns, &mut cdf);
        let mut enc = OdEcEnc::new();
        let mut ce = cdf.clone();
        write_selected_tx_size(&mut enc, &mut ce, bsize, depth, max_depths);
        let bytes = enc.done().to_vec();
        let mut dec = OdEcDec::new(&bytes);
        let mut cd = cdf.clone();
        let got = read_selected_tx_size(&mut dec, &mut cd, bsize, max_depths);
        assert_eq!(got, depth, "tx depth bsize={bsize} md={max_depths}");
        assert_eq!(ce, cd, "tx cdf");
    }
}

#[test]
fn read_filter_intra_mode_info_roundtrips() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{read_filter_intra_mode_info, write_filter_intra_mode_info};
    let mut rng = Rng(0x1e_f1a5_c0de_0051u64);
    let mk = |rng: &mut Rng, ns: usize, out: &mut [u16]| {
        let mut vals = [0i32; 8];
        for v in vals.iter_mut().take(ns - 1) { *v = 1 + (rng.next() % 32766) as i32; }
        vals[..ns - 1].sort_unstable();
        vals[..ns - 1].reverse();
        let mut prev = 32768i32;
        for i in 0..ns - 1 { let v = vals[i].min(prev - 1).max((ns - 1 - i) as i32); out[i] = v as u16; prev = v; }
        out[ns - 1] = 0; out[ns] = 0;
    };
    for _ in 0..200_000 {
        let allowed = rng.next() & 1 == 1;
        let use_fi = (rng.next() & 1) as i32;
        let mode = if use_fi != 0 { (rng.next() % 5) as i32 } else { 0 };
        let mut ucdf = [0u16; 3];
        let mut mcdf = [0u16; 6];
        mk(&mut rng, 2, &mut ucdf);
        mk(&mut rng, 5, &mut mcdf);
        let mut enc = OdEcEnc::new();
        let (mut ue, mut me) = (ucdf, mcdf);
        write_filter_intra_mode_info(&mut enc, &mut ue, &mut me, allowed, use_fi, mode);
        let bytes = enc.done().to_vec();
        let mut dec = OdEcDec::new(&bytes);
        let (mut ud, mut md) = (ucdf, mcdf);
        let (gu, gm) = read_filter_intra_mode_info(&mut dec, &mut ud, &mut md, allowed);
        if allowed {
            assert_eq!(gu, use_fi, "use_fi");
            if use_fi != 0 { assert_eq!(gm, mode, "fi mode"); }
        } else {
            assert_eq!((gu, gm), (0, 0), "not allowed");
        }
        assert_eq!(ue, ud, "use cdf");
        assert_eq!(me, md, "mode cdf");
    }
}

#[test]
fn read_tx_size_vartx_roundtrips_write() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{read_tx_size_vartx, write_tx_size_vartx};
    // max_txsize_rect_lookup[BLOCK_SIZES_ALL] — the block's top var-tx size.
    const MAX_TX_RECT: [usize; 22] =
        [0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18];
    let bsizes: [usize; 13] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 19];
    let nbr: [u8; 6] = [0, 4, 8, 16, 32, 64];
    let mut rng = Rng(0x7a12_b0de_5a1e_0060);
    for &bsize in &bsizes {
        let top = MAX_TX_RECT[bsize];
        for _ in 0..6000 {
            let its_usize: [usize; 16] = core::array::from_fn(|_| (rng.next() % 19) as usize);
            let mut above = [0u8; 32];
            let mut left = [0u8; 32];
            for i in 0..32 {
                above[i] = nbr[(rng.next() % 6) as usize];
                left[i] = nbr[(rng.next() % 6) as usize];
            }
            let re = -((rng.next() % 4) as i32) * 32;
            let be = -((rng.next() % 4) as i32) * 32;
            let mut cdf = [[0u16; 3]; 21];
            for c in cdf.iter_mut() {
                *c = [1 + (rng.next() % 32766) as u16, 0, 0];
            }
            // encode from arbitrary its
            let mut enc = OdEcEnc::new();
            let (mut a1, mut l1, mut c1) = (above, left, cdf);
            write_tx_size_vartx(&mut enc, &mut c1, bsize, &its_usize, re, be, &mut a1, &mut l1, top, 0, 0, 0);
            let bits1 = enc.done().to_vec();
            // decode: reconstruct its_dec + ctx + adapted cdf
            let mut dec = OdEcDec::new(&bits1);
            let (mut ad, mut ld, mut cd) = (above, left, cdf);
            let mut its_dec = [0usize; 16];
            read_tx_size_vartx(&mut dec, &mut cd, bsize, &mut its_dec, re, be, &mut ad, &mut ld, top, 0, 0, 0);
            // same tree walk => identical ctx + identical CDF adaptation
            assert_eq!(ad, a1, "above bsize={bsize} its={its_usize:?}");
            assert_eq!(ld, l1, "left bsize={bsize} its={its_usize:?}");
            assert_eq!(cd, c1, "cdf bsize={bsize} its={its_usize:?}");
            // reconstructed its re-encodes to the identical bitstream + ctx
            let mut enc2 = OdEcEnc::new();
            let (mut a2, mut l2, mut c2) = (above, left, cdf);
            write_tx_size_vartx(&mut enc2, &mut c2, bsize, &its_dec, re, be, &mut a2, &mut l2, top, 0, 0, 0);
            let bits2 = enc2.done().to_vec();
            assert_eq!(bits2, bits1, "re-encode bits bsize={bsize} its_dec={its_dec:?}");
            assert_eq!((a2, l2, c2), (a1, l1, c1), "re-encode ctx/cdf bsize={bsize}");
        }
    }
}

#[test]
fn read_map_tokens_roundtrips_pack() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{pack_map_tokens, read_map_tokens};
    let mut rng = Rng(0x1eba_15c0_de00_7000u64);
    // n-symbol icdf into a [u16;9]: out[0..n-1] descending in (0,32768), out[n-1]=0, out[n]=count=0.
    let mk = |rng: &mut Rng, n: usize, out: &mut [u16; 9]| {
        *out = [0u16; 9];
        let mut vals = [0i32; 8];
        for v in vals.iter_mut().take(n - 1) { *v = 1 + (rng.next() % 32766) as i32; }
        vals[..n - 1].sort_unstable();
        vals[..n - 1].reverse();
        let mut prev = 32768i32;
        for k in 0..n - 1 {
            let v = vals[k].min(prev - 1).max((n - 1 - k) as i32);
            out[k] = v as u16;
            prev = v;
        }
        out[n - 1] = 0;
        out[n] = 0;
    };
    for _ in 0..200_000 {
        let n = 2 + (rng.next() % 7) as i32; // palette size 2..=8
        let len = 1 + (rng.next() % 64) as usize; // map size 1..=64
        let mut tokens = vec![0i32; len];
        let mut color_ctxs = vec![0usize; len];
        tokens[0] = (rng.next() % n as u64) as i32;
        for i in 1..len {
            tokens[i] = (rng.next() % n as u64) as i32;
            color_ctxs[i] = (rng.next() % 5) as usize;
        }
        let mut map_cdf = [[0u16; 9]; 5];
        for c in map_cdf.iter_mut() { mk(&mut rng, n as usize, c); }
        let mut enc = OdEcEnc::new();
        let mut ce = map_cdf;
        pack_map_tokens(&mut enc, n, &tokens, &color_ctxs, &mut ce);
        let bytes = enc.done().to_vec();
        let mut dec = OdEcDec::new(&bytes);
        let mut cd = map_cdf;
        let mut got = vec![0i32; len];
        read_map_tokens(&mut dec, n, &color_ctxs, &mut cd, &mut got);
        assert_eq!(got, tokens, "tokens n={n} len={len}");
        assert_eq!(ce, cd, "map cdf n={n} len={len}");
    }
}

#[test]
fn read_delta_palette_colors_roundtrips() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{delta_encode_palette_colors, read_delta_palette_colors};
    let mut rng = Rng(0x1e_de17_a000_0080u64);
    for _ in 0..200_000 {
        let bit_depth = [8i32, 10, 12][(rng.next() % 3) as usize];
        let max_val = 1i32 << bit_depth;
        let min_val = (rng.next() % 2) as i32; // 1 luma, 0 chroma-U
        let num = 1 + (rng.next() % 8) as usize; // 1..=8
        // strictly-ascending colours in [0, max_val): sorted base + index.
        let mut base = vec![0i32; num];
        for b in base.iter_mut() { *b = (rng.next() % (max_val - num as i32) as u64) as i32; }
        base.sort_unstable();
        let colors: Vec<i32> = (0..num).map(|i| base[i] + i as i32).collect();
        let mut enc = OdEcEnc::new();
        delta_encode_palette_colors(&mut enc, &colors, bit_depth, min_val);
        let bytes = enc.done().to_vec();
        let mut dec = OdEcDec::new(&bytes);
        let got = read_delta_palette_colors(&mut dec, num, bit_depth, min_val);
        assert_eq!(got, colors, "bd={bit_depth} min={min_val} num={num}");
    }
}

#[test]
fn read_palette_colors_v_roundtrips() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{read_palette_colors_v, write_palette_colors_v};
    let mut rng = Rng(0x1e_c010_5f00_0081u64);
    for _ in 0..200_000 {
        let bit_depth = [8i32, 10, 12][(rng.next() % 3) as usize];
        let max_val = 1u64 << bit_depth;
        let n = 2 + (rng.next() % 7) as usize; // 2..=8
        let colors: Vec<u16> = (0..n).map(|_| (rng.next() % max_val) as u16).collect();
        let mut enc = OdEcEnc::new();
        write_palette_colors_v(&mut enc, &colors, bit_depth);
        let bytes = enc.done().to_vec();
        let mut dec = OdEcDec::new(&bytes);
        let got = read_palette_colors_v(&mut dec, n, bit_depth);
        assert_eq!(got, colors, "bd={bit_depth} n={n} colors={colors:?}");
    }
}

// ---- per-block leaf readers: is_inter / motion_mode / interp_filter / delta_q /
//      delta_lf / segment_id — roundtrips vs the C-validated writers ----

fn mk_ns_cdf(rng: &mut Rng, n: usize, out: &mut [u16]) {
    for v in out.iter_mut() {
        *v = 0;
    }
    let mut vals = [0i32; 16];
    for v in vals.iter_mut().take(n - 1) {
        *v = 1 + (rng.next() % 32766) as i32;
    }
    vals[..n - 1].sort_unstable();
    vals[..n - 1].reverse();
    let mut prev = 32768i32;
    for i in 0..n - 1 {
        let v = vals[i].min(prev - 1).max((n - 1 - i) as i32);
        out[i] = v as u16;
        prev = v;
    }
    out[n - 1] = 0;
    out[n] = 0;
}

#[test]
fn read_leaf_symbols_roundtrip() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        read_delta_lflevel, read_delta_qindex, read_is_inter, read_mb_interp_filter,
        read_motion_mode, read_segment_id, write_delta_lflevel, write_delta_qindex, write_is_inter,
        write_mb_interp_filter, write_motion_mode, write_segment_id,
    };
    let mut rng = Rng(0x1e_1eaf_c0de_00a0u64);
    for _ in 0..120_000 {
        // is_inter (coded path)
        {
            let is_inter = (rng.next() & 1) as i32;
            let mut c = [0u16; 3];
            mk_ns_cdf(&mut rng, 2, &mut c);
            let mut enc = OdEcEnc::new();
            let mut ce = c;
            write_is_inter(&mut enc, &mut ce, false, false, is_inter);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let mut cd = c;
            let got = read_is_inter(&mut dec, &mut cd, false, false);
            assert_eq!(got, is_inter, "is_inter");
            assert_eq!(ce, cd, "is_inter cdf");
        }
        // motion_mode
        {
            let last = (rng.next() % 3) as i32; // 0/1/2
            let mm = match last {
                0 => 0,
                1 => (rng.next() & 1) as i32,
                _ => (rng.next() % 3) as i32,
            };
            let mut obmc = [0u16; 3];
            let mut mmc = [0u16; 4];
            mk_ns_cdf(&mut rng, 2, &mut obmc);
            mk_ns_cdf(&mut rng, 3, &mut mmc);
            let mut enc = OdEcEnc::new();
            let (mut oe, mut me) = (obmc, mmc);
            write_motion_mode(&mut enc, &mut oe, &mut me, last, mm);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let (mut od, mut md) = (obmc, mmc);
            let got = read_motion_mode(&mut dec, &mut od, &mut md, last);
            assert_eq!(got, mm, "motion_mode last={last}");
            assert_eq!((oe, me), (od, md), "motion_mode cdf");
        }
        // interp_filter (needed + switchable)
        {
            let dual = rng.next() & 1 == 1;
            let f0 = (rng.next() % 3) as i32;
            let f1 = if dual { (rng.next() % 3) as i32 } else { f0 };
            let mut c0 = [0u16; 4];
            let mut c1 = [0u16; 4];
            mk_ns_cdf(&mut rng, 3, &mut c0);
            mk_ns_cdf(&mut rng, 3, &mut c1);
            let mut enc = OdEcEnc::new();
            let (mut e0, mut e1) = (c0, c1);
            write_mb_interp_filter(&mut enc, &mut e0, &mut e1, true, true, dual, f0, f1);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let (mut d0, mut d1) = (c0, c1);
            let (g0, g1) = read_mb_interp_filter(&mut dec, &mut d0, &mut d1, true, true, dual);
            assert_eq!((g0, g1), (f0, f1), "interp dual={dual}");
            assert_eq!((e0, e1), (d0, d1), "interp cdf");
        }
        // delta_q + delta_lf
        {
            let dq = (rng.next() % 511) as i32 - 255;
            let dl = (rng.next() % 511) as i32 - 255;
            let mut cq = [0u16; 5];
            let mut cl = [0u16; 5];
            mk_ns_cdf(&mut rng, 4, &mut cq);
            mk_ns_cdf(&mut rng, 4, &mut cl);
            let mut enc = OdEcEnc::new();
            let (mut eq, mut el) = (cq, cl);
            write_delta_qindex(&mut enc, &mut eq, dq);
            write_delta_lflevel(&mut enc, &mut el, dl);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let (mut dq_c, mut dl_c) = (cq, cl);
            let gq = read_delta_qindex(&mut dec, &mut dq_c);
            let gl = read_delta_lflevel(&mut dec, &mut dl_c);
            assert_eq!((gq, gl), (dq, dl), "delta q/lf");
            assert_eq!((eq, el), (dq_c, dl_c), "delta cdf");
        }
        // segment_id
        {
            let last = (rng.next() % 8) as i32; // last_active_segid 0..7
            let segment_id = (rng.next() % (last as u64 + 1)) as i32;
            let pred = (rng.next() % (last as u64 + 1)) as i32;
            let mut c = [0u16; 9];
            mk_ns_cdf(&mut rng, 8, &mut c);
            let mut enc = OdEcEnc::new();
            let mut ce = c;
            write_segment_id(&mut enc, &mut ce, true, true, false, segment_id, pred, last);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let mut cd = c;
            let got = read_segment_id(&mut dec, &mut cd, pred, last);
            assert_eq!(got, segment_id, "segment_id last={last} pred={pred}");
            assert_eq!(ce, cd, "segment_id cdf");
        }
    }
}

#[test]
fn read_composite_leaf_symbols_roundtrip() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        read_cdef, read_cfl_alphas, read_intrabc_info, read_skip_mode, write_cdef, write_cfl_alphas,
        write_intrabc_info, write_skip_mode,
    };
    let mut rng = Rng(0x1e_c0a1_c0de_00b0u64);
    // 69-u16 nmv component blob (sign/classes/class0/bits/fp/hp sub-CDFs).
    let mk_comp = |rng: &mut Rng| -> [u16; 69] {
        let mut c = [0u16; 69];
        mk_ns_cdf(rng, 2, &mut c[0..3]);
        mk_ns_cdf(rng, 11, &mut c[3..15]);
        mk_ns_cdf(rng, 2, &mut c[15..18]);
        for i in 0..10 {
            let o = 18 + i * 3;
            mk_ns_cdf(rng, 2, &mut c[o..o + 3]);
        }
        for i in 0..2 {
            let o = 48 + i * 5;
            mk_ns_cdf(rng, 4, &mut c[o..o + 5]);
        }
        mk_ns_cdf(rng, 4, &mut c[58..63]);
        mk_ns_cdf(rng, 2, &mut c[63..66]);
        mk_ns_cdf(rng, 2, &mut c[66..69]);
        c
    };
    for _ in 0..80_000 {
        // cfl_alphas
        {
            let js = (rng.next() % 8) as i32;
            let sign_u = (js + 1) / 3;
            let sign_v = (js + 1) % 3;
            let u = if sign_u != 0 { (rng.next() % 16) as i32 } else { 0 };
            let v = if sign_v != 0 { (rng.next() % 16) as i32 } else { 0 };
            let idx = (u << 4) | v;
            let mut sc = [0u16; 9];
            mk_ns_cdf(&mut rng, 8, &mut sc);
            let mut ac = [[0u16; 17]; 6];
            for c in ac.iter_mut() {
                mk_ns_cdf(&mut rng, 16, c);
            }
            let mut enc = OdEcEnc::new();
            let (mut se, mut ae) = (sc, ac);
            write_cfl_alphas(&mut enc, &mut se, &mut ae, idx, js);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let (mut sd, mut ad) = (sc, ac);
            let (gjs, gidx) = read_cfl_alphas(&mut dec, &mut sd, &mut ad);
            assert_eq!((gjs, gidx), (js, idx), "cfl js={js} idx={idx}");
            assert_eq!((se, ae), (sd, ad), "cfl cdf");
        }
        // skip_mode (coded path)
        {
            let skip_mode = (rng.next() & 1) as i32;
            let mut c = [0u16; 3];
            mk_ns_cdf(&mut rng, 2, &mut c);
            let mut enc = OdEcEnc::new();
            let mut ce = c;
            write_skip_mode(&mut enc, &mut ce, true, false, true, false, skip_mode);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let mut cd = c;
            let got = read_skip_mode(&mut dec, &mut cd, true, false, true, false);
            assert_eq!(got, skip_mode, "skip_mode");
            assert_eq!(ce, cd, "skip_mode cdf");
        }
        // intrabc (integer-pel DV rounds through MV_SUBPEL_NONE)
        {
            let use_intrabc = (rng.next() & 1) as i32;
            let dr = if use_intrabc != 0 { ((rng.next() % 401) as i32 - 200) * 8 } else { 0 };
            let dc = if use_intrabc != 0 { ((rng.next() % 401) as i32 - 200) * 8 } else { 0 };
            let mut ic = [0u16; 3];
            mk_ns_cdf(&mut rng, 2, &mut ic);
            let mut joints = [0u16; 5];
            mk_ns_cdf(&mut rng, 4, &mut joints);
            let comp0 = mk_comp(&mut rng);
            let comp1 = mk_comp(&mut rng);
            let mut enc = OdEcEnc::new();
            let (mut ie, mut je, mut c0e, mut c1e) = (ic, joints, comp0, comp1);
            write_intrabc_info(&mut enc, &mut ie, &mut je, &mut c0e, &mut c1e, use_intrabc, dr, dc);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let (mut id, mut jd, mut c0d, mut c1d) = (ic, joints, comp0, comp1);
            let (gu, gr, gc) = read_intrabc_info(&mut dec, &mut id, &mut jd, &mut c0d, &mut c1d);
            assert_eq!((gu, gr, gc), (use_intrabc, dr, dc), "intrabc dv=({dr},{dc})");
            assert_eq!((ie, je, c0e, c1e), (id, jd, c0d, c1d), "intrabc cdf");
        }
        // cdef (single SB-upper-left call)
        {
            let bits = (rng.next() % 4) as u32; // 0..3
            let strength = if bits > 0 { (rng.next() % (1u64 << bits)) as i32 } else { 0 };
            let skip = (rng.next() & 1) as i32;
            let sb128 = rng.next() & 1 == 1;
            let sb_size = if sb128 { 15usize } else { 12usize }; // BLOCK_128X128 / BLOCK_64X64
            let mib = if sb128 { 32 } else { 16 };
            let mut te = [false; 4];
            let mut enc = OdEcEnc::new();
            write_cdef(&mut enc, false, false, 0, 0, mib, sb_size, skip, &mut te, bits, strength);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let mut td = [false; 4];
            let got = read_cdef(&mut dec, false, false, 0, 0, mib, sb_size, skip, &mut td, bits);
            let expected = if skip == 0 { strength } else { -1 };
            assert_eq!(got, expected, "cdef bits={bits} skip={skip} sb128={sb128}");
            assert_eq!(te, td, "cdef transmitted state");
        }
    }
}

#[test]
fn read_inter_leaf_symbols_roundtrip() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        read_compound_type_info, read_interintra_info, read_palette_mode_info_flags,
        write_compound_type_info, write_interintra_info, write_palette_mode_info_flags,
    };
    let mut rng = Rng(0x1e_1eaf_2c0d_e0c0u64);
    for _ in 0..80_000 {
        // interintra (allowed path)
        {
            let interintra = (rng.next() & 1) as i32;
            let wedge_used = rng.next() & 1 == 1;
            let mode = if interintra != 0 { (rng.next() % 4) as i32 } else { 0 };
            let use_wedge = if interintra != 0 && wedge_used { (rng.next() & 1) as i32 } else { 0 };
            let widx = if use_wedge != 0 { (rng.next() % 16) as i32 } else { 0 };
            let mut ii = [0u16; 3];
            let mut im = [0u16; 5];
            let mut wi = [0u16; 3];
            let mut wx = [0u16; 17];
            mk_ns_cdf(&mut rng, 2, &mut ii);
            mk_ns_cdf(&mut rng, 4, &mut im);
            mk_ns_cdf(&mut rng, 2, &mut wi);
            mk_ns_cdf(&mut rng, 16, &mut wx);
            let mut enc = OdEcEnc::new();
            let (mut e0, mut e1, mut e2, mut e3) = (ii, im, wi, wx);
            write_interintra_info(&mut enc, true, interintra, &mut e0, mode, &mut e1, wedge_used, use_wedge, &mut e2, widx, &mut e3);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let (mut d0, mut d1, mut d2, mut d3) = (ii, im, wi, wx);
            let got = read_interintra_info(&mut dec, true, &mut d0, &mut d1, wedge_used, &mut d2, &mut d3);
            assert_eq!(got, (interintra, mode, use_wedge, widx), "interintra");
            assert_eq!((e0, e1, e2, e3), (d0, d1, d2, d3), "interintra cdf");
        }
        // compound_type
        {
            let masked = rng.next() & 1 == 1;
            let dist_wtd = rng.next() & 1 == 1;
            let wedge_used = rng.next() & 1 == 1;
            let cgi = if masked { (rng.next() & 1) as i32 } else { 0 };
            let (mut cidx, mut ctype) = (1i32, 0i32);
            let (mut widx, mut wsign, mut mask) = (0i32, 0i32, 0i32);
            if cgi == 0 {
                if dist_wtd { cidx = (rng.next() & 1) as i32; }
            } else {
                ctype = if wedge_used { 2 + (rng.next() & 1) as i32 } else { 3 };
                if ctype == 2 {
                    widx = (rng.next() % 16) as i32;
                    wsign = (rng.next() & 1) as i32;
                } else {
                    mask = (rng.next() & 1) as i32;
                }
            }
            let mut cg = [0u16; 3];
            let mut ci = [0u16; 3];
            let mut ct = [0u16; 3];
            let mut wx = [0u16; 17];
            mk_ns_cdf(&mut rng, 2, &mut cg);
            mk_ns_cdf(&mut rng, 2, &mut ci);
            mk_ns_cdf(&mut rng, 2, &mut ct);
            mk_ns_cdf(&mut rng, 16, &mut wx);
            let mut enc = OdEcEnc::new();
            let (mut cge, mut cie, mut cte, mut wxe) = (cg, ci, ct, wx);
            write_compound_type_info(&mut enc, masked, cgi, &mut cge, dist_wtd, cidx, &mut cie, wedge_used, ctype, &mut cte, widx, &mut wxe, wsign, mask);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let (mut cgd, mut cid, mut ctd, mut wxd) = (cg, ci, ct, wx);
            let got = read_compound_type_info(&mut dec, masked, &mut cgd, dist_wtd, &mut cid, wedge_used, &mut ctd, &mut wxd);
            assert_eq!(got, (cgi, cidx, ctype, widx, wsign, mask), "compound_type");
            assert_eq!((cge, cie, cte, wxe), (cgd, cid, ctd, wxd), "compound cdf");
        }
        // palette flags
        {
            let dc_y = rng.next() & 1 == 1;
            let dc_uv = rng.next() & 1 == 1;
            let n_y = if dc_y && rng.next() & 1 == 1 { 2 + (rng.next() % 7) as i32 } else { 0 };
            let n_uv = if dc_uv && rng.next() & 1 == 1 { 2 + (rng.next() % 7) as i32 } else { 0 };
            let mut ym = [0u16; 3];
            let mut ys = [0u16; 8];
            let mut um = [0u16; 3];
            let mut us = [0u16; 8];
            mk_ns_cdf(&mut rng, 2, &mut ym);
            mk_ns_cdf(&mut rng, 7, &mut ys);
            mk_ns_cdf(&mut rng, 2, &mut um);
            mk_ns_cdf(&mut rng, 7, &mut us);
            let mut enc = OdEcEnc::new();
            let (mut yme, mut yse, mut ume, mut use_) = (ym, ys, um, us);
            write_palette_mode_info_flags(&mut enc, dc_y, n_y, &mut yme, &mut yse, dc_uv, n_uv, &mut ume, &mut use_);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let (mut ymd, mut ysd, mut umd, mut usd) = (ym, ys, um, us);
            let got = read_palette_mode_info_flags(&mut dec, dc_y, &mut ymd, &mut ysd, dc_uv, &mut umd, &mut usd);
            assert_eq!(got, (n_y, n_uv), "palette flags");
            assert_eq!((yme, yse, ume, use_), (ymd, ysd, umd, usd), "palette cdf");
        }
    }
}

#[test]
fn read_modes_tile_roundtrips_write() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{get_partition_subsize, read_modes_tile, write_modes_tile};
    let mut rng = Rng(0x1e_7115_dec0_de1bu64);
    fn gen(rng: &mut Rng, bsize: usize, out: &mut Vec<i8>) {
        let cdf_len = if bsize == 3 { 4 } else if bsize == 15 { 8 } else { 10 };
        let p = (rng.next() % cdf_len as u64) as i32;
        out.push(p as i8);
        if p == 3 && bsize > 3 {
            let sub = get_partition_subsize(bsize, 3) as usize;
            for _ in 0..4 {
                gen(rng, sub, out);
            }
        }
    }
    let mk = |rng: &mut Rng, n: usize, out: &mut [u16]| {
        let mut prev = 32768i32;
        for e in out.iter_mut().take(n - 1) {
            let v = (prev - 1 - (rng.next() % 400) as i32).max(n as i32);
            *e = v as u16;
            prev = v;
        }
        out[n - 1] = 0;
        out[n] = 0;
    };
    for _ in 0..30_000 {
        let n_sb_rows = 1 + (rng.next() % 3) as i32;
        let n_sb_cols = 1 + (rng.next() % 3) as i32;
        let mut tree = Vec::new();
        for _ in 0..(n_sb_rows * n_sb_cols) {
            gen(&mut rng, 12, &mut tree); // BLOCK_64X64 SBs
        }
        let mut arena0 = [[0u16; 11]; 20];
        for (c, slot) in arena0.iter_mut().enumerate() {
            let bsl = c / 4;
            let ns = if bsl == 0 { 4 } else if bsl == 4 { 8 } else { 10 };
            mk(&mut rng, ns, slot);
        }
        let mut enc = OdEcEnc::new();
        let mut above_e = [0i8; 128];
        let mut arena_e = arena0;
        let consumed = write_modes_tile(&mut enc, &mut above_e, &mut arena_e, &tree, n_sb_rows, n_sb_cols, 16, 12);
        let bytes = enc.done().to_vec();
        let mut dec = OdEcDec::new(&bytes);
        let mut above_d = [0i8; 128];
        let mut arena_d = arena0;
        let out = read_modes_tile(&mut dec, &mut above_d, &mut arena_d, n_sb_rows, n_sb_cols, 16, 12);
        assert_eq!(out, tree, "tree {n_sb_rows}x{n_sb_cols} len={}", tree.len());
        assert_eq!(out.len(), consumed, "consumed count");
        assert_eq!(above_e, above_d, "above context");
        assert_eq!(arena_e, arena_d, "adapted arena");
    }
}

#[test]
fn read_intra_pred_mode_pieces_roundtrip() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        is_directional_mode, read_intra_uv_and_angle_delta, read_intra_y_and_angle_delta,
        use_angle_delta, write_intra_y_and_angle_delta, write_intra_uv_and_angle_delta,
    };
    let mut rng = Rng(0x1e_147a_c0de_00d0u64);
    // block sizes >= 8x8 (use_angle_delta true) + some 4x4 (false).
    let bsizes = [0usize, 3, 6, 9, 12, 15];
    for _ in 0..200_000 {
        let bsize = bsizes[(rng.next() % bsizes.len() as u64) as usize];
        // --- Y mode + angle ---
        {
            let mode = (rng.next() % 13) as i32;
            let ang_coded = use_angle_delta(bsize) && is_directional_mode(mode);
            let angle = if ang_coded { (rng.next() % 7) as i32 - 3 } else { 0 };
            let mut yc = [0u16; 14];
            let mut yac = [0u16; 8];
            mk_ns_cdf(&mut rng, 13, &mut yc);
            mk_ns_cdf(&mut rng, 7, &mut yac);
            let mut enc = OdEcEnc::new();
            let (mut yce, mut yace) = (yc, yac);
            write_intra_y_and_angle_delta(&mut enc, &mut yce, mode, bsize, angle, &mut yace);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let (mut ycd, mut yacd) = (yc, yac);
            let (gm, ga) = read_intra_y_and_angle_delta(&mut dec, &mut ycd, bsize, &mut yacd);
            assert_eq!(gm, mode, "y mode");
            if ang_coded {
                assert_eq!(ga, angle, "y angle");
            }
            assert_eq!((yce, yace), (ycd, yacd), "y cdf");
        }
        // --- UV mode + cfl + angle ---
        {
            let mono = rng.next() & 1 == 1;
            let chroma_ref = rng.next() & 1 == 1;
            let cfl_allowed = rng.next() & 1 == 1;
            let uv_n = if cfl_allowed { 14 } else { 13 };
            let uv_mode = (rng.next() % uv_n as u64) as i32;
            let js = if uv_mode == 13 { (rng.next() % 8) as i32 } else { 0 };
            let (su, sv) = ((js + 1) / 3, (js + 1) % 3);
            let u = if uv_mode == 13 && su != 0 { (rng.next() % 16) as i32 } else { 0 };
            let v = if uv_mode == 13 && sv != 0 { (rng.next() % 16) as i32 } else { 0 };
            let idx = (u << 4) | v;
            let intra_mode = aom_entropy::partition::get_uv_mode(uv_mode as usize);
            let uv_ang_coded = !mono && chroma_ref && use_angle_delta(bsize) && is_directional_mode(intra_mode);
            let angle_uv = if uv_ang_coded { (rng.next() % 7) as i32 - 3 } else { 0 };
            let mut uc = [0u16; 15];
            let mut sc = [0u16; 9];
            let mut ac = [[0u16; 17]; 6];
            let mut uac = [0u16; 8];
            mk_ns_cdf(&mut rng, uv_n, &mut uc);
            mk_ns_cdf(&mut rng, 8, &mut sc);
            for c in ac.iter_mut() { mk_ns_cdf(&mut rng, 16, c); }
            mk_ns_cdf(&mut rng, 7, &mut uac);
            let mut enc = OdEcEnc::new();
            let (mut uce, mut sce, mut ace, mut uace) = (uc, sc, ac, uac);
            write_intra_uv_and_angle_delta(&mut enc, mono, chroma_ref, uv_mode, cfl_allowed, bsize, idx, js, angle_uv, &mut uce, &mut sce, &mut ace, &mut uace);
            let b = enc.done().to_vec();
            let mut dec = OdEcDec::new(&b);
            let (mut ucd, mut scd, mut acd, mut uacd) = (uc, sc, ac, uac);
            let (guv, gidx, gjs, gang) = read_intra_uv_and_angle_delta(&mut dec, mono, chroma_ref, cfl_allowed, bsize, &mut ucd, &mut scd, &mut acd, &mut uacd);
            if !mono && chroma_ref {
                assert_eq!(guv, uv_mode, "uv mode");
                if uv_mode == 13 { assert_eq!((gidx, gjs), (idx, js), "cfl"); }
                if uv_ang_coded { assert_eq!(gang, angle_uv, "uv angle"); }
            }
            assert_eq!((uce, sce, ace, uace), (ucd, scd, acd, uacd), "uv cdf");
        }
    }
}

#[test]
fn read_intra_prediction_modes_roundtrips_nopalette() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        get_uv_mode, is_directional_mode, read_intra_prediction_modes, use_angle_delta,
        write_intra_prediction_modes,
    };
    let mut rng = Rng(0x1e_147a_dec0_de11u64);
    let bsizes = [0usize, 3, 6, 9, 12, 15];
    for _ in 0..200_000 {
        let bsize = bsizes[(rng.next() % bsizes.len() as u64) as usize];
        let mode = (rng.next() % 13) as i32;
        let y_ang_coded = use_angle_delta(bsize) && is_directional_mode(mode);
        let angle_y = if y_ang_coded { (rng.next() % 7) as i32 - 3 } else { 0 };
        let mono = rng.next() & 1 == 1;
        let chroma_ref = rng.next() & 1 == 1;
        let cfl_allowed = rng.next() & 1 == 1;
        let uv_n = if cfl_allowed { 14 } else { 13 };
        let uv_mode = if !mono && chroma_ref { (rng.next() % uv_n as u64) as i32 } else { 0 };
        let js = if uv_mode == 13 { (rng.next() % 8) as i32 } else { 0 };
        let (su, sv) = ((js + 1) / 3, (js + 1) % 3);
        let u = if uv_mode == 13 && su != 0 { (rng.next() % 16) as i32 } else { 0 };
        let v = if uv_mode == 13 && sv != 0 { (rng.next() % 16) as i32 } else { 0 };
        let cfl_idx = (u << 4) | v;
        let uv_intra = get_uv_mode(uv_mode as usize);
        let uv_ang_coded = !mono && chroma_ref && use_angle_delta(bsize) && is_directional_mode(uv_intra);
        let angle_uv = if uv_ang_coded { (rng.next() % 7) as i32 - 3 } else { 0 };
        let filter_allowed = rng.next() & 1 == 1;
        let use_fi = if filter_allowed { (rng.next() & 1) as i32 } else { 0 };
        let fi_mode = if use_fi != 0 { (rng.next() % 5) as i32 } else { 0 };

        let mut yc = [0u16; 14];
        let mut yac = [0u16; 8];
        let mut uc = [0u16; 15];
        let mut sc = [0u16; 9];
        let mut ac = [[0u16; 17]; 6];
        let mut uac = [0u16; 8];
        let mut fiu = [0u16; 3];
        let mut fim = [0u16; 6];
        let mut p0 = [0u16; 3];
        let mut p1 = [0u16; 8];
        let mut p2 = [0u16; 3];
        let mut p3 = [0u16; 8];
        mk_ns_cdf(&mut rng, 13, &mut yc);
        mk_ns_cdf(&mut rng, 7, &mut yac);
        mk_ns_cdf(&mut rng, uv_n, &mut uc);
        mk_ns_cdf(&mut rng, 8, &mut sc);
        for c in ac.iter_mut() { mk_ns_cdf(&mut rng, 16, c); }
        mk_ns_cdf(&mut rng, 7, &mut uac);
        mk_ns_cdf(&mut rng, 2, &mut fiu);
        mk_ns_cdf(&mut rng, 5, &mut fim);
        mk_ns_cdf(&mut rng, 2, &mut p0);
        mk_ns_cdf(&mut rng, 7, &mut p1);
        mk_ns_cdf(&mut rng, 2, &mut p2);
        mk_ns_cdf(&mut rng, 7, &mut p3);

        let mut enc = OdEcEnc::new();
        let (mut yce, mut yace, mut uce, mut sce, mut ace, mut uace, mut fiue, mut fime) =
            (yc, yac, uc, sc, ac, uac, fiu, fim);
        let (mut p0e, mut p1e, mut p2e, mut p3e) = (p0, p1, p2, p3);
        write_intra_prediction_modes(
            &mut enc, mode, bsize, &mut yce, angle_y, &mut yace, mono, chroma_ref, uv_mode,
            cfl_allowed, cfl_idx, js, angle_uv, &mut uce, &mut sce, &mut ace, &mut uace,
            false, 8, [0, 0], &[], 0, false, &[], [0, 0], false, &[], [0, 0],
            &mut p0e, &mut p1e, &mut p2e, &mut p3e, filter_allowed, use_fi, fi_mode,
            &mut fiue, &mut fime,
        );
        let b = enc.done().to_vec();
        let mut dec = OdEcDec::new(&b);
        let (mut ycd, mut yacd, mut ucd, mut scd, mut acd, mut uacd, mut fiud, mut fimd) =
            (yc, yac, uc, sc, ac, uac, fiu, fim);
        let (mut p0d, mut p1d, mut p2d, mut p3d) = (p0, p1, p2, p3);
        let (gm, ga, guv, gidx, gjs, gauv, _gps, _gpc, guf, gfm) = read_intra_prediction_modes(
            &mut dec, bsize, &mut ycd, &mut yacd, mono, chroma_ref, cfl_allowed, &mut ucd,
            &mut scd, &mut acd, &mut uacd, false, 8, &mut p0d, &mut p1d, &mut p2d,
            &mut p3d, 0, false, &[], [0, 0], false, &[], [0, 0], filter_allowed, &mut fiud, &mut fimd,
        );
        assert_eq!(gm, mode, "y mode");
        if y_ang_coded { assert_eq!(ga, angle_y, "y angle"); }
        if !mono && chroma_ref {
            assert_eq!(guv, uv_mode, "uv mode");
            if uv_mode == 13 { assert_eq!((gidx, gjs), (cfl_idx, js), "cfl"); }
            if uv_ang_coded { assert_eq!(gauv, angle_uv, "uv angle"); }
        }
        if filter_allowed {
            assert_eq!(guf, use_fi, "use_fi");
            if use_fi != 0 { assert_eq!(gfm, fi_mode, "fi mode"); }
        }
        assert_eq!((yce, yace, uce, sce, ace, uace, fiue, fime), (ycd, yacd, ucd, scd, acd, uacd, fiud, fimd), "cdfs");
    }
}

#[test]
fn read_kf_tail_roundtrips_write() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        get_uv_mode, is_directional_mode, read_kf_tail, use_angle_delta, write_kf_tail,
    };
    let mut rng = Rng(0x1e_c0de_de1f_00a0u64);
    let mk_comp = |rng: &mut Rng| -> [u16; 69] {
        let mut c = [0u16; 69];
        mk_ns_cdf(rng, 2, &mut c[0..3]);
        mk_ns_cdf(rng, 11, &mut c[3..15]);
        mk_ns_cdf(rng, 2, &mut c[15..18]);
        for i in 0..10 { let o = 18 + i * 3; mk_ns_cdf(rng, 2, &mut c[o..o + 3]); }
        for i in 0..2 { let o = 48 + i * 5; mk_ns_cdf(rng, 4, &mut c[o..o + 5]); }
        mk_ns_cdf(rng, 4, &mut c[58..63]);
        mk_ns_cdf(rng, 2, &mut c[63..66]);
        mk_ns_cdf(rng, 2, &mut c[66..69]);
        c
    };
    let bsizes = [3usize, 6, 9, 12, 15];
    for _ in 0..200_000 {
        let bsize = bsizes[(rng.next() % bsizes.len() as u64) as usize];
        let allow_intrabc = rng.next() & 1 == 1;
        let use_intrabc = if allow_intrabc { (rng.next() & 1) as i32 } else { 0 };
        let (dr, dc) = if use_intrabc != 0 {
            (((rng.next() % 201) as i32 - 100) * 8, ((rng.next() % 201) as i32 - 100) * 8)
        } else {
            (0, 0)
        };
        // intra fields (used only when not an intrabc block)
        let mode = (rng.next() % 13) as i32;
        let y_ang = if use_angle_delta(bsize) && is_directional_mode(mode) { (rng.next() % 7) as i32 - 3 } else { 0 };
        let mono = rng.next() & 1 == 1;
        let chroma_ref = rng.next() & 1 == 1;
        let cfl_allowed = rng.next() & 1 == 1;
        let uv_n = if cfl_allowed { 14 } else { 13 };
        let uv_mode = if !mono && chroma_ref { (rng.next() % uv_n as u64) as i32 } else { 0 };
        let js = if uv_mode == 13 { (rng.next() % 8) as i32 } else { 0 };
        let (su, sv) = ((js + 1) / 3, (js + 1) % 3);
        let u = if uv_mode == 13 && su != 0 { (rng.next() % 16) as i32 } else { 0 };
        let v = if uv_mode == 13 && sv != 0 { (rng.next() % 16) as i32 } else { 0 };
        let cfl_idx = (u << 4) | v;
        let uv_intra = get_uv_mode(uv_mode as usize);
        let uv_ang_c = !mono && chroma_ref && use_angle_delta(bsize) && is_directional_mode(uv_intra);
        let uv_ang = if uv_ang_c { (rng.next() % 7) as i32 - 3 } else { 0 };
        let fi_allowed = rng.next() & 1 == 1;
        let use_fi = if fi_allowed { (rng.next() & 1) as i32 } else { 0 };
        let fi_mode = if use_fi != 0 { (rng.next() % 5) as i32 } else { 0 };

        let mut ic = [0u16; 3]; let mut jc = [0u16; 5];
        let c0 = mk_comp(&mut rng); let c1 = mk_comp(&mut rng);
        let mut yc = [0u16; 14]; let mut yac = [0u16; 8]; let mut uc = [0u16; 15];
        let mut sc = [0u16; 9]; let mut ac = [[0u16; 17]; 6]; let mut uac = [0u16; 8];
        let mut fiu = [0u16; 3]; let mut fim = [0u16; 6];
        let (mut p0, mut p1, mut p2, mut p3) = ([0u16; 3], [0u16; 8], [0u16; 3], [0u16; 8]);
        mk_ns_cdf(&mut rng, 2, &mut ic); mk_ns_cdf(&mut rng, 4, &mut jc);
        mk_ns_cdf(&mut rng, 13, &mut yc); mk_ns_cdf(&mut rng, 7, &mut yac);
        mk_ns_cdf(&mut rng, uv_n, &mut uc); mk_ns_cdf(&mut rng, 8, &mut sc);
        for c in ac.iter_mut() { mk_ns_cdf(&mut rng, 16, c); }
        mk_ns_cdf(&mut rng, 7, &mut uac); mk_ns_cdf(&mut rng, 2, &mut fiu); mk_ns_cdf(&mut rng, 5, &mut fim);
        mk_ns_cdf(&mut rng, 2, &mut p0); mk_ns_cdf(&mut rng, 7, &mut p1);
        mk_ns_cdf(&mut rng, 2, &mut p2); mk_ns_cdf(&mut rng, 7, &mut p3);

        let mut enc = OdEcEnc::new();
        let (mut ice, mut jce, mut c0e, mut c1e, mut yce, mut yace, mut uce, mut sce, mut ace, mut uace, mut fiue, mut fime) =
            (ic, jc, c0, c1, yc, yac, uc, sc, ac, uac, fiu, fim);
        let (mut p0e, mut p1e, mut p2e, mut p3e) = (p0, p1, p2, p3);
        write_kf_tail(
            &mut enc, allow_intrabc, &mut ice, &mut jce, &mut c0e, &mut c1e, use_intrabc, dr, dc,
            mode, bsize, &mut yce, y_ang, &mut yace, mono, chroma_ref, uv_mode, cfl_allowed, cfl_idx,
            js, uv_ang, &mut uce, &mut sce, &mut ace, &mut uace, false, 8, [0, 0], &[], 0, false, &[],
            [0, 0], false, &[], [0, 0], &mut p0e, &mut p1e, &mut p2e, &mut p3e, fi_allowed, use_fi,
            fi_mode, &mut fiue, &mut fime,
        );
        let b = enc.done().to_vec();
        let mut dec = OdEcDec::new(&b);
        let (mut icd, mut jcd, mut c0d, mut c1d, mut ycd, mut yacd, mut ucd, mut scd, mut acd, mut uacd, mut fiud, mut fimd) =
            (ic, jc, c0, c1, yc, yac, uc, sc, ac, uac, fiu, fim);
        let (mut p0d, mut p1d, mut p2d, mut p3d) = (p0, p1, p2, p3);
        let g = read_kf_tail(
            &mut dec, allow_intrabc, &mut icd, &mut jcd, &mut c0d, &mut c1d, bsize, &mut ycd, &mut yacd,
            mono, chroma_ref, cfl_allowed, &mut ucd, &mut scd, &mut acd, &mut uacd, false, 8, &mut p0d,
            &mut p1d, &mut p2d, &mut p3d, 0, false, &[], [0, 0], false, &[], [0, 0], fi_allowed,
            &mut fiud, &mut fimd,
        );
        assert_eq!(g.use_intrabc, use_intrabc, "use_intrabc");
        if use_intrabc != 0 {
            assert_eq!((g.diff_row, g.diff_col), (dr, dc), "dv");
        } else {
            assert_eq!(g.mode, mode, "y mode");
            if !mono && chroma_ref {
                assert_eq!(g.uv_mode, uv_mode, "uv mode");
                if uv_mode == 13 { assert_eq!((g.cfl_alpha_idx, g.cfl_joint_sign), (cfl_idx, js), "cfl"); }
            }
            if fi_allowed { assert_eq!(g.use_filter_intra, use_fi, "use_fi"); }
        }
        assert_eq!((yce, uce, fiue), (ycd, ucd, fiud), "cdf adapt");
    }
}

#[test]
fn read_mb_modes_kf_prefix_roundtrips_write() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{read_mb_modes_kf_prefix, write_mb_modes_kf_prefix};
    let mut rng = Rng(0x1e_c0de_9e14_00b0u64);
    for _ in 0..200_000 {
        let segid_preskip = rng.next() & 1 == 1;
        let seg_enabled = rng.next() & 1 == 1;
        let update_map = seg_enabled && rng.next() & 1 == 1; // update_map => seg_enabled
        let last = (rng.next() % 8) as i32;
        let segment_id = (rng.next() % (last as u64 + 1)) as i32;
        let seg_pred = (rng.next() % (last as u64 + 1)) as i32;
        let seg_skip_active = rng.next() & 1 == 1;
        let skip_txfm = (rng.next() & 1) as i32;
        let coded_lossless = rng.next() & 1 == 1;
        let allow_intrabc = rng.next() & 1 == 1;
        let sb128 = rng.next() & 1 == 1;
        let (sb_size, mib) = if sb128 { (15usize, 32i32) } else { (12usize, 16i32) };
        let upper_left = rng.next() & 1 == 1;
        let (mi_row, mi_col) = if upper_left { (0, 0) } else { (mib, mib) };
        let bsize = [3usize, 12, 15][(rng.next() % 3) as usize];
        let cdef_bits = (rng.next() % 4) as u32;
        let cdef_strength = if cdef_bits > 0 { (rng.next() % (1u64 << cdef_bits)) as i32 } else { 0 };
        let dq_present = rng.next() & 1 == 1;
        let dlf_present = dq_present && rng.next() & 1 == 1;
        let dlf_multi = rng.next() & 1 == 1;
        let num_planes = if rng.next() & 1 == 1 { 3 } else { 1 };
        let dq_res = 1 << (rng.next() % 4);
        let dlf_res = 1 << (rng.next() % 4);
        let base0 = 100i32;
        // Encoder-valid delta targets only: the C write side asserts current_qindex > 0
        // and its RD layer keeps targets in [1, MAXQ] / [-63, 63] (bitstream.c,
        // av1_adjust_q_from_delta_q_res), so `base + reduced * res` never leaves range
        // and the reader's normative clamps (decodemv.c read_delta_q_params) stay
        // no-ops — lockstep carries require in-domain targets.
        let kq = ((rng.next() % 101) as i32 - 50).clamp(-((base0 - 1) / dq_res), (255 - base0) / dq_res);
        let cur_qindex = base0 + kq * dq_res;
        let mut xd_lf0 = [0i32; 4];
        for x in xd_lf0.iter_mut() { *x = (rng.next() % 21) as i32 - 10; }
        let xd_lfb0 = (rng.next() % 21) as i32 - 10;
        let mut mbmi_lf = [0i32; 4];
        for i in 0..4 {
            let k = ((rng.next() % 21) as i32 - 10)
                .clamp(-((63 + xd_lf0[i]) / dlf_res), (63 - xd_lf0[i]) / dlf_res);
            mbmi_lf[i] = xd_lf0[i] + k * dlf_res;
        }
        let klb = ((rng.next() % 21) as i32 - 10)
            .clamp(-((63 + xd_lfb0) / dlf_res), (63 - xd_lfb0) / dlf_res);
        let mbmi_lfb = xd_lfb0 + klb * dlf_res;

        let mut sc = [0u16; 9];
        let mut kc = [0u16; 3];
        let mut dqc = [0u16; 5];
        let mut dlmc = [[0u16; 5]; 4];
        let mut dlc = [0u16; 5];
        mk_ns_cdf(&mut rng, 8, &mut sc);
        mk_ns_cdf(&mut rng, 2, &mut kc);
        mk_ns_cdf(&mut rng, 4, &mut dqc);
        for c in dlmc.iter_mut() { mk_ns_cdf(&mut rng, 4, c); }
        mk_ns_cdf(&mut rng, 4, &mut dlc);

        // encode
        let mut enc = OdEcEnc::new();
        let (mut sce, mut kce, mut dqce, mut dlmce, mut dlce) = (sc, kc, dqc, dlmc, dlc);
        let mut cdef_te = [false; 4];
        let mut base_e = base0;
        let mut xd_lf_e = xd_lf0;
        let mut xd_lfb_e = xd_lfb0;
        let skip_e = write_mb_modes_kf_prefix(
            &mut enc, segid_preskip, seg_enabled, update_map, segment_id, seg_pred, last, &mut sce,
            seg_skip_active, skip_txfm, &mut kce, coded_lossless, allow_intrabc, mi_row, mi_col, mib,
            sb_size, &mut cdef_te, cdef_bits, cdef_strength, dq_present, dlf_present, dlf_multi,
            num_planes, bsize, cur_qindex, &mut base_e, dq_res, &mbmi_lf, &mut xd_lf_e, mbmi_lfb,
            &mut xd_lfb_e, dlf_res, &mut dqce, &mut dlmce, &mut dlce,
        );
        let b = enc.done().to_vec();
        // decode
        let mut dec = OdEcDec::new(&b);
        let (mut scd, mut kcd, mut dqcd, mut dlmcd, mut dlcd) = (sc, kc, dqc, dlmc, dlc);
        let mut cdef_td = [false; 4];
        let mut base_d = base0;
        let mut xd_lf_d = xd_lf0;
        let mut xd_lfb_d = xd_lfb0;
        let (skip_d, seg_d, cdef_d, cq_d) = read_mb_modes_kf_prefix(
            &mut dec, segid_preskip, seg_enabled, update_map, seg_pred, last, &mut scd,
            seg_skip_active, &mut kcd, coded_lossless, allow_intrabc, mi_row, mi_col, mib, sb_size,
            &mut cdef_td, cdef_bits, dq_present, dlf_present, dlf_multi, num_planes, bsize, &mut base_d,
            dq_res, &mut xd_lf_d, &mut xd_lfb_d, dlf_res, &mut dqcd, &mut dlmcd, &mut dlcd,
        );
        assert_eq!(skip_d, skip_e, "skip");
        let seg_coded = seg_enabled && update_map && (segid_preskip || skip_e == 0);
        if seg_coded {
            assert_eq!(seg_d, segment_id, "segment_id");
        }
        let cdef_coded = !coded_lossless && !allow_intrabc && skip_e == 0;
        assert_eq!(cdef_d, if cdef_coded { cdef_strength } else { -1 }, "cdef strength");
        assert_eq!(cdef_td, cdef_te, "cdef transmitted");
        assert_eq!((base_d, xd_lf_d, xd_lfb_d), (base_e, xd_lf_e, xd_lfb_e), "delta carries");
        assert_eq!(cq_d, base_e, "current_qindex == updated base");
        assert_eq!((scd, kcd, dqcd, dlmcd, dlcd), (sce, kce, dqce, dlmce, dlce), "cdf adapt");
    }
}

/// The normative decoder clamps of `read_delta_q_params` (av1/decoder/decodemv.c):
/// `current_base_qindex = clamp(base + reduced * res, 1, MAXQ)` ("Clamp to [1,MAXQ]
/// to not interfere with lossless mode") and per-lf-carry
/// `clamp(carry + r * res, -MAX_LOOP_FILTER, MAX_LOOP_FILTER)`. A libaom encoder
/// never produces out-of-range reconstructions (its write side asserts and aligns
/// targets), so these fire only on adversarial/foreign conformant-syntax streams —
/// pinned here with hand vectors at both bounds, plus interior/boundary no-ops.
#[test]
fn read_delta_q_params_normative_clamps() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        read_delta_q_params_sb, write_delta_lflevel, write_delta_qindex,
    };
    let mut rng = Rng(0xc1a3_c1a3_0000_0001);
    // (base, res, reduced, expected current_qindex after clamp)
    let dq_vectors = [
        (10, 8, -5, 1),    // 10 - 40 = -30 -> 1 (low clamp)
        (250, 8, 5, 255),  // 250 + 40 = 290 -> 255 (high clamp)
        (1, 1, -1, 1),     // 0 -> 1 (low clamp at the lossless boundary)
        (2, 1, -1, 1),     // exact bound, no clamp
        (100, 4, 7, 128),  // interior, no clamp
        (255, 2, 0, 255),  // zero delta at the top, no clamp
        (37, 8, 100, 255), // exp-Golomb-range positive delta -> high clamp
        (37, 8, -100, 1),  // exp-Golomb-range negative delta -> low clamp
    ];
    // (carry, res, r, expected carry after clamp)
    let dlf_vectors = [
        (-60, 4, -2, -63), // -68 -> -63 (low clamp)
        (60, 4, 2, 63),    // 68 -> 63 (high clamp)
        (-63, 1, 0, -63),  // bound, no clamp
        (10, 2, -5, 0),    // interior, no clamp
        (0, 8, 100, 63),   // exp-Golomb-range delta -> high clamp
    ];
    for &(base, dq_res, reduced, want_q) in &dq_vectors {
        for &(lf0, dlf_res, r_lf, want_lf) in &dlf_vectors {
            for &multi in &[false, true] {
                let mut dq_cdf = [0u16; 5];
                let mut dlf_cdf = [0u16; 5];
                let mut dlmc = [[0u16; 5]; 4];
                mk_ns_cdf(&mut rng, 4, &mut dq_cdf);
                mk_ns_cdf(&mut rng, 4, &mut dlf_cdf);
                for c in dlmc.iter_mut() {
                    mk_ns_cdf(&mut rng, 4, c);
                }
                // Raw symbol stream: the low-level writers code ANY reduced value
                // (no write-side clamp), exactly what a foreign encoder could emit.
                let mut enc = OdEcEnc::new();
                let (mut wq, mut wl, mut wlm) = (dq_cdf, dlf_cdf, dlmc);
                write_delta_qindex(&mut enc, &mut wq, reduced);
                if multi {
                    for c in wlm.iter_mut() {
                        write_delta_lflevel(&mut enc, c, r_lf);
                    }
                } else {
                    write_delta_lflevel(&mut enc, &mut wl, r_lf);
                }
                let b = enc.done().to_vec();

                let mut dec = OdEcDec::new(&b);
                let (mut rq, mut rl, mut rlm) = (dq_cdf, dlf_cdf, dlmc);
                let mut carry_q = base;
                let mut xd_lf = [lf0; 4];
                let mut xd_lfb = lf0;
                let got = read_delta_q_params_sb(
                    &mut dec, true, true, multi, 3, 3, 12, 0, true, &mut carry_q, dq_res,
                    &mut xd_lf, &mut xd_lfb, dlf_res, &mut rq, &mut rlm, &mut rl,
                );
                assert_eq!(
                    got, want_q,
                    "clamped current_qindex (base {base} res {dq_res} reduced {reduced})"
                );
                assert_eq!(carry_q, want_q, "carry == clamped current_qindex");
                if multi {
                    assert_eq!(
                        xd_lf,
                        [want_lf; 4],
                        "clamped delta_lf (carry {lf0} res {dlf_res} r {r_lf})"
                    );
                    assert_eq!(xd_lfb, lf0, "from_base carry untouched in multi mode");
                } else {
                    assert_eq!(
                        xd_lfb, want_lf,
                        "clamped delta_lf_from_base (carry {lf0} res {dlf_res} r {r_lf})"
                    );
                    assert_eq!(xd_lf, [lf0; 4], "multi carries untouched in single mode");
                }
            }
        }
    }
}

#[test]
fn read_mb_modes_kf_struct_driver_roundtrips() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        get_uv_mode, is_directional_mode, read_mb_modes_kf, use_angle_delta, write_mb_modes_kf,
        KfBlockState, KfCdfs, MbModeInfoKf,
    };
    let mut rng = Rng(0x1e_c0de_57c7_00c0u64);
    let mk_comp = |rng: &mut Rng| -> [u16; 69] {
        let mut c = [0u16; 69];
        mk_ns_cdf(rng, 2, &mut c[0..3]);
        mk_ns_cdf(rng, 11, &mut c[3..15]);
        mk_ns_cdf(rng, 2, &mut c[15..18]);
        for i in 0..10 { let o = 18 + i * 3; mk_ns_cdf(rng, 2, &mut c[o..o + 3]); }
        for i in 0..2 { let o = 48 + i * 5; mk_ns_cdf(rng, 4, &mut c[o..o + 5]); }
        mk_ns_cdf(rng, 4, &mut c[58..63]);
        mk_ns_cdf(rng, 2, &mut c[63..66]);
        mk_ns_cdf(rng, 2, &mut c[66..69]);
        c
    };
    let gen_cdfs = |rng: &mut Rng, cfl_allowed: bool| -> KfCdfs {
        let mut c = KfCdfs {
            seg: [0; 9], skip: [0; 3], delta_q: [0; 5], delta_lf_multi: [[0; 5]; 4], delta_lf: [0; 5],
            intrabc: [0; 3], ndvc_joints: [0; 5], ndvc_comp0: mk_comp(rng), ndvc_comp1: mk_comp(rng),
            y_mode: [0; 14], y_angle: [0; 8], uv_mode: [0; 15], cfl_sign: [0; 9], cfl_alpha: [[0; 17]; 6],
            uv_angle: [0; 8], pal_y_mode: [0; 3], pal_y_size: [0; 8], pal_uv_mode: [0; 3],
            pal_uv_size: [0; 8], fi_use: [0; 3], fi_mode: [0; 6],
        };
        mk_ns_cdf(rng, 8, &mut c.seg); mk_ns_cdf(rng, 2, &mut c.skip); mk_ns_cdf(rng, 4, &mut c.delta_q);
        for m in c.delta_lf_multi.iter_mut() { mk_ns_cdf(rng, 4, m); }
        mk_ns_cdf(rng, 4, &mut c.delta_lf); mk_ns_cdf(rng, 2, &mut c.intrabc);
        mk_ns_cdf(rng, 4, &mut c.ndvc_joints); mk_ns_cdf(rng, 13, &mut c.y_mode);
        mk_ns_cdf(rng, 7, &mut c.y_angle); mk_ns_cdf(rng, if cfl_allowed { 14 } else { 13 }, &mut c.uv_mode);
        mk_ns_cdf(rng, 8, &mut c.cfl_sign); for a in c.cfl_alpha.iter_mut() { mk_ns_cdf(rng, 16, a); }
        mk_ns_cdf(rng, 7, &mut c.uv_angle); mk_ns_cdf(rng, 2, &mut c.pal_y_mode);
        mk_ns_cdf(rng, 7, &mut c.pal_y_size); mk_ns_cdf(rng, 2, &mut c.pal_uv_mode);
        mk_ns_cdf(rng, 7, &mut c.pal_uv_size); mk_ns_cdf(rng, 2, &mut c.fi_use); mk_ns_cdf(rng, 5, &mut c.fi_mode);
        c
    };
    for _ in 0..100_000 {
        let seg_enabled = rng.next() & 1 == 1;
        let update_map = seg_enabled && rng.next() & 1 == 1;
        let last = (rng.next() % 8) as i32;
        let bsize = [3usize, 12, 15][(rng.next() % 3) as usize];
        let cfl_allowed = rng.next() & 1 == 1;
        let mono = rng.next() & 1 == 1;
        let chroma_ref = rng.next() & 1 == 1;
        let allow_intrabc = rng.next() & 1 == 1;
        let dq_res = 1 << (rng.next() % 4);
        let dlf_res = 1 << (rng.next() % 4);
        let mut st = KfBlockState {
            segid_preskip: rng.next() & 1 == 1, seg_enabled, update_map,
            seg_pred: (rng.next() % (last as u64 + 1)) as i32, last_active_segid: last,
            seg_skip_active: false, mi_row: 0, mi_col: 0, mib_size: 16, sb_size: 12, bsize,
            coded_lossless: false, allow_intrabc, cdef_bits: (rng.next() % 4) as u32,
            dq_present: rng.next() & 1 == 1, dlf_present: rng.next() & 1 == 1, dlf_multi: rng.next() & 1 == 1,
            num_planes: if rng.next() & 1 == 1 { 3 } else { 1 }, dq_res, dlf_res, monochrome: mono,
            is_chroma_ref: chroma_ref, cfl_allowed, allow_palette: false, bit_depth: 8,
            filter_allowed: rng.next() & 1 == 1, mb_to_top_edge: 0, has_above: false, has_left: false,
            cdef_transmitted: [false; 4], current_base_qindex: 100,
            xd_delta_lf: [(rng.next() % 11) as i32, (rng.next() % 11) as i32, (rng.next() % 11) as i32, (rng.next() % 11) as i32],
            xd_delta_lf_from_base: (rng.next() % 11) as i32,
        };
        st.dlf_present = st.dq_present && st.dlf_present;
        let skip = (rng.next() & 1) as i32;
        let use_intrabc = if allow_intrabc { (rng.next() & 1) as i32 } else { 0 };
        let (dr, dc) = if use_intrabc != 0 { (((rng.next() % 51) as i32 - 25) * 8, ((rng.next() % 51) as i32 - 25) * 8) } else { (0, 0) };
        let y_mode = (rng.next() % 13) as i32;
        let y_ang = if use_angle_delta(bsize) && is_directional_mode(y_mode) { (rng.next() % 7) as i32 - 3 } else { 0 };
        let uv_n = if cfl_allowed { 14 } else { 13 };
        let uv_mode = if !mono && chroma_ref { (rng.next() % uv_n as u64) as i32 } else { 0 };
        let js = if uv_mode == 13 { (rng.next() % 8) as i32 } else { 0 };
        let (su, sv) = ((js + 1) / 3, (js + 1) % 3);
        let u = if uv_mode == 13 && su != 0 { (rng.next() % 16) as i32 } else { 0 };
        let v = if uv_mode == 13 && sv != 0 { (rng.next() % 16) as i32 } else { 0 };
        let cfl_idx = (u << 4) | v;
        let uv_intra = get_uv_mode(uv_mode as usize);
        let uv_ang = if !mono && chroma_ref && use_angle_delta(bsize) && is_directional_mode(uv_intra) { (rng.next() % 7) as i32 - 3 } else { 0 };
        let use_fi = if st.filter_allowed { (rng.next() & 1) as i32 } else { 0 };
        let fi_mode = if use_fi != 0 { (rng.next() % 5) as i32 } else { 0 };
        // encoder-valid target: in [1, 255] so the reader's normative clamp is a no-op
        let kq = ((rng.next() % 41) as i32 - 20).clamp(-(99 / dq_res), 155 / dq_res);
        let cur_q = 100 + kq * dq_res;
        let dlf = [st.xd_delta_lf[0] + ((rng.next() % 11) as i32 - 5) * dlf_res, st.xd_delta_lf[1] + ((rng.next() % 11) as i32 - 5) * dlf_res, st.xd_delta_lf[2] + ((rng.next() % 11) as i32 - 5) * dlf_res, st.xd_delta_lf[3] + ((rng.next() % 11) as i32 - 5) * dlf_res];
        let dlfb = st.xd_delta_lf_from_base + ((rng.next() % 11) as i32 - 5) * dlf_res;
        let info = MbModeInfoKf {
            segment_id: (rng.next() % (last as u64 + 1)) as i32, skip, cdef_strength: if st.cdef_bits > 0 { (rng.next() % (1u64 << st.cdef_bits)) as i32 } else { 0 },
            current_qindex: cur_q, delta_lf: dlf, delta_lf_from_base: dlfb, use_intrabc, dv_row: dr, dv_col: dc,
            y_mode, angle_delta_y: y_ang, uv_mode, cfl_alpha_idx: cfl_idx, cfl_joint_sign: js, angle_delta_uv: uv_ang,
            palette_size: [0, 0], use_filter_intra: use_fi, filter_intra_mode: fi_mode,
        };
        let cdfs0 = gen_cdfs(&mut rng, cfl_allowed);
        let mut enc = OdEcEnc::new();
        let mut ce = cdfs0.clone();
        let mut se = st.clone();
        write_mb_modes_kf(&mut enc, &info, &mut ce, &mut se);
        let b = enc.done().to_vec();
        let mut dec = OdEcDec::new(&b);
        let mut cd = cdfs0.clone();
        let mut sd = st.clone();
        let g = read_mb_modes_kf(&mut dec, &mut cd, &mut sd);
        assert_eq!(g.skip, info.skip, "skip");
        assert_eq!(g.use_intrabc, info.use_intrabc, "use_intrabc");
        if use_intrabc != 0 {
            assert_eq!((g.dv_row, g.dv_col), (dr, dc), "dv");
        } else {
            assert_eq!(g.y_mode, info.y_mode, "y_mode");
        }
        // CDF adaptation + state carries lockstep = the field threading is correct.
        assert_eq!(ce.skip, cd.skip, "skip cdf");
        assert_eq!(ce.y_mode, cd.y_mode, "y_mode cdf");
        assert_eq!(ce.delta_q, cd.delta_q, "delta_q cdf");
        assert_eq!(ce.cfl_alpha, cd.cfl_alpha, "cfl cdf");
        assert_eq!((se.current_base_qindex, se.xd_delta_lf, se.cdef_transmitted), (sd.current_base_qindex, sd.xd_delta_lf, sd.cdef_transmitted), "state carries");
    }
}

#[test]
fn read_modes_b_tile_content_roundtrips() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        get_partition_subsize, get_uv_mode, is_directional_mode, read_modes_b, use_angle_delta,
        write_modes_b, KfBlockState, KfCdfs, MbModeInfoKf,
    };
    let mut rng = Rng(0x1e_c0de_b10c_00d0u64);
    let mk_comp = |rng: &mut Rng| -> [u16; 69] {
        let mut c = [0u16; 69];
        mk_ns_cdf(rng, 2, &mut c[0..3]); mk_ns_cdf(rng, 11, &mut c[3..15]); mk_ns_cdf(rng, 2, &mut c[15..18]);
        for i in 0..10 { let o = 18 + i * 3; mk_ns_cdf(rng, 2, &mut c[o..o + 3]); }
        for i in 0..2 { let o = 48 + i * 5; mk_ns_cdf(rng, 4, &mut c[o..o + 5]); }
        mk_ns_cdf(rng, 4, &mut c[58..63]); mk_ns_cdf(rng, 2, &mut c[63..66]); mk_ns_cdf(rng, 2, &mut c[66..69]);
        c
    };
    let gen_cdfs = |rng: &mut Rng, cfl: bool| -> KfCdfs {
        let mut c = KfCdfs {
            seg: [0; 9], skip: [0; 3], delta_q: [0; 5], delta_lf_multi: [[0; 5]; 4], delta_lf: [0; 5],
            intrabc: [0; 3], ndvc_joints: [0; 5], ndvc_comp0: mk_comp(rng), ndvc_comp1: mk_comp(rng),
            y_mode: [0; 14], y_angle: [0; 8], uv_mode: [0; 15], cfl_sign: [0; 9], cfl_alpha: [[0; 17]; 6],
            uv_angle: [0; 8], pal_y_mode: [0; 3], pal_y_size: [0; 8], pal_uv_mode: [0; 3], pal_uv_size: [0; 8],
            fi_use: [0; 3], fi_mode: [0; 6],
        };
        mk_ns_cdf(rng, 8, &mut c.seg); mk_ns_cdf(rng, 2, &mut c.skip); mk_ns_cdf(rng, 2, &mut c.intrabc);
        mk_ns_cdf(rng, 4, &mut c.ndvc_joints); mk_ns_cdf(rng, 13, &mut c.y_mode); mk_ns_cdf(rng, 7, &mut c.y_angle);
        mk_ns_cdf(rng, if cfl { 14 } else { 13 }, &mut c.uv_mode); mk_ns_cdf(rng, 8, &mut c.cfl_sign);
        for a in c.cfl_alpha.iter_mut() { mk_ns_cdf(rng, 16, a); } mk_ns_cdf(rng, 7, &mut c.uv_angle);
        mk_ns_cdf(rng, 2, &mut c.fi_use); mk_ns_cdf(rng, 5, &mut c.fi_mode);
        c
    };
    fn gen_tree(rng: &mut Rng, bsize: usize, tree: &mut Vec<i8>, leaves: &mut Vec<usize>) {
        if bsize > 3 && rng.next() & 1 == 1 {
            tree.push(3); // SPLIT
            let sub = get_partition_subsize(bsize, 3) as usize;
            for _ in 0..4 { gen_tree(rng, sub, tree, leaves); }
        } else {
            tree.push(0); // NONE leaf
            leaves.push(bsize);
        }
    }
    for _ in 0..40_000 {
        let seg_enabled = rng.next() & 1 == 1;
        let update_map = seg_enabled && rng.next() & 1 == 1;
        let last = (rng.next() % 8) as i32;
        let mono = rng.next() & 1 == 1;
        let chroma_ref = rng.next() & 1 == 1;
        let cfl_allowed = rng.next() & 1 == 1;
        let filter_allowed = rng.next() & 1 == 1;
        let mut tree = Vec::new();
        let mut leaves = Vec::new();
        gen_tree(&mut rng, 12, &mut tree, &mut leaves); // 64x64 SB
        // one MbModeInfoKf per leaf (intra: allow_intrabc=true + use_intrabc=0; no delta/cdef state)
        let mut infos = Vec::with_capacity(leaves.len());
        for &lb in &leaves {
            let y_mode = (rng.next() % 13) as i32;
            let y_ang = if use_angle_delta(lb) && is_directional_mode(y_mode) { (rng.next() % 7) as i32 - 3 } else { 0 };
            let uv_n = if cfl_allowed { 14 } else { 13 };
            let uv_mode = if !mono && chroma_ref { (rng.next() % uv_n as u64) as i32 } else { 0 };
            let js = if uv_mode == 13 { (rng.next() % 8) as i32 } else { 0 };
            let (su, sv) = ((js + 1) / 3, (js + 1) % 3);
            let u = if uv_mode == 13 && su != 0 { (rng.next() % 16) as i32 } else { 0 };
            let v = if uv_mode == 13 && sv != 0 { (rng.next() % 16) as i32 } else { 0 };
            let uv_intra = get_uv_mode(uv_mode as usize);
            let uv_ang = if !mono && chroma_ref && use_angle_delta(lb) && is_directional_mode(uv_intra) { (rng.next() % 7) as i32 - 3 } else { 0 };
            let use_fi = if filter_allowed { (rng.next() & 1) as i32 } else { 0 };
            infos.push(MbModeInfoKf {
                segment_id: (rng.next() % (last as u64 + 1)) as i32, skip: (rng.next() & 1) as i32,
                cdef_strength: 0, current_qindex: 100, delta_lf: [0; 4], delta_lf_from_base: 0,
                use_intrabc: 0, dv_row: 0, dv_col: 0, y_mode, angle_delta_y: y_ang, uv_mode,
                cfl_alpha_idx: (u << 4) | v, cfl_joint_sign: js, angle_delta_uv: uv_ang,
                palette_size: [0, 0], use_filter_intra: use_fi, filter_intra_mode: if use_fi != 0 { (rng.next() % 5) as i32 } else { 0 },
            });
        }
        let base_state = KfBlockState {
            segid_preskip: rng.next() & 1 == 1, seg_enabled, update_map,
            seg_pred: (rng.next() % (last as u64 + 1)) as i32, last_active_segid: last,
            seg_skip_active: false, mi_row: 0, mi_col: 0, mib_size: 16, sb_size: 12, bsize: 12,
            coded_lossless: false, allow_intrabc: true, cdef_bits: 0, dq_present: false,
            dlf_present: false, dlf_multi: false, num_planes: 3, dq_res: 1, dlf_res: 1,
            monochrome: mono, is_chroma_ref: chroma_ref, cfl_allowed, allow_palette: false, bit_depth: 8,
            filter_allowed, mb_to_top_edge: 0, has_above: false, has_left: false,
            cdef_transmitted: [false; 4], current_base_qindex: 100, xd_delta_lf: [0; 4], xd_delta_lf_from_base: 0,
        };
        let cdfs0 = gen_cdfs(&mut rng, cfl_allowed);
        let mut enc = OdEcEnc::new();
        let mut ce = cdfs0.clone();
        let mut se = base_state.clone();
        let mut above_e = [0i8; 128];
        let mut left_e = [0i8; 32];
        let mut arena_e = [[0u16; 11]; 20];
        for (c, slot) in arena_e.iter_mut().enumerate() {
            let bsl = c / 4;
            let ns = if bsl == 0 { 4 } else if bsl == 4 { 8 } else { 10 };
            mk_ns_cdf(&mut rng, ns, slot);
        }
        let arena0 = arena_e;
        let n = write_modes_b(&mut enc, &mut above_e, &mut left_e, &mut arena_e, &mut ce, &mut se, &tree, &infos, 0, 0, 12);
        let b = enc.done().to_vec();
        let mut dec = OdEcDec::new(&b);
        let mut cd = cdfs0.clone();
        let mut sd = base_state.clone();
        let mut above_d = [0i8; 128];
        let mut left_d = [0i8; 32];
        let mut arena_d = arena0;
        let (gtree, ginfos) = read_modes_b(&mut dec, &mut above_d, &mut left_d, &mut arena_d, &mut cd, &mut sd, 0, 0, 12);
        assert_eq!(gtree, tree, "partition tree");
        assert_eq!(n, infos.len(), "block count");
        assert_eq!(ginfos.len(), infos.len(), "decoded block count");
        for (gi, fi) in ginfos.iter().zip(&infos) {
            assert_eq!(gi.skip, fi.skip, "block skip");
            assert_eq!(gi.y_mode, fi.y_mode, "block y_mode");
            assert_eq!(gi.uv_mode, fi.uv_mode, "block uv_mode");
        }
        // CDF adaptation + partition arena + above context lockstep across the whole SB.
        assert_eq!(ce.y_mode, cd.y_mode, "y_mode cdf");
        assert_eq!(ce.skip, cd.skip, "skip cdf");
        assert_eq!(ce.cfl_alpha, cd.cfl_alpha, "cfl cdf");
        assert_eq!(arena_e, arena_d, "partition arena");
        assert_eq!(above_e, above_d, "above ctx");
    }
}

// ===================== FRAME_CONTEXT context-selection layer =====================

/// Random-valid fill for every CDF region of a KfFrameContext (coeff arena unused
/// by the mode-info path — length 0).
#[allow(clippy::items_after_test_module)]
fn mk_frame_ctx(rng: &mut Rng) -> aom_entropy::partition::KfFrameContext {
    use aom_entropy::partition::KfFrameContext;
    let mk_comp = |rng: &mut Rng| -> [u16; 69] {
        let mut c = [0u16; 69];
        mk_ns_cdf(rng, 2, &mut c[0..3]);
        mk_ns_cdf(rng, 11, &mut c[3..15]);
        mk_ns_cdf(rng, 2, &mut c[15..18]);
        for i in 0..10 { let o = 18 + i * 3; mk_ns_cdf(rng, 2, &mut c[o..o + 3]); }
        for i in 0..2 { let o = 48 + i * 5; mk_ns_cdf(rng, 4, &mut c[o..o + 5]); }
        mk_ns_cdf(rng, 4, &mut c[58..63]);
        mk_ns_cdf(rng, 2, &mut c[63..66]);
        mk_ns_cdf(rng, 2, &mut c[66..69]);
        c
    };
    let mut f = KfFrameContext::zeroed(0);
    for row in f.kf_y.iter_mut() {
        for cell in row.iter_mut() { mk_ns_cdf(rng, 13, cell); }
    }
    for (cfl, plane) in f.uv_mode.iter_mut().enumerate() {
        // ns = 14 with CfL / 13 without; slice covers ns+1 slots (count last).
        for cell in plane.iter_mut() { mk_ns_cdf(rng, 13 + cfl, &mut cell[..14 + cfl]); }
    }
    for a in f.angle_delta.iter_mut() { mk_ns_cdf(rng, 7, a); }
    for s in f.skip.iter_mut() { mk_ns_cdf(rng, 2, s); }
    for s in f.seg_spatial.iter_mut() { mk_ns_cdf(rng, 8, s); }
    for (c, slot) in f.partition.iter_mut().enumerate() {
        let bsl = c / 4;
        let ns = if bsl == 0 { 4 } else if bsl == 4 { 8 } else { 10 };
        mk_ns_cdf(rng, ns, slot);
    }
    for b in f.palette_y_mode.iter_mut() {
        for c in b.iter_mut() { mk_ns_cdf(rng, 2, c); }
    }
    for c in f.palette_uv_mode.iter_mut() { mk_ns_cdf(rng, 2, c); }
    for c in f.palette_y_size.iter_mut() { mk_ns_cdf(rng, 7, c); }
    for c in f.palette_uv_size.iter_mut() { mk_ns_cdf(rng, 7, c); }
    for c in f.filter_intra.iter_mut() { mk_ns_cdf(rng, 2, c); }
    mk_ns_cdf(rng, 5, &mut f.filter_intra_mode);
    mk_ns_cdf(rng, 8, &mut f.cfl_sign);
    for a in f.cfl_alpha.iter_mut() { mk_ns_cdf(rng, 16, a); }
    mk_ns_cdf(rng, 4, &mut f.delta_q);
    for m in f.delta_lf_multi.iter_mut() { mk_ns_cdf(rng, 4, m); }
    mk_ns_cdf(rng, 4, &mut f.delta_lf);
    mk_ns_cdf(rng, 2, &mut f.intrabc);
    mk_ns_cdf(rng, 4, &mut f.ndvc_joints);
    f.ndvc_comp0 = mk_comp(rng);
    f.ndvc_comp1 = mk_comp(rng);
    for cat in f.tx_size.iter_mut() {
        for c in cat.iter_mut() { mk_ns_cdf(rng, 3, c); }
    }
    for sq in f.ext_tx_1ddct.iter_mut() {
        for c in sq.iter_mut() { mk_ns_cdf(rng, 7, c); }
    }
    for sq in f.ext_tx_dtt4.iter_mut() {
        for c in sq.iter_mut() { mk_ns_cdf(rng, 5, c); }
    }
    f
}

/// Field-by-field KfFrameContext equality (named-field failure messages).
fn assert_fc_eq(
    a: &aom_entropy::partition::KfFrameContext,
    b: &aom_entropy::partition::KfFrameContext,
    what: &str,
) {
    assert_eq!(a.kf_y, b.kf_y, "{what}: kf_y cdf");
    assert_eq!(a.uv_mode, b.uv_mode, "{what}: uv_mode cdf");
    assert_eq!(a.angle_delta, b.angle_delta, "{what}: angle_delta cdf");
    assert_eq!(a.skip, b.skip, "{what}: skip cdf");
    assert_eq!(a.seg_spatial, b.seg_spatial, "{what}: seg_spatial cdf");
    assert_eq!(a.partition, b.partition, "{what}: partition arena");
    assert_eq!(a.palette_y_mode, b.palette_y_mode, "{what}: palette_y_mode cdf");
    assert_eq!(a.palette_uv_mode, b.palette_uv_mode, "{what}: palette_uv_mode cdf");
    assert_eq!(a.palette_y_size, b.palette_y_size, "{what}: palette_y_size cdf");
    assert_eq!(a.palette_uv_size, b.palette_uv_size, "{what}: palette_uv_size cdf");
    assert_eq!(a.filter_intra, b.filter_intra, "{what}: filter_intra cdf");
    assert_eq!(a.filter_intra_mode, b.filter_intra_mode, "{what}: filter_intra_mode cdf");
    assert_eq!(a.cfl_sign, b.cfl_sign, "{what}: cfl_sign cdf");
    assert_eq!(a.cfl_alpha, b.cfl_alpha, "{what}: cfl_alpha cdf");
    assert_eq!(a.delta_q, b.delta_q, "{what}: delta_q cdf");
    assert_eq!(a.delta_lf_multi, b.delta_lf_multi, "{what}: delta_lf_multi cdf");
    assert_eq!(a.delta_lf, b.delta_lf, "{what}: delta_lf cdf");
    assert_eq!(a.intrabc, b.intrabc, "{what}: intrabc cdf");
    assert_eq!(a.ndvc_joints, b.ndvc_joints, "{what}: ndvc_joints cdf");
    assert_eq!(a.ndvc_comp0, b.ndvc_comp0, "{what}: ndvc_comp0 cdf");
    assert_eq!(a.ndvc_comp1, b.ndvc_comp1, "{what}: ndvc_comp1 cdf");
    assert_eq!(a.tx_size, b.tx_size, "{what}: tx_size cdf");
    assert_eq!(a.ext_tx_1ddct, b.ext_tx_1ddct, "{what}: ext_tx_1ddct cdf");
    assert_eq!(a.ext_tx_dtt4, b.ext_tx_dtt4, "{what}: ext_tx_dtt4 cdf");
    assert_eq!(a.coeff, b.coeff, "{what}: coeff arena");
}

/// The `_fc` writer must be byte- and state-identical to the C-validated
/// `write_mb_modes_kf` when the latter is handed the SAME instances the selection
/// rules pick (selection functions each validated against C on their own). Cases
/// where the Y and UV angle deltas land on the same `angle_delta` instance are the
/// one shape the pre-selected API cannot express (one shared instance adapting
/// twice); those are counted and covered by the `_fc` roundtrip test below.
#[test]
fn mb_modes_kf_fc_matches_preselected_writer() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        filter_intra_allowed, get_uv_mode, get_y_mode_ctx, is_directional_mode,
        read_mb_modes_kf_fc, skip_txfm_context, use_angle_delta, write_mb_modes_kf,
        write_mb_modes_kf_fc, KfBlockState, KfCdfs, MbModeInfoKf, MiNbrKf,
    };
    let mut rng = Rng(0xfc_c0de_57c7_0001u64);
    let mut same_angle_cases = 0usize;
    let mut compared_cases = 0usize;
    for _ in 0..60_000 {
        let bsize = [0usize, 3, 6, 9, 12][(rng.next() % 5) as usize];
        let mono = rng.next() & 1 == 1;
        let chroma_ref = rng.next() & 1 == 1;
        let cfl_allowed = !mono && rng.next() & 1 == 1;
        let allow_intrabc = rng.next().is_multiple_of(8);
        let enable_fi = rng.next() & 1 == 1;
        let dq_res = 1 << (rng.next() % 4);
        let dlf_res = 1 << (rng.next() % 4);
        let st = KfBlockState {
            segid_preskip: false, seg_enabled: false, update_map: false, seg_pred: 0,
            last_active_segid: 0, seg_skip_active: false, mi_row: 0, mi_col: 0, mib_size: 16,
            sb_size: 12, bsize, coded_lossless: false, allow_intrabc,
            cdef_bits: (rng.next() % 4) as u32, dq_present: rng.next() & 1 == 1,
            dlf_present: rng.next() & 1 == 1, dlf_multi: rng.next() & 1 == 1,
            num_planes: if mono { 1 } else { 3 }, dq_res, dlf_res, monochrome: mono,
            is_chroma_ref: chroma_ref, cfl_allowed, allow_palette: false, bit_depth: 8,
            filter_allowed: false, mb_to_top_edge: 0, has_above: false, has_left: false,
            cdef_transmitted: [false; 4], current_base_qindex: 100,
            xd_delta_lf: [0; 4], xd_delta_lf_from_base: 0,
        };
        let mut st = st;
        st.dlf_present = st.dq_present && st.dlf_present;
        let above = (rng.next() & 1 == 1)
            .then(|| MiNbrKf { y_mode: (rng.next() % 13) as i32, skip_txfm: (rng.next() & 1) as i32 });
        let left = (rng.next() & 1 == 1)
            .then(|| MiNbrKf { y_mode: (rng.next() % 13) as i32, skip_txfm: (rng.next() & 1) as i32 });

        // random valid block info
        let skip = (rng.next() & 1) as i32;
        let use_intrabc = if allow_intrabc { (rng.next() & 1) as i32 } else { 0 };
        let (dr, dc) = if use_intrabc != 0 {
            (((rng.next() % 51) as i32 - 25) * 8, ((rng.next() % 51) as i32 - 25) * 8)
        } else {
            (0, 0)
        };
        // an intrabc block's mode is DC_PRED (read_intrabc_info sets it; not coded)
        let y_mode = if use_intrabc != 0 { 0 } else { (rng.next() % 13) as i32 };
        let y_ang_coded = use_intrabc == 0 && use_angle_delta(bsize) && is_directional_mode(y_mode);
        let y_ang = if y_ang_coded { (rng.next() % 7) as i32 - 3 } else { 0 };
        let uv_coded = use_intrabc == 0 && !mono && chroma_ref;
        let uv_n = if cfl_allowed { 14 } else { 13 };
        let uv_mode = if uv_coded { (rng.next() % uv_n as u64) as i32 } else { 0 };
        let js = if uv_coded && uv_mode == 13 { (rng.next() % 8) as i32 } else { 0 };
        let (su, sv) = ((js + 1) / 3, (js + 1) % 3);
        let u = if uv_coded && uv_mode == 13 && su != 0 { (rng.next() % 16) as i32 } else { 0 };
        let v = if uv_coded && uv_mode == 13 && sv != 0 { (rng.next() % 16) as i32 } else { 0 };
        let uv_intra = get_uv_mode(uv_mode as usize);
        let uv_ang_coded = uv_coded && use_angle_delta(bsize) && is_directional_mode(uv_intra);
        let uv_ang = if uv_ang_coded { (rng.next() % 7) as i32 - 3 } else { 0 };
        let fi_allowed = use_intrabc == 0 && filter_intra_allowed(enable_fi, bsize, y_mode, 0);
        let use_fi = if fi_allowed && rng.next() & 1 == 1 { 1 } else { 0 };
        let fi_mode = if use_fi != 0 { (rng.next() % 5) as i32 } else { 0 };
        // encoder invariant: current_qindex deviates from the carry only when the
        // delta-q gate ((bsize != sb_size || !skip) at the SB upper-left) codes it
        let dq_coded = st.dq_present && (bsize != st.sb_size || skip == 0);
        // encoder-valid target: in [1, 255] so the reader's normative clamp is a no-op
        let cur_q = if dq_coded {
            100 + ((rng.next() % 41) as i32 - 20).clamp(-(99 / dq_res), 155 / dq_res) * dq_res
        } else {
            100
        };
        let info = MbModeInfoKf {
            segment_id: 0, skip,
            cdef_strength: if st.cdef_bits > 0 { (rng.next() % (1u64 << st.cdef_bits)) as i32 } else { 0 },
            current_qindex: cur_q, delta_lf: [0; 4], delta_lf_from_base: 0, use_intrabc,
            dv_row: dr, dv_col: dc, y_mode, angle_delta_y: y_ang, uv_mode,
            cfl_alpha_idx: (u << 4) | v, cfl_joint_sign: js, angle_delta_uv: uv_ang,
            palette_size: [0, 0], use_filter_intra: use_fi, filter_intra_mode: fi_mode,
        };
        let f0 = mk_frame_ctx(&mut rng);

        // path 1: the _fc writer (inline selection)
        let mut f1 = f0.clone();
        let mut s1 = st.clone();
        let mut e1 = OdEcEnc::new();
        write_mb_modes_kf_fc(&mut e1, &info, &mut f1, &mut s1, enable_fi, above, left);
        let b1 = e1.done().to_vec();

        // the _fc reader roundtrips it (every case, including shared-angle ones)
        {
            let mut fr = f0.clone();
            let mut sr = st.clone();
            let mut dec = OdEcDec::new(&b1);
            let got = read_mb_modes_kf_fc(&mut dec, &mut fr, &mut sr, enable_fi, above, left);
            // read_cdef reports -1 when no strength is read (intrabc frame / skip block)
            let mut expected = info.clone();
            if allow_intrabc || skip != 0 {
                expected.cdef_strength = -1;
            }
            assert_eq!(got, expected, "fc roundtrip info");
            assert_fc_eq(&f1, &fr, "fc roundtrip");
            assert_eq!(
                (s1.current_base_qindex, s1.xd_delta_lf, s1.xd_delta_lf_from_base, s1.cdef_transmitted),
                (sr.current_base_qindex, sr.xd_delta_lf, sr.xd_delta_lf_from_base, sr.cdef_transmitted),
                "fc roundtrip carries"
            );
        }

        // path 2: the C-validated pre-selected writer on hand-picked instances
        let y_idx = if y_ang_coded { (y_mode - 1) as usize } else { 0 };
        let uv_idx = if uv_ang_coded { (uv_intra - 1) as usize } else { 0 };
        if y_ang_coded && uv_ang_coded && y_idx == uv_idx {
            // one shared instance adapting twice — inexpressible with pre-selected
            // per-symbol instances; covered by the roundtrip above.
            same_angle_cases += 1;
            continue;
        }
        compared_cases += 1;
        let skip_ctx = skip_txfm_context(
            above.map_or(0, |m| m.skip_txfm),
            left.map_or(0, |m| m.skip_txfm),
        ) as usize;
        let (ac, lc) = get_y_mode_ctx(above.map(|m| m.y_mode), left.map(|m| m.y_mode));
        let mut f2 = f0.clone();
        let mut c = KfCdfs {
            seg: f2.seg_spatial[0],
            skip: f2.skip[skip_ctx],
            delta_q: f2.delta_q,
            delta_lf_multi: f2.delta_lf_multi,
            delta_lf: f2.delta_lf,
            intrabc: f2.intrabc,
            ndvc_joints: f2.ndvc_joints,
            ndvc_comp0: f2.ndvc_comp0,
            ndvc_comp1: f2.ndvc_comp1,
            y_mode: f2.kf_y[ac][lc],
            y_angle: f2.angle_delta[y_idx],
            uv_mode: f2.uv_mode[cfl_allowed as usize][y_mode as usize],
            cfl_sign: f2.cfl_sign,
            cfl_alpha: f2.cfl_alpha,
            uv_angle: f2.angle_delta[uv_idx],
            pal_y_mode: [0; 3],
            pal_y_size: [0; 8],
            pal_uv_mode: [0; 3],
            pal_uv_size: [0; 8],
            fi_use: f2.filter_intra[bsize],
            fi_mode: f2.filter_intra_mode,
        };
        let mut s2 = st.clone();
        s2.filter_allowed = filter_intra_allowed(enable_fi, bsize, y_mode, 0);
        let mut e2 = OdEcEnc::new();
        write_mb_modes_kf(&mut e2, &info, &mut c, &mut s2);
        let b2 = e2.done().to_vec();
        assert_eq!(b1, b2, "fc writer vs pre-selected writer bytes");
        // fold the adapted instances back into the array model and compare whole
        f2.seg_spatial[0] = c.seg;
        f2.skip[skip_ctx] = c.skip;
        f2.delta_q = c.delta_q;
        f2.delta_lf_multi = c.delta_lf_multi;
        f2.delta_lf = c.delta_lf;
        f2.intrabc = c.intrabc;
        f2.ndvc_joints = c.ndvc_joints;
        f2.ndvc_comp0 = c.ndvc_comp0;
        f2.ndvc_comp1 = c.ndvc_comp1;
        f2.kf_y[ac][lc] = c.y_mode;
        if y_ang_coded {
            f2.angle_delta[y_idx] = c.y_angle;
        }
        if uv_coded {
            f2.uv_mode[cfl_allowed as usize][y_mode as usize] = c.uv_mode;
        }
        if uv_ang_coded {
            f2.angle_delta[uv_idx] = c.uv_angle;
        }
        f2.cfl_sign = c.cfl_sign;
        f2.cfl_alpha = c.cfl_alpha;
        f2.filter_intra[bsize] = c.fi_use;
        f2.filter_intra_mode = c.fi_mode;
        assert_fc_eq(&f1, &f2, "fc writer vs pre-selected writer CDFs");
        assert_eq!(
            (s1.current_base_qindex, s1.xd_delta_lf, s1.xd_delta_lf_from_base, s1.cdef_transmitted),
            (s2.current_base_qindex, s2.xd_delta_lf, s2.xd_delta_lf_from_base, s2.cdef_transmitted),
            "fc writer vs pre-selected writer carries"
        );
    }
    assert!(compared_cases > 40_000, "pre-selected comparison starved: {compared_cases}");
    assert!(same_angle_cases > 100, "shared-angle-instance cases never generated: {same_angle_cases}");
}

/// Multi-block `_fc` roundtrip: sequences of blocks adapt the shared per-context
/// arrays through one coder; decoded info + every CDF array + the state carries stay
/// in lockstep, and the selection is asserted NON-VACUOUS — many distinct context
/// instances must actually adapt across the sweep.
#[test]
fn mb_modes_kf_fc_multi_block_roundtrips_with_ctx_diversity() {
    use aom_entropy::dec::OdEcDec;
    use aom_entropy::enc::OdEcEnc;
    use aom_entropy::partition::{
        filter_intra_allowed, get_uv_mode, is_directional_mode, read_mb_modes_kf_fc,
        use_angle_delta, write_mb_modes_kf_fc, KfBlockState, MbModeInfoKf, MiNbrKf,
    };
    let mut rng = Rng(0xfc_c0de_57c7_0002u64);
    let mut kf_y_cells = [[false; 5]; 5];
    let mut skip_ctxs = [false; 3];
    let mut angle_insts = [false; 8];
    let mut uv_insts = [[false; 13]; 2];
    let mut fi_bsizes = [false; 22];
    for _ in 0..8_000 {
        let mono = rng.next() & 1 == 1;
        let enable_fi = rng.next() & 1 == 1;
        let f0 = mk_frame_ctx(&mut rng);
        let base = KfBlockState {
            segid_preskip: false, seg_enabled: false, update_map: false, seg_pred: 0,
            last_active_segid: 0, seg_skip_active: false, mi_row: 0, mi_col: 0, mib_size: 16,
            sb_size: 12, bsize: 12, coded_lossless: false, allow_intrabc: false,
            cdef_bits: (rng.next() % 4) as u32, dq_present: false, dlf_present: false,
            dlf_multi: false, num_planes: if mono { 1 } else { 3 }, dq_res: 1, dlf_res: 1,
            monochrome: mono, is_chroma_ref: false, cfl_allowed: false, allow_palette: false,
            bit_depth: 8, filter_allowed: false, mb_to_top_edge: 0, has_above: false,
            has_left: false, cdef_transmitted: [false; 4], current_base_qindex: 100,
            xd_delta_lf: [0; 4], xd_delta_lf_from_base: 0,
        };
        // choose a per-sequence block list with fresh neighbours per block
        let nblocks = 4 + (rng.next() % 5) as usize;
        let mut plan = Vec::new();
        for _ in 0..nblocks {
            let bsize = [3usize, 6, 9, 12][(rng.next() % 4) as usize];
            let chroma_ref = !mono && rng.next() & 1 == 1;
            let cfl_allowed = chroma_ref && rng.next() & 1 == 1;
            let above = (rng.next() & 1 == 1).then(|| MiNbrKf {
                y_mode: (rng.next() % 13) as i32,
                skip_txfm: (rng.next() & 1) as i32,
            });
            let left = (rng.next() & 1 == 1).then(|| MiNbrKf {
                y_mode: (rng.next() % 13) as i32,
                skip_txfm: (rng.next() & 1) as i32,
            });
            let y_mode = (rng.next() % 13) as i32;
            let y_ang = if use_angle_delta(bsize) && is_directional_mode(y_mode) {
                (rng.next() % 7) as i32 - 3
            } else {
                0
            };
            let uv_n = if cfl_allowed { 14 } else { 13 };
            let uv_mode = if chroma_ref { (rng.next() % uv_n as u64) as i32 } else { 0 };
            let js = if chroma_ref && uv_mode == 13 { (rng.next() % 8) as i32 } else { 0 };
            let (su, sv) = ((js + 1) / 3, (js + 1) % 3);
            let u = if chroma_ref && uv_mode == 13 && su != 0 { (rng.next() % 16) as i32 } else { 0 };
            let v = if chroma_ref && uv_mode == 13 && sv != 0 { (rng.next() % 16) as i32 } else { 0 };
            let uv_intra = get_uv_mode(uv_mode as usize);
            let uv_ang = if chroma_ref && use_angle_delta(bsize) && is_directional_mode(uv_intra) {
                (rng.next() % 7) as i32 - 3
            } else {
                0
            };
            let fi_allowed = filter_intra_allowed(enable_fi, bsize, y_mode, 0);
            let use_fi = if fi_allowed && rng.next() & 1 == 1 { 1 } else { 0 };
            let fi_mode = if use_fi != 0 { (rng.next() % 5) as i32 } else { 0 };
            let info = MbModeInfoKf {
                segment_id: 0, skip: (rng.next() & 1) as i32,
                cdef_strength: if base.cdef_bits > 0 { (rng.next() % (1u64 << base.cdef_bits)) as i32 } else { 0 },
                current_qindex: 100, delta_lf: [0; 4], delta_lf_from_base: 0, use_intrabc: 0,
                dv_row: 0, dv_col: 0, y_mode, angle_delta_y: y_ang, uv_mode,
                cfl_alpha_idx: (u << 4) | v, cfl_joint_sign: js, angle_delta_uv: uv_ang,
                palette_size: [0, 0], use_filter_intra: use_fi, filter_intra_mode: fi_mode,
            };
            plan.push((bsize, chroma_ref, cfl_allowed, above, left, info));
        }
        // encode the sequence over one coder
        let mut fe = f0.clone();
        let mut se = base.clone();
        let mut enc = OdEcEnc::new();
        for (bsize, chroma_ref, cfl_allowed, above, left, info) in &plan {
            se.bsize = *bsize;
            se.is_chroma_ref = *chroma_ref;
            se.cfl_allowed = *cfl_allowed;
            write_mb_modes_kf_fc(&mut enc, info, &mut fe, &mut se, enable_fi, *above, *left);
        }
        let b = enc.done().to_vec();
        // decode it back
        let mut fd = f0.clone();
        let mut sd = base.clone();
        let mut dec = OdEcDec::new(&b);
        for (i, (bsize, chroma_ref, cfl_allowed, above, left, info)) in plan.iter().enumerate() {
            sd.bsize = *bsize;
            sd.is_chroma_ref = *chroma_ref;
            sd.cfl_allowed = *cfl_allowed;
            let got = read_mb_modes_kf_fc(&mut dec, &mut fd, &mut sd, enable_fi, *above, *left);
            // every block sits at the SB upper-left (0,0): cdef_transmitted resets
            // per block, so non-skip blocks read the strength, skip blocks report -1
            let mut expected = info.clone();
            if info.skip != 0 {
                expected.cdef_strength = -1;
            }
            assert_eq!(&got, &expected, "block {i} info");
        }
        assert_fc_eq(&fe, &fd, "multi-block fc");
        assert_eq!(
            (se.cdef_transmitted, se.current_base_qindex),
            (sd.cdef_transmitted, sd.current_base_qindex),
            "multi-block carries"
        );
        // tally which context instances adapted (differ from the initial fill)
        for ((flag, new), old) in kf_y_cells
            .iter_mut()
            .flatten()
            .zip(fd.kf_y.iter().flatten())
            .zip(f0.kf_y.iter().flatten())
        {
            *flag |= new != old;
        }
        for ((flag, new), old) in skip_ctxs.iter_mut().zip(&fd.skip).zip(&f0.skip) {
            *flag |= new != old;
        }
        for ((flag, new), old) in angle_insts.iter_mut().zip(&fd.angle_delta).zip(&f0.angle_delta) {
            *flag |= new != old;
        }
        for ((flag, new), old) in uv_insts
            .iter_mut()
            .flatten()
            .zip(fd.uv_mode.iter().flatten())
            .zip(f0.uv_mode.iter().flatten())
        {
            *flag |= new != old;
        }
        for ((flag, new), old) in fi_bsizes.iter_mut().zip(&fd.filter_intra).zip(&f0.filter_intra) {
            *flag |= new != old;
        }
    }
    // the selection must be exercised across many distinct instances — a
    // single-shared-CDF regression would collapse these counts.
    let kf_y_n: usize = kf_y_cells.iter().flatten().filter(|&&x| x).count();
    let skip_n = skip_ctxs.iter().filter(|&&x| x).count();
    let angle_n = angle_insts.iter().filter(|&&x| x).count();
    let uv_n: usize = uv_insts.iter().flatten().filter(|&&x| x).count();
    let fi_n = fi_bsizes.iter().filter(|&&x| x).count();
    assert!(kf_y_n >= 20, "kf_y context diversity too low: {kf_y_n}/25");
    assert!(skip_n == 3, "skip context diversity too low: {skip_n}/3");
    assert!(angle_n == 8, "angle_delta instance diversity too low: {angle_n}/8");
    assert!(uv_n >= 20, "uv_mode instance diversity too low: {uv_n}/26");
    assert!(fi_n >= 3, "filter_intra bsize diversity too low: {fi_n}/22");
}
