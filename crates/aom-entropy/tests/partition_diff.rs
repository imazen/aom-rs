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
