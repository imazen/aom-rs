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
