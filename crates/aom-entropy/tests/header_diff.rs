//! Differential harness for the frame-header quantization params
//! (encode_quantization) vs C libaom's control flow (driven through the real
//! aom_wb primitives), plus an independent spec-layout anchor.

use aom_entropy::header::{encode_quantization, QuantParamsHeader};
use aom_entropy::wb::WriteBitBuffer;
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
    fn dq(&mut self) -> i32 {
        // delta-q is a 7-bit inverse-signed field: [-63, 63], often 0.
        if self.next().is_multiple_of(3) { 0 } else { (self.next() % 127) as i32 - 63 }
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

#[test]
fn encode_quantization_matches_c() {
    let mut rng = Rng(0x9a17_c0de_a11a_0009);
    for _ in 0..200_000 {
        let qp = QuantParamsHeader {
            base_qindex: rng.range(0, 256),
            y_dc_delta_q: rng.dq(),
            u_dc_delta_q: rng.dq(),
            u_ac_delta_q: rng.dq(),
            v_dc_delta_q: rng.dq(),
            v_ac_delta_q: rng.dq(),
            using_qmatrix: rng.next().is_multiple_of(2),
            qmatrix_level_y: rng.range(0, 16),
            qmatrix_level_u: rng.range(0, 16),
            qmatrix_level_v: rng.range(0, 16),
        };
        let num_planes = if rng.next().is_multiple_of(4) { 1 } else { 3 };
        let separate_uv = rng.next().is_multiple_of(2);

        let mut wb = WriteBitBuffer::new();
        encode_quantization(&mut wb, &qp, num_planes, separate_uv);
        let got = wb.bytes().to_vec();
        let want = c::ref_encode_quantization(
            qp.base_qindex, qp.y_dc_delta_q, qp.u_dc_delta_q, qp.u_ac_delta_q, qp.v_dc_delta_q,
            qp.v_ac_delta_q, qp.using_qmatrix, qp.qmatrix_level_y, qp.qmatrix_level_u,
            qp.qmatrix_level_v, num_planes, separate_uv,
        );
        assert_eq!(got, want, "encode_quantization {qp:?} np={num_planes} sep={separate_uv}");
    }
}

#[test]
fn encode_quantization_spec_anchor() {
    // Monochrome (num_planes=1), all deltas 0, no qm: base_qindex byte + two 0
    // bits (y_dc absent-flag, using_qmatrix) => [base, 0x00].
    let qp = QuantParamsHeader {
        base_qindex: 0x5a,
        y_dc_delta_q: 0,
        u_dc_delta_q: 0,
        u_ac_delta_q: 0,
        v_dc_delta_q: 0,
        v_ac_delta_q: 0,
        using_qmatrix: false,
        qmatrix_level_y: 0,
        qmatrix_level_u: 0,
        qmatrix_level_v: 0,
    };
    let mut wb = WriteBitBuffer::new();
    encode_quantization(&mut wb, &qp, 1, false);
    assert_eq!(wb.bytes(), &[0x5a, 0x00]);
}

#[test]
fn encode_loopfilter_matches_c() {
    use aom_entropy::header::{encode_loopfilter, LoopfilterHeader};
    let mut rng = Rng(0x10f1_c0de_a11a_0009);
    for _ in 0..200_000 {
        let deltas8 = |rng: &mut Rng| -> [i8; 8] {
            let mut a = [0i8; 8];
            for x in &mut a {
                *x = (rng.next() % 127) as i8 - 63;
            }
            a
        };
        let deltas2 = |rng: &mut Rng| -> [i8; 2] {
            [(rng.next() % 127) as i8 - 63, (rng.next() % 127) as i8 - 63]
        };
        // Sometimes make last == current so "changed"/"meaningful" go both ways.
        let ref_deltas = deltas8(&mut rng);
        let last_ref = if rng.next().is_multiple_of(3) { ref_deltas } else { deltas8(&mut rng) };
        let mode_deltas = deltas2(&mut rng);
        let last_mode = if rng.next().is_multiple_of(3) { mode_deltas } else { deltas2(&mut rng) };
        let lf = LoopfilterHeader {
            allow_intrabc: rng.next().is_multiple_of(7),
            filter_level: [rng.range(0, 64), rng.range(0, 64)],
            filter_level_u: rng.range(0, 64),
            filter_level_v: rng.range(0, 64),
            sharpness_level: rng.range(0, 8),
            mode_ref_delta_enabled: rng.next().is_multiple_of(2),
            mode_ref_delta_update: rng.next().is_multiple_of(2),
            ref_deltas,
            mode_deltas,
            last_ref_deltas: last_ref,
            last_mode_deltas: last_mode,
        };
        let num_planes = if rng.next().is_multiple_of(4) { 1 } else { 3 };
        let mut wb = WriteBitBuffer::new();
        encode_loopfilter(&mut wb, &lf, num_planes);
        let got = wb.bytes().to_vec();
        let want = c::ref_encode_loopfilter(
            lf.allow_intrabc, lf.filter_level, lf.filter_level_u, lf.filter_level_v,
            lf.sharpness_level, lf.mode_ref_delta_enabled, lf.mode_ref_delta_update, &lf.ref_deltas,
            &lf.mode_deltas, &lf.last_ref_deltas, &lf.last_mode_deltas, num_planes,
        );
        assert_eq!(got, want, "encode_loopfilter {lf:?} np={num_planes}");
    }
}

#[test]
fn encode_cdef_matches_c() {
    use aom_entropy::header::{encode_cdef, CdefHeader};
    let mut rng = Rng(0xcde1_c0de_a11a_0009);
    for _ in 0..200_000 {
        let cdef_bits = rng.range(0, 4); // nb_cdef_strengths = 1<<cdef_bits (1..8)
        let nb = 1usize << cdef_bits;
        let mut y = [0i32; 8];
        let mut uv = [0i32; 8];
        for k in 0..8 {
            y[k] = rng.range(0, 64);
            uv[k] = rng.range(0, 64);
        }
        let cdef = CdefHeader {
            enable_cdef: rng.next().is_multiple_of(5),
            allow_intrabc: rng.next().is_multiple_of(7),
            cdef_damping: rng.range(3, 7), // damping-3 fits 2 bits => damping 3..6
            cdef_bits,
            nb_cdef_strengths: nb,
            cdef_strengths: y,
            cdef_uv_strengths: uv,
        };
        let num_planes = if rng.next().is_multiple_of(4) { 1 } else { 3 };
        let mut wb = WriteBitBuffer::new();
        encode_cdef(&mut wb, &cdef, num_planes);
        let got = wb.bytes().to_vec();
        let want = c::ref_encode_cdef(cdef.enable_cdef, cdef.allow_intrabc, cdef.cdef_damping, cdef.cdef_bits, nb, &y, &uv, num_planes);
        assert_eq!(got, want, "encode_cdef {cdef:?} np={num_planes}");
    }
}

#[test]
fn encode_segmentation_matches_c() {
    use aom_entropy::header::{encode_segmentation, SegmentationHeader};
    let mut rng = Rng(0x5e91_c0de_a11a_0009);
    for _ in 0..200_000 {
        let mut feature_mask = [0u32; 8];
        let mut feature_data = [[0i32; 8]; 8];
        for (mask, row) in feature_mask.iter_mut().zip(feature_data.iter_mut()) {
            // random subset of the 8 features active
            *mask = (rng.next() as u32) & 0xff;
            for cell in row.iter_mut() {
                // span the clamp range on both signs (data_max up to 255)
                *cell = rng.range(-300, 301);
            }
        }
        let seg = SegmentationHeader {
            enabled: rng.next().is_multiple_of(4),
            has_primary_ref: rng.next().is_multiple_of(2),
            update_map: rng.next().is_multiple_of(2),
            temporal_update: rng.next().is_multiple_of(2),
            update_data: rng.next().is_multiple_of(2),
            feature_mask,
            feature_data,
        };
        let mut wb = WriteBitBuffer::new();
        encode_segmentation(&mut wb, &seg);
        let got = wb.bytes().to_vec();
        let want = c::ref_encode_segmentation(seg.enabled, seg.has_primary_ref, seg.update_map, seg.temporal_update, seg.update_data, &feature_mask, &feature_data);
        assert_eq!(got, want, "encode_segmentation {seg:?}");
    }
}

#[test]
fn frame_size_cluster_matches_c() {
    use aom_entropy::header::{
        write_frame_interp_filter, write_frame_size, write_render_size, write_superres_scale,
        FrameSizeHeader,
    };
    let mut rng = Rng(0xf5ce_c0de_a11a_0009);
    for _ in 0..200_000 {
        // interp filter: 0..=4 (4 = SWITCHABLE)
        let filter = rng.range(0, 5);
        let mut wb = WriteBitBuffer::new();
        write_frame_interp_filter(&mut wb, filter);
        assert_eq!(wb.bytes(), &c::ref_write_frame_interp_filter(filter)[..], "interp_filter {filter}");

        // superres: denom == 8 (no scale) or [9, 16)
        let enable_superres = rng.next().is_multiple_of(2);
        let denom = if rng.next().is_multiple_of(2) { 8 } else { rng.range(9, 17) };
        let mut wb = WriteBitBuffer::new();
        write_superres_scale(&mut wb, enable_superres, denom);
        assert_eq!(wb.bytes(), &c::ref_write_superres_scale(enable_superres, denom)[..], "superres en={enable_superres} d={denom}");

        // render size
        let scaling_active = rng.next().is_multiple_of(2);
        let rw = rng.range(1, 65536);
        let rh = rng.range(1, 65536);
        let mut wb = WriteBitBuffer::new();
        write_render_size(&mut wb, scaling_active, rw, rh);
        assert_eq!(wb.bytes(), &c::ref_write_render_size(scaling_active, rw, rh)[..], "render {scaling_active} {rw}x{rh}");

        // full frame size
        let fs = FrameSizeHeader {
            frame_size_override: rng.next().is_multiple_of(2),
            num_bits_width: rng.range(4, 17) as u32,
            num_bits_height: rng.range(4, 17) as u32,
            superres_upscaled_width: rng.range(1, 65536),
            superres_upscaled_height: rng.range(1, 65536),
            enable_superres,
            scale_denominator: denom,
            scaling_active,
            render_width: rw,
            render_height: rh,
        };
        let mut wb = WriteBitBuffer::new();
        write_frame_size(&mut wb, &fs);
        let want = c::ref_write_frame_size(fs.frame_size_override, fs.num_bits_width, fs.num_bits_height, fs.superres_upscaled_width, fs.superres_upscaled_height, fs.enable_superres, fs.scale_denominator, fs.scaling_active, fs.render_width, fs.render_height);
        assert_eq!(wb.bytes(), &want[..], "frame_size {fs:?}");
    }
}

#[test]
fn write_tile_info_matches_c() {
    use aom_entropy::header::{write_tile_info, TileInfoHeader};
    let mut rng = Rng(0x71fe_c0de_a11a_0009);
    for _ in 0..200_000 {
        let mib_size_log2 = rng.range(4, 6) as u32; // 4 or 5
        let uniform = rng.next().is_multiple_of(2);
        let mut col_start_sb = [0i32; 65];
        let mut row_start_sb = [0i32; 65];

        let (mi_cols, mi_rows, cols, rows, log2_cols, log2_rows, min_c, max_c, min_r, max_r, max_width_sb, max_height_sb);
        if uniform {
            // uniform spacing: log2 in [min, max]; the partition arrays are unused.
            min_c = rng.range(0, 3);
            max_c = min_c + rng.range(0, 4);
            log2_cols = min_c + rng.range(0, (max_c - min_c) + 1);
            min_r = rng.range(0, 3);
            max_r = min_r + rng.range(0, 4);
            log2_rows = min_r + rng.range(0, (max_r - min_r) + 1);
            cols = 1usize << log2_cols;
            rows = 1usize << log2_rows;
            mi_cols = rng.range(1, 4096);
            mi_rows = rng.range(1, 4096);
            max_width_sb = rng.range(1, 64);
            max_height_sb = rng.range(1, 64);
        } else {
            // explicit: build a valid partition summing to width_sb / height_sb.
            let ncols = rng.range(1, 8) as usize;
            let nrows = rng.range(1, 8) as usize;
            let max_tile = rng.range(1, 8);
            let mut wsum = 0;
            for i in 0..ncols {
                let s = rng.range(1, max_tile + 1);
                col_start_sb[i + 1] = col_start_sb[i] + s;
                wsum += s;
            }
            let mut hsum = 0;
            for i in 0..nrows {
                let s = rng.range(1, max_tile + 1);
                row_start_sb[i + 1] = row_start_sb[i] + s;
                hsum += s;
            }
            cols = ncols;
            rows = nrows;
            // mi_cols chosen so ceil_power_of_two(mi_cols, mib) == wsum exactly.
            mi_cols = wsum << mib_size_log2;
            mi_rows = hsum << mib_size_log2;
            max_width_sb = max_tile + rng.range(0, 4); // >= every tile size
            max_height_sb = max_tile + rng.range(0, 4);
            log2_cols = rng.range(0, 4);
            log2_rows = rng.range(0, 4);
            min_c = 0;
            max_c = 6;
            min_r = 0;
            max_r = 6;
        }

        let t = TileInfoHeader {
            mi_cols, mi_rows, mib_size_log2, uniform_spacing: uniform,
            log2_cols, min_log2_cols: min_c, max_log2_cols: max_c,
            log2_rows, min_log2_rows: min_r, max_log2_rows: max_r,
            cols, rows, col_start_sb, row_start_sb, max_width_sb, max_height_sb,
        };
        let mut wb = WriteBitBuffer::new();
        write_tile_info(&mut wb, &t);
        let got = wb.bytes().to_vec();
        let want = c::ref_write_tile_info(mi_cols, mi_rows, mib_size_log2, uniform, log2_cols, min_c, max_c, log2_rows, min_r, max_r, cols, rows, &col_start_sb, &row_start_sb, max_width_sb, max_height_sb);
        assert_eq!(got, want, "write_tile_info {t:?}");
    }
}

#[test]
fn encode_restoration_mode_matches_c() {
    use aom_entropy::header::{encode_restoration_mode, RestorationHeader};
    let mut rng = Rng(0x25e0_c0de_a11a_0009);
    for _ in 0..200_000 {
        let mut frt = [0u8; 3];
        for f in &mut frt {
            *f = rng.range(0, 4) as u8; // RESTORE_NONE..SWITCHABLE
        }
        // restoration_unit_size: exercise the >64 / >128 branches (valid: 64/128/256).
        let rus_pick = |rng: &mut Rng| -> i32 { [64, 128, 256][rng.range(0, 3) as usize] };
        let rus0 = rus_pick(&mut rng);
        let rus1 = if rng.next().is_multiple_of(2) { rus0 } else { rus_pick(&mut rng) };
        let r = RestorationHeader {
            enable_restoration: rng.next().is_multiple_of(2),
            allow_intrabc: rng.next().is_multiple_of(7),
            frame_restoration_type: frt,
            sb_size_128: rng.next().is_multiple_of(2),
            restoration_unit_size: [rus0, rus1, rus1],
            subsampling_x: rng.range(0, 2),
            subsampling_y: rng.range(0, 2),
        };
        let num_planes = if rng.next().is_multiple_of(4) { 1 } else { 3 };
        let mut wb = WriteBitBuffer::new();
        encode_restoration_mode(&mut wb, &r, num_planes);
        let got = wb.bytes().to_vec();
        let frt_i = [frt[0] as i32, frt[1] as i32, frt[2] as i32];
        let want = c::ref_encode_restoration_mode(r.enable_restoration, r.allow_intrabc, &frt_i, r.sb_size_128, &r.restoration_unit_size, r.subsampling_x, r.subsampling_y, num_planes);
        assert_eq!(got, want, "encode_restoration_mode {r:?} np={num_planes}");
    }
}

#[test]
fn write_delta_q_params_and_tx_mode_match_c() {
    use aom_entropy::header::{write_delta_q_params, write_tx_mode, DeltaQParams};
    let mut rng = Rng(0xd17a_c0de_a11a_0009);
    let pow2 = |rng: &mut Rng| -> i32 { 1 << rng.range(0, 4) }; // 1,2,4,8 -> get_msb 0..3
    for _ in 0..200_000 {
        let allow_intrabc = rng.next().is_multiple_of(3);
        // spec-valid: delta_lf_present is only written (and meaningful) when !intrabc.
        let delta_lf_present = !allow_intrabc && rng.next().is_multiple_of(2);
        let d = DeltaQParams {
            base_qindex: rng.range(0, 256),
            delta_q_present: rng.next().is_multiple_of(2),
            delta_q_res: pow2(&mut rng),
            allow_intrabc,
            delta_lf_present,
            delta_lf_res: pow2(&mut rng),
            delta_lf_multi: rng.next().is_multiple_of(2),
        };
        let mut wb = WriteBitBuffer::new();
        write_delta_q_params(&mut wb, &d);
        let got = wb.bytes().to_vec();
        let want = c::ref_write_delta_q_params(d.base_qindex, d.delta_q_present, d.delta_q_res, d.allow_intrabc, d.delta_lf_present, d.delta_lf_res, d.delta_lf_multi);
        assert_eq!(got, want, "write_delta_q_params {d:?}");

        // tx mode
        let coded_lossless = rng.next().is_multiple_of(4);
        let tx_mode_select = rng.next().is_multiple_of(2);
        let mut wb = WriteBitBuffer::new();
        write_tx_mode(&mut wb, coded_lossless, tx_mode_select);
        assert_eq!(wb.bytes(), &c::ref_write_tx_mode(coded_lossless, tx_mode_select)[..], "tx_mode cl={coded_lossless} sel={tx_mode_select}");
    }
}

#[test]
fn write_film_grain_params_matches_c() {
    use aom_entropy::header::{write_film_grain_params, FilmGrainParams};
    let mut rng = Rng(0xf11a_c0de_a11a_0009);
    for _ in 0..200_000 {
        let num_y_points = rng.range(0, 15); // max 14
        let num_cb_points = rng.range(0, 11); // max 10
        let num_cr_points = rng.range(0, 11);
        let ar_coeff_lag = rng.range(0, 4); // 0..3

        let mut scaling_points_y = [[0i32; 2]; 14];
        let mut scaling_points_cb = [[0i32; 2]; 10];
        let mut scaling_points_cr = [[0i32; 2]; 10];
        for pt in &mut scaling_points_y {
            pt[0] = rng.range(0, 256);
            pt[1] = rng.range(0, 256);
        }
        for pt in &mut scaling_points_cb {
            pt[0] = rng.range(0, 256);
            pt[1] = rng.range(0, 256);
        }
        for pt in &mut scaling_points_cr {
            pt[0] = rng.range(0, 256);
            pt[1] = rng.range(0, 256);
        }
        let mut ar_coeffs_y = [0i32; 24];
        let mut ar_coeffs_cb = [0i32; 25];
        let mut ar_coeffs_cr = [0i32; 25];
        for c in &mut ar_coeffs_y {
            *c = rng.range(-128, 128);
        }
        for c in &mut ar_coeffs_cb {
            *c = rng.range(-128, 128);
        }
        for c in &mut ar_coeffs_cr {
            *c = rng.range(-128, 128);
        }

        let p = FilmGrainParams {
            apply_grain: rng.next().is_multiple_of(2),
            random_seed: rng.range(0, 65536),
            is_inter_frame: rng.next().is_multiple_of(2),
            // mostly-true so the full parameter path is exercised more often
            update_parameters: !rng.next().is_multiple_of(4),
            ref_idx: rng.range(0, 8),
            num_y_points,
            scaling_points_y,
            monochrome: rng.next().is_multiple_of(3),
            chroma_scaling_from_luma: rng.next().is_multiple_of(2),
            subsampling_x: rng.range(0, 2),
            subsampling_y: rng.range(0, 2),
            num_cb_points,
            scaling_points_cb,
            num_cr_points,
            scaling_points_cr,
            scaling_shift: rng.range(8, 12), // -8 -> 0..3
            ar_coeff_lag,
            ar_coeffs_y,
            ar_coeffs_cb,
            ar_coeffs_cr,
            ar_coeff_shift: rng.range(6, 10), // -6 -> 0..3
            grain_scale_shift: rng.range(0, 4),
            cb_mult: rng.range(0, 256),
            cb_luma_mult: rng.range(0, 256),
            cb_offset: rng.range(0, 512),
            cr_mult: rng.range(0, 256),
            cr_luma_mult: rng.range(0, 256),
            cr_offset: rng.range(0, 512),
            overlap_flag: rng.next().is_multiple_of(2),
            clip_to_restricted_range: rng.next().is_multiple_of(2),
        };

        let mut wb = WriteBitBuffer::new();
        write_film_grain_params(&mut wb, &p);
        let got = wb.bytes().to_vec();

        let s = [
            p.apply_grain as i32, p.random_seed, p.is_inter_frame as i32,
            p.update_parameters as i32, p.ref_idx, p.num_y_points, p.monochrome as i32,
            p.chroma_scaling_from_luma as i32, p.subsampling_x, p.subsampling_y,
            p.num_cb_points, p.num_cr_points, p.scaling_shift, p.ar_coeff_lag,
            p.ar_coeff_shift, p.grain_scale_shift, p.cb_mult, p.cb_luma_mult, p.cb_offset,
            p.cr_mult, p.cr_luma_mult, p.cr_offset, p.overlap_flag as i32,
            p.clip_to_restricted_range as i32,
        ];
        let flat2 = |a: &[[i32; 2]]| -> Vec<i32> { a.iter().flatten().copied().collect() };
        let spy: [i32; 28] = flat2(&scaling_points_y).try_into().unwrap();
        let spcb: [i32; 20] = flat2(&scaling_points_cb).try_into().unwrap();
        let spcr: [i32; 20] = flat2(&scaling_points_cr).try_into().unwrap();
        let want = c::ref_write_film_grain_params(&s, &spy, &spcb, &spcr, &ar_coeffs_y, &ar_coeffs_cb, &ar_coeffs_cr);
        assert_eq!(got, want, "write_film_grain_params {p:?}");
    }
}

#[test]
fn write_global_motion_matches_c() {
    use aom_entropy::header::{write_global_motion, WarpedMotionParams};
    let mut rng = Rng(0x610b_c0de_a11a_0009);
    // Build wmmat[i] realizing a coded arg in [-4096,4096]: idx 0/1 use shift 10,
    // idx 2/3/4/5 shift 1; idx 2/5 subtract 1<<15. Recovered arg stays in i16 range.
    let shift = |idx: usize| -> u32 { if idx < 2 { 10 } else { 1 } };
    let subtract = |idx: usize| -> i32 { if idx == 2 || idx == 5 { 1 << 15 } else { 0 } };
    for _ in 0..300_000 {
        let allow_hp = rng.next().is_multiple_of(2);
        let mut wmtype = [0i32; 7];
        let mut wmmat = [0i32; 42];
        let mut refmat = [0i32; 42];
        for f in 0..7 {
            // encoder-reachable types: IDENTITY=0, ROTZOOM=2, AFFINE=3
            wmtype[f] = [0i32, 2, 3][rng.range(0, 3) as usize];
            for i in 0..6 {
                let a = rng.range(-4096, 4097);
                let ra = if rng.next().is_multiple_of(3) { a } else { rng.range(-4096, 4097) };
                wmmat[f * 6 + i] = (a + subtract(i)) << shift(i);
                refmat[f * 6 + i] = (ra + subtract(i)) << shift(i);
            }
        }
        let gm: [WarpedMotionParams; 7] = std::array::from_fn(|f| WarpedMotionParams {
            wmtype: wmtype[f] as u8,
            wmmat: std::array::from_fn(|i| wmmat[f * 6 + i]),
        });
        let refgm: [WarpedMotionParams; 7] = std::array::from_fn(|f| WarpedMotionParams {
            wmtype: 0,
            wmmat: std::array::from_fn(|i| refmat[f * 6 + i]),
        });
        let mut wb = WriteBitBuffer::new();
        write_global_motion(&mut wb, &gm, &refgm, allow_hp);
        let got = wb.bytes().to_vec();
        let want = c::ref_write_global_motion(&wmtype, &wmmat, &refmat, allow_hp);
        assert_eq!(got, want, "write_global_motion hp={allow_hp} types={wmtype:?}");
    }
}

#[test]
fn write_sequence_header_matches_c() {
    use aom_entropy::header::{write_sequence_header, SequenceHeaderParams};
    let mut rng = Rng(0x5e90_c0de_a11a_0009);
    for _ in 0..200_000 {
        let num_bits_width = rng.range(4, 17) as u32; // enough to hold max_frame_width-1
        let num_bits_height = rng.range(4, 17) as u32;
        // frame_id lengths: delta in [2, frame_id-1], frame_id such that fields fit.
        let delta_frame_id_length = rng.range(2, 18); // -2 fits 4 bits
        let frame_id_length = delta_frame_id_length + 1 + rng.range(0, 8); // -delta-1 fits 3 bits
        let force_sct = rng.range(0, 3); // 0,1,2
        // when force_sct == 0, force_integer_mv must be 2 (SELECT); else 0/1/2
        let force_integer_mv = if force_sct == 0 { 2 } else { rng.range(0, 3) };
        let s = SequenceHeaderParams {
            num_bits_width,
            num_bits_height,
            max_frame_width: rng.range(1, 1 << num_bits_width.min(20)),
            max_frame_height: rng.range(1, 1 << num_bits_height.min(20)),
            reduced_still_picture_hdr: rng.next().is_multiple_of(3),
            frame_id_numbers_present_flag: rng.next().is_multiple_of(2),
            delta_frame_id_length,
            frame_id_length,
            sb_size_128: rng.next().is_multiple_of(2),
            enable_filter_intra: rng.next().is_multiple_of(2),
            enable_intra_edge_filter: rng.next().is_multiple_of(2),
            enable_interintra_compound: rng.next().is_multiple_of(2),
            enable_masked_compound: rng.next().is_multiple_of(2),
            enable_warped_motion: rng.next().is_multiple_of(2),
            enable_dual_filter: rng.next().is_multiple_of(2),
            enable_order_hint: rng.next().is_multiple_of(2),
            enable_dist_wtd_comp: rng.next().is_multiple_of(2),
            enable_ref_frame_mvs: rng.next().is_multiple_of(2),
            force_screen_content_tools: force_sct,
            force_integer_mv,
            order_hint_bits_minus_1: rng.range(0, 8),
            enable_superres: rng.next().is_multiple_of(2),
            enable_cdef: rng.next().is_multiple_of(2),
            enable_restoration: rng.next().is_multiple_of(2),
        };
        let mut wb = WriteBitBuffer::new();
        write_sequence_header(&mut wb, &s);
        let got = wb.bytes().to_vec();
        let packed = [
            s.num_bits_width as i32, s.num_bits_height as i32, s.max_frame_width, s.max_frame_height,
            s.reduced_still_picture_hdr as i32, s.frame_id_numbers_present_flag as i32,
            s.delta_frame_id_length, s.frame_id_length, s.sb_size_128 as i32,
            s.enable_filter_intra as i32, s.enable_intra_edge_filter as i32, s.enable_interintra_compound as i32,
            s.enable_masked_compound as i32, s.enable_warped_motion as i32, s.enable_dual_filter as i32,
            s.enable_order_hint as i32, s.enable_dist_wtd_comp as i32, s.enable_ref_frame_mvs as i32,
            s.force_screen_content_tools, s.force_integer_mv, s.order_hint_bits_minus_1,
            s.enable_superres as i32, s.enable_cdef as i32, s.enable_restoration as i32,
        ];
        let want = c::ref_write_sequence_header(&packed);
        assert_eq!(got, want, "write_sequence_header {s:?}");
    }
}

#[test]
fn write_ext_tile_info_matches_c() {
    use aom_entropy::header::write_ext_tile_info;
    let mut rng = Rng(0xe471_c0de_a11a_0009);
    for _ in 0..200_000 {
        let pre_bits = rng.range(0, 40); // arbitrary starting offset
        let rows = rng.range(1, 9) as usize;
        let cols = rng.range(1, 9) as usize;
        let mut wb = WriteBitBuffer::new();
        for _ in 0..pre_bits {
            wb.write_bit(0);
        }
        write_ext_tile_info(&mut wb, rows, cols);
        let got = wb.bytes().to_vec();
        let want = c::ref_write_ext_tile_info(pre_bits, rows, cols);
        assert_eq!(got, want, "write_ext_tile_info pre={pre_bits} {rows}x{cols}");
    }
}

#[test]
fn write_color_config_matches_c() {
    use aom_entropy::header::{write_color_config, ColorConfigParams};
    let mut rng = Rng(0xc010_c0de_a11a_0009);
    for _ in 0..200_000 {
        let profile = rng.range(0, 3); // 0,1,2
        let bit_depth = if profile == 2 {
            [8, 10, 12][rng.range(0, 3) as usize]
        } else {
            [8, 10][rng.range(0, 2) as usize]
        };
        // profile 1 forbids monochrome (asserted); keep it spec-valid.
        let monochrome = profile != 1 && rng.next().is_multiple_of(3);
        // color description: unspecified triple, the sRGB triple, or random CICP.
        let (cp, tc, mc) = match rng.range(0, 3) {
            0 => (2, 2, 2),   // all unspecified -> no description
            1 => (1, 13, 0),  // sRGB special case
            _ => (rng.range(0, 256), rng.range(0, 256), rng.range(0, 256)),
        };
        let c = ColorConfigParams {
            bit_depth,
            profile,
            monochrome,
            color_primaries: cp,
            transfer_characteristics: tc,
            matrix_coefficients: mc,
            color_range: rng.next().is_multiple_of(2),
            subsampling_x: rng.range(0, 2),
            subsampling_y: rng.range(0, 2),
            chroma_sample_position: rng.range(0, 4),
            separate_uv_delta_q: rng.next().is_multiple_of(2),
        };
        let mut wb = WriteBitBuffer::new();
        write_color_config(&mut wb, &c);
        let got = wb.bytes().to_vec();
        let packed = [
            c.bit_depth, c.profile, c.monochrome as i32, c.color_primaries,
            c.transfer_characteristics, c.matrix_coefficients, c.color_range as i32,
            c.subsampling_x, c.subsampling_y, c.chroma_sample_position, c.separate_uv_delta_q as i32,
        ];
        let want = c::ref_write_color_config(&packed);
        assert_eq!(got, want, "write_color_config {c:?}");
    }
}

#[test]
fn timing_and_decoder_model_match_c() {
    use aom_entropy::header::{
        write_dec_model_op_parameters, write_decoder_model_info, write_timing_info_header,
        DecoderModelInfo, TimingInfoHeader,
    };
    let mut rng = Rng(0x71de_c0de_a11a_0009);
    for _ in 0..200_000 {
        // timing info
        let t = TimingInfoHeader {
            num_units_in_display_tick: rng.next() as u32,
            time_scale: rng.next() as u32,
            equal_picture_interval: rng.next().is_multiple_of(2),
            num_ticks_per_picture: (rng.next() as u32 & 0x00ff_fffe) + 1, // -1 safe, not u32::MAX
        };
        let mut wb = WriteBitBuffer::new();
        write_timing_info_header(&mut wb, &t);
        assert_eq!(wb.bytes(), &c::ref_write_timing_info(t.num_units_in_display_tick, t.time_scale, t.equal_picture_interval, t.num_ticks_per_picture)[..], "timing {t:?}");

        // decoder model info: the *_length fields are in [1, 32] (written -1 in 5 bits)
        let d = DecoderModelInfo {
            encoder_decoder_buffer_delay_length: rng.range(1, 33),
            num_units_in_decoding_tick: rng.next() as u32,
            buffer_removal_time_length: rng.range(1, 33),
            frame_presentation_time_length: rng.range(1, 33),
        };
        let mut wb = WriteBitBuffer::new();
        write_decoder_model_info(&mut wb, &d);
        assert_eq!(wb.bytes(), &c::ref_write_decoder_model_info(d.encoder_decoder_buffer_delay_length, d.num_units_in_decoding_tick, d.buffer_removal_time_length, d.frame_presentation_time_length)[..], "decoder_model {d:?}");

        // op parameters: delay_len in [1, 32]; delays fit that width
        let delay_len = rng.range(1, 33) as u32;
        let mask: u32 = if delay_len >= 32 { u32::MAX } else { (1u32 << delay_len) - 1 };
        let dec_delay = rng.next() as u32 & mask;
        let enc_delay = rng.next() as u32 & mask;
        let low_delay = rng.next().is_multiple_of(2);
        let mut wb = WriteBitBuffer::new();
        write_dec_model_op_parameters(&mut wb, dec_delay, enc_delay, low_delay, delay_len);
        assert_eq!(wb.bytes(), &c::ref_write_dec_model_op(dec_delay, enc_delay, low_delay, delay_len)[..], "op_params len={delay_len}");
    }
}
