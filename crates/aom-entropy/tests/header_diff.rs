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
