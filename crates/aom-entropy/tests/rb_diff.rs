//! Roundtrip harness for `ReadBitBuffer` (the byte-aligned MSB-first bit reader) and
//! the OBU-level parsers, all inverses of the C-validated `WriteBitBuffer` / OBU
//! writers. WriteBitBuffer is byte-identical to libaom's `aom_write_bit_buffer`, so a
//! clean read-back pins ReadBitBuffer to `aom_read_bit_buffer`.

use aom_entropy::header::{read_tile_group_header, write_tile_group_header};
use aom_entropy::obu::{read_obu_header, write_obu_header};
use aom_entropy::rb::ReadBitBuffer;
use aom_entropy::wb::WriteBitBuffer;

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
fn read_bit_buffer_inverts_write() {
    let mut rng = Rng(0x1e_5b12_c0de_0100);
    for _ in 0..200_000 {
        // literal (signed source, 1..=31 bits)
        let lb = 1 + (rng.next() % 31) as u32;
        let lv = (rng.next() % (1u64 << lb)) as i32;
        // unsigned literal (1..=32 bits)
        let ub = 1 + (rng.next() % 32) as u32;
        let uv = if ub == 32 { rng.next() as u32 } else { (rng.next() % (1u64 << ub)) as u32 };
        // inv-signed (bits+1 written; value in [-(2^bits), 2^bits))
        let ib = 1 + (rng.next() % 16) as u32;
        let iv = (rng.next() % (1u64 << (ib + 1))) as i32 - (1i32 << ib);
        // uvlc
        let vv = (rng.next() % (1u64 << 20)) as u32;

        let mut wb = WriteBitBuffer::new();
        wb.write_literal(lv, lb);
        wb.write_unsigned_literal(uv, ub);
        wb.write_inv_signed_literal(iv, ib);
        wb.write_uvlc(vv);
        let bytes = wb.bytes().to_vec();

        let mut rb = ReadBitBuffer::new(&bytes);
        assert_eq!(rb.read_literal(lb), lv, "literal {lv}@{lb}");
        assert_eq!(rb.read_unsigned_literal(ub), uv, "unsigned {uv}@{ub}");
        assert_eq!(rb.read_inv_signed_literal(ib), iv, "inv_signed {iv}@{ib}");
        assert_eq!(rb.read_uvlc(), vv, "uvlc {vv}");
        assert!(!rb.error, "no over-read");
    }
}

#[test]
fn read_obu_header_inverts_write() {
    let mut rng = Rng(0x1e_0b12_c0de_0101);
    for _ in 0..100_000 {
        let obu_type = (rng.next() % 16) as u32;
        let a = rng.next() & 1 == 1;
        let b = rng.next() & 1 == 1;
        let ext = (rng.next() % 256) as u8;
        let bytes = write_obu_header(obu_type, a, b, ext);
        let h = read_obu_header(&bytes).expect("valid header");
        let ext_flag = a && b;
        assert_eq!(h.obu_type, obu_type, "obu_type");
        assert_eq!(h.obu_extension_flag, ext_flag, "ext_flag");
        assert!(h.obu_has_size_field, "has_size always set");
        assert_eq!(h.obu_extension, if ext_flag { ext } else { 0 }, "ext byte");
        assert_eq!(h.header_len, if ext_flag { 2 } else { 1 }, "header_len");
        assert_eq!(h.header_len, bytes.len(), "consumes all header bytes");
    }
}

#[test]
fn read_tile_group_header_inverts_write() {
    let mut rng = Rng(0x1e_71c0_c0de_0102);
    for _ in 0..100_000 {
        let tiles_log2 = (rng.next() % 7) as i32; // 0..=6
        let present = rng.next() & 1 == 1;
        let (start, end) = if tiles_log2 > 0 {
            let m = 1u64 << tiles_log2;
            ((rng.next() % m) as i32, (rng.next() % m) as i32)
        } else {
            (0, 0)
        };
        let mut wb = WriteBitBuffer::new();
        write_tile_group_header(&mut wb, start, end, tiles_log2, present);
        let bytes = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&bytes);
        let (gs, ge, gp) = read_tile_group_header(&mut rb, tiles_log2);
        if tiles_log2 == 0 {
            assert_eq!((gs, ge, gp), (0, 0, false), "single tile");
        } else {
            assert_eq!(gp, present, "present flag t2={tiles_log2}");
            if present {
                assert_eq!((gs, ge), (start, end), "start/end t2={tiles_log2}");
            }
        }
    }
}

#[test]
fn read_header_components_invert_write() {
    use aom_entropy::header::{
        read_delta_q_params, read_frame_interp_filter, read_render_size, read_superres_scale,
        read_tx_mode, write_delta_q_params, write_frame_interp_filter, write_render_size,
        write_superres_scale, write_tx_mode, DeltaQParams,
    };
    let mut rng = Rng(0x1e_4ead_c0de_0110);
    for _ in 0..100_000 {
        // tx_mode
        {
            let lossless = rng.next() & 1 == 1;
            let sel = rng.next() & 1 == 1;
            let mut wb = WriteBitBuffer::new();
            write_tx_mode(&mut wb, lossless, sel);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let got = read_tx_mode(&mut rb, lossless);
            assert_eq!(got, if lossless { false } else { sel }, "tx_mode");
        }
        // delta_q_params
        {
            let base = if rng.next() & 1 == 1 { 1 + (rng.next() % 255) as i32 } else { 0 };
            let intrabc = rng.next() & 1 == 1;
            let dq_present = base > 0 && rng.next() & 1 == 1;
            let dq_res = if dq_present { 1 << (rng.next() % 4) } else { 1 };
            let dlf_present = dq_present && !intrabc && rng.next() & 1 == 1;
            let dlf_res = if dlf_present { 1 << (rng.next() % 4) } else { 1 };
            let dlf_multi = dlf_present && rng.next() & 1 == 1;
            let d = DeltaQParams {
                base_qindex: base,
                delta_q_present: dq_present,
                delta_q_res: dq_res,
                allow_intrabc: intrabc,
                delta_lf_present: dlf_present,
                delta_lf_res: dlf_res,
                delta_lf_multi: dlf_multi,
            };
            let mut wb = WriteBitBuffer::new();
            write_delta_q_params(&mut wb, &d);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let g = read_delta_q_params(&mut rb, base, intrabc);
            assert_eq!(
                (g.delta_q_present, g.delta_q_res, g.delta_lf_present, g.delta_lf_res, g.delta_lf_multi),
                (dq_present, dq_res, dlf_present, dlf_res, dlf_multi),
                "delta_q base={base} intrabc={intrabc}"
            );
        }
        // frame_interp_filter
        {
            let filter = (rng.next() % 5) as i32; // 0..3 fixed, 4=SWITCHABLE
            let mut wb = WriteBitBuffer::new();
            write_frame_interp_filter(&mut wb, filter);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            assert_eq!(read_frame_interp_filter(&mut rb), filter, "interp_filter");
        }
        // superres_scale
        {
            let enable = rng.next() & 1 == 1;
            // enable + coin => scaled [9,16]; else SCALE_NUMERATOR (8). `&&` short-circuits
            // so the coin draw is skipped when disabled, as the nested form would.
            let denom = if enable && rng.next() & 1 == 1 {
                9 + (rng.next() % 8) as i32
            } else {
                8
            };
            let mut wb = WriteBitBuffer::new();
            write_superres_scale(&mut wb, enable, denom);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let got = read_superres_scale(&mut rb, enable);
            assert_eq!(got, if enable { denom } else { 8 }, "superres enable={enable}");
        }
        // render_size
        {
            let active = rng.next() & 1 == 1;
            let w = 1 + (rng.next() % 65536) as i32;
            let h = 1 + (rng.next() % 65536) as i32;
            let mut wb = WriteBitBuffer::new();
            write_render_size(&mut wb, active, w, h);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let (ga, gw, gh) = read_render_size(&mut rb);
            assert_eq!(ga, active, "render active");
            if active {
                assert_eq!((gw, gh), (w, h), "render size");
            }
        }
    }
}

#[test]
fn read_quant_cdef_headers_invert_write() {
    use aom_entropy::header::{
        encode_cdef, encode_quantization, read_cdef_header, read_quantization, CdefHeader,
        QuantParamsHeader,
    };
    let mut rng = Rng(0x1e_4a2d_c0de_0120);
    let dgen = |rng: &mut Rng| -> i32 { (rng.next() % 127) as i32 - 63 };
    for _ in 0..100_000 {
        // quantization
        {
            let num_planes = if rng.next() & 1 == 1 { 3 } else { 1 };
            let separate = rng.next() & 1 == 1;
            let base = (rng.next() % 256) as i32;
            let ydc = dgen(&mut rng);
            let (udc, uac, vdc, vac) = if num_planes > 1 {
                let udc = dgen(&mut rng);
                let uac = dgen(&mut rng);
                let (vdc, vac) = if separate { (dgen(&mut rng), dgen(&mut rng)) } else { (udc, uac) };
                (udc, uac, vdc, vac)
            } else {
                (0, 0, 0, 0)
            };
            let using_qm = rng.next() & 1 == 1;
            let (qy, qu, qv) = if using_qm {
                let qy = (rng.next() % 16) as i32;
                let qu = (rng.next() % 16) as i32;
                let qv = if separate { (rng.next() % 16) as i32 } else { qu };
                (qy, qu, qv)
            } else {
                (0, 0, 0)
            };
            let qp = QuantParamsHeader {
                base_qindex: base,
                y_dc_delta_q: ydc,
                u_dc_delta_q: udc,
                u_ac_delta_q: uac,
                v_dc_delta_q: vdc,
                v_ac_delta_q: vac,
                using_qmatrix: using_qm,
                qmatrix_level_y: qy,
                qmatrix_level_u: qu,
                qmatrix_level_v: qv,
            };
            let mut wb = WriteBitBuffer::new();
            encode_quantization(&mut wb, &qp, num_planes, separate);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let g = read_quantization(&mut rb, num_planes, separate);
            assert_eq!(
                (g.base_qindex, g.y_dc_delta_q, g.u_dc_delta_q, g.u_ac_delta_q, g.v_dc_delta_q, g.v_ac_delta_q),
                (base, ydc, udc, uac, vdc, vac),
                "quant deltas np={num_planes} sep={separate}"
            );
            assert_eq!(
                (g.using_qmatrix, g.qmatrix_level_y, g.qmatrix_level_u, g.qmatrix_level_v),
                (using_qm, qy, qu, qv),
                "quant qm np={num_planes} sep={separate}"
            );
        }
        // cdef (coded path: enabled, no intrabc)
        {
            let num_planes = if rng.next() & 1 == 1 { 3 } else { 1 };
            let damping = 3 + (rng.next() % 4) as i32;
            let bits = (rng.next() % 4) as i32;
            let nb = 1usize << bits;
            let mut s = [0i32; 8];
            let mut uv = [0i32; 8];
            for i in 0..nb {
                s[i] = (rng.next() % 64) as i32;
                uv[i] = if num_planes > 1 { (rng.next() % 64) as i32 } else { 0 };
            }
            let c = CdefHeader {
                enable_cdef: true,
                allow_intrabc: false,
                cdef_damping: damping,
                cdef_bits: bits,
                nb_cdef_strengths: nb,
                cdef_strengths: s,
                cdef_uv_strengths: uv,
            };
            let mut wb = WriteBitBuffer::new();
            encode_cdef(&mut wb, &c, num_planes);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let g = read_cdef_header(&mut rb, true, false, num_planes);
            assert_eq!(
                (g.cdef_damping, g.cdef_bits, g.nb_cdef_strengths),
                (damping, bits, nb),
                "cdef hdr np={num_planes}"
            );
            assert_eq!(g.cdef_strengths, s, "cdef strengths");
            if num_planes > 1 {
                assert_eq!(g.cdef_uv_strengths, uv, "cdef uv strengths");
            }
        }
    }
}

#[test]
fn read_loopfilter_inverts_write() {
    use aom_entropy::header::{encode_loopfilter, read_loopfilter, LoopfilterHeader};
    let mut rng = Rng(0x1e_10f1_c0de_0130);
    let d8 = |rng: &mut Rng| -> i8 { ((rng.next() % 127) as i32 - 63) as i8 };
    for _ in 0..100_000 {
        let num_planes = if rng.next() & 1 == 1 { 3 } else { 1 };
        let fl0 = (rng.next() % 64) as i32;
        let fl1 = (rng.next() % 64) as i32;
        let uv_coded = num_planes > 1 && (fl0 != 0 || fl1 != 0);
        let flu = if uv_coded { (rng.next() % 64) as i32 } else { 0 };
        let flv = if uv_coded { (rng.next() % 64) as i32 } else { 0 };
        let sharp = (rng.next() % 8) as i32;
        let enabled = rng.next() & 1 == 1;

        let mut last_ref = [0i8; 8];
        let mut last_mode = [0i8; 2];
        for r in last_ref.iter_mut() { *r = d8(&mut rng); }
        for m in last_mode.iter_mut() { *m = d8(&mut rng); }

        // Scenario A: some deltas change (meaningful=update=true). Scenario B: none (update=false).
        let scenario_a = enabled && rng.next() & 1 == 1;
        let mut ref_d = last_ref;
        let mut mode_d = last_mode;
        let update;
        if scenario_a {
            update = true;
            // guarantee >=1 change vs last[0], staying in the su(6)=[-63,63] range.
            let mut nv = d8(&mut rng);
            if nv == last_ref[0] {
                nv = if last_ref[0] < 63 { last_ref[0] + 1 } else { last_ref[0] - 1 };
            }
            ref_d[0] = nv;
            for r in ref_d[1..].iter_mut() {
                if rng.next() & 1 == 1 {
                    *r = d8(&mut rng);
                }
            }
            for m in mode_d.iter_mut() {
                if rng.next() & 1 == 1 {
                    *m = d8(&mut rng);
                }
            }
        } else {
            update = false; // deltas == last => meaningful false
        }

        let lf = LoopfilterHeader {
            allow_intrabc: false,
            filter_level: [fl0, fl1],
            filter_level_u: flu,
            filter_level_v: flv,
            sharpness_level: sharp,
            mode_ref_delta_enabled: enabled,
            mode_ref_delta_update: update,
            ref_deltas: ref_d,
            mode_deltas: mode_d,
            last_ref_deltas: last_ref,
            last_mode_deltas: last_mode,
        };
        let mut wb = WriteBitBuffer::new();
        encode_loopfilter(&mut wb, &lf, num_planes);
        let b = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&b);
        let g = read_loopfilter(&mut rb, false, num_planes, last_ref, last_mode);
        assert_eq!(g.filter_level, [fl0, fl1], "filter levels");
        assert_eq!((g.filter_level_u, g.filter_level_v), (flu, flv), "uv levels");
        assert_eq!(g.sharpness_level, sharp, "sharpness");
        assert_eq!(g.mode_ref_delta_enabled, enabled, "enabled");
        assert_eq!(g.mode_ref_delta_update, update, "update np={num_planes} a={scenario_a}");
        assert_eq!(g.ref_deltas, ref_d, "ref_deltas");
        assert_eq!(g.mode_deltas, mode_d, "mode_deltas");
    }
}

#[test]
fn read_segmentation_inverts_write() {
    use aom_entropy::header::{encode_segmentation, read_segmentation, SegmentationHeader};
    const DATA_MAX: [i32; 8] = [255, 63, 63, 63, 63, 7, 0, 0];
    const SIGNED: [bool; 8] = [true, true, true, true, true, false, false, false];
    let mut rng = Rng(0x1e_5e62_c0de_0140);
    for _ in 0..100_000 {
        let enabled = rng.next() & 1 == 1;
        let has_pr = rng.next() & 1 == 1;
        let (mut update_map, mut temporal, mut update_data) = (false, false, false);
        let mut mask = [0u32; 8];
        let mut data = [[0i32; 8]; 8];
        if enabled {
            if has_pr {
                update_map = rng.next() & 1 == 1;
                temporal = update_map && rng.next() & 1 == 1;
                update_data = rng.next() & 1 == 1;
            } else {
                update_map = true;
                update_data = true;
            }
            if update_data {
                for i in 0..8 {
                    mask[i] = (rng.next() % 256) as u32;
                    for j in 0..8 {
                        if mask[i] & (1 << j) != 0 {
                            let dm = DATA_MAX[j];
                            data[i][j] = if dm == 0 {
                                0
                            } else if SIGNED[j] {
                                (rng.next() % (2 * dm as u64 + 1)) as i32 - dm
                            } else {
                                (rng.next() % (dm as u64 + 1)) as i32
                            };
                        }
                    }
                }
            }
        }
        let seg = SegmentationHeader {
            enabled,
            has_primary_ref: has_pr,
            update_map,
            temporal_update: temporal,
            update_data,
            feature_mask: mask,
            feature_data: data,
        };
        let mut wb = WriteBitBuffer::new();
        encode_segmentation(&mut wb, &seg);
        let b = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&b);
        let g = read_segmentation(&mut rb, has_pr);
        assert_eq!(g.enabled, enabled, "seg enabled");
        assert_eq!(
            (g.update_map, g.temporal_update, g.update_data),
            (update_map, temporal, update_data),
            "seg flags pr={has_pr}"
        );
        assert_eq!(g.feature_mask, mask, "seg mask");
        assert_eq!(g.feature_data, data, "seg data");
    }
}

#[test]
fn read_frame_size_inverts_write() {
    use aom_entropy::header::{read_frame_size, write_frame_size, FrameSizeHeader};
    let mut rng = Rng(0x1e_f512_c0de_0150);
    for _ in 0..100_000 {
        let nbw = 8 + (rng.next() % 9) as u32; // 8..=16
        let nbh = 8 + (rng.next() % 9) as u32;
        let over = rng.next() & 1 == 1;
        let w = 1 + (rng.next() % (1u64 << nbw)) as i32;
        let h = 1 + (rng.next() % (1u64 << nbh)) as i32;
        let en_sr = rng.next() & 1 == 1;
        let denom = if en_sr && rng.next() & 1 == 1 { 9 + (rng.next() % 8) as i32 } else { 8 };
        let sc_active = rng.next() & 1 == 1;
        let (rw, rh) = if sc_active {
            (1 + (rng.next() % 65536) as i32, 1 + (rng.next() % 65536) as i32)
        } else {
            (0, 0)
        };
        let fs = FrameSizeHeader {
            frame_size_override: over,
            num_bits_width: nbw,
            num_bits_height: nbh,
            superres_upscaled_width: w,
            superres_upscaled_height: h,
            enable_superres: en_sr,
            scale_denominator: denom,
            scaling_active: sc_active,
            render_width: rw,
            render_height: rh,
        };
        let mut wb = WriteBitBuffer::new();
        write_frame_size(&mut wb, &fs);
        let b = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&b);
        // for !override the size is inferred: pass the same (w,h).
        let g = read_frame_size(&mut rb, over, nbw, nbh, en_sr, w, h);
        assert_eq!(
            (g.superres_upscaled_width, g.superres_upscaled_height),
            (w, h),
            "frame size over={over}"
        );
        assert_eq!(g.scale_denominator, if en_sr { denom } else { 8 }, "superres");
        assert_eq!(g.scaling_active, sc_active, "render active");
        if sc_active {
            assert_eq!((g.render_width, g.render_height), (rw, rh), "render size");
        }
    }
}

#[test]
fn read_color_config_inverts_write() {
    use aom_entropy::header::{read_color_config, write_color_config, ColorConfigParams};
    let mut rng = Rng(0x1e_c010_c0de_0160);
    for _ in 0..100_000 {
        let profile = (rng.next() % 3) as i32;
        let bit_depth = if profile == 2 {
            [8, 10, 12][(rng.next() % 3) as usize]
        } else {
            [8, 10][(rng.next() % 2) as usize]
        };
        let monochrome = profile != 1 && rng.next() & 1 == 1;

        // sRGB draw only for non-mono profile-1 (short-circuits so RNG use matches the
        // per-branch form).
        let want_srgb = !monochrome && profile == 1 && rng.next() & 1 == 1;
        let (ssx, ssy) = if monochrome {
            (1, 1)
        } else if want_srgb {
            (0, 0)
        } else if profile == 0 {
            (1, 1)
        } else if profile == 1 {
            (0, 0)
        } else if bit_depth == 12 {
            let x = (rng.next() & 1) as i32;
            let y = if x == 1 { (rng.next() & 1) as i32 } else { 0 };
            (x, y)
        } else {
            (1, 0)
        };

        let (cp, tc, mc) = if want_srgb {
            (1, 13, 0)
        } else if rng.next() & 1 == 1 {
            let mut cp = (rng.next() % 256) as i32;
            let tc = (rng.next() % 256) as i32;
            let mc = (rng.next() % 256) as i32;
            if cp == 2 && tc == 2 && mc == 2 {
                cp = 3;
            }
            if cp == 1 && tc == 13 && mc == 0 {
                cp = 5; // avoid an accidental sRGB triple
            }
            (cp, tc, mc)
        } else {
            (2, 2, 2)
        };

        let chroma_pos = if !monochrome && !want_srgb && ssx == 1 && ssy == 1 {
            (rng.next() % 4) as i32
        } else {
            0
        };
        let sep_uv = !monochrome && rng.next() & 1 == 1;
        let color_range = want_srgb || rng.next() & 1 == 1;

        let c = ColorConfigParams {
            bit_depth,
            profile,
            monochrome,
            color_primaries: cp,
            transfer_characteristics: tc,
            matrix_coefficients: mc,
            color_range,
            subsampling_x: ssx,
            subsampling_y: ssy,
            chroma_sample_position: chroma_pos,
            separate_uv_delta_q: sep_uv,
        };
        let mut wb = WriteBitBuffer::new();
        write_color_config(&mut wb, &c);
        let b = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&b);
        let g = read_color_config(&mut rb, profile);
        let want = (
            bit_depth, monochrome, cp, tc, mc, color_range, ssx, ssy, chroma_pos, sep_uv,
        );
        let got = (
            g.bit_depth, g.monochrome, g.color_primaries, g.transfer_characteristics,
            g.matrix_coefficients, g.color_range, g.subsampling_x, g.subsampling_y,
            g.chroma_sample_position, g.separate_uv_delta_q,
        );
        assert_eq!(got, want, "color_config profile={profile} bd={bit_depth} mono={monochrome} srgb={want_srgb}");
    }
}

#[test]
fn read_trailing_and_restoration_invert_write() {
    use aom_entropy::header::{
        encode_restoration_mode, read_frame_header_trailing_flags, read_restoration_mode,
        write_frame_header_trailing_flags, RestorationHeader,
    };
    let mut rng = Rng(0x1e_712a_c0de_0170);
    for _ in 0..100_000 {
        // trailing flags
        {
            let intra_only = rng.next() & 1 == 1;
            let skip_allowed = rng.next() & 1 == 1;
            let might_warp = rng.next() & 1 == 1;
            let ref_sel = !intra_only && rng.next() & 1 == 1;
            let skip_flag = skip_allowed && rng.next() & 1 == 1;
            let warp = might_warp && rng.next() & 1 == 1;
            let reduced = rng.next() & 1 == 1;
            let mut wb = WriteBitBuffer::new();
            write_frame_header_trailing_flags(
                &mut wb, intra_only, ref_sel, skip_allowed, skip_flag, might_warp, warp, reduced,
            );
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let g = read_frame_header_trailing_flags(&mut rb, intra_only, skip_allowed, might_warp);
            assert_eq!(g, (ref_sel, skip_flag, warp, reduced), "trailing flags");
        }
        // restoration mode
        {
            let enable = rng.next() & 1 == 1;
            let intrabc = rng.next() & 1 == 1;
            let sb128 = rng.next() & 1 == 1;
            let ssx = (rng.next() % 2) as i32;
            let ssy = (rng.next() % 2) as i32;
            let num_planes = if rng.next() & 1 == 1 { 3 } else { 1 };
            let mut ft = [0u8; 3];
            // types are only in the bitstream when restoration is on; NONE otherwise.
            if enable && !intrabc {
                for t in ft.iter_mut().take(num_planes) {
                    *t = (rng.next() % 4) as u8;
                }
            }
            let all_none = ft[..num_planes].iter().all(|&t| t == 0);
            let chroma_none = ft[..num_planes].iter().enumerate().all(|(p, &t)| t == 0 || p == 0);
            let mut rus = [256i32; 3];
            if enable && !intrabc && !all_none {
                let rus0 = if sb128 {
                    [128, 256][(rng.next() % 2) as usize]
                } else {
                    [64, 128, 256][(rng.next() % 3) as usize]
                };
                rus[0] = rus0;
                let mut chroma = rus0;
                if num_planes > 1 && ssx.min(ssy) != 0 && !chroma_none && rng.next() & 1 == 1 {
                    chroma = rus0 >> 1;
                }
                rus[1] = chroma;
                rus[2] = chroma;
            }
            let r = RestorationHeader {
                enable_restoration: enable,
                allow_intrabc: intrabc,
                frame_restoration_type: ft,
                sb_size_128: sb128,
                restoration_unit_size: rus,
                subsampling_x: ssx,
                subsampling_y: ssy,
            };
            let mut wb = WriteBitBuffer::new();
            encode_restoration_mode(&mut wb, &r, num_planes);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let g = read_restoration_mode(&mut rb, enable, intrabc, sb128, ssx, ssy, num_planes);
            assert_eq!(g.frame_restoration_type, ft, "restoration types np={num_planes}");
            assert_eq!(g.restoration_unit_size, rus, "restoration unit size");
        }
    }
}

#[test]
fn read_signed_primitive_refsubexpfin_inverts_write() {
    // Same GM parameter ranges as wb_diff::signed_subexpfin_matches_c (n = GM_ALPHA_MAX+1
    // / (1<<trans_bits)+1, k = SUBEXPFIN_K = 3). write side is byte-identical to C, so a
    // clean read-back pins read_signed_primitive_refsubexpfin to the C reader.
    let mut rng = Rng(0x1e_5ec0_c0de_0180);
    for &n in &[4097i32, (1 << 12) + 1, (1 << 9) + 1] {
        for _ in 0..200_000 {
            let bound = 2 * n as u64 - 1;
            let r = (rng.next() % bound) as i32 - (n - 1);
            let v = (rng.next() % bound) as i32 - (n - 1);
            let mut wb = WriteBitBuffer::new();
            wb.write_signed_primitive_refsubexpfin(n as u16, 3, r as i16, v as i16);
            let bytes = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&bytes);
            let g = rb.read_signed_primitive_refsubexpfin(n as u16, 3, r as i16);
            assert_eq!(g, v as i16, "refsubexpfin n={n} ref={r} v={v}");
        }
    }
}

#[test]
fn read_global_motion_inverts_write() {
    use aom_entropy::header::{
        read_global_motion_params, write_global_motion_params, WarpedMotionParams,
    };
    let ident = WarpedMotionParams { wmtype: 0, wmmat: [0, 0, 1 << 16, 0, 0, 1 << 16] };
    let mut rng = Rng(0x1e_92c0_c0de_0190);
    for _ in 0..300_000 {
        let allow_hp = rng.next() & 1 == 1;
        let ty = (rng.next() % 4) as u8; // IDENTITY/TRANSLATION/ROTZOOM/AFFINE
        let alpha = |rng: &mut Rng| -> i32 { (rng.next() % 8193) as i32 - 4096 }; // [-GM_ALPHA_MAX, GM_ALPHA_MAX]
        let mut wm = [0i32, 0, 1 << 16, 0, 0, 1 << 16];
        if ty >= 2 {
            wm[2] = (alpha(&mut rng) + (1 << 15)) << 1;
            wm[3] = alpha(&mut rng) << 1;
        }
        if ty >= 3 {
            wm[4] = alpha(&mut rng) << 1;
            wm[5] = (alpha(&mut rng) + (1 << 15)) << 1;
        } else if ty == 2 {
            wm[4] = -wm[3];
            wm[5] = wm[2];
        }
        if ty >= 1 {
            let (tb, tpd) = if ty == 1 {
                (9 - !allow_hp as u32, 13 + !allow_hp as u32)
            } else {
                (12u32, 10u32)
            };
            let bound = 1i64 << tb;
            let tc = |rng: &mut Rng| -> i32 { (rng.next() % (2 * bound as u64 + 1)) as i32 - bound as i32 };
            wm[0] = tc(&mut rng) << tpd;
            wm[1] = tc(&mut rng) << tpd;
        }
        let params = WarpedMotionParams { wmtype: ty, wmmat: wm };
        let mut wb = WriteBitBuffer::new();
        write_global_motion_params(&mut wb, &params, &ident, allow_hp);
        let bytes = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&bytes);
        let g = read_global_motion_params(&mut rb, &ident, allow_hp);
        assert_eq!(g.wmtype, ty, "gm type hp={allow_hp}");
        assert_eq!(g.wmmat, wm, "gm wmmat ty={ty} hp={allow_hp}");
    }
}

#[test]
fn read_timing_and_decoder_model_invert_write() {
    use aom_entropy::header::{
        read_decoder_model_info, read_timing_info_header, write_decoder_model_info,
        write_timing_info_header, DecoderModelInfo, TimingInfoHeader,
    };
    let mut rng = Rng(0x1e_71de_c0de_01a0);
    for _ in 0..100_000 {
        // timing info
        {
            let equal = rng.next() & 1 == 1;
            let t = TimingInfoHeader {
                num_units_in_display_tick: rng.next() as u32,
                time_scale: rng.next() as u32,
                equal_picture_interval: equal,
                num_ticks_per_picture: if equal { 1 + (rng.next() % (1 << 20)) as u32 } else { 1 },
            };
            let mut wb = WriteBitBuffer::new();
            write_timing_info_header(&mut wb, &t);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let g = read_timing_info_header(&mut rb);
            assert_eq!(g.num_units_in_display_tick, t.num_units_in_display_tick, "timing units");
            assert_eq!(g.time_scale, t.time_scale, "timing scale");
            assert_eq!(g.equal_picture_interval, equal, "timing equal");
            if equal {
                assert_eq!(g.num_ticks_per_picture, t.num_ticks_per_picture, "timing ticks");
            }
        }
        // decoder model info
        {
            let d = DecoderModelInfo {
                encoder_decoder_buffer_delay_length: 1 + (rng.next() % 32) as i32,
                num_units_in_decoding_tick: rng.next() as u32,
                buffer_removal_time_length: 1 + (rng.next() % 32) as i32,
                frame_presentation_time_length: 1 + (rng.next() % 32) as i32,
            };
            let mut wb = WriteBitBuffer::new();
            write_decoder_model_info(&mut wb, &d);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let g = read_decoder_model_info(&mut rb);
            assert_eq!(
                (g.encoder_decoder_buffer_delay_length, g.num_units_in_decoding_tick,
                 g.buffer_removal_time_length, g.frame_presentation_time_length),
                (d.encoder_decoder_buffer_delay_length, d.num_units_in_decoding_tick,
                 d.buffer_removal_time_length, d.frame_presentation_time_length),
                "decoder model"
            );
        }
    }
}

#[test]
fn read_tile_info_inverts_write() {
    use aom_entropy::header::{read_tile_info, write_tile_info, TileInfoHeader};
    fn tlog2(blk: i32, target: i32) -> i32 {
        let mut k = 0;
        while (blk << k) < target {
            k += 1;
        }
        k
    }
    let mut rng = Rng(0x1e_71ce_c0de_01b0);
    for _ in 0..100_000 {
        let mib = [4u32, 5][(rng.next() % 2) as usize];
        let uniform = rng.next() & 1 == 1;
        if uniform {
            let min_c = (rng.next() % 3) as i32;
            let max_c = min_c + (rng.next() % 3) as i32;
            let log2_c = min_c + (rng.next() % (max_c - min_c + 1) as u64) as i32;
            let min_r = (rng.next() % 3) as i32;
            let max_r = min_r + (rng.next() % 3) as i32;
            let log2_r = min_r + (rng.next() % (max_r - min_r + 1) as u64) as i32;
            let (mi_cols, mi_rows) = (1 + (rng.next() % 4096) as i32, 1 + (rng.next() % 4096) as i32);
            let (cols, rows) = (1usize << log2_c, 1usize << log2_r);
            let t = TileInfoHeader {
                mi_cols, mi_rows, mib_size_log2: mib, uniform_spacing: true,
                log2_cols: log2_c, min_log2_cols: min_c, max_log2_cols: max_c,
                log2_rows: log2_r, min_log2_rows: min_r, max_log2_rows: max_r,
                cols, rows, col_start_sb: [0; 65], row_start_sb: [0; 65],
                max_width_sb: 64, max_height_sb: 64,
            };
            let mut wb = WriteBitBuffer::new();
            write_tile_info(&mut wb, &t);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let (g, ctx, tsb) = read_tile_info(&mut rb, mi_cols, mi_rows, mib, min_c, max_c, min_r, max_r, 64, 64);
            assert!(g.uniform_spacing, "uniform flag");
            assert_eq!((g.log2_cols, g.log2_rows), (log2_c, log2_r), "uniform log2");
            assert_eq!((g.cols, g.rows), (cols, rows), "uniform cols/rows");
            if rows * cols > 1 {
                assert_eq!((ctx, tsb), (0, 4), "ctx/tile_size_bytes");
            }
        } else {
            let sb_cols = 1 + (rng.next() % 24) as i32;
            let sb_rows = 1 + (rng.next() % 24) as i32;
            let mi_cols = sb_cols << mib; // ceil_power_of_two -> exactly sb_cols
            let mi_rows = sb_rows << mib;
            let max_w = 1 + (rng.next() % sb_cols as u64) as i32;
            let max_h = 1 + (rng.next() % sb_rows as u64) as i32;
            let partition = |rng: &mut Rng, total: i32, cap_max: i32| -> Vec<i32> {
                let mut starts = vec![0i32];
                let mut rem = total;
                while rem > 0 {
                    let cap = rem.min(cap_max);
                    let size = 1 + (rng.next() % cap as u64) as i32;
                    let last = *starts.last().unwrap();
                    starts.push(last + size);
                    rem -= size;
                }
                starts
            };
            let cs = partition(&mut rng, sb_cols, max_w);
            let rs = partition(&mut rng, sb_rows, max_h);
            let (cols, rows) = (cs.len() - 1, rs.len() - 1);
            let mut col_start_sb = [0i32; 65];
            let mut row_start_sb = [0i32; 65];
            col_start_sb[..cs.len()].copy_from_slice(&cs);
            row_start_sb[..rs.len()].copy_from_slice(&rs);
            let t = TileInfoHeader {
                mi_cols, mi_rows, mib_size_log2: mib, uniform_spacing: false,
                log2_cols: tlog2(1, cols as i32), min_log2_cols: 0, max_log2_cols: 6,
                log2_rows: tlog2(1, rows as i32), min_log2_rows: 0, max_log2_rows: 6,
                cols, rows, col_start_sb, row_start_sb,
                max_width_sb: max_w, max_height_sb: max_h,
            };
            let mut wb = WriteBitBuffer::new();
            write_tile_info(&mut wb, &t);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let (g, ctx, tsb) = read_tile_info(&mut rb, mi_cols, mi_rows, mib, 0, 6, 0, 6, max_w, max_h);
            assert!(!g.uniform_spacing, "non-uniform flag");
            assert_eq!((g.cols, g.rows), (cols, rows), "nonuniform cols/rows sb=({sb_cols},{sb_rows})");
            assert_eq!(g.col_start_sb, col_start_sb, "col_start_sb");
            assert_eq!(g.row_start_sb, row_start_sb, "row_start_sb");
            assert_eq!((g.log2_cols, g.log2_rows), (tlog2(1, cols as i32), tlog2(1, rows as i32)), "nonuniform log2");
            if rows * cols > 1 {
                assert_eq!((ctx, tsb), (0, 4), "ctx/tile_size_bytes");
            }
        }
    }
}
