//! Roundtrip harness for `ReadBitBuffer` (the byte-aligned MSB-first bit reader) and
//! the OBU-level parsers, all inverses of the C-validated `WriteBitBuffer` / OBU
//! writers. WriteBitBuffer is byte-identical to libaom's `aom_write_bit_buffer`, so a
//! clean read-back pins ReadBitBuffer to `aom_read_bit_buffer`.

use aom_dsp::entropy::header::{read_tile_group_header, write_tile_group_header};
use aom_dsp::entropy::obu::{read_obu_header, write_obu_header};
use aom_dsp::entropy::rb::ReadBitBuffer;
use aom_dsp::entropy::wb::WriteBitBuffer;

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
        let uv = if ub == 32 {
            rng.next() as u32
        } else {
            (rng.next() % (1u64 << ub)) as u32
        };
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
    use aom_dsp::entropy::header::{
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
            let base = if rng.next() & 1 == 1 {
                1 + (rng.next() % 255) as i32
            } else {
                0
            };
            let intrabc = rng.next() & 1 == 1;
            let dq_present = base > 0 && rng.next() & 1 == 1;
            let dq_res = if dq_present { 1 << (rng.next() % 4) } else { 1 };
            let dlf_present = dq_present && !intrabc && rng.next() & 1 == 1;
            let dlf_res = if dlf_present {
                1 << (rng.next() % 4)
            } else {
                1
            };
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
                (
                    g.delta_q_present,
                    g.delta_q_res,
                    g.delta_lf_present,
                    g.delta_lf_res,
                    g.delta_lf_multi
                ),
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
            assert_eq!(
                got,
                if enable { denom } else { 8 },
                "superres enable={enable}"
            );
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
    use aom_dsp::entropy::header::{
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
                let (vdc, vac) = if separate {
                    (dgen(&mut rng), dgen(&mut rng))
                } else {
                    (udc, uac)
                };
                (udc, uac, vdc, vac)
            } else {
                (0, 0, 0, 0)
            };
            let using_qm = rng.next() & 1 == 1;
            let (qy, qu, qv) = if using_qm {
                let qy = (rng.next() % 16) as i32;
                let qu = (rng.next() % 16) as i32;
                let qv = if separate {
                    (rng.next() % 16) as i32
                } else {
                    qu
                };
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
                (
                    g.base_qindex,
                    g.y_dc_delta_q,
                    g.u_dc_delta_q,
                    g.u_ac_delta_q,
                    g.v_dc_delta_q,
                    g.v_ac_delta_q
                ),
                (base, ydc, udc, uac, vdc, vac),
                "quant deltas np={num_planes} sep={separate}"
            );
            assert_eq!(
                (
                    g.using_qmatrix,
                    g.qmatrix_level_y,
                    g.qmatrix_level_u,
                    g.qmatrix_level_v
                ),
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
                uv[i] = if num_planes > 1 {
                    (rng.next() % 64) as i32
                } else {
                    0
                };
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
    use aom_dsp::entropy::header::{encode_loopfilter, read_loopfilter, LoopfilterHeader};
    let mut rng = Rng(0x1e_10f1_c0de_0130);
    let d8 = |rng: &mut Rng| -> i8 { ((rng.next() % 127) as i32 - 63) as i8 };
    for _ in 0..100_000 {
        let num_planes = if rng.next() & 1 == 1 { 3 } else { 1 };
        let fl0 = (rng.next() % 64) as i32;
        let fl1 = (rng.next() % 64) as i32;
        let uv_coded = num_planes > 1 && (fl0 != 0 || fl1 != 0);
        let flu = if uv_coded {
            (rng.next() % 64) as i32
        } else {
            0
        };
        let flv = if uv_coded {
            (rng.next() % 64) as i32
        } else {
            0
        };
        let sharp = (rng.next() % 8) as i32;
        let enabled = rng.next() & 1 == 1;

        let mut last_ref = [0i8; 8];
        let mut last_mode = [0i8; 2];
        for r in last_ref.iter_mut() {
            *r = d8(&mut rng);
        }
        for m in last_mode.iter_mut() {
            *m = d8(&mut rng);
        }

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
                nv = if last_ref[0] < 63 {
                    last_ref[0] + 1
                } else {
                    last_ref[0] - 1
                };
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
        assert_eq!(
            (g.filter_level_u, g.filter_level_v),
            (flu, flv),
            "uv levels"
        );
        assert_eq!(g.sharpness_level, sharp, "sharpness");
        assert_eq!(g.mode_ref_delta_enabled, enabled, "enabled");
        assert_eq!(
            g.mode_ref_delta_update, update,
            "update np={num_planes} a={scenario_a}"
        );
        assert_eq!(g.ref_deltas, ref_d, "ref_deltas");
        assert_eq!(g.mode_deltas, mode_d, "mode_deltas");
    }
}

#[test]
fn read_segmentation_inverts_write() {
    use aom_dsp::entropy::header::{encode_segmentation, read_segmentation, SegmentationHeader};
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

/// The NORMATIVE feature-data clamp of `setup_segmentation` (decodeframe.c):
/// `data = clamp(data, -data_max, data_max)` after the read. Signed features
/// read `inv_signed_literal(ubits)` in `[-(2^ubits), 2^ubits - 1]`, so exactly
/// `-(2^ubits)` (= `-(data_max+1)`) clamps — a value the C ENCODER (which
/// clamps before writing) can never emit, hand-vectored here per the
/// hand-vector-the-unreachable-normative-clamp methodology.
#[test]
fn read_segmentation_normative_data_clamp() {
    use aom_dsp::entropy::header::read_segmentation;
    // enabled=1; no primary ref (update flags forced, none coded); then the
    // 8x8 (segment, feature) active-bit grid:
    //   seg0 feature0 (ALT_Q, signed, max 255, ubits 8): active,
    //     inv_signed_literal(8) = 9-bit two's complement (sign-extend read),
    //     raw 0b1_0000_0000 = 256 -> -256;
    //   seg1 feature1 (ALT_LF_Y_V, signed, max 63, ubits 6): active,
    //     7-bit two's complement raw 0b100_0000 = 64 -> -64;
    //   every other active bit 0.
    let mut w = WriteBitBuffer::new();
    w.write_bit(1); // enabled
    w.write_bit(1); // seg0 feature0 active
    w.write_literal(256, 9); // -256 (two's-complement raw)
    for _ in 1..8 {
        w.write_bit(0); // seg0 features 1..7
    }
    w.write_bit(0); // seg1 feature0
    w.write_bit(1); // seg1 feature1 active
    w.write_literal(64, 7); // -64 (two's-complement raw)
    for _ in 2..8 {
        w.write_bit(0); // seg1 features 2..7
    }
    for _ in 2 * 8..8 * 8 {
        w.write_bit(0); // segments 2..7, all features
    }
    let bytes = w.bytes().to_vec();
    let mut rb = ReadBitBuffer::new(&bytes);
    let g = read_segmentation(&mut rb, false);
    assert!(g.enabled && g.update_map && g.update_data);
    assert_eq!(g.feature_mask[0], 1, "seg0 ALT_Q active");
    assert_eq!(g.feature_data[0][0], -255, "ALT_Q -256 clamps to -255");
    assert_eq!(g.feature_mask[1], 1 << 1, "seg1 ALT_LF_Y_V active");
    assert_eq!(g.feature_data[1][1], -63, "ALT_LF -64 clamps to -63");
}

#[test]
fn read_frame_size_inverts_write() {
    use aom_dsp::entropy::header::{read_frame_size, write_frame_size, FrameSizeHeader};
    let mut rng = Rng(0x1e_f512_c0de_0150);
    for _ in 0..100_000 {
        let nbw = 8 + (rng.next() % 9) as u32; // 8..=16
        let nbh = 8 + (rng.next() % 9) as u32;
        let over = rng.next() & 1 == 1;
        let w = 1 + (rng.next() % (1u64 << nbw)) as i32;
        let h = 1 + (rng.next() % (1u64 << nbh)) as i32;
        let en_sr = rng.next() & 1 == 1;
        let denom = if en_sr && rng.next() & 1 == 1 {
            9 + (rng.next() % 8) as i32
        } else {
            8
        };
        let sc_active = rng.next() & 1 == 1;
        let (rw, rh) = if sc_active {
            (
                1 + (rng.next() % 65536) as i32,
                1 + (rng.next() % 65536) as i32,
            )
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
        assert_eq!(
            g.scale_denominator,
            if en_sr { denom } else { 8 },
            "superres"
        );
        assert_eq!(g.scaling_active, sc_active, "render active");
        if sc_active {
            assert_eq!((g.render_width, g.render_height), (rw, rh), "render size");
        }
    }
}

#[test]
fn read_color_config_inverts_write() {
    use aom_dsp::entropy::header::{read_color_config, write_color_config, ColorConfigParams};
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
            bit_depth,
            monochrome,
            cp,
            tc,
            mc,
            color_range,
            ssx,
            ssy,
            chroma_pos,
            sep_uv,
        );
        let got = (
            g.bit_depth,
            g.monochrome,
            g.color_primaries,
            g.transfer_characteristics,
            g.matrix_coefficients,
            g.color_range,
            g.subsampling_x,
            g.subsampling_y,
            g.chroma_sample_position,
            g.separate_uv_delta_q,
        );
        assert_eq!(
            got, want,
            "color_config profile={profile} bd={bit_depth} mono={monochrome} srgb={want_srgb}"
        );
    }
}

#[test]
fn read_trailing_and_restoration_invert_write() {
    use aom_dsp::entropy::header::{
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
                &mut wb,
                intra_only,
                ref_sel,
                skip_allowed,
                skip_flag,
                might_warp,
                warp,
                reduced,
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
            let chroma_none = ft[..num_planes]
                .iter()
                .enumerate()
                .all(|(p, &t)| t == 0 || p == 0);
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
            assert_eq!(
                g.frame_restoration_type, ft,
                "restoration types np={num_planes}"
            );
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
    use aom_dsp::entropy::header::{
        read_global_motion_params, write_global_motion_params, WarpedMotionParams,
    };
    let ident = WarpedMotionParams {
        wmtype: 0,
        wmmat: [0, 0, 1 << 16, 0, 0, 1 << 16],
    };
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
            let tc = |rng: &mut Rng| -> i32 {
                (rng.next() % (2 * bound as u64 + 1)) as i32 - bound as i32
            };
            wm[0] = tc(&mut rng) << tpd;
            wm[1] = tc(&mut rng) << tpd;
        }
        let params = WarpedMotionParams {
            wmtype: ty,
            wmmat: wm,
        };
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
    use aom_dsp::entropy::header::{
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
                num_ticks_per_picture: if equal {
                    1 + (rng.next() % (1 << 20)) as u32
                } else {
                    1
                },
            };
            let mut wb = WriteBitBuffer::new();
            write_timing_info_header(&mut wb, &t);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let g = read_timing_info_header(&mut rb);
            assert_eq!(
                g.num_units_in_display_tick, t.num_units_in_display_tick,
                "timing units"
            );
            assert_eq!(g.time_scale, t.time_scale, "timing scale");
            assert_eq!(g.equal_picture_interval, equal, "timing equal");
            if equal {
                assert_eq!(
                    g.num_ticks_per_picture, t.num_ticks_per_picture,
                    "timing ticks"
                );
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
                (
                    g.encoder_decoder_buffer_delay_length,
                    g.num_units_in_decoding_tick,
                    g.buffer_removal_time_length,
                    g.frame_presentation_time_length
                ),
                (
                    d.encoder_decoder_buffer_delay_length,
                    d.num_units_in_decoding_tick,
                    d.buffer_removal_time_length,
                    d.frame_presentation_time_length
                ),
                "decoder model"
            );
        }
    }
}

#[test]
fn read_tile_info_inverts_write() {
    use aom_dsp::entropy::header::{read_tile_info, write_tile_info, TileInfoHeader};
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
            let (mi_cols, mi_rows) = (
                1 + (rng.next() % 4096) as i32,
                1 + (rng.next() % 4096) as i32,
            );
            // av1_calculate_tile_cols / _rows (tile_common.c): the REAL C
            // decoder's read_tile_info_max_tile (decodeframe.c) calls these
            // UNCONDITIONALLY after reading log2_cols/log2_rows — even in the
            // uniform branch — deriving col/row_start_sb (and re-deriving
            // cols/rows as the resulting loop count) from mi_cols/mi_rows, NOT
            // simply `1 << log2_cols`. A real ENCODER populates its own
            // TileInfoHeader the same way before ever calling write_tile_info
            // (so the writer's `t.rows * t.cols > 1` ctx/tile_size_bytes gate
            // agrees with what the reader re-derives); mirror that derivation
            // here for a self-consistent writer-side fixture instead of the
            // `1 << log2_cols` shortcut (min_c/max_c/log2_c are constructed
            // independently of mi_cols above, unlike av1_get_tile_limits, so
            // the two need not coincide).
            let sb_cols = (mi_cols + (1 << mib) - 1) >> mib;
            let sb_rows = (mi_rows + (1 << mib) - 1) >> mib;
            let size_sb_c = (sb_cols + (1 << log2_c) - 1) >> log2_c;
            let mut exp_col_start_sb = [0i32; 65];
            let (mut start_sb, mut i) = (0, 0usize);
            while start_sb < sb_cols {
                exp_col_start_sb[i] = start_sb;
                start_sb += size_sb_c;
                i += 1;
            }
            exp_col_start_sb[i] = sb_cols;
            let exp_cols = i;
            let size_sb_r = (sb_rows + (1 << log2_r) - 1) >> log2_r;
            let mut exp_row_start_sb = [0i32; 65];
            let (mut start_sb, mut j) = (0, 0usize);
            while start_sb < sb_rows {
                exp_row_start_sb[j] = start_sb;
                start_sb += size_sb_r;
                j += 1;
            }
            exp_row_start_sb[j] = sb_rows;
            let exp_rows = j;
            let t = TileInfoHeader {
                mi_cols,
                mi_rows,
                mib_size_log2: mib,
                uniform_spacing: true,
                log2_cols: log2_c,
                min_log2_cols: min_c,
                max_log2_cols: max_c,
                log2_rows: log2_r,
                min_log2_rows: min_r,
                max_log2_rows: max_r,
                cols: exp_cols,
                rows: exp_rows,
                col_start_sb: exp_col_start_sb,
                row_start_sb: exp_row_start_sb,
                max_width_sb: 64,
                max_height_sb: 64,
            };
            let mut wb = WriteBitBuffer::new();
            write_tile_info(&mut wb, &t);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let (g, ctx, tsb) = read_tile_info(
                &mut rb, mi_cols, mi_rows, mib, min_c, max_c, min_r, max_r, 64, 64,
            );
            assert!(g.uniform_spacing, "uniform flag");
            assert_eq!((g.log2_cols, g.log2_rows), (log2_c, log2_r), "uniform log2");
            assert_eq!(
                (g.cols, g.rows),
                (exp_cols, exp_rows),
                "uniform cols/rows sb=({sb_cols},{sb_rows}) log2=({log2_c},{log2_r})"
            );
            assert_eq!(g.col_start_sb, exp_col_start_sb, "uniform col_start_sb");
            assert_eq!(g.row_start_sb, exp_row_start_sb, "uniform row_start_sb");
            if exp_rows * exp_cols > 1 {
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
                mi_cols,
                mi_rows,
                mib_size_log2: mib,
                uniform_spacing: false,
                log2_cols: tlog2(1, cols as i32),
                min_log2_cols: 0,
                max_log2_cols: 6,
                log2_rows: tlog2(1, rows as i32),
                min_log2_rows: 0,
                max_log2_rows: 6,
                cols,
                rows,
                col_start_sb,
                row_start_sb,
                max_width_sb: max_w,
                max_height_sb: max_h,
            };
            let mut wb = WriteBitBuffer::new();
            write_tile_info(&mut wb, &t);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let (g, ctx, tsb) =
                read_tile_info(&mut rb, mi_cols, mi_rows, mib, 0, 6, 0, 6, max_w, max_h);
            assert!(!g.uniform_spacing, "non-uniform flag");
            assert_eq!(
                (g.cols, g.rows),
                (cols, rows),
                "nonuniform cols/rows sb=({sb_cols},{sb_rows})"
            );
            assert_eq!(g.col_start_sb, col_start_sb, "col_start_sb");
            assert_eq!(g.row_start_sb, row_start_sb, "row_start_sb");
            assert_eq!(
                (g.log2_cols, g.log2_rows),
                (tlog2(1, cols as i32), tlog2(1, rows as i32)),
                "nonuniform log2"
            );
            if rows * cols > 1 {
                assert_eq!((ctx, tsb), (0, 4), "ctx/tile_size_bytes");
            }
        }
    }
}

#[test]
fn read_frame_size_with_refs_inverts_write() {
    use aom_dsp::entropy::header::{
        read_frame_size_with_refs, write_frame_size_with_refs, FrameSizeHeader, FrameSizeWithRefs,
    };
    let dummy_fs = FrameSizeHeader {
        frame_size_override: true,
        num_bits_width: 16,
        num_bits_height: 16,
        superres_upscaled_width: 1,
        superres_upscaled_height: 1,
        enable_superres: false,
        scale_denominator: 8,
        scaling_active: false,
        render_width: 0,
        render_height: 0,
    };
    let mut rng = Rng(0x1e_f52e_c0de_01c0);
    for _ in 0..100_000 {
        let en_sr = rng.next() & 1 == 1;
        let denom = if en_sr && rng.next() & 1 == 1 {
            9 + (rng.next() % 8) as i32
        } else {
            8
        };
        if rng.next() & 1 == 1 {
            // matched-ref path
            let fi = (rng.next() % 7) as usize;
            let (w, h) = (
                1 + (rng.next() % 4096) as i32,
                1 + (rng.next() % 4096) as i32,
            );
            let (rw, rh) = (
                1 + (rng.next() % 4096) as i32,
                1 + (rng.next() % 4096) as i32,
            );
            let mut rcw = [0i32; 7];
            let mut rch = [0i32; 7];
            let mut rrw = [0i32; 7];
            let mut rrh = [0i32; 7];
            rcw[fi] = w;
            rch[fi] = h;
            rrw[fi] = rw;
            rrh[fi] = rh;
            let mut valid = [false; 7];
            valid[fi] = true;
            let fx = FrameSizeWithRefs {
                superres_upscaled_width: w,
                superres_upscaled_height: h,
                render_width: rw,
                render_height: rh,
                ref_cfg_valid: valid,
                ref_y_crop_width: rcw,
                ref_y_crop_height: rch,
                ref_render_width: rrw,
                ref_render_height: rrh,
                enable_superres: en_sr,
                scale_denominator: denom,
                frame_size: dummy_fs,
            };
            let mut wb = WriteBitBuffer::new();
            write_frame_size_with_refs(&mut wb, &fx);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let g = read_frame_size_with_refs(&mut rb, &rcw, &rch, &rrw, &rrh, en_sr, 16, 16);
            assert_eq!(g, (w, h, rw, rh, denom, fi as i32), "matched ref fi={fi}");
        } else {
            // fallback path (no ref matches)
            let (w, h) = (
                1 + (rng.next() % 65536) as i32,
                1 + (rng.next() % 65536) as i32,
            );
            let scaling = rng.next() & 1 == 1;
            let (rw, rh) = if scaling {
                (
                    1 + (rng.next() % 65536) as i32,
                    1 + (rng.next() % 65536) as i32,
                )
            } else {
                (w, h)
            };
            let fs = FrameSizeHeader {
                frame_size_override: true,
                num_bits_width: 16,
                num_bits_height: 16,
                superres_upscaled_width: w,
                superres_upscaled_height: h,
                enable_superres: en_sr,
                scale_denominator: denom,
                scaling_active: scaling,
                render_width: rw,
                render_height: rh,
            };
            let fx = FrameSizeWithRefs {
                superres_upscaled_width: w,
                superres_upscaled_height: h,
                render_width: rw,
                render_height: rh,
                ref_cfg_valid: [false; 7],
                ref_y_crop_width: [0; 7],
                ref_y_crop_height: [0; 7],
                ref_render_width: [0; 7],
                ref_render_height: [0; 7],
                enable_superres: en_sr,
                scale_denominator: denom,
                frame_size: fs,
            };
            let mut wb = WriteBitBuffer::new();
            write_frame_size_with_refs(&mut wb, &fx);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let z = [0i32; 7];
            let g = read_frame_size_with_refs(&mut rb, &z, &z, &z, &z, en_sr, 16, 16);
            assert_eq!(g, (w, h, rw, rh, denom, -1), "fallback scaling={scaling}");
        }
    }
}

#[test]
fn read_inter_ref_signaling_inverts_write() {
    use aom_dsp::entropy::header::{
        read_inter_ref_signaling, write_inter_ref_signaling, InterRefSignaling,
    };
    let mut rng = Rng(0x1e_12e5_c0de_01d0);
    for _ in 0..100_000 {
        let short = rng.next() & 1 == 1;
        // short signalling requires order hint; else force order-hint random.
        let enable_order_hint = short || rng.next() & 1 == 1;
        // distinct ref_map_idx in [0,8) so frame-id deltas don't alias a map slot.
        let mut pool = [0i32, 1, 2, 3, 4, 5, 6, 7];
        for i in (1..8).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            pool.swap(i, j);
        }
        let mut ref_map_idx = [0i32; 7];
        ref_map_idx.copy_from_slice(&pool[..7]);

        // frame ids only exercised on the not-short path (short needs av1_set_frame_refs).
        let frame_id_present = !short && rng.next() & 1 == 1;
        let frame_id_length = 3 + (rng.next() % 13) as u32; // [3,15]
        let m = 1i32 << frame_id_length;
        let delta_frame_id_length = 2 + (rng.next() % (frame_id_length - 2).max(1) as u64) as u32; // [2, fidl-1]
        let current_frame_id = (rng.next() % m as u64) as i32;
        let mut want_delta = [0i32; 7];
        let mut ref_frame_id = [0i32; 8];
        if frame_id_present {
            for r in 0..7 {
                let d = (rng.next() % (1u64 << delta_frame_id_length)) as i32; // [0, 2^dfidl)
                want_delta[r] = d;
                // set ref_frame_id[map_idx] so the encoder recomputes exactly d.
                ref_frame_id[ref_map_idx[r] as usize] =
                    (current_frame_id - (d + 1) + m).rem_euclid(m);
            }
        }

        let s = InterRefSignaling {
            enable_order_hint,
            frame_refs_short_signaling: short,
            ref_map_idx,
            set_ref_frame_config: false,
            rtc_reference: [0; 7],
            rtc_ref_idx: [0; 7],
            number_spatial_layers: 1,
            frame_id_numbers_present_flag: frame_id_present,
            frame_id_length,
            current_frame_id,
            ref_frame_id,
            delta_frame_id_length,
        };
        let mut wb = WriteBitBuffer::new();
        write_inter_ref_signaling(&mut wb, &s);
        let b = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&b);
        let (gshort, glst, ggld, gremap, gdelta) = read_inter_ref_signaling(
            &mut rb,
            enable_order_hint,
            frame_id_present,
            delta_frame_id_length,
        );
        assert_eq!(gshort, short, "short flag");
        if short {
            assert_eq!(
                (glst, ggld),
                (ref_map_idx[0], ref_map_idx[3]),
                "short lst/gld"
            );
        } else {
            assert_eq!(gremap, ref_map_idx, "remapped idx");
            if frame_id_present {
                assert_eq!(gdelta, want_delta, "frame-id deltas");
            }
        }
    }
}

#[test]
fn read_refresh_and_film_grain_invert_write() {
    use aom_dsp::entropy::header::{
        read_film_grain_params, read_refresh_frame_context, write_film_grain_params,
        write_refresh_frame_context, FilmGrainParams,
    };
    let mut rng = Rng(0x1e_f11a_c0de_01e0);
    for _ in 0..100_000 {
        // refresh_frame_context
        {
            let reduced = rng.next() & 1 == 1;
            let disable = rng.next() & 1 == 1;
            let might = !reduced && !disable;
            let disabled = if might { rng.next() & 1 == 1 } else { true };
            let mut wb = WriteBitBuffer::new();
            write_refresh_frame_context(&mut wb, reduced, disable, disabled);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            assert_eq!(
                read_refresh_frame_context(&mut rb, reduced, disable),
                disabled,
                "refresh ctx"
            );
        }
        // film grain
        {
            let is_inter = rng.next() & 1 == 1;
            let mono = rng.next() & 1 == 1;
            let (ssx, ssy) = ((rng.next() % 2) as i32, (rng.next() % 2) as i32);
            let l8 = |rng: &mut Rng| -> i32 { (rng.next() % 256) as i32 };
            let mut fg = FilmGrainParams {
                apply_grain: false,
                random_seed: 0,
                is_inter_frame: is_inter,
                update_parameters: false,
                ref_idx: 0,
                num_y_points: 0,
                scaling_points_y: [[0; 2]; 14],
                monochrome: mono,
                chroma_scaling_from_luma: false,
                subsampling_x: ssx,
                subsampling_y: ssy,
                num_cb_points: 0,
                scaling_points_cb: [[0; 2]; 10],
                num_cr_points: 0,
                scaling_points_cr: [[0; 2]; 10],
                scaling_shift: 8,
                ar_coeff_lag: 0,
                ar_coeffs_y: [0; 24],
                ar_coeffs_cb: [0; 25],
                ar_coeffs_cr: [0; 25],
                ar_coeff_shift: 6,
                grain_scale_shift: 0,
                cb_mult: 0,
                cb_luma_mult: 0,
                cb_offset: 0,
                cr_mult: 0,
                cr_luma_mult: 0,
                cr_offset: 0,
                overlap_flag: false,
                clip_to_restricted_range: false,
            };
            fg.apply_grain = rng.next() & 1 == 1;
            if fg.apply_grain {
                fg.random_seed = (rng.next() % 65536) as i32;
                fg.update_parameters = if is_inter { rng.next() & 1 == 1 } else { true };
                if !fg.update_parameters {
                    fg.ref_idx = (rng.next() % 8) as i32;
                } else {
                    fg.num_y_points = (rng.next() % 15) as i32;
                    for i in 0..fg.num_y_points as usize {
                        fg.scaling_points_y[i] = [l8(&mut rng), l8(&mut rng)];
                    }
                    if !mono {
                        fg.chroma_scaling_from_luma = rng.next() & 1 == 1;
                    }
                    let chroma_absent = mono
                        || fg.chroma_scaling_from_luma
                        || (ssx == 1 && ssy == 1 && fg.num_y_points == 0);
                    if !chroma_absent {
                        fg.num_cb_points = (rng.next() % 11) as i32;
                        for i in 0..fg.num_cb_points as usize {
                            fg.scaling_points_cb[i] = [l8(&mut rng), l8(&mut rng)];
                        }
                        fg.num_cr_points = (rng.next() % 11) as i32;
                        for i in 0..fg.num_cr_points as usize {
                            fg.scaling_points_cr[i] = [l8(&mut rng), l8(&mut rng)];
                        }
                    }
                    fg.scaling_shift = 8 + (rng.next() % 4) as i32;
                    fg.ar_coeff_lag = (rng.next() % 4) as i32;
                    let npl = 2 * fg.ar_coeff_lag * (fg.ar_coeff_lag + 1);
                    let npc = npl + i32::from(fg.num_y_points > 0);
                    if fg.num_y_points != 0 {
                        for i in 0..npl as usize {
                            fg.ar_coeffs_y[i] = l8(&mut rng) - 128;
                        }
                    }
                    if fg.num_cb_points != 0 || fg.chroma_scaling_from_luma {
                        for i in 0..npc as usize {
                            fg.ar_coeffs_cb[i] = l8(&mut rng) - 128;
                        }
                    }
                    if fg.num_cr_points != 0 || fg.chroma_scaling_from_luma {
                        for i in 0..npc as usize {
                            fg.ar_coeffs_cr[i] = l8(&mut rng) - 128;
                        }
                    }
                    fg.ar_coeff_shift = 6 + (rng.next() % 4) as i32;
                    fg.grain_scale_shift = (rng.next() % 4) as i32;
                    if fg.num_cb_points != 0 {
                        fg.cb_mult = l8(&mut rng);
                        fg.cb_luma_mult = l8(&mut rng);
                        fg.cb_offset = (rng.next() % 512) as i32;
                    }
                    if fg.num_cr_points != 0 {
                        fg.cr_mult = l8(&mut rng);
                        fg.cr_luma_mult = l8(&mut rng);
                        fg.cr_offset = (rng.next() % 512) as i32;
                    }
                    fg.overlap_flag = rng.next() & 1 == 1;
                    fg.clip_to_restricted_range = rng.next() & 1 == 1;
                }
            }
            let mut wb = WriteBitBuffer::new();
            write_film_grain_params(&mut wb, &fg);
            let b = wb.bytes().to_vec();
            let mut rb = ReadBitBuffer::new(&b);
            let g = read_film_grain_params(&mut rb, is_inter, mono, ssx, ssy);
            assert_eq!(
                (
                    g.apply_grain,
                    g.random_seed,
                    g.update_parameters,
                    g.ref_idx,
                    g.num_y_points,
                    g.chroma_scaling_from_luma,
                    g.num_cb_points,
                    g.num_cr_points
                ),
                (
                    fg.apply_grain,
                    fg.random_seed,
                    fg.update_parameters,
                    fg.ref_idx,
                    fg.num_y_points,
                    fg.chroma_scaling_from_luma,
                    fg.num_cb_points,
                    fg.num_cr_points
                ),
                "film grain scalars A"
            );
            assert_eq!(
                (
                    g.scaling_shift,
                    g.ar_coeff_lag,
                    g.ar_coeff_shift,
                    g.grain_scale_shift,
                    g.overlap_flag,
                    g.clip_to_restricted_range
                ),
                (
                    fg.scaling_shift,
                    fg.ar_coeff_lag,
                    fg.ar_coeff_shift,
                    fg.grain_scale_shift,
                    fg.overlap_flag,
                    fg.clip_to_restricted_range
                ),
                "film grain scalars B"
            );
            assert_eq!(g.scaling_points_y, fg.scaling_points_y, "fg pts y");
            assert_eq!(
                (g.scaling_points_cb, g.scaling_points_cr),
                (fg.scaling_points_cb, fg.scaling_points_cr),
                "fg pts uv"
            );
            assert_eq!(
                (g.ar_coeffs_y, g.ar_coeffs_cb, g.ar_coeffs_cr),
                (fg.ar_coeffs_y, fg.ar_coeffs_cb, fg.ar_coeffs_cr),
                "fg ar coeffs"
            );
            assert_eq!(
                (
                    g.cb_mult,
                    g.cb_luma_mult,
                    g.cb_offset,
                    g.cr_mult,
                    g.cr_luma_mult,
                    g.cr_offset
                ),
                (
                    fg.cb_mult,
                    fg.cb_luma_mult,
                    fg.cb_offset,
                    fg.cr_mult,
                    fg.cr_luma_mult,
                    fg.cr_offset
                ),
                "fg mults"
            );
        }
    }
}

#[test]
fn read_sequence_header_inverts_write() {
    use aom_dsp::entropy::header::{read_sequence_header, write_sequence_header, SequenceHeaderParams};
    let mut rng = Rng(0x1e_5e90_c0de_01f0);
    for _ in 0..100_000 {
        let nbw = 1 + (rng.next() % 16) as u32;
        let nbh = 1 + (rng.next() % 16) as u32;
        let mfw = 1 + (rng.next() % (1u64 << nbw)) as i32;
        let mfh = 1 + (rng.next() % (1u64 << nbh)) as i32;
        let reduced = rng.next() & 1 == 1;
        let (mut fidp, mut dfidl, mut fidl) = (false, 0, 0);
        let (mut eic, mut emc, mut ewm, mut edf, mut eoh, mut edwc, mut erfm) =
            (false, false, false, false, false, false, false);
        let (mut fsct, mut fimv, mut ohbm1) = (2i32, 2i32, -1i32);
        if !reduced {
            fidp = rng.next() & 1 == 1;
            if fidp {
                dfidl = 2 + (rng.next() % 16) as i32;
                fidl = dfidl + 1 + (rng.next() % 8) as i32;
            }
            eic = rng.next() & 1 == 1;
            emc = rng.next() & 1 == 1;
            ewm = rng.next() & 1 == 1;
            edf = rng.next() & 1 == 1;
            eoh = rng.next() & 1 == 1;
            if eoh {
                edwc = rng.next() & 1 == 1;
                erfm = rng.next() & 1 == 1;
            }
            fsct = (rng.next() % 3) as i32;
            if fsct > 0 {
                fimv = (rng.next() % 3) as i32;
            }
            if eoh {
                ohbm1 = (rng.next() % 8) as i32;
            }
        }
        let s = SequenceHeaderParams {
            num_bits_width: nbw,
            num_bits_height: nbh,
            max_frame_width: mfw,
            max_frame_height: mfh,
            reduced_still_picture_hdr: reduced,
            frame_id_numbers_present_flag: fidp,
            delta_frame_id_length: dfidl,
            frame_id_length: fidl,
            sb_size_128: rng.next() & 1 == 1,
            enable_filter_intra: rng.next() & 1 == 1,
            enable_intra_edge_filter: rng.next() & 1 == 1,
            enable_interintra_compound: eic,
            enable_masked_compound: emc,
            enable_warped_motion: ewm,
            enable_dual_filter: edf,
            enable_order_hint: eoh,
            enable_dist_wtd_comp: edwc,
            enable_ref_frame_mvs: erfm,
            force_screen_content_tools: fsct,
            force_integer_mv: fimv,
            order_hint_bits_minus_1: ohbm1,
            enable_superres: rng.next() & 1 == 1,
            enable_cdef: rng.next() & 1 == 1,
            enable_restoration: rng.next() & 1 == 1,
        };
        let mut wb = WriteBitBuffer::new();
        write_sequence_header(&mut wb, &s);
        let b = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&b);
        let g = read_sequence_header(&mut rb, reduced);
        assert_eq!(
            (
                g.num_bits_width,
                g.num_bits_height,
                g.max_frame_width,
                g.max_frame_height,
                g.frame_id_numbers_present_flag,
                g.delta_frame_id_length,
                g.frame_id_length,
                g.sb_size_128
            ),
            (nbw, nbh, mfw, mfh, fidp, dfidl, fidl, s.sb_size_128),
            "seq hdr A red={reduced}"
        );
        assert_eq!(
            (
                g.enable_filter_intra,
                g.enable_intra_edge_filter,
                g.enable_interintra_compound,
                g.enable_masked_compound,
                g.enable_warped_motion,
                g.enable_dual_filter,
                g.enable_order_hint
            ),
            (
                s.enable_filter_intra,
                s.enable_intra_edge_filter,
                eic,
                emc,
                ewm,
                edf,
                eoh
            ),
            "seq hdr B"
        );
        assert_eq!(
            (
                g.enable_dist_wtd_comp,
                g.enable_ref_frame_mvs,
                g.force_screen_content_tools,
                g.force_integer_mv,
                g.order_hint_bits_minus_1,
                g.enable_superres,
                g.enable_cdef,
                g.enable_restoration
            ),
            (
                edwc,
                erfm,
                fsct,
                fimv,
                ohbm1,
                s.enable_superres,
                s.enable_cdef,
                s.enable_restoration
            ),
            "seq hdr C"
        );
    }
}

#[test]
fn read_frame_header_prefix_inverts_write() {
    use aom_dsp::entropy::header::{
        read_frame_header_prefix, write_frame_header_prefix, FrameHeaderPrefix,
    };
    let mut rng = Rng(0x1e_f5e1_c0de_0200);
    for _ in 0..300_000 {
        let reduced = rng.next().is_multiple_of(8); // exercise the degenerate reduced path occasionally
        let dmi = rng.next() & 1 == 1;
        let eqpi = rng.next() & 1 == 1;
        let fptl = 1 + (rng.next() % 20) as u32;
        let fidp = rng.next() & 1 == 1;
        let fidlen = 8 + (rng.next() % 8) as u32;
        let force_sct = (rng.next() % 3) as i32;
        let force_int = (rng.next() % 3) as i32;
        let eoh = rng.next() & 1 == 1;
        let ohbm1 = (rng.next() % 8) as i32;
        let opcnt = (rng.next() % 4) as i32;
        let brtl = 1 + (rng.next() % 20) as u32;
        let tlid = (rng.next() % 8) as i32;
        let slid = (rng.next() % 4) as i32;
        let mut opdmpp = [false; 32];
        let mut opidc = [0i32; 32];
        for op in 0..=opcnt as usize {
            opdmpp[op] = rng.next() & 1 == 1;
            opidc[op] = (rng.next() % 4096) as i32;
        }

        let show_existing = !reduced && rng.next().is_multiple_of(6);
        let (mut frame_type, mut show_frame, mut showable, mut err_res) =
            (0i32, true, false, false);
        let (mut efb, mut fpt, mut dfid) = (0i32, 0u32, 0i32);
        if reduced {
            // KEY/show/no-err inferred
        } else if show_existing {
            efb = (rng.next() % 8) as i32;
            fpt = if dmi && !eqpi {
                (rng.next() % (1u64 << fptl)) as u32
            } else {
                0
            };
            dfid = if fidp {
                (rng.next() % (1u64 << fidlen)) as i32
            } else {
                0
            };
        } else {
            frame_type = (rng.next() % 4) as i32;
            show_frame = rng.next() & 1 == 1;
            if show_frame {
                fpt = if dmi && !eqpi {
                    (rng.next() % (1u64 << fptl)) as u32
                } else {
                    0
                };
                showable = frame_type != 0;
            } else {
                showable = rng.next() & 1 == 1;
            }
            err_res = if frame_type == 3 {
                true
            } else if frame_type == 0 && show_frame {
                false
            } else {
                rng.next() & 1 == 1
            };
        }
        let sframe = frame_type == 3;
        let intra_only = frame_type == 0 || frame_type == 2;
        let disable_cdf = rng.next() & 1 == 1;
        let allow_sct = if force_sct == 2 {
            rng.next() & 1 == 1
        } else {
            force_sct != 0
        };
        let cur_int_mv = if allow_sct {
            if force_int == 2 {
                rng.next() & 1 == 1
            } else {
                force_int != 0
            }
        } else {
            false
        };
        let (max_w, max_h) = (
            1 + (rng.next() % 1024) as i32,
            1 + (rng.next() % 1024) as i32,
        );
        let want_override = if sframe { 1 } else { (rng.next() & 1) as i32 };
        let (sup_w, sup_h) = if want_override == 1 && !sframe {
            (max_w + 1, max_h)
        } else {
            (max_w, max_h)
        };
        let current_frame_id = if !reduced && fidp {
            (rng.next() % (1u64 << fidlen)) as i32
        } else {
            0
        };
        let order_hint = if !reduced && eoh {
            (rng.next() % (1u64 << (ohbm1 + 1))) as i32
        } else {
            0
        };
        let primary_ref = if !reduced && !err_res && !intra_only {
            (rng.next() % 8) as i32
        } else {
            7
        };
        let brt_present = dmi && rng.next() & 1 == 1;
        let mut brts = [0u32; 32];
        if brt_present {
            for op in 0..=opcnt as usize {
                if opdmpp[op] {
                    let idc = opidc[op];
                    if idc == 0 || (((idc >> tlid) & 1) != 0 && ((idc >> (slid + 8)) & 1) != 0) {
                        brts[op] = (rng.next() % (1u64 << brtl)) as u32;
                    }
                }
            }
        }
        let refresh_written =
            (frame_type == 0 && !show_frame) || frame_type == 1 || frame_type == 2;
        let refresh = if refresh_written {
            (rng.next() % 256) as i32
        } else {
            0xff
        };
        let rfmoh_written = (!intra_only || refresh != 0xff) && err_res && eoh;
        let mut rfmoh = [0i32; 8];
        if rfmoh_written {
            for oh in rfmoh.iter_mut() {
                *oh = (rng.next() % (1u64 << (ohbm1 + 1))) as i32;
            }
        }

        let p = FrameHeaderPrefix {
            reduced_still_picture_hdr: reduced,
            show_existing_frame: show_existing,
            existing_fb_idx_to_show: efb,
            decoder_model_info_present_flag: dmi,
            equal_picture_interval: eqpi,
            frame_presentation_time: fpt,
            frame_presentation_time_length: fptl,
            frame_id_numbers_present_flag: fidp,
            frame_id_length: fidlen,
            display_frame_id: dfid,
            frame_type,
            show_frame,
            showable_frame: showable,
            error_resilient_mode: err_res,
            disable_cdf_update: disable_cdf,
            force_screen_content_tools: force_sct,
            allow_screen_content_tools: allow_sct,
            force_integer_mv: force_int,
            cur_frame_force_integer_mv: cur_int_mv,
            superres_upscaled_width: sup_w,
            superres_upscaled_height: sup_h,
            max_frame_width: max_w,
            max_frame_height: max_h,
            current_frame_id,
            enable_order_hint: eoh,
            order_hint,
            order_hint_bits_minus_1: ohbm1,
            primary_ref_frame: primary_ref,
            buffer_removal_time_present: brt_present,
            operating_points_cnt_minus_1: opcnt,
            op_decoder_model_param_present: opdmpp,
            operating_point_idc: opidc,
            temporal_layer_id: tlid,
            spatial_layer_id: slid,
            buffer_removal_times: brts,
            buffer_removal_time_length: brtl,
            refresh_frame_flags: refresh,
            ref_frame_map_order_hint: rfmoh,
        };
        let mut wb = WriteBitBuffer::new();
        let (wover, wearly) = write_frame_header_prefix(&mut wb, &p);
        let b = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&b);
        let (g, gover, gearly) = read_frame_header_prefix(&mut rb, &p);
        assert_eq!(
            (gover, gearly),
            (wover, wearly),
            "override/early ft={frame_type} se={show_existing}"
        );
        assert_eq!(g.show_existing_frame, show_existing, "show_existing");
        if show_existing {
            assert_eq!(
                (
                    g.existing_fb_idx_to_show,
                    g.frame_presentation_time,
                    g.display_frame_id
                ),
                (efb, fpt, dfid),
                "show_existing fields"
            );
            continue;
        }
        assert_eq!(
            (
                g.frame_type,
                g.show_frame,
                g.showable_frame,
                g.error_resilient_mode,
                g.frame_presentation_time
            ),
            (frame_type, show_frame, showable, err_res, fpt),
            "frame type block red={reduced}"
        );
        assert_eq!(
            (
                g.disable_cdf_update,
                g.allow_screen_content_tools,
                g.cur_frame_force_integer_mv,
                g.current_frame_id,
                g.order_hint,
                g.primary_ref_frame
            ),
            (
                disable_cdf,
                allow_sct,
                cur_int_mv,
                current_frame_id,
                order_hint,
                primary_ref
            ),
            "tools/id block"
        );
        assert_eq!(
            (g.buffer_removal_time_present, g.buffer_removal_times),
            (brt_present, brts),
            "buffer removal"
        );
        assert_eq!(
            (g.refresh_frame_flags, g.ref_frame_map_order_hint),
            (refresh, rfmoh),
            "refresh/order hints"
        );
    }
}

#[test]
fn read_sequence_header_obu_inverts_write() {
    use aom_dsp::entropy::header::{
        read_sequence_header_obu, write_sequence_header_obu, ColorConfigParams, DecoderModelInfo,
        SequenceHeaderObu, SequenceHeaderParams, TimingInfoHeader,
    };
    let mut rng = Rng(0x1e_5e0b_c0de_0210);
    for _ in 0..100_000 {
        let profile = (rng.next() % 3) as i32;
        let bit_depth = if profile == 2 {
            [8, 10, 12][(rng.next() % 3) as usize]
        } else {
            [8, 10][(rng.next() % 2) as usize]
        };
        // color config (valid per profile, following read_color_config inference)
        let mono = profile != 1 && rng.next() & 1 == 1;
        let want_srgb = !mono && profile == 1 && rng.next() & 1 == 1;
        let (ssx, ssy) = if mono {
            (1, 1)
        } else if want_srgb {
            (0, 0)
        } else if profile == 0 {
            (1, 1)
        } else if profile == 1 {
            (0, 0)
        } else if bit_depth == 12 {
            let x = (rng.next() & 1) as i32;
            (x, if x == 1 { (rng.next() & 1) as i32 } else { 0 })
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
                cp = 5;
            }
            (cp, tc, mc)
        } else {
            (2, 2, 2)
        };
        let chroma_pos = if !mono && !want_srgb && ssx == 1 && ssy == 1 {
            (rng.next() % 4) as i32
        } else {
            0
        };
        let sep_uv = !mono && rng.next() & 1 == 1;
        let color_config = ColorConfigParams {
            bit_depth,
            profile,
            monochrome: mono,
            color_primaries: cp,
            transfer_characteristics: tc,
            matrix_coefficients: mc,
            color_range: want_srgb || rng.next() & 1 == 1,
            subsampling_x: ssx,
            subsampling_y: ssy,
            chroma_sample_position: chroma_pos,
            separate_uv_delta_q: sep_uv,
        };

        let reduced = rng.next().is_multiple_of(4);
        let still_picture = reduced || rng.next() & 1 == 1;
        // sequence header (reduced consistent)
        let nbw = 1 + (rng.next() % 16) as u32;
        let nbh = 1 + (rng.next() % 16) as u32;
        let eoh = !reduced && rng.next() & 1 == 1;
        let seq_header = SequenceHeaderParams {
            num_bits_width: nbw,
            num_bits_height: nbh,
            max_frame_width: 1 + (rng.next() % (1u64 << nbw)) as i32,
            max_frame_height: 1 + (rng.next() % (1u64 << nbh)) as i32,
            reduced_still_picture_hdr: reduced,
            frame_id_numbers_present_flag: false,
            delta_frame_id_length: 0,
            frame_id_length: 0,
            sb_size_128: rng.next() & 1 == 1,
            enable_filter_intra: rng.next() & 1 == 1,
            enable_intra_edge_filter: rng.next() & 1 == 1,
            enable_interintra_compound: !reduced && rng.next() & 1 == 1,
            enable_masked_compound: !reduced && rng.next() & 1 == 1,
            enable_warped_motion: !reduced && rng.next() & 1 == 1,
            enable_dual_filter: !reduced && rng.next() & 1 == 1,
            enable_order_hint: eoh,
            enable_dist_wtd_comp: eoh && rng.next() & 1 == 1,
            enable_ref_frame_mvs: eoh && rng.next() & 1 == 1,
            force_screen_content_tools: if reduced { 2 } else { (rng.next() % 3) as i32 },
            force_integer_mv: 2,
            order_hint_bits_minus_1: if eoh { (rng.next() % 8) as i32 } else { -1 },
            enable_superres: rng.next() & 1 == 1,
            enable_cdef: rng.next() & 1 == 1,
            enable_restoration: rng.next() & 1 == 1,
        };
        let sh = SequenceHeaderParams {
            force_integer_mv: if seq_header.force_screen_content_tools > 0 {
                (rng.next() % 3) as i32
            } else {
                2
            },
            ..seq_header
        };

        // operating points
        let opcnt = if reduced { 0 } else { (rng.next() % 4) as i32 };
        let ti_present = !reduced && rng.next() & 1 == 1;
        let dmi_present = ti_present && rng.next() & 1 == 1;
        let disp_present = !reduced && rng.next() & 1 == 1;
        let ti = TimingInfoHeader {
            num_units_in_display_tick: rng.next() as u32,
            time_scale: rng.next() as u32,
            equal_picture_interval: rng.next() & 1 == 1,
            num_ticks_per_picture: 1 + (rng.next() % (1 << 20)) as u32,
        };
        let ti = if ti.equal_picture_interval {
            ti
        } else {
            TimingInfoHeader {
                num_ticks_per_picture: 1,
                ..ti
            }
        };
        let bdl = 1 + (rng.next() % 16) as i32;
        let dmi = DecoderModelInfo {
            encoder_decoder_buffer_delay_length: bdl,
            num_units_in_decoding_tick: rng.next() as u32,
            buffer_removal_time_length: 1 + (rng.next() % 16) as i32,
            frame_presentation_time_length: 1 + (rng.next() % 16) as i32,
        };
        let mut op_idc = [0i32; 32];
        let mut sli = [0i32; 32];
        let mut tier = [0i32; 32];
        let mut op_dmpp = [false; 32];
        let mut op_dispp = [false; 32];
        let mut op_dbd = [0u32; 32];
        let mut op_ebd = [0u32; 32];
        let mut op_ldmf = [false; 32];
        let mut op_idd = [0i32; 32];
        if reduced {
            sli[0] = (rng.next() % 32) as i32;
        } else {
            for i in 0..=opcnt as usize {
                op_idc[i] = (rng.next() % 4096) as i32;
                sli[i] = (rng.next() % 32) as i32;
                if sli[i] >= 8 {
                    tier[i] = (rng.next() & 1) as i32;
                }
                if dmi_present {
                    op_dmpp[i] = rng.next() & 1 == 1;
                    if op_dmpp[i] {
                        op_dbd[i] = (rng.next() % (1u64 << bdl)) as u32;
                        op_ebd[i] = (rng.next() % (1u64 << bdl)) as u32;
                        op_ldmf[i] = rng.next() & 1 == 1;
                    }
                }
                if disp_present {
                    op_dispp[i] = rng.next() & 1 == 1;
                    if op_dispp[i] {
                        op_idd[i] = 1 + (rng.next() % 10) as i32;
                    }
                }
            }
        }

        let obu = SequenceHeaderObu {
            profile,
            still_picture,
            reduced_still_picture_hdr: reduced,
            timing_info_present: ti_present,
            timing_info: ti,
            decoder_model_info_present_flag: dmi_present,
            decoder_model_info: dmi,
            display_model_info_present_flag: disp_present,
            operating_points_cnt_minus_1: opcnt,
            operating_point_idc: op_idc,
            seq_level_idx: sli,
            tier,
            op_decoder_model_param_present: op_dmpp,
            op_display_model_param_present: op_dispp,
            op_decoder_buffer_delay: op_dbd,
            op_encoder_buffer_delay: op_ebd,
            op_low_delay_mode_flag: op_ldmf,
            op_initial_display_delay: op_idd,
            seq_header: sh,
            color_config,
            film_grain_params_present: rng.next() & 1 == 1,
        };
        let mut wb = WriteBitBuffer::new();
        write_sequence_header_obu(&mut wb, &obu);
        let b = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&b);
        let g = read_sequence_header_obu(&mut rb);
        assert_eq!(
            (
                g.profile,
                g.still_picture,
                g.reduced_still_picture_hdr,
                g.timing_info_present,
                g.decoder_model_info_present_flag,
                g.display_model_info_present_flag,
                g.operating_points_cnt_minus_1,
                g.film_grain_params_present
            ),
            (
                profile,
                still_picture,
                reduced,
                ti_present,
                dmi_present,
                disp_present,
                opcnt,
                obu.film_grain_params_present
            ),
            "obu top red={reduced}"
        );
        assert_eq!(
            (g.operating_point_idc, g.seq_level_idx, g.tier),
            (op_idc, sli, tier),
            "obu op arrays"
        );
        assert_eq!(
            (
                g.op_decoder_model_param_present,
                g.op_decoder_buffer_delay,
                g.op_encoder_buffer_delay
            ),
            (op_dmpp, op_dbd, op_ebd),
            "obu op dmi"
        );
        assert_eq!(
            (g.op_display_model_param_present, g.op_initial_display_delay),
            (op_dispp, op_idd),
            "obu op disp"
        );
        assert_eq!(
            (
                g.seq_header.num_bits_width,
                g.seq_header.enable_order_hint,
                g.seq_header.enable_superres
            ),
            (nbw, eoh, sh.enable_superres),
            "obu seq_header"
        );
        assert_eq!(
            (
                g.color_config.bit_depth,
                g.color_config.subsampling_x,
                g.color_config.subsampling_y
            ),
            (bit_depth, ssx, ssy),
            "obu color_config"
        );
    }
}

#[test]
fn read_uncompressed_header_key_frame_roundtrips() {
    use aom_dsp::entropy::header::{
        read_uncompressed_header, write_frame_header_obu, CdefHeader, ColorConfigParams,
        DeltaQParams, FilmGrainParams, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader,
        FrameSizeWithRefs, InterRefSignaling, LoopfilterHeader, QuantParamsHeader,
        RestorationHeader, SegmentationHeader, TileInfoHeader, TimingInfoHeader,
        WarpedMotionParams,
    };
    let _ = (
        TimingInfoHeader {
            num_units_in_display_tick: 0,
            time_scale: 0,
            equal_picture_interval: false,
            num_ticks_per_picture: 1,
        },
        ColorConfigParams {
            bit_depth: 8,
            profile: 0,
            monochrome: false,
            color_primaries: 2,
            transfer_characteristics: 2,
            matrix_coefficients: 2,
            color_range: false,
            subsampling_x: 1,
            subsampling_y: 1,
            chroma_sample_position: 0,
            separate_uv_delta_q: false,
        },
    );
    let ident = WarpedMotionParams {
        wmtype: 0,
        wmmat: [0, 0, 1 << 16, 0, 0, 1 << 16],
    };
    let empty_prefix = |ft: i32, show: bool, allow_sct: bool| FrameHeaderPrefix {
        reduced_still_picture_hdr: false,
        show_existing_frame: false,
        existing_fb_idx_to_show: 0,
        decoder_model_info_present_flag: false,
        equal_picture_interval: false,
        frame_presentation_time: 0,
        frame_presentation_time_length: 1,
        frame_id_numbers_present_flag: false,
        frame_id_length: 0,
        display_frame_id: 0,
        frame_type: ft,
        show_frame: show,
        showable_frame: false,
        error_resilient_mode: false,
        disable_cdf_update: false,
        force_screen_content_tools: 2,
        allow_screen_content_tools: allow_sct,
        force_integer_mv: 2,
        cur_frame_force_integer_mv: false,
        superres_upscaled_width: 256,
        superres_upscaled_height: 256,
        max_frame_width: 256,
        max_frame_height: 256,
        current_frame_id: 0,
        enable_order_hint: false,
        order_hint: 0,
        order_hint_bits_minus_1: -1,
        primary_ref_frame: 7,
        buffer_removal_time_present: false,
        operating_points_cnt_minus_1: 0,
        op_decoder_model_param_present: [false; 32],
        operating_point_idc: [0; 32],
        temporal_layer_id: 0,
        spatial_layer_id: 0,
        buffer_removal_times: [0; 32],
        buffer_removal_time_length: 1,
        refresh_frame_flags: 0xff,
        ref_frame_map_order_hint: [0; 8],
    };
    let mut rng = Rng(0x1e_f5e0_c0de_0220);
    for _ in 0..20_000 {
        let allow_sct = rng.next() & 1 == 1;
        let superres_scaled = rng.next() & 1 == 1;
        let base_q = 1 + (rng.next() % 255) as i32; // nonzero => not coded-lossless
        let intrabc_possible = allow_sct && !superres_scaled;
        let allow_intrabc = intrabc_possible && rng.next() & 1 == 1;
        let fg_present = rng.next() & 1 == 1;

        let fh = FrameHeaderObu {
            prefix: empty_prefix(0, true, allow_sct),
            allow_screen_content_tools: allow_sct,
            superres_scaled,
            allow_intrabc,
            frame_size: FrameSizeHeader {
                frame_size_override: false,
                num_bits_width: 16,
                num_bits_height: 16,
                superres_upscaled_width: 256,
                superres_upscaled_height: 256,
                enable_superres: false,
                scale_denominator: 8,
                scaling_active: false,
                render_width: 0,
                render_height: 0,
            },
            inter_ref: InterRefSignaling {
                enable_order_hint: false,
                frame_refs_short_signaling: false,
                ref_map_idx: [0; 7],
                set_ref_frame_config: false,
                rtc_reference: [0; 7],
                rtc_ref_idx: [0; 7],
                number_spatial_layers: 1,
                frame_id_numbers_present_flag: false,
                frame_id_length: 0,
                current_frame_id: 0,
                ref_frame_id: [0; 8],
                delta_frame_id_length: 0,
            },
            frame_size_with_refs: FrameSizeWithRefs {
                superres_upscaled_width: 256,
                superres_upscaled_height: 256,
                render_width: 256,
                render_height: 256,
                ref_cfg_valid: [false; 7],
                ref_y_crop_width: [0; 7],
                ref_y_crop_height: [0; 7],
                ref_render_width: [0; 7],
                ref_render_height: [0; 7],
                enable_superres: false,
                scale_denominator: 8,
                frame_size: FrameSizeHeader {
                    frame_size_override: true,
                    num_bits_width: 16,
                    num_bits_height: 16,
                    superres_upscaled_width: 1,
                    superres_upscaled_height: 1,
                    enable_superres: false,
                    scale_denominator: 8,
                    scaling_active: false,
                    render_width: 0,
                    render_height: 0,
                },
            },
            cur_frame_force_integer_mv: false,
            allow_high_precision_mv: false,
            interp_filter: 0,
            switchable_motion_mode: false,
            might_allow_ref_frame_mvs: false,
            allow_ref_frame_mvs: false,
            refresh_frame_context_disabled: rng.next() & 1 == 1,
            tile_info: TileInfoHeader {
                mi_cols: 64,
                mi_rows: 64,
                mib_size_log2: 4,
                uniform_spacing: true,
                log2_cols: 0,
                min_log2_cols: 0,
                max_log2_cols: 0,
                log2_rows: 0,
                min_log2_rows: 0,
                max_log2_rows: 0,
                cols: 1,
                rows: 1,
                col_start_sb: [0; 65],
                row_start_sb: [0; 65],
                max_width_sb: 64,
                max_height_sb: 64,
            },
            context_update_tile_id: 0,
            tile_size_bytes: 1,
            quant: QuantParamsHeader {
                base_qindex: base_q,
                y_dc_delta_q: 0,
                u_dc_delta_q: 0,
                u_ac_delta_q: 0,
                v_dc_delta_q: 0,
                v_ac_delta_q: 0,
                using_qmatrix: false,
                qmatrix_level_y: 0,
                qmatrix_level_u: 0,
                qmatrix_level_v: 0,
            },
            num_planes: 3,
            separate_uv_delta_q: false,
            segmentation: SegmentationHeader {
                enabled: false,
                has_primary_ref: false,
                update_map: false,
                temporal_update: false,
                update_data: false,
                feature_mask: [0; 8],
                feature_data: [[0; 8]; 8],
            },
            delta_q: DeltaQParams {
                base_qindex: base_q,
                delta_q_present: rng.next() & 1 == 1,
                delta_q_res: 1,
                allow_intrabc,
                delta_lf_present: false,
                delta_lf_res: 1,
                delta_lf_multi: false,
            },
            all_lossless: false,
            coded_lossless: false,
            loopfilter: LoopfilterHeader {
                // enabled=1 with no delta update: the encoder's unconditional `meaningful`
                // bit then coincides with the decoder's gated update bit (libaom only ever
                // emits enabled=1; enabled=0 desyncs — see read_loopfilter).
                allow_intrabc,
                filter_level: [(rng.next() % 64) as i32, (rng.next() % 64) as i32],
                filter_level_u: 0,
                filter_level_v: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: true,
                mode_ref_delta_update: false,
                ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
                mode_deltas: [0, 0],
                last_ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
                last_mode_deltas: [0, 0],
            },
            cdef: CdefHeader {
                enable_cdef: true,
                allow_intrabc,
                cdef_damping: 3 + (rng.next() % 4) as i32,
                cdef_bits: 0,
                nb_cdef_strengths: 1,
                cdef_strengths: [(rng.next() % 64) as i32, 0, 0, 0, 0, 0, 0, 0],
                cdef_uv_strengths: [(rng.next() % 64) as i32, 0, 0, 0, 0, 0, 0, 0],
            },
            restoration: RestorationHeader {
                enable_restoration: true,
                allow_intrabc,
                frame_restoration_type: [0; 3],
                sb_size_128: false,
                restoration_unit_size: [256; 3],
                subsampling_x: 1,
                subsampling_y: 1,
            },
            tx_mode_select: rng.next() & 1 == 1,
            reference_mode_select: false,
            skip_mode_allowed: false,
            derive_skip_mode_allowed: false,
            skip_mode_flag: false,
            might_allow_warped_motion: false,
            allow_warped_motion: false,
            reduced_tx_set_used: rng.next() & 1 == 1,
            global_motion: [ident; 7],
            ref_global_motion: [ident; 7],
            film_grain_params_present: fg_present,
            film_grain: FilmGrainParams {
                apply_grain: fg_present && rng.next() & 1 == 1,
                random_seed: (rng.next() % 65536) as i32,
                is_inter_frame: false,
                update_parameters: true,
                ref_idx: 0,
                num_y_points: 0,
                scaling_points_y: [[0; 2]; 14],
                monochrome: false,
                chroma_scaling_from_luma: false,
                subsampling_x: 1,
                subsampling_y: 1,
                num_cb_points: 0,
                scaling_points_cb: [[0; 2]; 10],
                num_cr_points: 0,
                scaling_points_cr: [[0; 2]; 10],
                scaling_shift: 8,
                ar_coeff_lag: 0,
                ar_coeffs_y: [0; 24],
                ar_coeffs_cb: [0; 25],
                ar_coeffs_cr: [0; 25],
                ar_coeff_shift: 6,
                grain_scale_shift: 0,
                cb_mult: 0,
                cb_luma_mult: 0,
                cb_offset: 0,
                cr_mult: 0,
                cr_luma_mult: 0,
                cr_offset: 0,
                overlap_flag: false,
                clip_to_restricted_range: false,
            },
            large_scale: false,
        };
        let mut wb = WriteBitBuffer::new();
        write_frame_header_obu(&mut wb, &fh);
        let b = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&b);
        let g = read_uncompressed_header(&mut rb, &fh);
        assert_eq!(g.allow_intrabc, allow_intrabc, "allow_intrabc");
        assert_eq!(
            g.refresh_frame_context_disabled, fh.refresh_frame_context_disabled,
            "refresh ctx"
        );
        assert_eq!(g.quant.base_qindex, base_q, "quant base");
        assert_eq!(
            g.delta_q.delta_q_present, fh.delta_q.delta_q_present,
            "delta_q present"
        );
        // loopfilter/cdef are coded only when intrabc is off (they early-return otherwise).
        if !allow_intrabc {
            assert_eq!(
                g.loopfilter.filter_level, fh.loopfilter.filter_level,
                "loopfilter level"
            );
            assert_eq!(g.cdef.cdef_damping, fh.cdef.cdef_damping, "cdef damping");
            assert_eq!(
                g.cdef.cdef_strengths, fh.cdef.cdef_strengths,
                "cdef strengths"
            );
        }
        assert_eq!(g.tx_mode_select, fh.tx_mode_select, "tx_mode");
        assert_eq!(g.reduced_tx_set_used, fh.reduced_tx_set_used, "reduced_tx");
        // film grain is parsed only when present+shown; otherwise the field is untouched.
        if fg_present {
            assert_eq!(
                g.film_grain.apply_grain, fh.film_grain.apply_grain,
                "film grain apply"
            );
            assert_eq!(
                g.film_grain.random_seed,
                if fh.film_grain.apply_grain {
                    fh.film_grain.random_seed
                } else {
                    0
                },
                "film grain seed"
            );
        }
        assert!(!rb.error, "no over-read (desync)");
    }
}

#[test]
fn read_uncompressed_header_inter_frame_roundtrips() {
    use aom_dsp::entropy::header::{
        read_uncompressed_header, write_frame_header_obu, CdefHeader, DeltaQParams,
        FilmGrainParams, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, FrameSizeWithRefs,
        InterRefSignaling, LoopfilterHeader, QuantParamsHeader, RestorationHeader,
        SegmentationHeader, TileInfoHeader, WarpedMotionParams,
    };
    let ident = WarpedMotionParams {
        wmtype: 0,
        wmmat: [0, 0, 1 << 16, 0, 0, 1 << 16],
    };
    let mut rng = Rng(0x1e_12e5_c0de_0230);
    for _ in 0..20_000 {
        let base_q = 1 + (rng.next() % 255) as i32;
        let refresh = (rng.next() % 256) as i32;
        let primary_ref = (rng.next() % 8) as i32;
        let might_ref_mvs = rng.next() & 1 == 1;
        let might_warp = rng.next() & 1 == 1;
        let skip_allowed = rng.next() & 1 == 1;
        let mut prefix = FrameHeaderPrefix {
            reduced_still_picture_hdr: false,
            show_existing_frame: false,
            existing_fb_idx_to_show: 0,
            decoder_model_info_present_flag: false,
            equal_picture_interval: false,
            frame_presentation_time: 0,
            frame_presentation_time_length: 1,
            frame_id_numbers_present_flag: false,
            frame_id_length: 0,
            display_frame_id: 0,
            frame_type: 1,
            show_frame: true,
            showable_frame: true,
            error_resilient_mode: false,
            disable_cdf_update: false,
            force_screen_content_tools: 2,
            allow_screen_content_tools: false,
            force_integer_mv: 2,
            cur_frame_force_integer_mv: false,
            superres_upscaled_width: 256,
            superres_upscaled_height: 256,
            max_frame_width: 256,
            max_frame_height: 256,
            current_frame_id: 0,
            enable_order_hint: false,
            order_hint: 0,
            order_hint_bits_minus_1: -1,
            primary_ref_frame: primary_ref,
            buffer_removal_time_present: false,
            operating_points_cnt_minus_1: 0,
            op_decoder_model_param_present: [false; 32],
            operating_point_idc: [0; 32],
            temporal_layer_id: 0,
            spatial_layer_id: 0,
            buffer_removal_times: [0; 32],
            buffer_removal_time_length: 1,
            refresh_frame_flags: refresh,
            ref_frame_map_order_hint: [0; 8],
        };
        prefix.showable_frame = true; // frame_type != KEY
        let fh = FrameHeaderObu {
            prefix,
            allow_screen_content_tools: false,
            superres_scaled: false,
            allow_intrabc: false,
            frame_size: FrameSizeHeader {
                frame_size_override: false,
                num_bits_width: 16,
                num_bits_height: 16,
                superres_upscaled_width: 256,
                superres_upscaled_height: 256,
                enable_superres: false,
                scale_denominator: 8,
                scaling_active: false,
                render_width: 0,
                render_height: 0,
            },
            inter_ref: InterRefSignaling {
                enable_order_hint: false,
                frame_refs_short_signaling: false,
                ref_map_idx: [
                    (rng.next() % 8) as i32,
                    (rng.next() % 8) as i32,
                    (rng.next() % 8) as i32,
                    (rng.next() % 8) as i32,
                    (rng.next() % 8) as i32,
                    (rng.next() % 8) as i32,
                    (rng.next() % 8) as i32,
                ],
                set_ref_frame_config: false,
                rtc_reference: [0; 7],
                rtc_ref_idx: [0; 7],
                number_spatial_layers: 1,
                frame_id_numbers_present_flag: false,
                frame_id_length: 0,
                current_frame_id: 0,
                ref_frame_id: [0; 8],
                delta_frame_id_length: 0,
            },
            frame_size_with_refs: FrameSizeWithRefs {
                superres_upscaled_width: 256,
                superres_upscaled_height: 256,
                render_width: 256,
                render_height: 256,
                ref_cfg_valid: [false; 7],
                ref_y_crop_width: [0; 7],
                ref_y_crop_height: [0; 7],
                ref_render_width: [0; 7],
                ref_render_height: [0; 7],
                enable_superres: false,
                scale_denominator: 8,
                frame_size: FrameSizeHeader {
                    frame_size_override: true,
                    num_bits_width: 16,
                    num_bits_height: 16,
                    superres_upscaled_width: 1,
                    superres_upscaled_height: 1,
                    enable_superres: false,
                    scale_denominator: 8,
                    scaling_active: false,
                    render_width: 0,
                    render_height: 0,
                },
            },
            cur_frame_force_integer_mv: false,
            allow_high_precision_mv: rng.next() & 1 == 1,
            interp_filter: (rng.next() % 5) as i32,
            switchable_motion_mode: rng.next() & 1 == 1,
            might_allow_ref_frame_mvs: might_ref_mvs,
            allow_ref_frame_mvs: might_ref_mvs && rng.next() & 1 == 1,
            refresh_frame_context_disabled: rng.next() & 1 == 1,
            tile_info: TileInfoHeader {
                mi_cols: 64,
                mi_rows: 64,
                mib_size_log2: 4,
                uniform_spacing: true,
                log2_cols: 0,
                min_log2_cols: 0,
                max_log2_cols: 0,
                log2_rows: 0,
                min_log2_rows: 0,
                max_log2_rows: 0,
                cols: 1,
                rows: 1,
                col_start_sb: [0; 65],
                row_start_sb: [0; 65],
                max_width_sb: 64,
                max_height_sb: 64,
            },
            context_update_tile_id: 0,
            tile_size_bytes: 1,
            quant: QuantParamsHeader {
                base_qindex: base_q,
                y_dc_delta_q: 0,
                u_dc_delta_q: 0,
                u_ac_delta_q: 0,
                v_dc_delta_q: 0,
                v_ac_delta_q: 0,
                using_qmatrix: false,
                qmatrix_level_y: 0,
                qmatrix_level_u: 0,
                qmatrix_level_v: 0,
            },
            num_planes: 3,
            separate_uv_delta_q: false,
            segmentation: SegmentationHeader {
                enabled: false,
                has_primary_ref: primary_ref != 7,
                update_map: false,
                temporal_update: false,
                update_data: false,
                feature_mask: [0; 8],
                feature_data: [[0; 8]; 8],
            },
            delta_q: DeltaQParams {
                base_qindex: base_q,
                delta_q_present: rng.next() & 1 == 1,
                delta_q_res: 1,
                allow_intrabc: false,
                delta_lf_present: false,
                delta_lf_res: 1,
                delta_lf_multi: false,
            },
            all_lossless: false,
            coded_lossless: false,
            loopfilter: LoopfilterHeader {
                allow_intrabc: false,
                filter_level: [(rng.next() % 64) as i32, (rng.next() % 64) as i32],
                filter_level_u: 0,
                filter_level_v: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: true,
                mode_ref_delta_update: false,
                ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
                mode_deltas: [0, 0],
                last_ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
                last_mode_deltas: [0, 0],
            },
            cdef: CdefHeader {
                enable_cdef: true,
                allow_intrabc: false,
                cdef_damping: 3 + (rng.next() % 4) as i32,
                cdef_bits: 0,
                nb_cdef_strengths: 1,
                cdef_strengths: [(rng.next() % 64) as i32, 0, 0, 0, 0, 0, 0, 0],
                cdef_uv_strengths: [(rng.next() % 64) as i32, 0, 0, 0, 0, 0, 0, 0],
            },
            restoration: RestorationHeader {
                enable_restoration: true,
                allow_intrabc: false,
                frame_restoration_type: [0; 3],
                sb_size_128: false,
                restoration_unit_size: [256; 3],
                subsampling_x: 1,
                subsampling_y: 1,
            },
            tx_mode_select: rng.next() & 1 == 1,
            reference_mode_select: rng.next() & 1 == 1,
            skip_mode_allowed: skip_allowed,
            derive_skip_mode_allowed: false,
            skip_mode_flag: skip_allowed && rng.next() & 1 == 1,
            might_allow_warped_motion: might_warp,
            allow_warped_motion: might_warp && rng.next() & 1 == 1,
            reduced_tx_set_used: rng.next() & 1 == 1,
            global_motion: [ident; 7],
            ref_global_motion: [ident; 7],
            film_grain_params_present: false,
            film_grain: FilmGrainParams {
                apply_grain: false,
                random_seed: 0,
                is_inter_frame: true,
                update_parameters: false,
                ref_idx: 0,
                num_y_points: 0,
                scaling_points_y: [[0; 2]; 14],
                monochrome: false,
                chroma_scaling_from_luma: false,
                subsampling_x: 1,
                subsampling_y: 1,
                num_cb_points: 0,
                scaling_points_cb: [[0; 2]; 10],
                num_cr_points: 0,
                scaling_points_cr: [[0; 2]; 10],
                scaling_shift: 8,
                ar_coeff_lag: 0,
                ar_coeffs_y: [0; 24],
                ar_coeffs_cb: [0; 25],
                ar_coeffs_cr: [0; 25],
                ar_coeff_shift: 6,
                grain_scale_shift: 0,
                cb_mult: 0,
                cb_luma_mult: 0,
                cb_offset: 0,
                cr_mult: 0,
                cr_luma_mult: 0,
                cr_offset: 0,
                overlap_flag: false,
                clip_to_restricted_range: false,
            },
            large_scale: false,
        };
        let mut wb = WriteBitBuffer::new();
        write_frame_header_obu(&mut wb, &fh);
        let b = wb.bytes().to_vec();
        let mut rb = ReadBitBuffer::new(&b);
        let g = read_uncompressed_header(&mut rb, &fh);
        assert_eq!(g.prefix.frame_type, 1, "inter frame_type");
        assert_eq!(g.prefix.refresh_frame_flags, refresh, "refresh flags");
        assert_eq!(g.prefix.primary_ref_frame, primary_ref, "primary ref");
        assert_eq!(
            g.inter_ref.ref_map_idx, fh.inter_ref.ref_map_idx,
            "remapped idx"
        );
        assert_eq!(
            g.allow_high_precision_mv, fh.allow_high_precision_mv,
            "allow_hp"
        );
        assert_eq!(g.interp_filter, fh.interp_filter, "interp filter");
        assert_eq!(
            g.switchable_motion_mode, fh.switchable_motion_mode,
            "switchable motion"
        );
        if might_ref_mvs {
            assert_eq!(
                g.allow_ref_frame_mvs, fh.allow_ref_frame_mvs,
                "ref frame mvs"
            );
        }
        assert_eq!(
            g.reference_mode_select, fh.reference_mode_select,
            "reference mode select"
        );
        assert_eq!(g.reduced_tx_set_used, fh.reduced_tx_set_used, "reduced tx");
        assert_eq!(g.global_motion[0].wmtype, 0, "gm identity");
        assert!(!rb.error, "no over-read (inter desync)");
    }
}
