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
