//! Byte-match [`aom_dsp::entropy::header::write_sequence_header_obu`] against the
//! REAL sequence-header OBU produced by `shim_encode_av1_kf` (the real
//! `aom_codec_av1_cx` public API — the same path `aomenc` drives). This is a
//! first slice of the encoder gate's step 3 ("byte-match the smallest frame
//! vs shim_encode_av1_kf"): the sequence-header OBU is content-independent
//! (doesn't depend on the partition/mode RDO decisions this port's search
//! makes), so it's the part of the byte-match reachable WITHOUT first
//! porting aomenc's loop-filter-level search, CDEF-strength search, or
//! deriving real (CDF-based, not this crate's synthetic-but-valid) RD cost
//! tables — all still-missing pieces for the frame-header + tile-group
//! portions of the byte match (see `pack_tile_roundtrip.rs` and the pack
//! module docs for that status).
//!
//! Method: encode a real minimal KEY frame via `ref_encode_av1_kf`, walk its
//! OBU stream (`read_obu_header` + `uleb_decode`, both already bit-exact vs
//! C) to find the sequence-header OBU's payload, parse it with the
//! ALREADY-VALIDATED `read_sequence_header_obu` (aom-entropy, decoder-owned,
//! independently bit-exact-tested against real libaom in
//! `header_diff.rs::write_sequence_header_obu_matches_real_c`'s round trip
//! and elsewhere), then re-serialize the parsed struct with
//! `write_sequence_header_obu` and assert the bytes are identical to the
//! real OBU's payload. Any mismatch here would be a genuine ordering/field
//! bug in the writer (the parsed values are real aomenc's own choices, not
//! a guess).

use aom_dsp::entropy::header::{read_sequence_header_obu, write_sequence_header_obu};
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::rb::ReadBitBuffer;
use aom_dsp::entropy::wb::WriteBitBuffer;
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;

/// Split a real AV1 byte stream into `(obu_type, payload)` pairs (OBU header
/// + leb128 size framing only -- no payload interpretation).
fn walk_obus(bytes: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        assert!(
            hdr.obu_has_size_field,
            "shim_encode_av1_kf always sets has_size_field"
        );
        let (size, size_bytes) =
            aom_dsp::entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
        let payload_start = after_header + size_bytes;
        let payload_end = payload_start + size as usize;
        out.push((hdr.obu_type, &bytes[payload_start..payload_end]));
        pos = payload_end;
    }
    out
}

#[test]
fn sequence_header_matches_real_aomenc_output() {
    c::ref_init();

    // Sweep the two ALLINTRA-directive-relevant usages (usage=2 primary,
    // usage=0 secondary) x mono/color x a couple of sizes -- the sequence
    // header depends on profile (from bd/mono/subsampling), sb-size, and the
    // tool-enable flags aomenc's speed/usage config selects, all
    // content-independent so a flat mid-gray source is fine.
    let cases: &[(usize, usize, bool, usize, usize, u32, i32)] = &[
        (64, 64, true, 1, 1, 2, 32),   // mono, ALLINTRA, sb64
        (64, 64, false, 1, 1, 2, 40),  // 420, ALLINTRA
        (128, 96, false, 0, 0, 0, 60), // 444, GOOD
        (64, 64, false, 1, 0, 2, 20),  // 422, ALLINTRA
    ];

    for &(w, h, mono, ss_x, ss_y, usage, cq_level) in cases {
        let y = vec![128u16; w * h];
        let (cw, ch) = if mono {
            (0, 0)
        } else {
            ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
        };
        let u = vec![128u16; cw * ch];
        let v = vec![128u16; cw * ch];

        let bytes = c::ref_encode_av1_kf(
            &y,
            &u,
            &v,
            w,
            h,
            8,
            mono,
            ss_x as i32,
            ss_y as i32,
            cq_level,
            0,
            false,
            false,
            usage,
            0,
            false,
        );
        assert!(
            !bytes.is_empty(),
            "shim_encode_av1_kf must produce a real stream"
        );

        let obus = walk_obus(&bytes);
        let seq_payload = obus
            .iter()
            .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
            .map(|(_, p)| *p)
            .unwrap_or_else(|| panic!("no sequence-header OBU in stream (w={w} h={h})"));

        let mut rb = ReadBitBuffer::new(seq_payload);
        let real_seq = read_sequence_header_obu(&mut rb);

        let mut wb = WriteBitBuffer::new();
        write_sequence_header_obu(&mut wb, &real_seq);
        assert_eq!(
            wb.bytes(),
            seq_payload,
            "w={w} h={h} mono={mono} ss=({ss_x},{ss_y}) usage={usage} cq={cq_level}: our \
             write_sequence_header_obu must reproduce the real aomenc sequence-header OBU bytes \
             exactly, given the exact field values real aomenc chose"
        );
    }
}
