//! **Independent multi-tile header serialization + assembly byte-match gate
//! (#16).** Proves [`assemble_multitile_frame_obu_payload_derived`] reconstructs
//! a REAL libaom multi-tile `OBU_FRAME` payload BYTE-FOR-BYTE by serializing the
//! frame header ITSELF (via `write_frame_header_obu`) and filling in the real
//! `context_update_tile_id` + `tile_size_bytes` the way the C encoder does after
//! tile packing (`av1/encoder/bitstream.c` `write_tile_obu_size`, :4053/:4068) --
//! NOT bootstrapped from the reference header's bytes (the limitation the old
//! `encoder_gate_multitile` gate documented).
//!
//! For each real `ref_encode_av1_kf_tiles` stream this:
//!   1. extracts the `OBU_FRAME` payload and parses the header (`p`);
//!   2. proves the ANTI-VACUOUS precondition: the header serialized with
//!      `write_tile_info`'s PLACEHOLDERS (`context_update_tile_id == 0`,
//!      `tile_size_bytes_minus_1 == 3`) DIFFERS from C's real header bytes -- so
//!      matching C is only possible if the overwrites actually fired;
//!   3. extracts C's real per-tile payloads and feeds them to the DERIVED
//!      assembler, asserting its output equals C's `OBU_FRAME` payload byte-for-
//!      byte (header re-serialization + derived `tile_size_bytes`/`largest_tile_id`
//!      overwrites + length-prefixed tile-group assembly, all vs C).
//!
//! Content is skewed noisier toward the bottom-right so a non-first tile is the
//! largest -- exercising a NON-ZERO `context_update_tile_id` (= `largest_tile_id`)
//! overwrite, not just the trivial 0 the placeholder already holds.

use aom_decode::frame::decode_frame_obus_prefilter;
use aom_encode::obu_assemble::assemble_multitile_frame_obu_payload_derived;
use aom_dsp::entropy::header::write_frame_header_obu;
use aom_dsp::entropy::leb128::uleb_decode;
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::wb::WriteBitBuffer;
use aom_sys_ref as c;

const OBU_FRAME: u32 = 6;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Photographic-ish luma with high-frequency detail INCREASING toward the
/// bottom-right, so later (raster-order) tiles encode to larger payloads and the
/// largest tile is usually not tile 0 (exercises a non-zero context_update_tile_id).
fn gen_luma(w: usize, h: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed | 1);
    let mut p = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let fx = col as f64 / w.max(1) as f64;
            let fy = r as f64 / h.max(1) as f64;
            let base = 40.0 + 120.0 * (0.6 * fx + 0.4 * fy);
            // detail amplitude grows toward the bottom-right corner.
            let detail = 6.0 + 44.0 * (0.5 * fx + 0.5 * fy);
            let noise = (((rng.next() >> 40) % 1000) as f64 / 1000.0 - 0.5) * 2.0 * detail;
            p[r * w + col] = (base + noise).clamp(0.0, 255.0) as u16;
        }
    }
    p
}

/// Extract the `OBU_FRAME` (type 6) payload bytes from a real encoded stream.
fn obu_frame_payload(stream: &[u8]) -> Vec<u8> {
    let mut pos = 0usize;
    while pos < stream.len() {
        let h = read_obu_header(&stream[pos..]).expect("bad OBU header");
        assert!(h.obu_has_size_field, "OBU must carry a size field");
        let (size, size_len) =
            uleb_decode(&stream[pos + h.header_len..]).expect("bad OBU size leb128");
        let body = pos + h.header_len + size_len;
        let end = body + size as usize;
        if h.obu_type == OBU_FRAME {
            return stream[body..end].to_vec();
        }
        pos = end;
    }
    panic!("no OBU_FRAME in the encoded stream");
}

/// Split a `num_tg == 1` tile-group payload (leading `tile_start_and_end_present`
/// = one `0x00` byte, then `tile_size_bytes`-byte LE `len-1` prefixes on every
/// tile but the last) into the raw per-tile payloads. Mirrors the decoder's
/// `split_tiles` / `get_tile_buffer`.
fn split_tiles(tile_group: &[u8], num_tiles: usize, tsb: usize) -> Vec<Vec<u8>> {
    let mut tiles = Vec::with_capacity(num_tiles);
    let mut pos = 1; // skip the tile-group header (present_flag 0 + byte_alignment)
    for i in 0..num_tiles {
        if i + 1 < num_tiles {
            let mut v = 0u64;
            for b in 0..tsb {
                v |= (tile_group[pos + b] as u64) << (8 * b);
            }
            pos += tsb;
            let len = (v + 1) as usize; // + AV1_MIN_TILE_SIZE_BYTES
            tiles.push(tile_group[pos..pos + len].to_vec());
            pos += len;
        } else {
            tiles.push(tile_group[pos..].to_vec());
        }
    }
    tiles
}

struct Case {
    w: usize,
    h: usize,
    ss_x: i32,
    ss_y: i32,
    mono: bool,
    usage: u32,
    cq: i32,
    tile_columns_log2: i32,
    tile_rows_log2: i32,
}

/// Run one case; returns `(tile_size_bytes, context_update_tile_id, num_tiles)`
/// observed on the REAL stream (for the coverage floors).
fn run_case(cs: &Case) -> (i32, i32, usize) {
    let (cw, ch) = if cs.mono {
        (0, 0)
    } else {
        (
            (cs.w + cs.ss_x as usize) >> cs.ss_x,
            (cs.h + cs.ss_y as usize) >> cs.ss_y,
        )
    };
    let seed =
        ((cs.w as u64) << 40) ^ ((cs.h as u64) << 24) ^ ((cs.cq as u64) << 8) ^ cs.usage as u64;
    let y = gen_luma(cs.w, cs.h, seed);
    let u = vec![128u16; cw * ch];
    let v = vec![128u16; cw * ch];

    let stream = c::ref_encode_av1_kf_tiles(
        &y,
        &u,
        &v,
        cs.w,
        cs.h,
        8,
        cs.mono,
        cs.ss_x,
        cs.ss_y,
        cs.cq,
        0,
        false,
        false,
        cs.usage,
        0,
        false,
        false, // sb_size_128
        cs.tile_columns_log2,
        cs.tile_rows_log2,
    );
    assert!(!stream.is_empty(), "oracle must produce a real stream");

    let frame_payload = obu_frame_payload(&stream);
    let (_t, _cfg, p) =
        decode_frame_obus_prefilter(&stream).expect("port prefilter decode of C stream");
    let num_tiles = p.tile_info.rows * p.tile_info.cols;
    let ctx = format!(
        "{}x{} ss=({},{}) mono={} usage={} cq={} grid={}x{}",
        cs.w, cs.h, cs.ss_x, cs.ss_y, cs.mono, cs.usage, cs.cq, p.tile_info.cols, p.tile_info.rows
    );
    assert!(
        num_tiles > 1,
        "{ctx}: case must be multi-tile (got {num_tiles})"
    );

    // The frame header re-serialized with write_tile_info's PLACEHOLDERS
    // (context_update_tile_id == 0, tile_size_bytes_minus_1 == 3). Its LENGTH is
    // byte-exact vs C (only the two placeholder VALUES differ), so it locates the
    // header/tile-group boundary.
    let mut hwb = WriteBitBuffer::new();
    write_frame_header_obu(&mut hwb, &p);
    hwb.byte_align_zeros();
    let placeholder_header = hwb.bytes().to_vec();
    let header_end = placeholder_header.len();

    // ANTI-VACUOUS: the placeholder header must DIFFER from C's real header bytes
    // (i.e. context_update_tile_id and/or tile_size_bytes_minus_1 are really
    // overwritten). Otherwise the byte-match below would prove nothing.
    assert_ne!(
        placeholder_header.as_slice(),
        &frame_payload[0..header_end],
        "{ctx}: placeholder multi-tile header already equals C — overwrite gate is vacuous"
    );

    // Extract C's REAL per-tile payloads and reconstruct via the DERIVED assembler
    // (independent header serialization + derived tile_size_bytes / largest_tile_id
    // overwrites). Must reproduce C's OBU_FRAME payload byte-for-byte.
    let tiles = split_tiles(
        &frame_payload[header_end..],
        num_tiles,
        p.tile_size_bytes as usize,
    );
    assert_eq!(tiles.len(), num_tiles, "{ctx}: tile split count");
    let our_payload = assemble_multitile_frame_obu_payload_derived(&p, &tiles);

    if our_payload != frame_payload {
        let first_diff = our_payload
            .iter()
            .zip(frame_payload.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(our_payload.len().min(frame_payload.len()));
        panic!(
            "{ctx}: DERIVED multi-tile OBU_FRAME payload != C at byte {first_diff} \
             (our={:?} c={:?}); our.len()={} c.len()={} tile_size_bytes={} \
             context_update_tile_id={} tile_lens={:?}",
            our_payload.get(first_diff),
            frame_payload.get(first_diff),
            our_payload.len(),
            frame_payload.len(),
            p.tile_size_bytes,
            p.context_update_tile_id,
            tiles.iter().map(|t| t.len()).collect::<Vec<_>>(),
        );
    }
    println!(
        "{ctx}: BYTE MATCH — tile_size_bytes={} context_update_tile_id={} tile_lens={:?}",
        p.tile_size_bytes,
        p.context_update_tile_id,
        tiles.iter().map(|t| t.len()).collect::<Vec<_>>()
    );
    (p.tile_size_bytes, p.context_update_tile_id, num_tiles)
}

#[test]
fn derived_multitile_obu_frame_byte_matches_c() {
    // Grid shapes (2x1 col-only, 1x2 row-only, 2x2 both, 4x1 four-col) x formats
    // x cq. Low cq (high quality) grows tile payloads toward tile_size_bytes >= 2;
    // the bottom-right-heavy content pushes largest_tile_id off 0.
    let cases = [
        // column-only / row-only tiling
        Case {
            w: 256,
            h: 128,
            ss_x: 1,
            ss_y: 1,
            mono: false,
            usage: 2,
            cq: 8,
            tile_columns_log2: 1,
            tile_rows_log2: 0,
        },
        Case {
            w: 128,
            h: 256,
            ss_x: 1,
            ss_y: 1,
            mono: false,
            usage: 2,
            cq: 8,
            tile_columns_log2: 0,
            tile_rows_log2: 1,
        },
        // 2x2 both-axes tiling, several formats
        Case {
            w: 256,
            h: 256,
            ss_x: 1,
            ss_y: 1,
            mono: false,
            usage: 2,
            cq: 6,
            tile_columns_log2: 1,
            tile_rows_log2: 1,
        },
        Case {
            w: 256,
            h: 256,
            ss_x: 0,
            ss_y: 0,
            mono: false,
            usage: 0,
            cq: 12,
            tile_columns_log2: 1,
            tile_rows_log2: 1,
        },
        Case {
            w: 256,
            h: 256,
            ss_x: 1,
            ss_y: 1,
            mono: true,
            usage: 2,
            cq: 6,
            tile_columns_log2: 1,
            tile_rows_log2: 1,
        },
        // 4x1 four tile columns
        Case {
            w: 256,
            h: 64,
            ss_x: 1,
            ss_y: 1,
            mono: false,
            usage: 2,
            cq: 20,
            tile_columns_log2: 2,
            tile_rows_log2: 0,
        },
        // larger low-cq frame to push tile_size_bytes to 2
        Case {
            w: 384,
            h: 384,
            ss_x: 1,
            ss_y: 1,
            mono: false,
            usage: 2,
            cq: 4,
            tile_columns_log2: 1,
            tile_rows_log2: 1,
        },
    ];

    let mut n = 0u32;
    let mut tsb_seen = std::collections::BTreeSet::new();
    let mut saw_ctx_nonzero = false;
    let mut saw_tsb_gt1 = false;
    for cs in &cases {
        let (tsb, ctx_id, _num) = run_case(cs);
        tsb_seen.insert(tsb);
        saw_ctx_nonzero |= ctx_id != 0;
        saw_tsb_gt1 |= tsb > 1;
        n += 1;
    }
    println!(
        "derived multi-tile gate: {n} cases all byte-match; tile_size_bytes seen={tsb_seen:?} \
         saw_ctx_nonzero={saw_ctx_nonzero} saw_tsb_gt1={saw_tsb_gt1}"
    );
    assert_eq!(n as usize, cases.len(), "every case must run");
    // Anti-vacuous coverage: the overwrite mechanism is only meaningfully proven
    // if it wrote a NON-ZERO context_update_tile_id (largest_tile_id != 0) at
    // least once (the trivial 0 already matches the placeholder), and a
    // tile_size_bytes > 1 at least once (the multi-byte LE prefix path).
    assert!(
        saw_ctx_nonzero,
        "no case produced a non-zero context_update_tile_id — the largest_tile_id \
         overwrite is only exercised at its trivial 0 value"
    );
    assert!(
        saw_tsb_gt1,
        "no case produced tile_size_bytes > 1 — the multi-byte length-prefix path \
         is untested; add a larger/lower-cq case"
    );
}
