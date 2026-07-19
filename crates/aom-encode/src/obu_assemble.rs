//! Assemble a real AV1 `OBU_FRAME` (frame header + tile group, the
//! `cpi->num_tg == 1` combined form real aomenc's default config always
//! produces) from the already bit-exact pieces:
//! [`aom_dsp::entropy::header::write_frame_header_obu`] +
//! [`aom_dsp::entropy::header::write_tile_group_header`] (both aom-entropy,
//! decoder-owned but CALLED here, not modified) +
//! [`crate::pack::pack_tile`]'s raw entropy-coded tile bytes +
//! [`aom_dsp::entropy::obu::write_obu_header`] / [`aom_dsp::entropy::leb128::uleb_encode`]
//! (OBU-level wrapping, also aom-entropy).
//!
//! # Bit layout (AV1 spec, `frame_obu( sz )`)
//!
//! ```text
//! frame_obu(sz) {
//!     frame_header_obu()      // write_frame_header_obu -- already bit-exact
//!     byte_alignment()        // pad to the next byte boundary
//!     tile_group_obu(sz)      // tile_start_and_end_present_flag [+ start/end]
//!                             // byte_alignment()  (tile_group_obu's OWN)
//!                             // tile data
//! }
//! ```
//!
//! For `num_tiles == 1` (`tiles_log2 == 0`), `write_tile_group_header`'s own
//! C twin has a hard early return (`if (!tiles_log2) return;` —
//! `av1/encoder/bitstream.c`) and reads/writes ZERO bits, so
//! `tile_group_obu`'s own trailing `byte_alignment()` is *also* a no-op
//! (nothing was written since the frame header's own alignment already
//! landed on a byte boundary). This collapses the whole tile-group envelope
//! to: frame-header bits, one byte-align, then the raw tile bytes verbatim
//! -- the sole/last tile carries no `tile_size_bytes`-byte length prefix
//! (matching the decoder's `split_tiles`: only tiles BEFORE the last one are
//! length-prefixed). Multi-tile (`tiles_log2 > 0`, length-prefixed non-last
//! tiles) is NOT implemented -- the natural next lift once the envelope
//! needs more than one tile.

use aom_dsp::entropy::header::{FrameHeaderObu, write_frame_header_obu, write_tile_group_header};
use aom_dsp::entropy::leb128::uleb_encode;
use aom_dsp::entropy::obu::write_obu_header;
use aom_dsp::entropy::wb::WriteBitBuffer;

/// `OBU_FRAME` (`av1/common/enums.h` `OBU_TYPE`).
pub const OBU_FRAME: u32 = 6;

/// Assemble the `OBU_FRAME` PAYLOAD ONLY (frame-header bits + byte-align +
/// tile-group header + byte-align + tile data) -- everything
/// `write_frame_header_obu`/`write_tile_group_header` produce, without the
/// OBU header/leb128-size wrapper. Exposed separately from
/// [`assemble_obu_frame_single_tile`] so callers that only need the payload
/// (e.g. to size a leb128 field themselves) don't pay for an extra copy.
/// `num_tiles == 1` only (`tiles_log2` MUST be 0 -- see module docs).
pub fn assemble_frame_obu_payload_single_tile(
    frame_header: &FrameHeaderObu,
    tiles_log2: i32,
    tile_bytes: &[u8],
) -> Vec<u8> {
    assert_eq!(
        tiles_log2, 0,
        "assemble_frame_obu_payload_single_tile: multi-tile (tiles_log2 > 0) not implemented"
    );
    let mut wb = WriteBitBuffer::new();
    write_frame_header_obu(&mut wb, frame_header);
    wb.byte_align_zeros(); // frame_obu()'s byte_alignment(), between frame_header_obu() and tile_group_obu()
    // tile_group_obu()'s own header: a hard no-op at tiles_log2 == 0 (see
    // module docs) -- called anyway so the tiles_log2 > 0 shape is honestly
    // represented in the composition, even though this function asserts
    // that arm unreachable for now.
    write_tile_group_header(&mut wb, 0, 0, tiles_log2, false);
    wb.byte_align_zeros(); // tile_group_obu()'s OWN byte_alignment(); a no-op here (nothing written above)
    let mut payload = wb.bytes().to_vec();
    payload.extend_from_slice(tile_bytes); // the sole/last tile: no length prefix
    payload
}

/// Assemble ONE complete, OBU-wrapped `OBU_FRAME` (header byte(s) + leb128
/// size + payload) for the `num_tiles == 1` envelope -- see module docs.
/// `has_nonzero_operating_point_idc` + `obu_extension` mirror the sequence
/// header's own values; `is_layer_specific_obu = true` for a frame OBU
/// (matches real aomenc's `av1_write_obu_tg_tile_headers`, which always
/// passes `true` there).
pub fn assemble_obu_frame_single_tile(
    frame_header: &FrameHeaderObu,
    tiles_log2: i32,
    tile_bytes: &[u8],
    has_nonzero_operating_point_idc: bool,
    obu_extension: u8,
) -> Vec<u8> {
    let payload = assemble_frame_obu_payload_single_tile(frame_header, tiles_log2, tile_bytes);
    let mut out = write_obu_header(
        OBU_FRAME,
        has_nonzero_operating_point_idc,
        true,
        obu_extension,
    );
    let size_bytes =
        uleb_encode(payload.len() as u64, 8).expect("OBU_FRAME payload size fits a leb128 varint");
    out.extend_from_slice(&size_bytes);
    out.extend_from_slice(&payload);
    out
}

/// Assemble the `OBU_FRAME` PAYLOAD for a MULTI-tile frame (`num_tiles > 1`) in
/// the default `num_tg == 1` form real aomenc produces, from a PRE-SERIALIZED,
/// byte-aligned frame header plus the per-tile raw entropy payloads.
///
/// Real aomenc codes a multi-tile keyframe as a SINGLE `OBU_FRAME` with all
/// tiles in one tile group and `tile_start_and_end_present_flag == 0` --
/// verified against `av1/encoder/bitstream.c`: `obu_type = (num_tg == 1) ?
/// OBU_FRAME : OBU_TILE_GROUP` (`:3815`) and the present flag passed to
/// `write_tile_group_header` is `cpi->num_tg > 1` (`:3830`), both false for the
/// default `num_tg == 1`. Layout, per the AV1 spec's `frame_obu(sz)`:
///
/// ```text
/// frame_header_obu(); byte_alignment()   // == `frame_header_bytes` (caller-supplied, byte-aligned)
/// tile_group_obu(sz) {
///     tile_start_and_end_present_flag    // one 0 bit (num_tg == 1)
///     byte_alignment()                   // -> the single trailing byte is 0x00
///     for each tile in raster (tile-row-major) order:
///         if not last: tile_size_minus_1 // tile_size_bytes-byte LE
///         tile_data                      // raw entropy bytes
/// }
/// ```
///
/// Because the tile group starts byte-aligned (the frame header already ended on
/// a byte boundary), `tile_start_and_end_present_flag == 0` followed by
/// `byte_alignment()` is exactly one `0x00` byte -- appended here directly.
///
/// `frame_header_bytes` is the complete, byte-aligned `frame_header_obu()`
/// output. It is taken as raw bytes (not a `FrameHeaderObu`) DELIBERATELY: the
/// aom-entropy `write_tile_info` multi-tile branch currently hardcodes
/// `context_update_tile_id`/`tile_size_bytes_minus_1`
/// (`crates/aom-entropy/src/header.rs`), so re-serializing a multi-tile
/// `FrameHeaderObu` does not yet round-trip -- the caller supplies the header
/// bytes (e.g. bootstrapped from the parsed real header) until that writer takes
/// the real values.
///
/// Every tile EXCEPT the last is prefixed by a `tile_size_bytes`-byte
/// little-endian `tile_size_minus_1` (= payload len - 1; the encoder's
/// `AV1_MIN_TILE_SIZE_BYTES == 1` offset); the last tile is the remainder with
/// no prefix. Inverse of the decoder's `split_tiles`
/// (`av1/decoder/decodeframe.c` `get_tile_buffer`). `tile_size_bytes` is the
/// header's decoded field (1..=4). `tiles` holds each tile's raw entropy bytes
/// (from a per-tile `pack_tile` + `OdEcEnc::done()`), raster order.
pub fn assemble_multitile_frame_obu_payload(
    frame_header_bytes: &[u8],
    tile_size_bytes: i32,
    tiles: &[Vec<u8>],
) -> Vec<u8> {
    assert!(tiles.len() > 1, "multi-tile assembler requires > 1 tile");
    assert!(
        (1..=4).contains(&tile_size_bytes),
        "tile_size_bytes must be 1..=4 (got {tile_size_bytes})"
    );
    let mut payload = frame_header_bytes.to_vec();
    // tile_group_obu() header for num_tg == 1: tile_start_and_end_present_flag
    // (one 0 bit) + byte_alignment() -> one 0x00 byte (header already aligned).
    payload.push(0);
    let n = tiles.len();
    let tsb = tile_size_bytes as usize;
    for (i, tb) in tiles.iter().enumerate() {
        if i + 1 < n {
            // Non-last tile: tile_size_bytes-byte little-endian (payload_len - 1).
            let v = (tb.len() as u64)
                .checked_sub(1)
                .expect("a coded tile payload is never empty");
            for b in 0..tsb {
                payload.push(((v >> (8 * b)) & 0xff) as u8);
            }
        }
        payload.extend_from_slice(tb);
    }
    payload
}

/// `choose_size_bytes(size, spare_msbs=0)` (`av1/encoder/bitstream.c:3244`): the
/// minimum number of bytes (1..=4) needed to hold `size`. NOTE: libaom sizes the
/// `tile_size_bytes` field from the RAW maximum tile payload length, NOT
/// `max - 1`, even though each written prefix value is `len - 1` — so a max tile
/// of exactly 256 bytes yields `tile_size_bytes == 2` although every prefix
/// value (<= 255) fits in one byte. Reproduced exactly for byte-identity.
fn choose_size_bytes(size: u32) -> i32 {
    if size >> 24 != 0 {
        4
    } else if size >> 16 != 0 {
        3
    } else if size >> 8 != 0 {
        2
    } else {
        1
    }
}

/// Assemble the `OBU_FRAME` PAYLOAD for a MULTI-tile frame (`num_tiles > 1`,
/// `num_tg == 1`) by INDEPENDENTLY serializing the frame header from
/// `frame_header` (not bootstrapped from a reference header's bytes) and filling
/// in the real `context_update_tile_id` + `tile_size_bytes` the way libaom's
/// encoder does after tile packing.
///
/// `write_frame_header_obu` writes `write_tile_info`'s placeholders (`0`,
/// `tile_size_bytes_minus_1 == 3`) and records the saved bit position
/// (`WriteBitBuffer::saved_position`, the port's `*saved_wb = *wb`). This
/// function then, matching `av1/encoder/bitstream.c`'s `write_tile_obu_size`
/// (`:4053` + `:4068`):
///   * derives `tile_size_bytes = choose_size_bytes(max_tile_size)` over the RAW
///     tile payload lengths (the last tile's length participates too, even
///     though it carries no prefix), and `largest_tile_id` = the raster index of
///     the first tile of that maximum length;
///   * overwrites `context_update_tile_id` with `largest_tile_id`
///     (`log2_cols + log2_rows` bits) and `tile_size_bytes_minus_1` with
///     `tile_size_bytes - 1` (2 bits).
///
/// Each non-last tile is then prefixed by its `len - 1`
/// (`- AV1_MIN_TILE_SIZE_BYTES`) as a `tile_size_bytes`-byte little-endian
/// value; the last tile carries no prefix. `tiles` are the raw per-tile entropy
/// payloads in raster (row-major) order.
pub fn assemble_multitile_frame_obu_payload_derived(
    frame_header: &FrameHeaderObu,
    tiles: &[Vec<u8>],
) -> Vec<u8> {
    let n = tiles.len();
    assert!(n > 1, "multi-tile assembler requires > 1 tile");
    let ti = &frame_header.tile_info;
    assert_eq!(
        ti.rows * ti.cols,
        n,
        "tile count {n} != tile_info.rows*cols {}",
        ti.rows * ti.cols
    );

    // max_tile_size over the RAW payload lengths of ALL tiles (incl. the last),
    // and largest_tile_id = raster index of the first tile of that max length
    // (strict `>`, matching bitstream.c:4024).
    let mut max_tile_size = 0u32;
    let mut largest_tile_id = 0i32;
    for (i, tb) in tiles.iter().enumerate() {
        let sz = tb.len() as u32;
        if sz > max_tile_size {
            max_tile_size = sz;
            largest_tile_id = i as i32;
        }
    }
    let tile_size_bytes = choose_size_bytes(max_tile_size);

    // Serialize the header; write_frame_header_obu records the saved position
    // right before the context_update_tile_id + tile_size_bytes_minus_1 fields.
    let mut wb = WriteBitBuffer::new();
    write_frame_header_obu(&mut wb, frame_header);
    let saved = wb
        .saved_position()
        .expect("multi-tile frame header must record the tile-info saved position");
    let ctx_bits = (ti.log2_cols + ti.log2_rows) as u32;
    // C write_tile_obu_size: overwrite context_update_tile_id first (its
    // aom_wb_overwrite_literal auto-advances saved_wb by ctx_bits), then
    // tile_size_bytes_minus_1. Single buffer here -> explicit positions.
    wb.overwrite_literal(saved, largest_tile_id, ctx_bits);
    wb.overwrite_literal(saved + ctx_bits as usize, tile_size_bytes - 1, 2);

    wb.byte_align_zeros(); // frame_obu()'s byte_alignment()
    let mut payload = wb.bytes().to_vec();
    // tile_group_obu() for num_tg == 1: tile_start_and_end_present_flag (one 0
    // bit) + byte_alignment() -> one 0x00 byte (header already aligned).
    payload.push(0);
    let tsb = tile_size_bytes as usize;
    for (i, tb) in tiles.iter().enumerate() {
        if i + 1 < n {
            let v = (tb.len() as u64)
                .checked_sub(1)
                .expect("a coded tile payload is never empty");
            for b in 0..tsb {
                payload.push(((v >> (8 * b)) & 0xff) as u8);
            }
        }
        payload.extend_from_slice(tb);
    }
    payload
}
