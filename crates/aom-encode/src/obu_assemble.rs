//! Assemble a real AV1 `OBU_FRAME` (frame header + tile group, the
//! `cpi->num_tg == 1` combined form real aomenc's default config always
//! produces) from the already bit-exact pieces:
//! [`aom_entropy::header::write_frame_header_obu`] +
//! [`aom_entropy::header::write_tile_group_header`] (both aom-entropy,
//! decoder-owned but CALLED here, not modified) +
//! [`crate::pack::pack_tile`]'s raw entropy-coded tile bytes +
//! [`aom_entropy::obu::write_obu_header`] / [`aom_entropy::leb128::uleb_encode`]
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

use aom_entropy::header::{write_frame_header_obu, write_tile_group_header, FrameHeaderObu};
use aom_entropy::leb128::uleb_encode;
use aom_entropy::obu::write_obu_header;
use aom_entropy::wb::WriteBitBuffer;

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
    let mut out = write_obu_header(OBU_FRAME, has_nonzero_operating_point_idc, true, obu_extension);
    let size_bytes =
        uleb_encode(payload.len() as u64, 8).expect("OBU_FRAME payload size fits a leb128 varint");
    out.extend_from_slice(&size_bytes);
    out.extend_from_slice(&payload);
    out
}
