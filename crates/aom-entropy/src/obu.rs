//! OBU header packing (libaom `av1/encoder/bitstream.c` `av1_write_obu_header`):
//! the 1-2 byte header that prefixes every Open Bitstream Unit. Byte-identical to
//! C libaom. (The level-stats side effect in the C function does not affect the
//! bytes and is out of scope here.)

/// `av1_write_obu_header` (byte output): `obu_type` in bits 6..3, the extension
/// flag in bit 2 (= `has_nonzero_operating_point_idc && is_layer_specific_obu`),
/// `obu_has_size_field` (always 1) in bit 1; then the extension byte if flagged.
pub fn write_obu_header(
    obu_type: u32,
    has_nonzero_operating_point_idc: bool,
    is_layer_specific_obu: bool,
    obu_extension: u8,
) -> Vec<u8> {
    let obu_extension_flag = has_nonzero_operating_point_idc && is_layer_specific_obu;
    let obu_has_size_field = 1u32;
    let mut out = Vec::with_capacity(2);
    out.push(
        ((obu_type << 3) | ((obu_extension_flag as u32) << 2) | (obu_has_size_field << 1)) as u8,
    );
    if obu_extension_flag {
        out.push(obu_extension);
    }
    out
}

/// Parsed OBU header fields (inverse of [`write_obu_header`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObuHeader {
    /// OBU type (bits 6..3 of byte 0).
    pub obu_type: u32,
    /// Extension flag (bit 2): whether the 1-byte extension header follows.
    pub obu_extension_flag: bool,
    /// `obu_has_size_field` (bit 1).
    pub obu_has_size_field: bool,
    /// The extension byte (temporal/spatial id), 0 when no extension.
    pub obu_extension: u8,
    /// Header length in bytes (1, or 2 with the extension).
    pub header_len: usize,
}

/// `aom_read_obu_header` — inverse of [`write_obu_header`]: parse the 1–2 byte OBU
/// header. Returns `None` if the buffer is too short or the forbidden bit is set.
pub fn read_obu_header(data: &[u8]) -> Option<ObuHeader> {
    let byte0 = *data.first()?;
    if byte0 & 0x80 != 0 {
        return None; // obu_forbidden_bit must be 0
    }
    let obu_type = ((byte0 >> 3) & 0xF) as u32;
    let obu_extension_flag = (byte0 >> 2) & 1 != 0;
    let obu_has_size_field = (byte0 >> 1) & 1 != 0;
    let (obu_extension, header_len) = if obu_extension_flag {
        (*data.get(1)?, 2)
    } else {
        (0, 1)
    };
    Some(ObuHeader {
        obu_type,
        obu_extension_flag,
        obu_has_size_field,
        obu_extension,
        header_len,
    })
}
