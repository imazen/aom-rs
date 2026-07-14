//! Differential harness for the OBU header packing vs C libaom. Exhaustive over
//! obu_type 0..8 x the two flag inputs x extension bytes; also spot-checks the
//! spec byte layout directly (obu_type<<3 | ext<<2 | has_size<<1).

use aom_entropy::obu::write_obu_header;
use aom_sys_ref as c;

#[test]
fn obu_header_matches_c() {
    for obu_type in 0..8u32 {
        for &has_nz in &[false, true] {
            for &is_layer in &[false, true] {
                for ext in [0u8, 1, 0x55, 0xff] {
                    let got = write_obu_header(obu_type, has_nz, is_layer, ext);
                    let want = c::ref_write_obu_header(obu_type, has_nz, is_layer, ext);
                    assert_eq!(
                        got, want,
                        "obu ty={obu_type} nz={has_nz} layer={is_layer} ext={ext}"
                    );
                }
            }
        }
    }
}

#[test]
fn obu_header_spec_layout() {
    // Direct spec anchor (independent of the C transcription): no extension.
    for obu_type in 0..8u32 {
        let bytes = write_obu_header(obu_type, false, false, 0);
        assert_eq!(bytes.len(), 1);
        // has_size_field = 1 (bit 1); ext flag 0; obu_type in bits 6..3.
        assert_eq!(bytes[0], ((obu_type << 3) | 0b10) as u8, "ty={obu_type}");
    }
    // With extension (both flags true): 2 bytes, ext bit set.
    let bytes = write_obu_header(6, true, true, 0xab);
    assert_eq!(bytes, vec![((6u32 << 3) | 0b110) as u8, 0xab]);
}
