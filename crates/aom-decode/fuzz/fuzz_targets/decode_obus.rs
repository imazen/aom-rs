#![no_main]
//! Fuzz the single-KEY-frame OBU decode entry point.
//!
//! `decode_frame_obus` decodes one AV1 temporal unit (temporal delimiter +
//! sequence header + frame) to cropped planes — the exact function the Gate-1
//! conformance harness drives and the natural entry for a still AVIF image.
//! On ANY malformed input it must return `Err`, never panic and never allocate
//! without bound.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = aom_decode::frame::decode_frame_obus(data);
});
