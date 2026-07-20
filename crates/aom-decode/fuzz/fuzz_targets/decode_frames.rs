#![no_main]
//! Fuzz the multi-frame OBU decode entry point.
//!
//! `decode_frames` parses a stream of raw AV1 OBU temporal units (a KEY frame
//! optionally followed by inter frames — the exact bytes an AVIF `mdat` /
//! animated-AVIF track carries, and what zenavif hands the decoder). On ANY
//! malformed input it must return `Err`, never panic (unwrap / expect /
//! out-of-bounds slice index / `assert!` / arithmetic overflow) and never
//! allocate without bound. This target holds that contract.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = aom_decode::frame::decode_frames(data);
});
