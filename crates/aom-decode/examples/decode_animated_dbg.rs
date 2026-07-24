//! Debug driver for the animated-AVIF differential work: decode a raw OBU
//! stream with [`decode_frames`]; set `AOM_DBG_BLOCKS=1` to trace the block
//! walk (positions, modes, `tell_frac`) for comparison against the
//! instrumented libaom accounting dump (`/root/aom-inspect` + `-a`).
use aom_decode::frame::decode_frames;
fn main() {
    let path = std::env::args().nth(1).expect("usage: decode_animated_dbg <stream.obu>");
    let data = std::fs::read(path).expect("read stream");
    let frames = decode_frames(&data).expect("decode");
    eprintln!("decoded {} shown frames", frames.len());
}
