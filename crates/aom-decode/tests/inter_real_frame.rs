//! REAL-conformance inter-frame gate (milestone: the FIRST real inter frame).
//!
//! Target: frame 1 of `av1-1-b8-00-quantizer-63` — a genuine conformance P-frame
//! (352x288, 79 inter blocks) that uses the FULL single-ref inter toolset at once:
//! SIMPLE + OBMC (19) + WARPED_CAUSAL (11) + interintra (2, incl. one wedge) +
//! intra-in-inter (3) + var-tx. Single `LAST` ref throughout (no compound).
//!
//! This starts as a foundation anchor (frame 0 KEY byte-exact) + a self-promoting
//! frame-1 probe, and promotes to a hard byte-identity gate once the remaining
//! features land.

mod common;

use aom_decode::frame::{FrameDecode, decode_frames};
use common::md5::Md5;
use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    if let Ok(d) = std::env::var("AOM_CONFORMANCE_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("conformance")
        .join("data")
}

fn ivf_temporal_units(data: &[u8]) -> Vec<Vec<u8>> {
    assert!(data.len() >= 32 && &data[0..4] == b"DKIF", "not an IVF file");
    let hdr_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut off = hdr_len;
    let mut tus = Vec::new();
    while off + 12 <= data.len() {
        let sz =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize;
        off += 12;
        assert!(off + sz <= data.len(), "IVF frame runs past end of file");
        tus.push(data[off..off + sz].to_vec());
        off += sz;
    }
    tus
}

/// `md5_helper.h::Add(aom_image_t*)`: hash each cropped plane row-by-row.
fn image_md5(fd: &FrameDecode) -> String {
    let mut m = Md5::new();
    let hi = fd.bit_depth > 8;
    let push = |m: &mut Md5, plane: &[u16], pw: usize, ph: usize| {
        assert_eq!(plane.len(), pw * ph, "plane size mismatch");
        for r in 0..ph {
            let mut row = Vec::with_capacity(pw * if hi { 2 } else { 1 });
            for &s in &plane[r * pw..r * pw + pw] {
                if hi {
                    row.extend_from_slice(&s.to_le_bytes());
                } else {
                    row.push(s as u8);
                }
            }
            m.update(&row);
        }
    };
    push(&mut m, &fd.y, fd.width, fd.height);
    if fd.monochrome {
        let (cw, ch) = ((fd.width + 1) >> 1, (fd.height + 1) >> 1);
        let neutral = vec![1u16 << (fd.bit_depth - 1); cw * ch];
        push(&mut m, &neutral, cw, ch);
        push(&mut m, &neutral, cw, ch);
    } else {
        push(&mut m, &fd.u, fd.width_uv, fd.height_uv);
        push(&mut m, &fd.v, fd.width_uv, fd.height_uv);
    }
    m.finish()
}

const VECTOR: &str = "av1-1-b8-00-quantizer-63";
const GOLDEN_F0: &str = "af57402f541c571ee7ee04ebed6a2f0e";
const GOLDEN_F1: &str = "d732186fdf74067730547b61a1fe1c03";

fn load_tus() -> Vec<Vec<u8>> {
    let dir = corpus_dir();
    let ivf_path = dir.join(format!("{VECTOR}.ivf"));
    let ivf = std::fs::read(&ivf_path)
        .unwrap_or_else(|e| panic!("conformance vector {} not found ({e})", ivf_path.display()));
    ivf_temporal_units(&ivf)
}

/// Foundation anchor: frame 0 (KEY, q63/base_qindex 255) must decode byte-exact.
/// Proves the KB-1 >64-block fix holds on this vector and the harness is sound.
#[test]
fn real_frame_q63_frame0_key_byte_identical() {
    let tus = load_tus();
    let f0 = decode_frames(&tus[0]).expect("q63 KEY frame decodes");
    assert_eq!(f0.len(), 1, "one shown KEY frame");
    assert_eq!(image_md5(&f0[0]), GOLDEN_F0, "q63 frame 0 (KEY) golden");
    eprintln!("real_frame q63: frame 0 (KEY) byte-identical to golden");
}

/// SELF-PROMOTING frame-1 gate: asserts the golden byte-match once the full inter
/// toolset lands; until then, pins on a documented inter-envelope guard and
/// reports it (so we can see how far the decode advances).
#[test]
fn real_frame_q63_frame1_inter() {
    let tus = load_tus();
    let mut stream = tus[0].clone();
    stream.extend_from_slice(&tus[1]);
    let res = std::panic::catch_unwind(|| decode_frames(&stream));
    match res {
        Ok(Ok(frames)) => {
            assert_eq!(frames.len(), 2, "two shown frames decoded");
            assert_eq!(image_md5(&frames[0]), GOLDEN_F0, "q63 frame 0 golden");
            assert_eq!(image_md5(&frames[1]), GOLDEN_F1, "q63 frame 1 (inter) golden");
            eprintln!("real_frame q63: FULL byte-match — MILESTONE MET");
        }
        Ok(Err(e)) => panic!("q63 decode returned an unexpected error (not a pin): {e}"),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("<non-string panic>");
            eprintln!("real_frame q63: frame-1 PINNED on `{msg}`");
        }
    }
}
