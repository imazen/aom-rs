//! REAL-conformance inter-frame gate (milestone: the FIRST real inter frame).
//!
//! Target: frame 1 of `av1-1-b8-00-quantizer-63` — a genuine conformance P-frame
//! (352x288, 79 inter blocks) that uses the FULL single-ref inter toolset at once:
//! SIMPLE (49) + OBMC (19) + WARPED_CAUSAL (11) + interintra (2, incl. one wedge,
//! at mi(56,76)/(56,78)) + intra-in-inter (3, all DC; one filter-intra, at
//! mi(32,80)/(36,80)/(44,18)) + inter var-tx. Single `LAST` ref throughout
//! (NO compound — verified by census, so the single-ref driver suffices).
//!
//! STATUS: **MILESTONE MET** — frame 0 (KEY) and frame 1 (inter) both decode
//! BYTE-IDENTICAL to the golden MD5s. This file was a self-promoting probe that
//! pinned on whatever feature the decode reached next; it is now a hard gate.
//!
//! The last landing closed FOUR gaps, not the two originally scoped. The extra
//! two only became reachable once the first was fixed — each was masking the next:
//!
//!   1. **intra-in-inter** (3 DC blocks, at mi(32,80)/(36,80)/(44,18)): the
//!      `is_inter == 0` arm of read_inter_frame_mode_info (decodemv.c:1550). It is
//!      the EXISTING byte-exact KEY intra decode with exactly TWO parse
//!      differences: (a) the Y mode reads `y_mode_cdf[size_group_lookup[bsize]]`
//!      (:1077) instead of the neighbour-context `kf_y_cdf` (:815), and (b) no
//!      intrabc read (`av1_allow_intrabc` is frame_is_intra_only-gated). Landed by
//!      SHARING the KEY mode-info tail and the whole post-mode-info body rather
//!      than transcribing a second copy — C shares them too (one
//!      frame-type-independent intra recon visitor pair, decodeframe.c:2756/:2761).
//!   2. **`get_tx_size_context`'s inter-neighbour override**: C replaces the
//!      txfm-context term with the neighbour's BLOCK size whenever that neighbour
//!      `is_inter_block` (`use_intrabc || ref_frame[0] > INTRA_FRAME`). The port
//!      tested `use_intrabc` alone — correct on a KEY frame (nothing else is ever
//!      inter there), wrong for an intra block inside an inter frame.
//!   3. **the skip-block entropy-context reset on the INTER path**: C runs
//!      `av1_reset_entropy_context` for EVERY skip block (decodeframe.c:1219); the
//!      port did it only for intra ones. A skipped inter block therefore left stale
//!      culs in its footprint, and the next block that read across them picked a
//!      different `txb_skip_cdf` row — the same symbol VALUES on different
//!      probabilities, so the decode drifted silently and desynced later. This one
//!      was worth 41 of the frame's 79 blocks on its own.
//!   4. **interintra prediction wiring** (2 blocks at mi(56,76)/(56,78), one
//!      wedge): the read (`read_interintra_info`) + per-plane build-intra +
//!      `combine_interintra` blend, replacing the `assert interintra == 0` guard.
//!      An interintra block also reads NO motion-mode symbol — C gates that read on
//!      `ref_frame[1] != INTRA_FRAME` (decodemv.c:1421), so interintra and
//!      OBMC/WARPED_CAUSAL are mutually exclusive. Kernels were already
//!      differential-locked (`aom-dsp/src/inter/interintra.rs`,
//!      `aom-dsp/tests/interintra_diff.rs`).
//!
//! Gaps 2 and 3 were found by diffing the port's block walk against the REAL
//! libaom decoder's own per-block dump and per-symbol accounting
//! (`CONFIG_INSPECTION=1 CONFIG_ACCOUNTING=1`), which is the tool to reach for
//! first on any future inter desync: it gives C's exact block list AND its exact
//! symbol sequence, so a "same value, wrong CDF row" drift is visible immediately
//! instead of being inferred. All 79 of C's frame-1 blocks now match the port's
//! walk exactly. All prior inter ratchet gates (16x16/18/34/66, 64x66) stay
//! byte-exact.

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
    assert!(
        data.len() >= 32 && &data[0..4] == b"DKIF",
        "not an IVF file"
    );
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

/// HARD frame-1 gate (PROMOTED from the self-promoting pin — MILESTONE MET):
/// frame 1 must decode to the golden MD5, byte for byte.
///
/// This was a pin that caught the panic and merely reported how far the decode
/// advanced. Both remaining features have landed, so it is now an ordinary
/// assertion: any regression — a desync, a panic, or a single wrong pixel — fails
/// the test instead of being printed and swallowed.
#[test]
fn real_frame_q63_frame1_inter() {
    let tus = load_tus();
    let mut stream = tus[0].clone();
    stream.extend_from_slice(&tus[1]);
    let frames = decode_frames(&stream).expect("q63 frame 0 + frame 1 decode");
    assert_eq!(frames.len(), 2, "two shown frames decoded");
    assert_eq!(image_md5(&frames[0]), GOLDEN_F0, "q63 frame 0 golden");
    assert_eq!(
        image_md5(&frames[1]),
        GOLDEN_F1,
        "q63 frame 1 (inter) golden"
    );
    eprintln!("real_frame q63: frame 1 (inter) byte-identical to golden");
}
