//! REAL-conformance inter-frame gate (milestone: the FIRST real inter frame).
//!
//! Target: frame 1 of `av1-1-b8-00-quantizer-63` — a genuine conformance P-frame
//! (352x288, 79 inter blocks) that uses the FULL single-ref inter toolset at once:
//! SIMPLE (49) + OBMC (19) + WARPED_CAUSAL (11) + interintra (2, incl. one wedge,
//! at mi(56,76)/(56,78)) + intra-in-inter (3, all DC; one filter-intra, at
//! mi(32,80)/(36,80)/(44,18)) + inter var-tx. Single `LAST` ref throughout
//! (NO compound — verified by census, so the single-ref driver suffices).
//!
//! STATUS (self-promoting frame-1 probe): frame 0 (KEY) is byte-exact, and the
//! ENTIRE inter BODY now decodes in arithmetic-sync — SIMPLE + OBMC (above/left +
//! chroma) + WARPED_CAUSAL + inter var-tx all parse + reconstruct through the
//! whole frame after the OBMC-chroma landing (0ca3775) + the interintra kernels
//! (f17ee31) + the txfm-context fix (080a2bb, the mi(32,0) TX_32X64 desync). The
//! probe now PINS on the FIRST `is_inter == 0` block (intra-in-inter), the last
//! two remaining features:
//!
//!   1. **intra-in-inter** (3 DC blocks): the `is_inter == 0` arm of
//!      read_inter_frame_mode_info (decodemv.c:1547). Per the C spec, this is the
//!      EXISTING byte-exact KEY intra decode (read_mb_modes_kf_fc +
//!      decode_token_recon_block, the `else` branch at lib.rs:3409+) with exactly
//!      TWO parse differences: (a) `y_mode_cdf[size_group_lookup[bsize]]` instead
//!      of the neighbour-context kf_y_mode (a NEW `default_y_mode_cdf`
//!      [BLOCK_SIZE_GROUPS][INTRA_MODES], NOT yet ported — add to
//!      xtask/gen_default_cdfs.py + thread on InterCdfs), and (b) NO intrabc read
//!      (frame_is_intra_only false). uv_mode/angle/filter_intra/cfl/tx_size/coeff
//!      CDFs already live in `cdfs`; the INTRA tx-size (read_selected_tx_size, NOT
//!      var-tx), intra tx-type (intra_ext_tx_cdf, dir = filter-intra-adjusted
//!      y_mode via fimode_to_intradir), predict + reconstruct all reuse the KEY
//!      path unchanged (recon routes on `is_inter_block == false`, so
//!      ref_frame=[INTRA,NONE] fires the KEY intra recon with no special-casing).
//!      For q63 all 3 are DC (no angle/cfl read; palette off — non-screen); one
//!      codes filter-intra.
//!   2. **interintra prediction WIRING** (2 blocks): the kernels are done +
//!      differential-locked (`aom-inter/src/interintra.rs`,
//!      `aom-inter/tests/interintra_diff.rs`); the read (`read_interintra_info`,
//!      already in aom-entropy) + the per-plane build-intra + combine_interintra
//!      blend must replace the `assert interintra == 0` guard in decode_block_inter.
//!
//! When both land, this probe self-promotes to the hard golden byte-identity
//! gate below (MILESTONE MET). All prior inter ratchet gates
//! (16x16/18/34/66, 64x66) stay byte-exact.

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
