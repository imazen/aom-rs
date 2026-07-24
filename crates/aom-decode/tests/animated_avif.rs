//! Animated-AVIF inter-decode differential gate (INTER_DECODE_ENVELOPE.md).
//!
//! Fixtures: `tests/data/animated/<vector>.<track>.obu` — the concatenated
//! per-track AV1 temporal-unit streams extracted from libavif's
//! `colors-animated-*.avif` test vectors (via `tools/avif-extract`; the
//! `-audio`/`-depth` color tracks are byte-identical to their siblings and
//! kept for per-vector coverage). Goldens: `<vector>.<track>.md5` — ONE md5
//! per SHOWN frame, computed over aomdec 3.14.1 `--rawvideo` output split at
//! frame boundaries (planar Y[,U,V] at the coded bit depth, little-endian
//! 16-bit above 8, cropped dims, luma-only for monochrome).
//!
//! The harness decodes every track with [`decode_frames`] and reports
//! per-frame pass/fail; `EXPECTED_EXACT` is the ratchet — every listed track
//! MUST be fully byte-exact (the hard gate), and any track beyond the list
//! that becomes byte-exact should be promoted into it in the same commit as
//! the chunk that fixed it.

use aom_decode::frame::{FrameDecode, decode_frames};

mod common;
use common::md5::Md5;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/animated");

const TRACKS: &[&str] = &[
    "colors-animated-8bpc.color",
    "colors-animated-8bpc-audio.color",
    "colors-animated-8bpc-alpha-exif-xmp.color",
    "colors-animated-8bpc-alpha-exif-xmp.alpha",
    "colors-animated-8bpc-depth-exif-xmp.color",
    "colors-animated-8bpc-depth-exif-xmp.alpha",
    "colors-animated-12bpc-keyframes-0-2-3.color",
    "colors-animated-12bpc-keyframes-0-2-3.alpha",
];

/// The ratchet: tracks that MUST decode fully byte-exact (all shown frames)
/// on the current envelope. Grow this list with each landed chunk; never
/// shrink it.
const EXPECTED_EXACT: &[&str] = &[
    "colors-animated-8bpc.color",
    "colors-animated-8bpc-audio.color",
    "colors-animated-8bpc-alpha-exif-xmp.color",
    "colors-animated-8bpc-alpha-exif-xmp.alpha",
    "colors-animated-8bpc-depth-exif-xmp.color",
    "colors-animated-8bpc-depth-exif-xmp.alpha",
    "colors-animated-12bpc-keyframes-0-2-3.color",
    "colors-animated-12bpc-keyframes-0-2-3.alpha",
];

/// The per-frame ratchet for tracks not yet fully exact: `(track, exact
/// frame indices)` that MUST stay byte-exact. Chunk-1 state (2026-07-23):
/// - 8bpc color: KEY + hidden-ALTREF chain + the shown primary_ref=NONE
///   inter frame + show_existing of a hidden frame (frames 0-2). Frames 3-4
///   (primary_ref != NONE + live temporal-MV field over stored zero-MV
///   grids) await chunk 3.
/// - 8bpc alpha (mono, all intra-in-inter, primary_ref chain): frames 0-2;
///   3-4 diverge deeper in the chain (under investigation, chunk 2/3).
/// (Empty since the temporal-MV-field chunk: every track is fully exact and
/// lives in `EXPECTED_EXACT`. Use this for partial-progress ratcheting when
/// new, harder vectors join the corpus.)
const EXPECTED_FRAMES: &[(&str, &[usize])] = &[];

/// Raw-plane md5 in the golden layout: Y then (unless mono) U, V; cropped
/// dims; 1 byte/sample at bd8, 2 bytes LE above.
fn frame_md5(fd: &FrameDecode) -> String {
    let mut m = Md5::new();
    let hi = fd.bit_depth > 8;
    let push = |m: &mut Md5, plane: &[u16], pw: usize, ph: usize| {
        assert_eq!(plane.len(), pw * ph, "plane size mismatch");
        let mut bytes = Vec::with_capacity(pw * ph * if hi { 2 } else { 1 });
        for &s in plane {
            if hi {
                bytes.extend_from_slice(&s.to_le_bytes());
            } else {
                bytes.push(s as u8);
            }
        }
        m.update(&bytes);
    };
    push(&mut m, &fd.y, fd.width, fd.height);
    if !fd.monochrome {
        push(&mut m, &fd.u, fd.width_uv, fd.height_uv);
        push(&mut m, &fd.v, fd.width_uv, fd.height_uv);
    }
    m.finish()
}

struct TrackResult {
    track: &'static str,
    /// `Err(decode error)` or per-frame `(ok, got, want)`.
    outcome: Result<Vec<(bool, String, String)>, String>,
}

impl TrackResult {
    fn fully_exact(&self) -> bool {
        matches!(&self.outcome, Ok(v) if !v.is_empty() && v.iter().all(|(ok, _, _)| *ok))
    }
}

fn run_track(track: &'static str) -> TrackResult {
    let stream = std::fs::read(format!("{DATA}/{track}.obu"))
        .unwrap_or_else(|e| panic!("fixture {track}.obu missing: {e}"));
    let goldens: Vec<String> = std::fs::read_to_string(format!("{DATA}/{track}.md5"))
        .unwrap_or_else(|e| panic!("golden {track}.md5 missing: {e}"))
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    let outcome = match decode_frames(&stream) {
        Err(e) => Err(format!("{e:?}")),
        Ok(frames) => {
            let mut rows = Vec::new();
            for (i, g) in goldens.iter().enumerate() {
                let got = frames
                    .get(i)
                    .map(frame_md5)
                    .unwrap_or_else(|| "<missing frame>".into());
                rows.push((got == *g, got, g.clone()));
            }
            if frames.len() != goldens.len() {
                rows.push((
                    false,
                    format!("<{} frames decoded>", frames.len()),
                    format!("<{} shown frames expected>", goldens.len()),
                ));
            }
            Ok(rows)
        }
    };
    TrackResult { track, outcome }
}

/// The differential gate: full per-track / per-frame status report, hard
/// assertion on the `EXPECTED_EXACT` ratchet.
#[test]
fn animated_tracks_byte_exact_ratchet() {
    let mut failures = Vec::new();
    for track in TRACKS {
        let r = run_track(track);
        match &r.outcome {
            Err(e) => eprintln!("[{track}] DECODE ERROR: {e}"),
            Ok(rows) => {
                for (i, (ok, got, want)) in rows.iter().enumerate() {
                    eprintln!(
                        "[{track}] frame {i}: {} (got {got}, want {want})",
                        if *ok { "OK" } else { "DIFF" }
                    );
                }
            }
        }
        let required = EXPECTED_EXACT.contains(track);
        if required && !r.fully_exact() {
            failures.push(format!("{track} (full-track ratchet)"));
        }
        if !required && r.fully_exact() {
            eprintln!("[{track}] NOTE: fully byte-exact but not in EXPECTED_EXACT — promote it");
        }
        if let Some((_, need)) = EXPECTED_FRAMES.iter().find(|(t, _)| t == track) {
            match &r.outcome {
                Err(e) => failures.push(format!("{track} (frame ratchet; decode error {e})")),
                Ok(rows) => {
                    for &fi in *need {
                        if !rows.get(fi).is_some_and(|(ok, _, _)| *ok) {
                            failures.push(format!("{track} frame {fi} (frame ratchet)"));
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "ratcheted tracks regressed: {failures:?}"
    );
}
