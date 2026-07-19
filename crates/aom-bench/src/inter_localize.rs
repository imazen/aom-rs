//! INTER-ENCODE Chunk 0 — the decode-both localizer.
//!
//! The verification tool the inter-encode skeleton (chunk 2) uses to PIN a
//! bitstream divergence. Given two AV1 streams (aomenc's 2-frame `[KEY, P]` and,
//! from chunk 2 on, the port's re-encode of the same source), it decodes BOTH
//! with the port's byte-exact decoder ([`aom_decode::frame::decode_frames`], the
//! multi-frame ref-managed path) and reports the FIRST divergent
//! `(frame, tile, superblock, sample)` — the localization the encode skeleton
//! walks down to a partition / mode / MV / coeff root.
//!
//! The comparison core ([`first_frameset_divergence`]) is a pure function over
//! decoded frame-sets, so it also localizes a port-decode vs a C-decode of the
//! SAME stream (the [`FrameView::of_ref_decoded`] constructor) — how chunk 0
//! measures the port decoder's inter envelope against `aomenc`.
//!
//! Tile granularity: the chunk-0 envelope is single-tile, so `tile` is always 0;
//! the superblock coordinate (luma pixels / `sb_px`) is the load-bearing
//! locator. Finer partition/mode/MV localization is the single-frame
//! [`aom_decode::frame::decode_frame_obus_prefilter`] trace applied to the
//! pinned frame — this tool delivers the (frame, SB, sample) the trace starts
//! from.

use aom_decode::frame::FrameDecode;
use aom_sys_ref::RefDecodedFrame;
use std::fmt;

/// Default superblock edge in luma pixels (`--sb-size=64`; the chunk-0 config).
pub const SB64_PX: usize = 64;

/// Which plane a sample divergence is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plane {
    Y,
    U,
    V,
}

impl fmt::Display for Plane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Plane::Y => "Y",
            Plane::U => "U",
            Plane::V => "V",
        })
    }
}

/// The first divergence between two decoded frame-sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Divergence {
    /// The two sides decoded a different number of shown frames.
    FrameCount { a: usize, b: usize },
    /// A frame's decoded geometry (luma + chroma dims) differs. `(w, h, w_uv,
    /// h_uv)` per side.
    Geometry {
        frame: usize,
        a: (usize, usize, usize, usize),
        b: (usize, usize, usize, usize),
    },
    /// One side failed to decode (or panicked) where the other succeeded — the
    /// port decoder rejects a feature the reference codes (chunk-2 signal).
    DecodeError { a_ok: bool, b_ok: bool, msg: String },
    /// A reconstructed sample differs. `row`/`col` are in-plane; `sb_row`/
    /// `sb_col` are the superblock grid coordinates (in LUMA pixels / `sb_px`,
    /// so chroma samples map through subsampling to their covering SB).
    Sample {
        frame: usize,
        plane: Plane,
        row: usize,
        col: usize,
        sb_row: usize,
        sb_col: usize,
        a: u16,
        b: u16,
    },
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Divergence::FrameCount { a, b } => {
                write!(f, "frame-count divergence: side A decoded {a}, side B {b}")
            }
            Divergence::Geometry { frame, a, b } => write!(
                f,
                "frame {frame}: geometry divergence: A (w={} h={} w_uv={} h_uv={}) vs B (w={} h={} w_uv={} h_uv={})",
                a.0, a.1, a.2, a.3, b.0, b.1, b.2, b.3
            ),
            Divergence::DecodeError { a_ok, b_ok, msg } => write!(
                f,
                "decode divergence: A decoded={a_ok}, B decoded={b_ok} ({msg})"
            ),
            Divergence::Sample {
                frame,
                plane,
                row,
                col,
                sb_row,
                sb_col,
                a,
                b,
            } => write!(
                f,
                "frame {frame} tile 0 SB({sb_row},{sb_col}) plane {plane} @({row},{col}): A={a} B={b} (Δ={})",
                *a as i32 - *b as i32
            ),
        }
    }
}

/// A borrowed view of one decoded frame's cropped planes + geometry — the
/// common shape the comparator consumes, so it works over BOTH a port
/// [`FrameDecode`] and a C [`RefDecodedFrame`].
#[derive(Debug, Clone, Copy)]
pub struct FrameView<'a> {
    pub y: &'a [u16],
    pub u: &'a [u16],
    pub v: &'a [u16],
    pub width: usize,
    pub height: usize,
    pub width_uv: usize,
    pub height_uv: usize,
    pub mono: bool,
    pub ss_x: usize,
    pub ss_y: usize,
}

impl<'a> FrameView<'a> {
    /// View of a port-decoded frame.
    pub fn of_decode(f: &'a FrameDecode) -> Self {
        FrameView {
            y: &f.y,
            u: &f.u,
            v: &f.v,
            width: f.width,
            height: f.height,
            width_uv: f.width_uv,
            height_uv: f.height_uv,
            mono: f.monochrome,
            ss_x: f.subsampling_x,
            ss_y: f.subsampling_y,
        }
    }

    /// View of a C-decoded frame (`RefDecodedFrame::info` = `[bd, mono, ss_x,
    /// ss_y, w, h]`; chroma dims derived from the coded subsampling).
    pub fn of_ref_decoded(f: &'a RefDecodedFrame) -> Self {
        let mono = f.info[1] != 0;
        let ss_x = f.info[2] as usize;
        let ss_y = f.info[3] as usize;
        let width = f.info[4] as usize;
        let height = f.info[5] as usize;
        let (width_uv, height_uv) = if mono {
            (0, 0)
        } else {
            ((width + ss_x) >> ss_x, (height + ss_y) >> ss_y)
        };
        FrameView {
            y: &f.y,
            u: &f.u,
            v: &f.v,
            width,
            height,
            width_uv,
            height_uv,
            mono,
            ss_x,
            ss_y,
        }
    }

    fn geom(&self) -> (usize, usize, usize, usize) {
        (self.width, self.height, self.width_uv, self.height_uv)
    }
}

/// Compare two decoded frame-sets sample-by-sample (Y then U then V, raster
/// order) and return the FIRST divergence, or `None` if every shown frame is
/// byte-identical. `sb_px` is the superblock luma edge (64 for the chunk-0
/// config) used only to label the divergence's superblock.
pub fn first_frameset_divergence(
    a: &[FrameView],
    b: &[FrameView],
    sb_px: usize,
) -> Option<Divergence> {
    if a.len() != b.len() {
        return Some(Divergence::FrameCount {
            a: a.len(),
            b: b.len(),
        });
    }
    for (i, (fa, fb)) in a.iter().zip(b.iter()).enumerate() {
        if fa.geom() != fb.geom() {
            return Some(Divergence::Geometry {
                frame: i,
                a: fa.geom(),
                b: fb.geom(),
            });
        }
        // Luma: sample (row, col) sits in SB (row / sb_px, col / sb_px).
        if let Some(p) = fa.y.iter().zip(fb.y.iter()).position(|(x, y)| x != y) {
            let (row, col) = (p / fa.width, p % fa.width);
            return Some(Divergence::Sample {
                frame: i,
                plane: Plane::Y,
                row,
                col,
                sb_row: row / sb_px,
                sb_col: col / sb_px,
                a: fa.y[p],
                b: fb.y[p],
            });
        }
        if fa.mono {
            continue;
        }
        // Chroma: map (row, col) up to luma coords for the SB label.
        for (plane, (pa, pb)) in [(Plane::U, (fa.u, fb.u)), (Plane::V, (fa.v, fb.v))] {
            if let Some(p) = pa.iter().zip(pb.iter()).position(|(x, y)| x != y) {
                let (row, col) = (p / fa.width_uv, p % fa.width_uv);
                let luma_row = row << fa.ss_y;
                let luma_col = col << fa.ss_x;
                return Some(Divergence::Sample {
                    frame: i,
                    plane,
                    row,
                    col,
                    sb_row: luma_row / sb_px,
                    sb_col: luma_col / sb_px,
                    a: pa[p],
                    b: pb[p],
                });
            }
        }
    }
    None
}

/// Decode `stream` with the port's multi-frame decoder, catching a decode
/// `Err` OR an internal panic (an unimplemented inter feature) and mapping both
/// to a `String` — so the localizer reports a decoder gap instead of aborting.
pub fn try_decode_frames(stream: &[u8]) -> Result<Vec<FrameDecode>, String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // silence the backtrace for the probe
    let res = std::panic::catch_unwind(|| aom_decode::frame::decode_frames(stream));
    std::panic::set_hook(hook);
    match res {
        Ok(Ok(frames)) => Ok(frames),
        Ok(Err(e)) => Err(format!("decode error: {e}")),
        Err(_) => Err("decoder panicked (unimplemented inter feature)".to_string()),
    }
}

/// Decode BOTH streams with the port decoder and localize the first divergence.
/// `None` = the two streams decode to byte-identical frame-sets. An asymmetric
/// decode failure (one side rejects/panics, the other decodes) is itself a
/// [`Divergence::DecodeError`]; if BOTH fail the `Err` carries both messages.
pub fn decode_both(
    stream_a: &[u8],
    stream_b: &[u8],
    sb_px: usize,
) -> Result<Option<Divergence>, String> {
    let ra = try_decode_frames(stream_a);
    let rb = try_decode_frames(stream_b);
    match (ra, rb) {
        (Ok(fa), Ok(fb)) => {
            let va: Vec<FrameView> = fa.iter().map(FrameView::of_decode).collect();
            let vb: Vec<FrameView> = fb.iter().map(FrameView::of_decode).collect();
            Ok(first_frameset_divergence(&va, &vb, sb_px))
        }
        (Ok(_), Err(e)) => Ok(Some(Divergence::DecodeError {
            a_ok: true,
            b_ok: false,
            msg: format!("B: {e}"),
        })),
        (Err(e), Ok(_)) => Ok(Some(Divergence::DecodeError {
            a_ok: false,
            b_ok: true,
            msg: format!("A: {e}"),
        })),
        (Err(ea), Err(eb)) => Err(format!("both sides failed to decode (A: {ea}; B: {eb})")),
    }
}
