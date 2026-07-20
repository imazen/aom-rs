//! Caller-supplied decode configuration — resource limits (and, in later
//! chunks, the allocation mode and stop token) threaded through the
//! config-carrying decode entries (`CLAUDE.md` §1). The bare entries
//! (`decode_frame_obus`, `decode_frames`, `decode_frame_obus_prefilter`) apply
//! [`DecodeConfig::default`], which preserves the historical behavior: the
//! hardcoded [`DEFAULT_MAX_DECODE_PIXELS`] pixel ceiling and no width/height cap.

use crate::{DecodeError, LimitKind};

/// The default per-frame pixel ceiling (~268 Mpx) applied when a caller does
/// not set [`DecodeLimits::max_pixels`]. A crafted header declaring more than
/// this is rejected before any width×height-scaled buffer is allocated.
pub const DEFAULT_MAX_DECODE_PIXELS: u64 = 1 << 28;

/// Resource limits a caller may impose on a decode. Every field is `Option`;
/// `None` means "no caller cap" — `max_pixels == None` falls back to
/// [`DEFAULT_MAX_DECODE_PIXELS`], and the other dimensions are unbounded
/// (still subject to the pixel ceiling). Checked after the frame header is
/// parsed and before the first frame-sized allocation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Maximum total pixels (`width * height`). `None` → [`DEFAULT_MAX_DECODE_PIXELS`].
    pub max_pixels: Option<u64>,
    /// Maximum frame width in pixels. `None` → unbounded (pixel cap still applies).
    pub max_width: Option<u32>,
    /// Maximum frame height in pixels. `None` → unbounded (pixel cap still applies).
    pub max_height: Option<u32>,
    /// Advisory peak-memory cap in bytes, for the caller's own estimation /
    /// gating. `None` → unbounded. (Not yet enforced against a running total;
    /// exposed so a caller can record its budget alongside the dim caps.)
    pub max_memory_bytes: Option<u64>,
}

impl DecodeLimits {
    /// Construct with every field unset (`None`) — the same effect as
    /// [`DecodeLimits::default`]: default pixel ceiling, no dimension caps.
    pub const fn new() -> Self {
        DecodeLimits {
            max_pixels: None,
            max_width: None,
            max_height: None,
            max_memory_bytes: None,
        }
    }

    /// The effective pixel ceiling — the caller's `max_pixels`, or
    /// [`DEFAULT_MAX_DECODE_PIXELS`] when unset.
    pub fn effective_max_pixels(&self) -> u64 {
        self.max_pixels.unwrap_or(DEFAULT_MAX_DECODE_PIXELS)
    }

    /// Reject a frame whose declared dimensions exceed any configured limit.
    /// Called after header parse and before the recon / mi / segment
    /// allocations, so an over-budget header never drives a large allocation.
    /// Width/height are the header's (possibly negative on a malformed stream)
    /// signed dims; they are clamped to non-negative before comparison.
    pub(crate) fn check_dims(&self, width: i32, height: i32) -> Result<(), DecodeError> {
        let w = width.max(0) as u64;
        let h = height.max(0) as u64;
        if let Some(mw) = self.max_width {
            if w > mw as u64 {
                return Err(DecodeError::LimitExceeded { kind: LimitKind::Width, actual: w, max: mw as u64 });
            }
        }
        if let Some(mh) = self.max_height {
            if h > mh as u64 {
                return Err(DecodeError::LimitExceeded { kind: LimitKind::Height, actual: h, max: mh as u64 });
            }
        }
        let px = w.saturating_mul(h);
        let max_px = self.effective_max_pixels();
        if px > max_px {
            return Err(DecodeError::LimitExceeded { kind: LimitKind::Pixels, actual: px, max: max_px });
        }
        Ok(())
    }
}

/// Configuration for a decode. Passed to the `*_with` entry points; the bare
/// entries use [`DecodeConfig::default`]. `#[non_exhaustive]` + the builder
/// methods keep additive growth (allocation mode, stop token) non-breaking.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct DecodeConfig {
    /// Resource limits to enforce (default: the hardcoded pixel ceiling only).
    pub limits: DecodeLimits,
}

impl DecodeConfig {
    /// A config with default limits (the bare-entry behavior).
    pub fn new() -> Self {
        DecodeConfig::default()
    }

    /// Set the resource limits (builder style).
    pub fn with_limits(mut self, limits: DecodeLimits) -> Self {
        self.limits = limits;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DecodeError, LimitKind};

    fn kind_of(e: &DecodeError) -> Option<LimitKind> {
        match e {
            DecodeError::LimitExceeded { kind, .. } => Some(*kind),
            _ => None,
        }
    }

    #[test]
    fn default_limits_apply_the_pixel_ceiling() {
        let d = DecodeLimits::default();
        assert_eq!(d.effective_max_pixels(), DEFAULT_MAX_DECODE_PIXELS);
        // A normal frame passes under the default ceiling.
        assert!(d.check_dims(1920, 1080).is_ok());
        // Exactly at the ceiling passes; one pixel over is rejected.
        assert!(d.check_dims(1 << 14, 1 << 14).is_ok()); // 2^28 exactly
        let over = d.check_dims((1 << 14) + 1, 1 << 14).unwrap_err();
        assert_eq!(kind_of(&over), Some(LimitKind::Pixels));
        // A malformed 65535x65535 header (~4.29 Gpx) is rejected before alloc.
        assert_eq!(kind_of(&d.check_dims(65535, 65535).unwrap_err()), Some(LimitKind::Pixels));
    }

    #[test]
    fn caller_max_pixels_rejects_a_larger_header() {
        // The §1 acceptance case: max_pixels = 1_000_000 rejects a 2 Mpx header.
        let d = DecodeLimits { max_pixels: Some(1_000_000), ..Default::default() };
        let e = d.check_dims(1920, 1080).unwrap_err(); // ~2.07 Mpx
        match e {
            DecodeError::LimitExceeded { kind: LimitKind::Pixels, actual, max } => {
                assert_eq!(actual, 1920 * 1080);
                assert_eq!(max, 1_000_000);
            }
            other => panic!("expected Pixels LimitExceeded, got {other:?}"),
        }
        // A frame under the cap passes.
        assert!(d.check_dims(640, 480).is_ok()); // 307200 < 1_000_000
    }

    #[test]
    fn width_and_height_caps_are_enforced_independently() {
        let d = DecodeLimits { max_width: Some(1280), max_height: Some(720), ..Default::default() };
        assert!(d.check_dims(1280, 720).is_ok());
        assert_eq!(kind_of(&d.check_dims(1281, 720).unwrap_err()), Some(LimitKind::Width));
        assert_eq!(kind_of(&d.check_dims(1280, 721).unwrap_err()), Some(LimitKind::Height));
    }

    #[test]
    fn negative_dims_clamp_to_zero_and_do_not_panic() {
        // A malformed header could carry negative signed dims; clamp, don't panic.
        let d = DecodeLimits::default();
        assert!(d.check_dims(-1, -1).is_ok());
        assert!(d.check_dims(i32::MIN, i32::MIN).is_ok());
    }
}
