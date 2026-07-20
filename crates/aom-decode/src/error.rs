//! Structured, category-bearing decode errors — the zen cross-cutting error
//! contract (`CLAUDE.md` §3/§4). This replaces the former stringly-typed
//! `Result<_, String>` on the public decode entry points.
//!
//! The variants partition every failure into the categories a consumer must
//! be able to distinguish — corrupt vs truncated vs unsupported-*type* vs
//! unsupported-*feature* vs limit-exceeded vs out-of-memory vs cancelled vs
//! internal-bug — so the integration layer (zenavif) can translate each
//! variant onto its own `ErrorCategory` without collapsing every failure into
//! one opaque code. This crate deliberately does **not** depend on `zencodec`;
//! it exposes a self-contained error type and lets the seam map it.

use alloc::borrow::Cow;
use alloc::string::String;
use core::fmt;

use enough::StopReason;

/// Which caller-configured resource limit a [`DecodeError::LimitExceeded`]
/// refers to. Maps onto the integration layer's limit categories at the seam.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitKind {
    /// Total pixel count (`width * height`).
    Pixels,
    /// Frame width in pixels.
    Width,
    /// Frame height in pixels.
    Height,
    /// Peak decode memory in bytes.
    MemoryBytes,
}

impl LimitKind {
    /// A stable lowercase identifier, for messages and the seam mapping.
    pub fn as_str(self) -> &'static str {
        match self {
            LimitKind::Pixels => "pixels",
            LimitKind::Width => "width",
            LimitKind::Height => "height",
            LimitKind::MemoryBytes => "memory_bytes",
        }
    }
}

/// A categorized AV1 decode failure.
///
/// `#[non_exhaustive]`: new variants may be added without a breaking change,
/// so consumers must include a wildcard arm when matching.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Input ended mid-structure — a short OBU, tile payload, or leb128 size.
    /// Maps to an "unexpected end of file" category.
    Truncated(&'static str),
    /// The bitstream violates AV1 syntax or is otherwise corrupt. Maps to a
    /// "malformed" category.
    Malformed(Cow<'static, str>),
    /// A syntactically-valid but unsupported bitstream *type* (e.g. a
    /// subsampling or `frame_type` this decoder does not handle). Maps to an
    /// "unsupported type" category.
    UnsupportedType(Cow<'static, str>),
    /// A well-formed AV1 tool this decoder does not yet implement — distinct
    /// from corruption: the stream is valid, the feature is out of scope.
    /// Maps to an "unsupported feature" category.
    UnsupportedFeature(&'static str),
    /// A caller-configured resource limit was exceeded, rejected before the
    /// corresponding allocation. Maps to a "resource limit" category.
    LimitExceeded {
        /// Which limit was hit.
        kind: LimitKind,
        /// The value the header declared.
        actual: u64,
        /// The configured maximum.
        max: u64,
    },
    /// A fallible allocation failed (out of memory) under fallible allocation
    /// mode. Maps to an "out of memory" category.
    AllocFailed {
        /// The requested size in bytes.
        bytes: usize,
    },
    /// The decode was cancelled at a coarse boundary via the caller's stop
    /// token, carrying the [`StopReason`]. Maps to a "cancelled/stopped"
    /// category.
    Cancelled(StopReason),
    /// A broken internal invariant — a bug in this decoder, not attacker
    /// input. Maps to an "internal" category.
    Internal(&'static str),
}

impl DecodeError {
    /// A short, stable category name — the coarse bucket this error belongs
    /// to, for logging and for the zenavif seam's variant→category mapping.
    pub fn category(&self) -> &'static str {
        match self {
            DecodeError::Truncated(_) => "truncated",
            DecodeError::Malformed(_) => "malformed",
            DecodeError::UnsupportedType(_) => "unsupported-type",
            DecodeError::UnsupportedFeature(_) => "unsupported-feature",
            DecodeError::LimitExceeded { .. } => "limit-exceeded",
            DecodeError::AllocFailed { .. } => "alloc-failed",
            DecodeError::Cancelled(_) => "cancelled",
            DecodeError::Internal(_) => "internal",
        }
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Truncated(m) => write!(f, "truncated bitstream: {m}"),
            DecodeError::Malformed(m) => write!(f, "malformed bitstream: {m}"),
            DecodeError::UnsupportedType(m) => write!(f, "unsupported type: {m}"),
            DecodeError::UnsupportedFeature(m) => write!(f, "unsupported feature: {m}"),
            DecodeError::LimitExceeded { kind, actual, max } => {
                write!(f, "decode limit exceeded: {} = {actual} > {max}", kind.as_str())
            }
            DecodeError::AllocFailed { bytes } => {
                write!(f, "allocation of {bytes} bytes failed (out of memory)")
            }
            DecodeError::Cancelled(r) => write!(f, "decode cancelled by stop token: {r}"),
            DecodeError::Internal(m) => write!(f, "internal decoder error: {m}"),
        }
    }
}

impl core::error::Error for DecodeError {}

/// Bridge for the many sites that produce a `&'static str` reason — they
/// default to [`DecodeError::Malformed`] (the safe category for an
/// uncategorized syntax failure). Sites with a more precise category
/// construct the variant directly.
impl From<&'static str> for DecodeError {
    fn from(s: &'static str) -> Self {
        DecodeError::Malformed(Cow::Borrowed(s))
    }
}

/// Bridge for the sites that produce an owned `String` reason (e.g. the
/// `corrupt` poison channel and `format!` messages) — likewise default to
/// [`DecodeError::Malformed`].
impl From<String> for DecodeError {
    fn from(s: String) -> Self {
        DecodeError::Malformed(Cow::Owned(s))
    }
}

/// A [`StopReason`] from a cancelled decode becomes [`DecodeError::Cancelled`],
/// so a polled `stop.check()?` propagates cleanly.
impl From<StopReason> for DecodeError {
    fn from(r: StopReason) -> Self {
        DecodeError::Cancelled(r)
    }
}
