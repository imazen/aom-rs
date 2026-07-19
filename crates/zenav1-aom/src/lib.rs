//! `zenav1-aom` — a pure-Rust, bit-exact reimplementation of libaom v3.14.1.
//!
//! This is a thin facade that re-exports the workspace's published crates as a
//! single dependency:
//!
//! - [`dsp`] — the DSP / transform / quant / entropy kernels (always present).
//! - [`decode`] — the AV1 decoder (feature `decode`, enabled by default).
//! - [`encode`] — the AV1 encoder (feature `encode`, enabled by default).
//!
//! Size-sensitive consumers can build a decode-only stack with
//! `default-features = false, features = ["decode"]` — the encoder crate is then
//! never compiled.
//!
//! ```toml
//! # decoder + encoder (default)
//! zenav1-aom = "0.0.1"
//! # decoder only (wasm / size-sensitive)
//! zenav1-aom = { version = "0.0.1", default-features = false, features = ["decode"] }
//! ```
#![forbid(unsafe_code)]

pub use aom_dsp as dsp;

#[cfg(feature = "decode")]
pub use aom_decode as decode;

#[cfg(feature = "encode")]
pub use aom_encode as encode;
