//! aom-dsp — consolidated DSP + entropy kernels for the pure-Rust, bit-exact
//! libaom v3.14.1 port.
//!
//! Each former per-kernel crate is a module here: [`transform`], [`quant`],
//! [`txb`], [`cdef`], [`restore`], [`intra`], [`loopfilter`], [`dist`],
//! [`inter`], [`convolve`], [`recon`], [`dispatch`], and the MSAC range coder
//! [`entropy`]. Consolidating them into one publishable crate keeps the
//! release surface small (a single `cargo publish` / version bump) while
//! preserving the exact kernel byte-for-byte — the module paths are the only
//! thing that changed (`aom_transform::X` → `aom_dsp::transform::X`).
//!
//! During the consolidation the sub-crates are re-exported below via
//! `pub use aom_X as X` and then physically absorbed into `src/X/`, one family
//! at a time, so the differential gates stay green through every step.
#![forbid(unsafe_code)]

pub use aom_cdef as cdef;
pub use aom_convolve as convolve;
pub use aom_dispatch as dispatch;
pub use aom_dist as dist;
pub use aom_entropy as entropy;
pub use aom_inter as inter;
pub use aom_intra as intra;
pub use aom_loopfilter as loopfilter;
pub use aom_quant as quant;
pub use aom_recon as recon;
pub use aom_restore as restore;
pub use aom_transform as transform;
pub use aom_txb as txb;
