//! aom-entropy — bit-exact AV1 Daala range coder (port of libaom v3.14.1).
//!
//! Encoder (`od_ec_enc`) produces byte-identical output to C; decoder
//! (`od_ec_dec`) recovers identical symbols. Foundational to both tracks.

#![forbid(unsafe_code)]
pub mod cdf;
pub mod dec;
pub mod default_cdfs;
pub mod enc;
pub mod header;
pub mod leb128;
pub mod lr;
pub mod obu;
pub mod partition;
pub mod rb;
pub mod wb;

pub use cdf::{read_symbol, update_cdf, write_symbol};
pub use dec::OdEcDec;
pub use enc::OdEcEnc;
