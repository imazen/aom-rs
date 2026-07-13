//! aom-entropy — bit-exact AV1 Daala range coder (port of libaom v3.14.1).
//!
//! Encoder (`od_ec_enc`) produces byte-identical output to C; decoder
//! (`od_ec_dec`) recovers identical symbols. Foundational to both tracks.

pub mod dec;
pub mod enc;

pub use dec::OdEcDec;
pub use enc::OdEcEnc;
