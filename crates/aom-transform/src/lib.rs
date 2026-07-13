//! aom-transform — bit-exact AV1 transform kernels (port of libaom v3.14.1).
//!
//! Every public function is validated byte-for-byte against the C reference by
//! a differential harness in `tests/`. Scalar-first; SIMD specializations must
//! match this scalar output exactly (the same contract libaom holds internally).

pub mod cospi;
pub mod fdct;
pub mod inv_txfm1d_gen;
pub mod inv_txfm2d;
pub mod special;
pub mod txfm1d_gen;
pub mod txfm2d;

pub use fdct::{av1_fdct4, clamp_value, half_btf, round_shift};
pub use special::{
    av1_fadst4, av1_fidentity16, av1_fidentity32, av1_fidentity4, av1_fidentity8, av1_iadst4,
    av1_iidentity16, av1_iidentity32, av1_iidentity4, av1_iidentity8,
};
pub use txfm1d_gen::{av1_fadst16, av1_fadst8, av1_fdct16, av1_fdct32, av1_fdct64, av1_fdct8};
pub use inv_txfm1d_gen::{
    av1_iadst16, av1_iadst8, av1_idct16, av1_idct32, av1_idct4, av1_idct64, av1_idct8,
};
