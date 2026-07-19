//! Task §F (self-derived sequence header): port + differential of the
//! **operating-points level/tier** algorithm — the last self-derivable
//! sequence-header field that the port currently bootstraps from a real parse.
//!
//! # Which function actually writes the header level?
//!
//! The coverage audit named `av1_get_seq_level_idx` (`av1/encoder/level.c:1366`)
//! as the missing algorithm. Traced first-hand against `reference/libaom`, that
//! is the **wrong** target for the written header:
//!
//! - `av1_get_seq_level_idx` computes the *achieved* level from per-operating-
//!   point accumulated `AV1LevelInfo` statistics (`check_level_constraints`
//!   over `level_info[op]`), and it runs **only** when
//!   `level_params->keep_level_stats >> op & 1` — which is **0 by default**
//!   (`encoder.c:912`, set only under `--target-seq-level-idx` /
//!   `SEQ_LEVEL_KEEP_STATS`). It never feeds the written sequence header.
//! - The **written** `seq_level_idx[op]` (and `tier[op]`, `op_params[].bitrate`)
//!   is produced by **`set_bitstream_level_tier`** (`encoder.c:464`, called at
//!   init `encoder.c:649`), whose core is the pure function
//!   `does_level_match(width, height, fps, …)` (`encoder.c:451`). That is what
//!   this file ports.
//!
//! # Differential ground truth (real C output, no new shim needed)
//!
//! For a still image the shim's encode sets `g_limit == 1`
//! (`crates/aom-sys-ref/shim/dec_shim.c`) → `seq->still_picture` (encoder.c:594)
//! → `seq->reduced_still_picture_hdr` (encoder.c:596, `full_still_picture_hdr`
//! is 0). The **reduced** still-picture header STILL codes `seq_level_idx[0]`
//! as a 5-bit field (`av1_write_sequence_header_obu`, `bitstream.c:3515`:
//! `write_bitstream_level(seq_params->seq_level_idx[0], &wb)`) — i.e. the exact
//! value `set_bitstream_level_tier` computed. So the existing
//! `ref_encode_av1_kf` output already carries the real-C level in
//! `seq.seq_level_idx[0]`; the port is diffed against it directly. (The level
//! is content- and speed-independent, so the witnesses use the fastest flat
//! mono encodes.) `tier` is not coded in a reduced header (default main tier 0).
//!
//! # Productionization (deferred, one line)
//!
//! The port lives in this test file rather than `aom-encode/src` because
//! wiring a new `pub mod seq_level;` requires editing `aom-encode/src/lib.rs`,
//! which a concurrent `cargo fmt` WIP change is holding; keeping this additive
//! (new file only) avoids the conflict. Lifting `seq_header_seq_level_idx` /
//! `does_level_match` into `crates/aom-encode/src/seq_level.rs` + one
//! `pub mod seq_level;` line is the trivial follow-up once that lands.

use aom_dsp::entropy::header::read_sequence_header_obu;
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::rb::ReadBitBuffer;
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;

fn walk_obus(bytes: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        let (size, size_bytes) =
            aom_dsp::entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
        let payload_start = after_header + size_bytes;
        let payload_end = payload_start + size as usize;
        out.push((hdr.obu_type, &bytes[payload_start..payload_end]));
        pos = payload_end;
    }
    out
}

// ---- AV1_LEVEL enum values (av1/common/enums.h) -- only the subset
//      set_bitstream_level_tier can produce, plus the sentinels. ----
const SEQ_LEVEL_2_0: i32 = 0;
const SEQ_LEVEL_2_1: i32 = 1;
const SEQ_LEVEL_3_0: i32 = 4;
const SEQ_LEVEL_3_1: i32 = 5;
const SEQ_LEVEL_4_0: i32 = 8;
const SEQ_LEVEL_4_1: i32 = 9;
const SEQ_LEVEL_5_0: i32 = 12;
const SEQ_LEVEL_5_1: i32 = 13;
const SEQ_LEVEL_5_2: i32 = 14;
const SEQ_LEVEL_6_0: i32 = 16;
const SEQ_LEVEL_6_1: i32 = 17;
const SEQ_LEVEL_6_2: i32 = 18;
const SEQ_LEVELS: i32 = 28;
const SEQ_LEVEL_MAX: i32 = 31;

/// `does_level_match` (`av1/encoder/encoder.c:451`). Pure integer/double
/// arithmetic on the frame dimensions + framerate vs a level's caps.
fn does_level_match(
    width: i32,
    height: i32,
    fps: f64,
    lvl_width: i32,
    lvl_height: i32,
    lvl_fps: f64,
    lvl_dim_mult: i32,
) -> bool {
    let lvl_luma_pels = lvl_width as i64 * lvl_height as i64;
    let lvl_display_sample_rate = lvl_luma_pels as f64 * lvl_fps;
    let luma_pels = width as i64 * height as i64;
    let display_sample_rate = luma_pels as f64 * fps;
    luma_pels <= lvl_luma_pels
        && display_sample_rate <= lvl_display_sample_rate
        && width <= lvl_width * lvl_dim_mult
        && height <= lvl_height * lvl_dim_mult
}

/// The `set_bitstream_level_tier` level ladder (`encoder.c:472-509`): the
/// lowest level whose dimension/display-rate caps the frame fits under, else
/// `SEQ_LEVEL_MAX`. The `CONFIG_CWG_C013` 7.x/8.x arm (encoder.c:512-535) is
/// gated on `target_seq_level_idx[0]` in `[SEQ_LEVEL_7_0, SEQ_LEVEL_8_3]` and
/// is therefore unreachable at the default `target == SEQ_LEVEL_MAX`; it is
/// omitted here (documented, not silently dropped).
fn inferred_seq_level_from_dims(width: i32, height: i32, fps: f64) -> i32 {
    if does_level_match(width, height, fps, 512, 288, 30.0, 4) {
        SEQ_LEVEL_2_0
    } else if does_level_match(width, height, fps, 704, 396, 30.0, 4) {
        SEQ_LEVEL_2_1
    } else if does_level_match(width, height, fps, 1088, 612, 30.0, 4) {
        SEQ_LEVEL_3_0
    } else if does_level_match(width, height, fps, 1376, 774, 30.0, 4) {
        SEQ_LEVEL_3_1
    } else if does_level_match(width, height, fps, 2048, 1152, 30.0, 3) {
        SEQ_LEVEL_4_0
    } else if does_level_match(width, height, fps, 2048, 1152, 60.0, 3) {
        SEQ_LEVEL_4_1
    } else if does_level_match(width, height, fps, 4096, 2176, 30.0, 2) {
        SEQ_LEVEL_5_0
    } else if does_level_match(width, height, fps, 4096, 2176, 60.0, 2) {
        SEQ_LEVEL_5_1
    } else if does_level_match(width, height, fps, 4096, 2176, 120.0, 2) {
        SEQ_LEVEL_5_2
    } else if does_level_match(width, height, fps, 8192, 4352, 30.0, 2) {
        SEQ_LEVEL_6_0
    } else if does_level_match(width, height, fps, 8192, 4352, 60.0, 2) {
        SEQ_LEVEL_6_1
    } else if does_level_match(width, height, fps, 8192, 4352, 120.0, 2) {
        SEQ_LEVEL_6_2
    } else {
        SEQ_LEVEL_MAX
    }
}

/// `set_bitstream_level_tier`'s written `seq_level_idx[op]` (encoder.c:541-545):
/// a higher explicit `target_seq_level_idx[op]` overrides the inferred level;
/// at the default `target == SEQ_LEVEL_MAX` (>= SEQ_LEVELS) the inferred level
/// is used unchanged.
fn seq_header_seq_level_idx(width: i32, height: i32, fps: f64, target_seq_level_idx: i32) -> i32 {
    let level = inferred_seq_level_from_dims(width, height, fps);
    if target_seq_level_idx < SEQ_LEVELS && target_seq_level_idx > level {
        target_seq_level_idx
    } else {
        level
    }
}

/// The default init framerate for the shim encode: `g_timebase == {1, 30}`
/// (`av1/av1_cx_iface.c:5265`, ALL_INTRA usage) → `init_framerate =
/// den/num = 30` (`av1_cx_iface.c:1197`).
const SHIM_FPS: f64 = 30.0;

fn real_reduced_seq_level(w: usize, h: usize) -> (i32, bool) {
    // Flat mono, cpu-used 9 (fastest): the coded seq_level_idx is content- and
    // speed-independent (set at init from dims + fps only).
    let y = vec![128u16; w * h];
    let bytes = c::ref_encode_av1_kf(
        &y, &[], &[], w, h, 8, /*mono*/ true, 1, 1, /*cq*/ 32, /*cpu*/ 9,
        /*cdef*/ false, /*restoration*/ false, /*usage=ALLINTRA*/ 2, /*aq*/ 0,
        /*two_pass*/ false,
    );
    assert!(!bytes.is_empty(), "shim_encode_av1_kf must produce a stream");
    let obus = walk_obus(&bytes);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, p)| *p)
        .unwrap_or_else(|| panic!("no sequence-header OBU ({w}x{h})"));
    let mut rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut rb);
    (seq.seq_level_idx[0], seq.reduced_still_picture_hdr)
}

/// Differential: the ported `set_bitstream_level_tier` seq-level derivation vs
/// real aomenc's coded `seq_level_idx[0]`, across a dimension sweep that
/// exercises the level ladder (2.0 / 2.1 / 3.0 / 3.1 / 4.0) AND all three
/// `does_level_match` clauses (luma-pels, and BOTH the width and height
/// `dim_mult` caps). Every witness is diffed against the REAL C header value,
/// and cross-checked against a hand-derived expectation for regression clarity.
#[test]
fn seq_level_idx_matches_real_reduced_header() {
    c::ref_init();
    // (w, h, hand-derived expected level, why)
    let cases: &[(usize, usize, i32, &str)] = &[
        (64, 64, SEQ_LEVEL_2_0, "tiny"),
        (512, 288, SEQ_LEVEL_2_0, "2.0 luma boundary (512*288)"),
        (513, 288, SEQ_LEVEL_2_1, "just over 2.0 luma cap -> 2.1"),
        (704, 396, SEQ_LEVEL_2_1, "2.1 luma boundary (704*396)"),
        (705, 396, SEQ_LEVEL_3_0, "just over 2.1 luma cap -> 3.0"),
        (64, 1160, SEQ_LEVEL_2_1, "height dim_mult: 1160 > 288*4 fails 2.0"),
        (2050, 64, SEQ_LEVEL_2_1, "width dim_mult: 2050 > 512*4 fails 2.0"),
        (800, 600, SEQ_LEVEL_3_0, "3.0 by luma (480000)"),
        (1280, 720, SEQ_LEVEL_3_1, "720p -> 3.1"),
        (1408, 768, SEQ_LEVEL_4_0, "over 3.1 luma cap -> 4.0 (dim_mult 3)"),
    ];

    let mut seen = std::collections::BTreeSet::new();
    for &(w, h, expected, why) in cases {
        let (real, reduced) = real_reduced_seq_level(w, h);
        assert!(
            reduced,
            "{w}x{h}: the still-picture shim encode must use a reduced header (so \
             seq_level_idx[0] is the coded set_bitstream_level_tier value)"
        );
        let port = seq_header_seq_level_idx(w as i32, h as i32, SHIM_FPS, SEQ_LEVEL_MAX);

        assert_eq!(
            real, expected,
            "{w}x{h} ({why}): REAL C seq_level_idx[0]={real} vs hand-derived {expected}"
        );
        assert_eq!(
            port, real,
            "{w}x{h} ({why}): PORT set_bitstream_level_tier seq_level_idx={port} vs \
             REAL C coded seq_level_idx[0]={real}"
        );
        eprintln!("seq_level_idx {w}x{h}: level={real} (port==real==expected) [{why}]");
        seen.insert(real);
    }

    // Anti-vacuous: the sweep must span multiple distinct levels, not trivially
    // return one constant.
    assert!(
        seen.len() >= 4,
        "sweep must exercise multiple levels (got {seen:?})"
    );
    eprintln!(
        "seq_level_idx_matches_real_reduced_header: {} cells, {} distinct levels {seen:?}",
        cases.len(),
        seen.len()
    );
}

/// Unit lock on the two override branches of `set_bitstream_level_tier`
/// (encoder.c:541-545) that the reduced-header witnesses above cannot reach
/// (they always run at the default `target == SEQ_LEVEL_MAX`): an explicit
/// higher target overrides the inferred level; a target `<= level` or
/// `>= SEQ_LEVELS` does not.
#[test]
fn seq_level_idx_target_override_semantics() {
    // 64x64 infers SEQ_LEVEL_2_0 (0).
    let (w, h) = (64, 64);
    assert_eq!(inferred_seq_level_from_dims(w, h, 30.0), SEQ_LEVEL_2_0);

    // Default target (MAX): inferred level unchanged.
    assert_eq!(
        seq_header_seq_level_idx(w, h, 30.0, SEQ_LEVEL_MAX),
        SEQ_LEVEL_2_0
    );
    // A higher, valid explicit target (< SEQ_LEVELS and > level) overrides.
    assert_eq!(
        seq_header_seq_level_idx(w, h, 30.0, SEQ_LEVEL_4_0),
        SEQ_LEVEL_4_0
    );
    // A target not greater than the inferred level does NOT override.
    assert_eq!(
        seq_header_seq_level_idx(w, h, 30.0, SEQ_LEVEL_2_0),
        SEQ_LEVEL_2_0
    );
    // A target >= SEQ_LEVELS (e.g. MAX) never overrides even if "greater".
    assert_eq!(
        seq_header_seq_level_idx(1408, 768, 30.0, SEQ_LEVEL_MAX),
        SEQ_LEVEL_4_0
    );
}
