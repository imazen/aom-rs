//! DEFAULT-config loop-restoration parity gate — the port's DEFAULT allintra
//! encode vs a plain `aomenc --allintra` with NO coding-tool flags.
//!
//! Loop-restoration is **ON by default in allintra** (verified first-hand:
//! `default_extra_cfg.enable_restoration = 1`, av1_cx_iface.c:286; NOT cleared
//! by the :3065 allintra override; kept non-realtime at :1273; the encoder's
//! own shim comment says "the encoder falls back to the ALLINTRA defaults
//! (restoration ON, etc.)"). C runs `av1_pick_filter_restoration` and emits the
//! seq/frame restoration syntax even when every unit resolves RESTORE_NONE —
//! so a plain `aomenc --allintra` stream differs from `--enable-restoration=0`
//! at the header level regardless of the search outcome.
//!
//! The port's byte-exact LR search (PARITY.md C2 / `lr_restoration_gate`, 8/8)
//! is wired into the DEFAULT path: `EncodeCell::port_encode` runs it whenever
//! the (bootstrap) sequence header has `enable_restoration = 1` — C's
//! `is_restoration_used`. This gate proves that default path byte-matches a
//! genuinely flagless reference. It is the counterpart to the explicit-off
//! `encoder_gate_e2e_*` gates (which test the `--enable-restoration=0`
//! NON-default config and stay valid as-is).
//!
//! Per cell:
//!  - `plain`  = `c_encode_defaults()` — plain `aomenc --allintra`, no tool flags;
//!  - `lr_ref` = `c_encode_lr()` — explicit `--enable-restoration=1`;
//!  - `off`    = `c_encode()` — explicit `--enable-restoration=0` (the old ref);
//!  - assert `plain == lr_ref`: proves restoration IS the default (=1) AND that
//!    the other non-restoration tool defaults — palette / intrabc / deltaq —
//!    are byte-inert on this (non-screen) content;
//!  - track `plain != off`: the default genuinely differs from restoration-off;
//!  - assert `port_encode(&plain)` frame-OBU == `plain`'s frame-OBU: **the port's
//!    default encode byte-matches a plain `aomenc --allintra`** (default parity).
//!
//! Anti-vacuous floors: at least one cell where the reference actually restores
//! a plane (LR syntax exercised, not just an all-NONE header), and at least one
//! where the default stream differs from the restoration-off stream.

use aom_bench::{EncodeCell, parse_restoration_decision};

/// Real conformance-decoded content across the quality range (the KB-6 /
/// `lr_restoration_gate` recipe: 1-SB, multi-SB + partial edges, multi-unit
/// size-descent grids, and a 10-bit arm). Speed 0 — where the port's real-
/// content base encode is byte-exact (KB-6 30/30), so any divergence here is
/// the restoration wiring's.
fn cells() -> Vec<EncodeCell> {
    vec![
        EncodeCell::real_content("def_size64_cq12", "av1-1-b8-01-size-64x64", None, 12, 0),
        EncodeCell::real_content("def_size64_cq32", "av1-1-b8-01-size-64x64", None, 32, 0),
        EncodeCell::real_content("def_size64_cq48", "av1-1-b8-01-size-64x64", None, 48, 0),
        EncodeCell::real_content("def_size196_cq20", "av1-1-b8-01-size-196x196", None, 20, 0),
        EncodeCell::real_content("def_size196_cq48", "av1-1-b8-01-size-196x196", None, 48, 0),
        EncodeCell::real_content("def_quant00_cq32", "av1-1-b8-00-quantizer-00", None, 32, 0),
        EncodeCell::real_content("def_quant00_cq55", "av1-1-b8-00-quantizer-00", None, 55, 0),
        EncodeCell::real_content(
            "def_b10_quant00_cq32",
            "av1-1-b10-00-quantizer-00",
            None,
            32,
            0,
        ),
    ]
}

#[test]
fn port_default_matches_plain_aomenc_allintra() {
    let mut real_active = 0usize;
    let mut differs_from_off = 0usize;
    let n = cells().len();

    for cell in cells() {
        // The plain no-flags allintra reference (restoration ON by default).
        let plain = cell.c_encode_defaults();
        assert!(
            !plain.is_empty(),
            "{}: plain --allintra encode failed",
            cell.label
        );

        // Finding verification: a genuinely flagless `--allintra` encode is
        // byte-identical to an explicit `--enable-restoration=1` encode — i.e.
        // restoration's default IS on, and the other non-restoration tool
        // defaults (palette/intrabc/deltaq) are byte-inert on this content.
        let lr_ref = cell.c_encode_lr();
        assert_eq!(
            plain, lr_ref,
            "{}: plain `aomenc --allintra` must equal explicit `--enable-restoration=1` \
             (restoration IS the allintra default; palette/intrabc/deltaq inert here)",
            cell.label
        );

        // Anti-vacuous: the default really differs from restoration-off.
        let off = cell.c_encode();
        if plain != off {
            differs_from_off += 1;
        }

        // The reference's restoration decision (diagnostic + anti-vacuous).
        let (frt, us) = parse_restoration_decision(&plain);
        if frt.iter().any(|&t| t != 0) {
            real_active += 1;
        }

        // THE GATE: the port's DEFAULT path (auto-derives the LR stage from
        // `plain`'s enable_restoration=1 seq bit) byte-matches the plain stream.
        let port_payload = cell.port_encode(&plain);
        let real_payload = EncodeCell::frame_obu_payload(&plain);
        let exact = port_payload == real_payload;
        eprintln!(
            "{}: frt={frt:?} us={us:?} | default_differs_from_off={} | {}",
            cell.label,
            plain != off,
            if exact { "BYTE-EXACT" } else { "DIVERGES" },
        );
        assert_eq!(
            port_payload, real_payload,
            "{}: the port's DEFAULT allintra encode must be BYTE-IDENTICAL to a plain \
             `aomenc --allintra` (restoration on) — default-config parity",
            cell.label
        );
    }

    eprintln!(
        "LR default-parity: {n}/{n} port==plain byte-exact, {real_active}/{n} reference-LR-active, \
         {differs_from_off}/{n} default!=restoration-off"
    );

    // Anti-vacuous: the reference must actually exercise LR on this grid (else
    // "matches the default" would be an empty statement about the feature), and
    // the default must actually differ from the restoration-off config (else
    // the whole default-parity question would be moot).
    assert!(
        real_active >= 1,
        "no cell made the default encoder restore a plane — the LR syntax is unexercised"
    );
    assert!(
        differs_from_off >= 1,
        "the default stream never differed from --enable-restoration=0 — restoration is inert here, \
         so this grid cannot witness default parity"
    );
}
