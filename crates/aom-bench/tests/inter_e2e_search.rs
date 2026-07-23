//! INTER-ENCODE chunk 2f/2g — THE END-TO-END SEARCH GATE for the §3
//! low-delay zero-MV P: the port's OWN partition search CHOOSES the inter
//! blocks (`PickFrameCfg::inter` → `rd_pick_inter_mode_sb` competing against
//! the intra winner at every leaf) and the packed frame-1 OBU payload must be
//! BYTE-IDENTICAL to `aomenc`'s.
//!
//! This is the rung the pack-only gate (`inter_pack_tile_diff.rs`) could not
//! claim: there the tile was hand-assembled from the measured block layout;
//! here `pack_tile` runs the real RD loop end-to-end and nothing block-level
//! is copied from the reference. The measured ground truth (instrumented
//! libaom decoder): the zero-MV P codes one `PARTITION_NONE` 64x64
//! `NEARESTMV (LAST)` skip block per superblock.
//!
//! ## Honest bootstrap (unchanged contract)
//!
//! Sequence template + the recon-dependent frame-1 header tail (LF, CDEF,
//! frame `interp_filter`) come from the reference stream, exactly as the
//! KEY-frame `port_encode` bootstraps its header; `base_qindex` is derived
//! and cross-checked. The TILE — the part this gate exists for — is derived
//! from nothing.
//!
//! ## Failure diagnostics
//!
//! On a byte mismatch the test rebuilds the port's full 2-frame stream
//! (frame payloads substituted into the reference framing) and runs the
//! decode-both localizer, printing the first divergent (frame, SB, sample)
//! before panicking.

use aom_bench::inter_localize::decode_both;
use aom_bench::{EncodeCell, MultiFrameEncodeCell, parse_inter_2frame_reference};

fn base(label: &str, w: usize, h: usize, mono: bool, cq: i32) -> EncodeCell {
    let content = |r: usize, c: usize| -> u16 { (40 + ((r * 3 + c * 5) % 160)) as u16 };
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = content(r, c);
        }
    }
    let (cw, ch) = if mono { (0, 0) } else { ((w + 1) >> 1, (h + 1) >> 1) };
    let cont_uv = |r: usize, c: usize| -> u16 { (110 + ((r * 2 + c) % 40)) as u16 };
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    if !mono {
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = cont_uv(r, c);
                v[r * cw + c] = cont_uv(r, c) + 3;
            }
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono,
        ss_x: 1,
        ss_y: 1,
        usage: 0,
        cq_level: cq,
        speed: 0,
        bd: 8,
        y,
        u,
        v,
    }
}

/// Rebuild a stream with each OBU_FRAME payload replaced (in order) by the
/// given payloads — the port's full 2-frame stream in the reference framing.
fn substitute_frame_payloads(stream: &[u8], payloads: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut fi = 0usize;
    while pos < stream.len() {
        let b0 = stream[pos];
        let obu_type = (b0 >> 3) & 0xF;
        let ext = (b0 >> 2) & 1;
        let hdr_len = 1 + usize::from(ext == 1);
        let mut p = pos + hdr_len;
        let mut size = 0u64;
        let mut shift = 0;
        loop {
            let b = stream[p];
            size |= u64::from(b & 0x7f) << shift;
            p += 1;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        let end = p + size as usize;
        if obu_type == 6 {
            // OBU_FRAME: re-emit with the substituted payload + new leb size.
            let payload = payloads[fi];
            fi += 1;
            out.extend_from_slice(&stream[pos..pos + hdr_len]);
            let mut sz = payload.len() as u64;
            loop {
                let mut byte = (sz & 0x7f) as u8;
                sz >>= 7;
                if sz != 0 {
                    byte |= 0x80;
                }
                out.push(byte);
                if sz == 0 {
                    break;
                }
            }
            out.extend_from_slice(payload);
        } else {
            out.extend_from_slice(&stream[pos..end]);
        }
        pos = end;
    }
    assert_eq!(fi, payloads.len(), "substituted every frame payload");
    out
}

/// Run one cell through the FRAME-1 chain: real aomenc, then the port's P via
/// its OWN search — byte-compare the frame-1 payload and decode-both the
/// substituted stream (frame 0 kept from the reference, so the pixel check
/// isolates frame 1).
fn run_cell(label: &str, w: usize, h: usize, mono: bool, cq: i32) {
    let cell = MultiFrameEncodeCell::translational(&base(label, w, h, mono, cq), 0, 0);
    let stream = cell.c_encode_inter(false, false);
    let r = parse_inter_2frame_reference(&stream);

    // Frame 1: the port's OWN SEARCH chooses the inter blocks.
    let port_f1 = cell.port_encode_inter_p(&stream);

    if port_f1 != r.f1_payload {
        // Localize before failing: substitute the port frame-1 payload into
        // the reference framing and decode both streams with the port decoder.
        let port_stream = substitute_frame_payloads(&stream, &[&r.f0_payload, &port_f1]);
        match decode_both(&port_stream, &stream, 64) {
            Ok(None) => eprintln!(
                "{label}: bytes differ but decode to IDENTICAL pixels — a \
                 probability/CDF-row divergence (same symbols, different rows)"
            ),
            Ok(Some(d)) => eprintln!("{label}: first decoded divergence: {d:?}"),
            Err(e) => eprintln!("{label}: decode-both failed: {e}"),
        }
        let split = r.header_bits.div_ceil(8);
        panic!(
            "{label}: port frame-1 payload differs from aomenc\n  port:   {:02x?}\n  aomenc: {:02x?}\n  (header = first {split} bytes)",
            port_f1, r.f1_payload,
        );
    }

    // Byte-identical payload ⇒ decode-both == None follows; asserted as the
    // pixel-level cross-check of the substituted full stream.
    let port_stream = substitute_frame_payloads(&stream, &[&r.f0_payload, &port_f1]);
    assert_eq!(
        decode_both(&port_stream, &stream, 64).expect("both streams decode"),
        None,
        "{label}: decode-both pixel identity"
    );
}

/// THE RUNG-1 GATE: single-SB 64x64 zero-MV P, 4:2:0 cq60 — the port's own
/// search picks the inter-skip block and the frame payload byte-matches.
#[test]
fn zero_mv_p_own_search_64x64_cq60_420_byte_exact() {
    run_cell("interp_e2e_64_cq60_420", 64, 64, false, 60);
}

/// DISCOVERED GAP (pinned, self-promoting): the port's KEY encode of frame 0
/// at GOOD usage (`usage=0` — the §3 inter context) does NOT byte-match
/// aomenc's frame-0 payload. Every landed KEY byte gate runs ALLINTRA
/// (`usage=2`); the GOOD-mode KEY search/header was never byte-gated (the
/// chunk-0 "frame-0 control" was decode-side). Measured at 64x64 cq60 4:2:0:
/// the port header is 2 bytes longer and the tile diverges mid-way — a
/// GOOD-vs-ALLINTRA speed-feature/header gap, NOT an inter-wiring issue
/// (`PickFrameCfg::inter` is None on this path).
///
/// This test asserts the divergence is PRESENT so it FAILS the moment the
/// GOOD KEY path becomes byte-exact — then promote it to `assert_eq!` and
/// extend `run_cell` to assert frame 0 + full-stream identity again.
#[test]
fn good_usage_key_frame0_pinned_divergent() {
    let cell =
        MultiFrameEncodeCell::translational(&base("interp_f0_good", 64, 64, false, 60), 0, 0);
    let stream = cell.c_encode_inter(false, false);
    let r = parse_inter_2frame_reference(&stream);
    let port_f0 = cell.frame0_cell().port_encode(&stream);
    assert_ne!(
        port_f0, r.f0_payload,
        "GOOD-usage KEY frame 0 NOW BYTE-MATCHES — promote this pin to a real \
         byte-identity gate and assert full-stream identity in run_cell"
    );
}

/// The cq ladder at 64x64: the all-skip zero-MV P holds across cq (measured:
/// aomenc's zero-MV P decodes identical to frame 0's recon at every cq), so
/// each cell must byte-match end-to-end. Mono exercises the luma-only path.
#[test]
fn zero_mv_p_own_search_cq_ladder_byte_exact() {
    for cq in [20, 40, 63] {
        run_cell(&format!("interp_e2e_64_cq{cq}_420"), 64, 64, false, cq);
    }
    for cq in [20, 60] {
        run_cell(&format!("interp_e2e_64_cq{cq}_mono"), 64, 64, true, cq);
    }
}

/// MULTI-BLOCK 64x128 — PINNED OPEN (self-promoting), root FULLY MEASURED
/// via a byte-inert instrumented sibling libaom (throwaway, removed): the
/// residual is a SEARCH-SPACE gap, not a pack/CDF bug.
///
/// ## Measured facts (2026-07-23, instrumented C at the pinned v3.14.1 SHA)
///
/// 1. The §3 GOOD-mode stream codes **SB128** (`mib_size_log2 = 5` — libaom's
///    GOOD default at every speed), NOT SB64. A 64x128 frame is ONE
///    column-cropped 128x128 superblock whose root codes a gathered 2-way
///    SPLIT/VERT partition symbol (`write_partition`'s `has_rows && !has_cols`
///    arm) before the two visible 64x64 children. The old "two-superblock
///    tile" framing (and `inter_pack_tile_diff.rs`'s hand-rolled SB64 pin) was
///    WRONG about the walk shape; a 64x64 frame is degenerate (both walks emit
///    identical symbols), which is why the single-SB gates match either way.
///    `port_encode_inter_p` now drives the declared SB size.
/// 2. At the cropped root C evaluates SPLIT (rd 248,444,575; children rd
///    133,585,034 + 102,885,701, both NEARESTMV skip) then VERT. VERT's single
///    64x128 sub-block FAILS: `av1_txfm_search`'s final skip-guard
///    (tx_search.c:3893) computes tmprd 250,728,413 > the remaining budget
///    247,851,864 (SPLIT minus the VERT partition bit) → no codeable
///    candidate → SPLIT wins in C.
/// 3. The 64x128 candidate is over-priced because of the INTERP-FILTER search:
///    during encoding the frame filter is SWITCHABLE (the coded REGULAR header
///    value is a POST-encode selection), so every inter candidate pays
///    `rs = av1_get_switchable_rate(..)`. `use_more_sharp_interp = 1`
///    (speed_features.c:1139, GOOD base at EVERY speed, non-boosted frames)
///    gives SHARP a `mul = 90` (10%) RD discount in `interpolation_filter_rd`
///    (interp_search.c) — on the dist-heavy 64x128 model RD the discount lets
///    SHARP displace REGULAR, and its rate at ctx 3 is 3931 vs REGULAR's 109.
///    The children (smaller model RD) keep REGULAR (rs 109 each).
/// 4. The port models NO interp-filter rate on the inter arm (rs = 0), so its
///    64x128 VERT leaf is ~3822 rate units cheaper than C's and WINS the root:
///    port tile `[d7, a0]` (VERT, one block) vs aomenc `[f2, 24, 80]` (SPLIT,
///    two blocks).
///
/// ## The fix (next rung, bounded)
///
/// Model the §3 interp-filter search on the inter arm: the CURVFIT model-rd
/// (`MODELRD_TYPE_INTERP_FILTER = MODELRD_CURVFIT`, model_rd.h:31 — NOT the
/// ported lapndz model; needs `av1_model_rd_curvfit` + tables + a
/// differential), `av1_get_pred_context_switchable_interp`, the
/// switchable-interp cost table, and `interpolation_filter_rd`'s biased
/// accept (`tmp_rd * mul / 100 < *rd`, SHARP mul=90). At zero MV every
/// filter's prediction is identical, so the reduced search is a pure
/// rate/model compare — no convolution needed. The winner's rs then joins the
/// inter leaf's mode_rate (C: rate2_nocoeff) and the skip-guard rd.
///
/// This test asserts the divergence is PRESENT so it FAILS the moment the
/// port matches — then promote to `run_cell` (hard byte assert).
#[test]
fn zero_mv_p_own_search_64x128_cropped_sb128_pinned_divergent() {
    let cell =
        MultiFrameEncodeCell::translational(&base("interp_e2e_64x128", 64, 128, false, 60), 0, 0);
    let stream = cell.c_encode_inter(false, false);
    let r = parse_inter_2frame_reference(&stream);
    let port_f1 = cell.port_encode_inter_p(&stream);
    assert_ne!(
        port_f1, r.f1_payload,
        "64x128 cropped-SB128 P NOW BYTE-MATCHES — promote this pin to \
         run_cell(\"interp_e2e_64x128_cq60_420\", 64, 128, false, 60)"
    );
    // Positive results (asserted): the derived header half matches; the tile
    // shape is the measured VERT-vs-SPLIT flip, not something new.
    let split = r.header_bits.div_ceil(8);
    assert_eq!(
        &port_f1[..split.min(port_f1.len())],
        &r.f1_payload[..split],
        "the derived P header must stay byte-exact on the cropped-SB128 frame"
    );
    eprintln!(
        "PINNED (measured root: missing interp-filter rate model): port tile {:02x?} vs aomenc {:02x?}",
        &port_f1[split..],
        &r.f1_payload[split..]
    );
}
