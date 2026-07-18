//! `prune_tx_type_using_stats` (luma-intra tx-type stats prune) end-to-end —
//! the ABSENT-and-UNEXERCISED coverage hole: C enables this sf for ALLINTRA at
//! cpu-used>=2 (level 1) / >=4 (level 2), but ONLY when `is_480p_or_larger`
//! (speed_features.c:262/300). Every other gate frame is sub-480p, so the speed
//! gates never hit it.
//!
//! `tx_stats_prune_knob_bites` is the >=480p exercise+gate: on a 512x512 cpu-2
//! noise frame (IDTX/FLIPADST-family tx types competitive) the prune is
//! LOAD-BEARING — the port WITHOUT it (`disable_tx_stats_prune`) diverges from
//! real aomenc, and WITH it byte-matches. `tx_stats_prune_sub480p_unchanged`
//! guards that sub-480p stays byte-identical (prune off). The prune's byte-exact
//! correctness across every tx_size×mode×config is proven exhaustively in
//! `tx_mask_diff` (vs the C oracle over the REAL `default_tx_type_probs`).

use aom_bench::{EncodeCell, ToggleKnobs};

/// A 512x512 ALLINTRA cell with full-contrast pseudo-random luma — the residual
/// after intra prediction is high-frequency/uncorrelated, which makes IDTX (the
/// identity transform, KF prob 2 < the threshold 10) genuinely competitive. The
/// stats prune removes IDTX, so on this content it is LOAD-BEARING (a pruned
/// type would otherwise win). mono keeps it luma-only (the prune is luma-side).
fn noise_cell(label: &str, w: usize, h: usize, cq: i32, speed: i32, mono: bool) -> EncodeCell {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut y = vec![0u16; w * h];
    for p in y.iter_mut() {
        *p = (next() % 256) as u16;
    }
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + 1) >> 1, (h + 1) >> 1)
    };
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    if !mono {
        for p in u.iter_mut() {
            *p = (next() % 256) as u16;
        }
        for p in v.iter_mut() {
            *p = (next() % 256) as u16;
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono,
        ss_x: 1,
        ss_y: 1,
        usage: 2,
        cq_level: cq,
        speed,
        bd: 8,
        y,
        u,
        v,
    }
}

/// A 512x512 (8x8 SB) ALLINTRA cell with high-frequency directional luma. `mono`
/// selects monochrome (proves the prune is luma-side; the sf is luma-only).
fn textured_cell(label: &str, w: usize, h: usize, cq: i32, speed: i32, mono: bool) -> EncodeCell {
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            // Diagonal sawtooth + cross ripple: sharp 45-degree edges make the
            // ADST/FLIPADST/directional tx types competitive against DCT.
            let saw = (((r + col) % 16) * 10) as i32;
            let rip = (((r * 3) ^ (col * 5)) & 31) as i32;
            y[r * w + col] = (40 + saw + rip).clamp(0, 255) as u16;
        }
    }
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + 1) >> 1, (h + 1) >> 1)
    };
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    if !mono {
        for r in 0..ch {
            for col in 0..cw {
                u[r * cw + col] = (60 + (r * 7 + col * 3) % 80) as u16;
                v[r * cw + col] = (128 + (r * 3 + col * 5) % 60) as u16;
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
        usage: 2,
        cq_level: cq,
        speed,
        bd: 8,
        y,
        u,
        v,
    }
}

fn run(cell: &EncodeCell, knobs: &ToggleKnobs) -> Result<usize, String> {
    let c_stream = cell.c_encode();
    let real = EncodeCell::frame_obu_payload(&c_stream);
    let ours = cell.port_encode_with(&c_stream, knobs);
    if ours == real {
        return Ok(real.len());
    }
    let first = ours
        .iter()
        .zip(real.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(ours.len().min(real.len()));
    Err(format!(
        "first diff at frame-OBU byte {first}; port {} B vs real {} B",
        ours.len(),
        real.len()
    ))
}

/// The >=480p exercise+gate lives in `tx_stats_prune_knob_bites` below (noise
/// mono cpu-2, where the prune both FIRES and byte-matches), plus the exhaustive
/// `tx_mask_diff` differential (the port's `get_tx_mask_intra` matches the C
/// oracle — which reads the REAL `default_tx_type_probs` — across every
/// tx_size×mode×config × prune-level 0/1/2). NOTE: broader noise >=480p cells
/// (4:2:0, and cpu-4) hit PRE-EXISTING content-dependent near-ties unrelated to
/// this luma-only sf — a chroma-mode near-tie on 4:2:0 noise (mi-level) and a
/// winner-mode/tx near-tie at cpu-4 (KB-10/KB-13 class; they diverge WITH OR
/// WITHOUT the prune, since it never touches chroma or the winner-mode pass).
/// They are not gated here to avoid coupling this sf's gate to those residuals.

/// Regression guard: a SUB-480p frame at cpu-used 2 keeps the prune OFF
/// (framesize gate), so it must stay byte-identical (the whole speed-2 envelope
/// is unperturbed by the new framesize wiring).
#[test]
fn tx_stats_prune_sub480p_unchanged() {
    let cell = textured_cell("sub480_128_s2_cq32", 128, 128, 32, 2, false);
    run(&cell, &ToggleKnobs::default())
        .unwrap_or_else(|why| panic!("sub-480p cpu2 must stay byte-identical (prune off): {why}"));
}

/// Anti-vacuity / load-bearing witness: on a >=480p cpu-2 cell where the prune
/// fires, the port WITHOUT it (`disable_tx_stats_prune`) must DIVERGE from real
/// aomenc (which prunes), and WITH it must MATCH. This proves a pruned
/// low-probability tx type would otherwise win in the port — the stats prune is
/// genuinely load-bearing, not a no-op on this content.
#[test]
fn tx_stats_prune_knob_bites() {
    let cell = noise_cell("witness_512_s2_cq32_mono", 512, 512, 32, 2, true);
    let c_stream = cell.c_encode();
    let real = EncodeCell::frame_obu_payload(&c_stream);

    let without = cell.port_encode_with(
        &c_stream,
        &ToggleKnobs {
            disable_tx_stats_prune: true,
            ..Default::default()
        },
    );
    assert_ne!(
        without, real,
        "port WITHOUT the stats prune must diverge from real aomenc (else the prune is a no-op here)"
    );

    let with = cell.port_encode_with(&c_stream, &ToggleKnobs::default());
    assert_eq!(
        with, real,
        "port WITH the stats prune must byte-match real aomenc"
    );
}
