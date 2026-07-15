//! THE REAL-BITSTREAM GATE TEST: decode bitstreams produced by the REAL
//! libaom v3.14.1 encoder (`aom_codec_av1_cx`, the same library+path the
//! aomenc CLI drives) with `decode_frame_obus`, and compare every plane
//! BYTE-IDENTICALLY against the REAL C decoder (`aom_codec_av1_dx`).
//!
//! ENVELOPE (every constraint, honestly):
//! - one shown KEY frame per stream (`g_limit=1`, forced KF) — all-intra.
//! - encoder flags: `--cpu-used=0 --end-usage=q --cq-level=<q>
//!   --enable-cdef={0,1} --enable-restoration={0,1} --sb-size=64
//!   --deltaq-mode=0 --aq-mode={0,1,2} --enable-palette=0 --enable-intrabc=0`
//!   over usage GOOD + ALL_INTRA, one-pass and two-pass.
//! - SEGMENTATION IS IN THE ENVELOPE: the `--aq-mode={1,2} --passes=2` arms
//!   carry 8-segment `SEG_LVL_ALT_Q` segmentation (variance/complexity AQ;
//!   the two-pass recode loop is REQUIRED — one-pass never runs the aq
//!   setup) — per-block segment-id symbols, spatial-pred contexts over the
//!   decoded segment map, and per-segment dequants, all byte-identical.
//!   SEG_LVL_SKIP / SEG_LVL_ALT_LF / segid_preskip streams are NOT
//!   producible by this encoder path (ROI maps are realtime-speed>=7-only
//!   in v3.14.1); those paths are covered by the symbol-level C-diffed
//!   roundtrips in aom-entropy instead.
//! - DEBLOCKING IS IN THE ENVELOPE: whatever loop-filter levels /
//!   sharpness / mode-ref deltas the encoder picks are applied
//!   (aom-loopfilter::frame, C-diffed in lf_apply_diff.rs). The low-cq arms
//!   still come out level 0 (deblock is a gated no-op there — same gate as
//!   the C decoder); the high-cq arms pick NONZERO levels and pin the real
//!   deblock application byte-for-byte. The run asserts both populations
//!   are present (`nonzero_lf` / luma+chroma coverage guards).
//! - CDEF IS IN THE ENVELOPE: the `--enable-cdef=1` arms decode whatever
//!   damping / strength grids / per-SB strength indices the speed-0 CDEF
//!   search picks, applied after deblocking (aom-cdef::frame, C-diffed in
//!   cdef_frame_diff.rs). The run recomposes each cdef stream WITHOUT the
//!   CDEF stage and counts streams whose pixels genuinely changed —
//!   `cdef_applied` floors keep that population from silently vanishing.
//! - single tile, no superres / film grain / screen-content tools / qm /
//!   lossless / 128x128 SBs (decode_frame_obus hard-errors).
//! - sizes include non-multiple-of-SB (96x80) and non-multiple-of-8 (100x76,
//!   the decoder's 8px-aligned mi grid cropped back).
//! - 4:2:0 + 4:4:4 + monochrome + 4:2:2, bit depths 8 and 10. 4:2:2 chroma
//!   deblocking (nonzero U/V filter levels) IS in the envelope and decodes
//!   byte-identical to C — see `deblocked_422_chroma_byte_identical_to_c`
//!   (the past "unportable BLOCK_INVALID OOB" claim was verified false: the
//!   portrait luma sizes that map to BLOCK_INVALID are never chroma-reference
//!   blocks, so the OOB branch is dead for conformant 4:2:2 streams).

use aom_decode::frame::{
    apply_cdef, apply_deblock, apply_restoration, decode_frame_obus, decode_frame_obus_prefilter,
};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Photographic-ish content: smooth gradients + sinusoidal structure + noise
/// (deliberately NOT flat/synthetic-few-colors, which could trip the
/// encoder's screen-content detection — screen-content tools are out of the
/// decode envelope and would hard-error).
fn gen_plane(w: usize, h: usize, bd: i32, seed: u64, chroma: bool) -> Vec<u16> {
    let mut rng = Rng(seed | 1);
    let maxv = (1i64 << bd) - 1;
    let mut p = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let fx = col as f64 / w.max(1) as f64;
            let fy = r as f64 / h.max(1) as f64;
            let base = 0.25 + 0.5 * (0.6 * fx + 0.4 * fy);
            let wave = 0.12 * ((fx * 9.0).sin() * (fy * 7.0).cos());
            let noise = ((rng.next() >> 40) as i64 % 33 - 16) as f64 / maxv as f64;
            let mut v = base + wave + noise * if chroma { 2.0 } else { 4.0 };
            v = v.clamp(0.0, 1.0);
            p[r * w + col] = (v * maxv as f64).round() as u16;
        }
    }
    p
}

struct Cfg {
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss: (i32, i32),
    cq: i32,
    /// `--enable-cdef` for the encode.
    cdef: bool,
    /// `--enable-restoration` for the encode.
    restoration: bool,
    /// `--usage`: 0 = GOOD, 2 = ALL_INTRA (the zenavif/avifenc still mode).
    usage: u32,
    /// `--aq-mode`: 1 = VARIANCE_AQ / 2 = COMPLEXITY_AQ — 8-segment
    /// `SEG_LVL_ALT_Q` SEGMENTATION on intra frames, but only through the
    /// two-pass recode loop (one-pass takes `encode_without_recode`, which
    /// never runs the aq segmentation setup).
    aq: u32,
    /// `--passes=2` (firstpass stats + last pass).
    two_pass: bool,
    /// `--sb-size=128` (`AOM_SUPERBLOCK_SIZE_128X128`) via
    /// `ref_encode_av1_kf_sb128` instead of `--sb-size=64` via
    /// `ref_encode_av1_kf`.
    sb128: bool,
    /// `--tile-columns=<log2>` / `--tile-rows=<log2>` via
    /// `ref_encode_av1_kf_tiles` instead of `ref_encode_av1_kf[_sb128]` — 0,0
    /// (the default for every existing arm) is single-tile, byte-identical
    /// to the non-tiles encode paths (`AV1E_SET_TILE_COLUMNS`/`_ROWS` default
    /// to 0 either way).
    tile_columns_log2: i32,
    tile_rows_log2: i32,
}

/// Facts the sweep asserts floors over.
struct RunFacts {
    len: usize,
    tx_select: bool,
    base_qindex: i32,
    filter_level: [i32; 4],
    /// The stream carries CDEF syntax (the decoder's do_cdef gate held).
    cdef_gated: bool,
    /// CDEF genuinely changed at least one pixel (recomposed without the
    /// CDEF stage and compared).
    cdef_applied: bool,
    /// The stream carries restoration syntax (any plane type non-NONE).
    lr_gated: bool,
    /// Decoded RU populations `(wiener, sgrproj)` summed over planes.
    lr_units: (usize, usize),
    /// Restoration genuinely changed at least one pixel.
    lr_applied: bool,
    /// The stream carries SEGMENTATION (seg.enabled in the frame header).
    seg_enabled: bool,
    /// `av1_calculate_segdata`'s last_active_segid (7 = all 8 segments).
    seg_last_active: i32,
    /// `TileInfoHeader::{cols,rows}` as coded.
    tile_cols: usize,
    tile_rows: usize,
}

/// Reproduce the REAL encoder bytes for a config (deterministic seed from
/// (w,h,bd,cq)). Factored out of `run_config` so gates that need to recompose
/// the decode pipeline from the SAME bytes can't drift from the gated encode.
fn encode_stream(cfg: &Cfg) -> Vec<u8> {
    let (cw, ch) = if cfg.mono {
        (0, 0)
    } else {
        (
            (cfg.w + cfg.ss.0 as usize) >> cfg.ss.0,
            (cfg.h + cfg.ss.1 as usize) >> cfg.ss.1,
        )
    };
    let seed =
        ((cfg.w as u64) << 32) ^ ((cfg.h as u64) << 16) ^ ((cfg.bd as u64) << 8) ^ cfg.cq as u64;
    let y = gen_plane(cfg.w, cfg.h, cfg.bd, seed ^ 0x1111, false);
    let u = gen_plane(cw, ch, cfg.bd, seed ^ 0x2222, true);
    let v = gen_plane(cw, ch, cfg.bd, seed ^ 0x3333, true);

    // REAL encoder bytes (cpu-used=0). tile_columns_log2/tile_rows_log2 != 0
    // routes through the multi-tile oracle entry point (--tile-columns/-rows,
    // sb128 controllable too); else sb128 routes through the SB128 oracle
    // entry point (--sb-size=128); otherwise --sb-size=64 as before.
    if cfg.tile_columns_log2 != 0 || cfg.tile_rows_log2 != 0 {
        c::ref_encode_av1_kf_tiles(
            &y,
            &u,
            &v,
            cfg.w,
            cfg.h,
            cfg.bd,
            cfg.mono,
            cfg.ss.0,
            cfg.ss.1,
            cfg.cq,
            0,
            cfg.cdef,
            cfg.restoration,
            cfg.usage,
            cfg.aq,
            cfg.two_pass,
            cfg.sb128,
            cfg.tile_columns_log2,
            cfg.tile_rows_log2,
        )
    } else if cfg.sb128 {
        c::ref_encode_av1_kf_sb128(
            &y,
            &u,
            &v,
            cfg.w,
            cfg.h,
            cfg.bd,
            cfg.mono,
            cfg.ss.0,
            cfg.ss.1,
            cfg.cq,
            0,
            cfg.cdef,
            cfg.restoration,
            cfg.usage,
            cfg.aq,
            cfg.two_pass,
            true,
        )
    } else {
        c::ref_encode_av1_kf(
            &y,
            &u,
            &v,
            cfg.w,
            cfg.h,
            cfg.bd,
            cfg.mono,
            cfg.ss.0,
            cfg.ss.1,
            cfg.cq,
            0,
            cfg.cdef,
            cfg.restoration,
            cfg.usage,
            cfg.aq,
            cfg.two_pass,
        )
    }
}

fn run_config(cfg: &Cfg) -> RunFacts {
    let bytes = encode_stream(cfg);

    // Rust decode (hard-errors outside the envelope — that FAILS the test).
    let rust = decode_frame_obus(&bytes).unwrap_or_else(|e| {
        panic!(
            "decode_frame_obus rejected {}x{} bd{} mono={} ss={:?} cq={}: {e}",
            cfg.w, cfg.h, cfg.bd, cfg.mono, cfg.ss, cfg.cq
        )
    });

    // Envelope facts verified from OUR OWN parse.
    assert!(rust.base_qindex > 0);
    assert_eq!((rust.width, rust.height), (cfg.w, cfg.h));
    assert_eq!(rust.bit_depth, cfg.bd);
    assert_eq!(rust.monochrome, cfg.mono);
    if !cfg.mono {
        assert_eq!(
            (rust.subsampling_x as i32, rust.subsampling_y as i32),
            cfg.ss
        );
    }

    // Gold oracle: the REAL C decoder on the same bytes.
    let cref = c::ref_decode_av1_kf(&bytes, cfg.w, cfg.h);
    assert_eq!(cref.info[0], cfg.bd);
    assert_eq!(cref.info[1] != 0, cfg.mono);

    assert_eq!(
        rust.y, cref.y,
        "LUMA mismatch {}x{} bd{} ss={:?} cq={}",
        cfg.w, cfg.h, cfg.bd, cfg.ss, cfg.cq
    );
    if cfg.mono {
        assert!(rust.u.is_empty() && rust.v.is_empty());
    } else {
        assert_eq!(
            rust.u, cref.u,
            "U mismatch {}x{} bd{} ss={:?} cq={}",
            cfg.w, cfg.h, cfg.bd, cfg.ss, cfg.cq
        );
        assert_eq!(
            rust.v, cref.v,
            "V mismatch {}x{} bd{} ss={:?} cq={}",
            cfg.w, cfg.h, cfg.bd, cfg.ss, cfg.cq
        );
    }

    // CDEF application detection: recompose the pipeline WITHOUT the CDEF
    // stage and compare the mi-aligned planes. (A stream can carry CDEF
    // syntax whose strengths/skip flags end up changing nothing — the floors
    // below need the genuinely-changed population.)
    let cdef_gated =
        rust.cdef_bits != 0 || rust.cdef_strengths[0] != 0 || rust.cdef_uv_strengths[0] != 0;
    assert!(
        cfg.cdef || !cdef_gated,
        "--enable-cdef=0 stream carries CDEF syntax"
    );
    let cdef_applied = if cdef_gated {
        let (mut t, tcfg, header) = decode_frame_obus_prefilter(&bytes).unwrap();
        if header.loopfilter.filter_level != [0, 0] {
            apply_deblock(&mut t, &tcfg, &header);
        }
        let no_cdef = t.clone();
        apply_cdef(&mut t, &tcfg, &header);
        t.recon != no_cdef.recon || t.recon_u != no_cdef.recon_u || t.recon_v != no_cdef.recon_v
    } else {
        false
    };
    // Restoration application detection: recompose the pipeline WITHOUT the
    // restoration stage and compare (the decoder's exact ordering — deblock,
    // pre-CDEF snapshot, CDEF, restoration — reproduced via the hidden
    // stage entry points).
    let lr_gated = rust.lr_frame_restoration_type.iter().any(|&t| t != 0);
    assert!(
        cfg.restoration || !lr_gated,
        "--enable-restoration=0 stream carries restoration syntax"
    );
    assert!(
        cfg.aq > 0 || !rust.seg_enabled,
        "--aq-mode=0 stream carries segmentation"
    );
    let lr_applied = if lr_gated {
        let (mut t, tcfg, header) = decode_frame_obus_prefilter(&bytes).unwrap();
        if header.loopfilter.filter_level != [0, 0] {
            apply_deblock(&mut t, &tcfg, &header);
        }
        let pre_cdef = cdef_gated.then(|| (t.recon.clone(), t.recon_u.clone(), t.recon_v.clone()));
        if cdef_gated {
            apply_cdef(&mut t, &tcfg, &header);
        }
        let no_lr = t.clone();
        apply_restoration(&mut t, &tcfg, pre_cdef.as_ref(), !cdef_gated);
        t.recon != no_lr.recon || t.recon_u != no_lr.recon_u || t.recon_v != no_lr.recon_v
    } else {
        false
    };
    RunFacts {
        len: bytes.len(),
        tx_select: rust.tx_mode_select,
        base_qindex: rust.base_qindex,
        filter_level: rust.filter_level,
        cdef_gated,
        cdef_applied,
        lr_gated,
        lr_units: (rust.lr_unit_counts.0, rust.lr_unit_counts.1),
        lr_applied,
        seg_enabled: rust.seg_enabled,
        seg_last_active: rust.seg_last_active_segid,
        tile_cols: rust.tile_cols,
        tile_rows: rust.tile_rows,
    }
}

#[test]
fn real_bitstreams_decode_byte_identical_to_c() {
    // cq grid PROBED 2026-07-14 (this content, libaom v3.14.1 build,
    // cpu-used=0). Low-cq arms come out loop-filter level 0 (bd10 picks
    // nonzero from cq>=8; bd8 from around cq>=40 on this content); the added
    // high-cq arms (bd8 cq 44/52, bd10 cq 16/36, plus the previously-excluded
    // (100x76,444) cq 36 which picks [0,2,5,0]) pick NONZERO levels — the
    // deblock application is pinned byte-identically on those. bd8 bands:
    // cq 2 -> q 8 (band 0), 6 -> 24 (band 1), 16 -> 64 (band 2),
    // 28 -> 112 (band 2), 36 -> 144 (band 3).
    let sizes = [(64usize, 64usize), (96, 80), (100, 76)];
    let combos = [
        (8i32, (1i32, 1i32), false),
        (8, (0, 0), false),
        (10, (1, 1), false),
        (10, (0, 0), false),
        (8, (1, 1), true), // monochrome
    ];
    let mut n = 0u32;
    let mut select_seen = 0u32;
    let mut bands = [0u32; 4];
    let mut zero_lf = 0u32;
    let mut nonzero_luma_lf = 0u32;
    let mut nonzero_chroma_lf = 0u32;
    let mut cdef_gated = 0u32;
    let mut cdef_applied = 0u32;
    let mut lr_gated = 0u32;
    let mut lr_applied = 0u32;
    let mut lr_wiener_units = 0usize;
    let mut lr_sgrproj_units = 0usize;
    let mut allintra_arms = 0u32;
    #[allow(clippy::too_many_arguments)]
    let mut seg_gated = 0u32;
    let mut seg_full_alphabet = 0u32;
    let mut seg_deblocked = 0u32;
    let mut seg_cdef = 0u32;
    let mut seg_allintra = 0u32;
    #[allow(clippy::too_many_arguments)]
    let mut run_full = |w: usize,
                        h: usize,
                        bd: i32,
                        ss: (i32, i32),
                        mono: bool,
                        cq: i32,
                        cdef: bool,
                        restoration: bool,
                        usage: u32,
                        aq: u32,
                        two_pass: bool|
     -> RunFacts {
        let f = run_config(&Cfg {
            w,
            h,
            bd,
            mono,
            ss,
            cq,
            cdef,
            restoration,
            usage,
            aq,
            two_pass,
            sb128: false,
            tile_columns_log2: 0,
            tile_rows_log2: 0,
        });
        assert!(f.len > 50, "suspiciously small stream ({} bytes)", f.len);
        select_seen += f.tx_select as u32;
        let q = f.base_qindex;
        bands[if q <= 20 {
            0
        } else if q <= 60 {
            1
        } else if q <= 120 {
            2
        } else {
            3
        }] += 1;
        if f.filter_level == [0; 4] {
            zero_lf += 1;
        }
        if f.filter_level[0] != 0 || f.filter_level[1] != 0 {
            nonzero_luma_lf += 1;
        }
        if f.filter_level[2] != 0 || f.filter_level[3] != 0 {
            nonzero_chroma_lf += 1;
        }
        cdef_gated += f.cdef_gated as u32;
        cdef_applied += f.cdef_applied as u32;
        lr_gated += f.lr_gated as u32;
        lr_applied += f.lr_applied as u32;
        lr_wiener_units += f.lr_units.0;
        lr_sgrproj_units += f.lr_units.1;
        allintra_arms += (usage == 2) as u32;
        seg_gated += f.seg_enabled as u32;
        seg_full_alphabet += (f.seg_enabled && f.seg_last_active == 7) as u32;
        seg_deblocked +=
            (f.seg_enabled && (f.filter_level[0] != 0 || f.filter_level[1] != 0)) as u32;
        seg_cdef += (f.seg_enabled && f.cdef_gated) as u32;
        seg_allintra += (f.seg_enabled && usage == 2) as u32;
        n += 1;
        f
    };
    for &(w, h) in &sizes {
        for &(bd, ss, mono) in &combos {
            for &cq in &[2i32, 6] {
                run_full(w, h, bd, ss, mono, cq, false, false, 0, 0, false);
            }
            if bd == 8 {
                for &cq in &[16i32, 28] {
                    run_full(w, h, bd, ss, mono, cq, false, false, 0, 0, false);
                }
                // Deblocked arms: aggressive q picks nonzero filter levels.
                for &cq in &[44i32, 52] {
                    run_full(w, h, bd, ss, mono, cq, false, false, 0, 0, false);
                }
            } else {
                // bd10 picks nonzero levels from cq>=8 — previously excluded,
                // now the deblocked bd10 arm.
                for &cq in &[16i32, 36] {
                    run_full(w, h, bd, ss, mono, cq, false, false, 0, 0, false);
                }
            }
            // CDEF arms (--enable-cdef=1): the speed-0 CDEF search picks the
            // damping + strength grids; moderate/aggressive q picks NONZERO
            // strengths (probed 2026-07-14: 29/39 arms carry CDEF syntax —
            // ten cq-6 high-quality arms pick all-zero and code none — and
            // every gated arm changes pixels; cq 52 bd8 chains deblock+CDEF).
            for &cq in &[6i32, 36] {
                run_full(w, h, bd, ss, mono, cq, true, false, 0, 0, false);
            }
            if bd == 8 {
                run_full(w, h, bd, ss, mono, 52, true, false, 0, 0, false);
            }
        }
    }
    // Band-3 (q>120) arm at cq 36, incl. the (100x76, 4:4:4) combo that
    // picks [0,2,5,0] (was excluded when deblocking was out of envelope).
    for &(w, h, ss, mono) in &[
        (64usize, 64usize, (1i32, 1i32), false),
        (64, 64, (0, 0), false),
        (64, 64, (1, 1), true),
        (96, 80, (1, 1), false),
        (96, 80, (0, 0), false),
        (96, 80, (1, 1), true),
        (100, 76, (1, 1), false),
        (100, 76, (0, 0), false),
        (100, 76, (1, 1), true),
    ] {
        run_full(w, h, 8, ss, mono, 36, false, false, 0, 0, false);
    }
    // LOOP-RESTORATION arms. Three pipeline shapes, all vs the C decoder:
    //  (a) --enable-restoration=1 --enable-cdef=0: the OPTIMIZED LR path
    //      (no boundary saves; decodeframe.c:5470);
    //  (b) restoration+cdef both on: the boundary-swapped path (deblocked
    //      pre-CDEF rows feed internal stripe boundaries);
    //  (c) usage=2 (ALL_INTRA — the zenavif/avifenc still mode; encoder
    //      defaults CDEF off for allintra but an explicit control overrides,
    //      av1_cx_iface.c:3065) x both shapes — ALLINTRA-headline coverage.
    for &(w, h) in &sizes {
        for &(bd, ss, mono) in &combos {
            for &cq in &[6i32, 36] {
                run_full(w, h, bd, ss, mono, cq, false, true, 0, 0, false); // (a)
            }
            run_full(w, h, bd, ss, mono, 36, true, true, 0, 0, false); // (b)
            for &cq in &[6i32, 36] {
                run_full(w, h, bd, ss, mono, cq, true, true, 2, 0, false); // (c) lr+cdef
            }
            run_full(w, h, bd, ss, mono, 36, false, true, 2, 0, false); // (c) optimized
            if bd == 8 {
                // deblock + CDEF + restoration chained, GOOD and ALLINTRA.
                run_full(w, h, bd, ss, mono, 52, true, true, 0, 0, false);
                run_full(w, h, bd, ss, mono, 52, true, true, 2, 0, false);
            }
        }
    }
    // SEGMENTATION arms (--aq-mode>0 --passes=2). VARIANCE_AQ (1) /
    // COMPLEXITY_AQ (2) enable 8-segment SEG_LVL_ALT_Q segmentation on intra
    // frames — ONLY through the two-pass recode loop (one-pass encodes take
    // encode_without_recode, which never runs the aq setup; PROBED 2026-07-14:
    // 108/108 two-pass aq arms carry seg with last_active_segid=7, one-pass
    // arms never do). Every arm decodes per-block segment-id symbols
    // (spatial-pred coded + spatial prediction on skipped blocks) and
    // per-segment ALT_Q dequants byte-identically vs the C decoder.
    //  - the aq1 grid sweeps GOOD + ALLINTRA at cq {6,36};
    //  - the aq2 slice pins COMPLEXITY_AQ;
    //  - cdef=true arms chain segmentation + CDEF;
    //  - the bd10 cq52 arms chain segmentation + REAL deblocking (bd10 picks
    //    nonzero LF levels there; bd8 seg streams stay level 0).
    for &(w, h) in &sizes {
        for &(bd, ss, mono) in &combos {
            for &cq in &[6i32, 36] {
                run_full(w, h, bd, ss, mono, cq, false, false, 0, 1, true);
                run_full(w, h, bd, ss, mono, cq, false, false, 2, 1, true);
            }
            run_full(w, h, bd, ss, mono, 36, false, false, 0, 2, true);
            run_full(w, h, bd, ss, mono, 36, true, false, 2, 1, true);
            if bd == 10 {
                run_full(w, h, bd, ss, mono, 52, false, false, 0, 1, true);
                run_full(w, h, bd, ss, mono, 52, false, false, 2, 1, true);
            }
        }
    }
    assert_eq!(
        n,
        30 + 18 + 18 + 12 + 9 + 30 + 9 + 90 + 18 + 102,
        "15 combos x cq{{2,6}} + 9 bd8 x cq{{16,28}} + 9 bd8 x cq{{44,52}} + 6 bd10 x cq{{16,36}} \
         + 9 band-3 + CDEF arms (15 combos x cq{{6,36}} + 9 bd8 cq52) \
         + LR arms (15 combos x [2 opt-GOOD + 1 swap-GOOD + 2 allintra-cdef + 1 allintra-opt] \
         + 9 bd8 x cq52 x {{GOOD,ALLINTRA}}) \
         + SEG arms (15 combos x [aq1 cq{{6,36}} x {{GOOD,ALLINTRA}} + aq2 + cdef-allintra] \
         + 6 bd10 x cq52 x {{GOOD,ALLINTRA}})"
    );
    // Speed-0 allintra codes TX_MODE_SELECT — the multi-txb path must be live.
    assert!(select_seen > 0, "no TX_MODE_SELECT stream decoded");
    // All four coefficient-CDF qindex bands exercised on real streams.
    assert!(bands.iter().all(|&b| b > 0), "band coverage {bands:?}");
    // BOTH loop-filter populations must be present: level-0 streams (the
    // application is a gated no-op — the original 56-stream envelope stays
    // green) and genuinely deblocked streams, luma AND chroma.
    println!(
        "lf coverage: zero={zero_lf} nonzero_luma={nonzero_luma_lf} nonzero_chroma={nonzero_chroma_lf} of {n}"
    );
    println!("cdef coverage: gated={cdef_gated} applied={cdef_applied} of {n}");
    // Observed on this deterministic content (2026-07-14 probe): 76+ zero,
    // 11+ nonzero-luma of which 6+ nonzero-chroma — the floors keep both
    // populations from silently vanishing. (The CDEF arms add more nonzero-lf
    // streams at cq 36/52.)
    assert!(zero_lf >= 40, "level-0 population collapsed ({zero_lf})");
    assert!(
        nonzero_luma_lf >= 10,
        "deblocked-luma population too small ({nonzero_luma_lf})"
    );
    assert!(
        nonzero_chroma_lf >= 5,
        "deblocked-chroma population too small ({nonzero_chroma_lf})"
    );
    // CDEF floors (probed 2026-07-14 on this content: 29 of the 39
    // --enable-cdef=1 arms carry CDEF syntax — the speed-0 search picks
    // all-zero strengths on ten of the cq-6 high-quality arms and then
    // writes none — and all 29 gated streams genuinely change pixels).
    assert!(
        cdef_gated >= 25,
        "CDEF-gated population too small ({cdef_gated})"
    );
    assert!(
        cdef_applied >= 20,
        "CDEF-applied population too small ({cdef_applied})"
    );
    // Restoration floors (PROBED 2026-07-14 on this content, cpu-used=0:
    // 71 of the 108 restoration arms carry LR syntax — the speed-0 search
    // picks all-NONE on the rest — ALL 71 genuinely change pixels, and the
    // decoded RU populations are 44 wiener + 85 sgrproj): the sweep must
    // keep streams that CARRY restoration syntax, BOTH RU kernel families,
    // and genuinely-pixel-changing restoration from silently vanishing.
    println!(
        "lr coverage: gated={lr_gated} applied={lr_applied} wiener_units={lr_wiener_units} \
         sgrproj_units={lr_sgrproj_units} allintra_arms={allintra_arms} of {n}"
    );
    assert!(lr_gated >= 55, "LR-gated population too small ({lr_gated})");
    assert!(
        lr_applied >= 55,
        "LR-applied population too small ({lr_applied})"
    );
    assert!(
        lr_wiener_units >= 30 && lr_sgrproj_units >= 60,
        "RU kernel populations too small (wiener={lr_wiener_units} sgrproj={lr_sgrproj_units})"
    );
    // SEGMENTATION floors (PROBED 2026-07-14 on this content: every two-pass
    // aq arm carries segmentation with the full 8-segment alphabet; the bd10
    // cq52 seg arms pick nonzero deblock levels; the cdef seg arms carry CDEF
    // syntax at cq36). Byte-identity of every segmented stream is asserted
    // inside run_config like all other arms.
    println!(
        "seg coverage: gated={seg_gated} full_alphabet={seg_full_alphabet} \
         deblocked={seg_deblocked} cdef={seg_cdef} allintra={seg_allintra} of {n}"
    );
    assert!(
        seg_gated >= 95,
        "segmented population too small ({seg_gated})"
    );
    // VARIANCE_AQ arms carry the full 8-segment alphabet (last_active 7);
    // the 15 COMPLEXITY_AQ arms use a SHORTER alphabet (av1_setup_in_frame_q_adj
    // enables fewer segments) — both alphabet shapes must stay present.
    assert!(
        seg_full_alphabet >= 80,
        "full-alphabet (8-segment) population too small ({seg_full_alphabet})"
    );
    assert!(
        seg_gated > seg_full_alphabet,
        "short-alphabet (COMPLEXITY_AQ) population vanished"
    );
    assert!(
        seg_deblocked >= 8,
        "segmentation+deblock population too small ({seg_deblocked})"
    );
    assert!(
        seg_cdef >= 8,
        "segmentation+CDEF population too small ({seg_cdef})"
    );
    assert!(
        seg_allintra >= 40,
        "ALL_INTRA segmented population too small ({seg_allintra})"
    );
    assert_eq!(allintra_arms, 54 + 51, "ALL_INTRA arm count");
}

#[test]
fn sb128_streams_decode_byte_identical_to_c() {
    // SB128 (`--sb-size=128`, `AOM_SUPERBLOCK_SIZE_128X128`) gate: real
    // libaom-encoded streams with 128x128 superblocks decode byte-identical
    // to the C decoder. Exercises the genuinely-SB128-specific code paths
    // (everything else in the driver is size-generic, verified against the
    // C reference by direct source comparison): the partition tree rooting
    // at BLOCK_128X128 (has_top_right/has_bottom_left's >64x64-block special
    // case, partition_cdf_length's HORZ_4/VERT_4 exclusion at 128x128), the
    // per-64x64-unit CDEF `cdef_transmitted[4]` 4-way index
    // (`cdef_unit_col + 2*cdef_unit_row`, only live when sb_size==128 — a
    // 64 SB always indexes 0), and loop-restoration's corners-in-sb SB
    // extent at `mib_size=32` (hardcoded 16,16 before this chunk). ALLINTRA
    // (usage=2) is the primary arm per the zenavif/avifenc still-image
    // product path; GOOD (usage=0) is retained as a smaller cross-check.
    //
    // Sizes: 128x128 (one exactly-fitting SB — clean root recursion),
    // 192x160 and 256x224 (SB grids >1x1 on at least one axis, the latter
    // 2x2 with a partial bottom row — exercises the tile SB row/col walk
    // stepping by mib_size=32 across multiple superblocks), 100x76 (smaller
    // than one SB — the whole frame is a single clipped root node forcing
    // early SPLITs; reuses the main gate's non-multiple-of-8 size).
    let sizes = [(128usize, 128usize), (192, 160), (256, 224), (100, 76)];
    let combos = [
        (8i32, (1i32, 1i32), false),
        (8, (0, 0), false),
        (10, (1, 1), false),
        (10, (0, 0), false),
        (8, (1, 1), true), // monochrome
    ];
    let mut n = 0u32;
    let mut allintra_arms = 0u32;
    let mut good_arms = 0u32;
    let mut select_seen = 0u32;
    let mut bands = [0u32; 4];
    let mut nonzero_lf = 0u32;
    let mut cdef_gated = 0u32;
    let mut cdef_applied = 0u32;
    let mut lr_gated = 0u32;
    let mut lr_applied = 0u32;
    let mut multi_sb_arms = 0u32; // sizes spanning >1 SB on either axis
    #[allow(clippy::too_many_arguments)]
    let mut run = |w: usize,
                   h: usize,
                   bd: i32,
                   ss: (i32, i32),
                   mono: bool,
                   cq: i32,
                   cdef: bool,
                   restoration: bool,
                   usage: u32| {
        let f = run_config(&Cfg {
            w,
            h,
            bd,
            mono,
            ss,
            cq,
            cdef,
            restoration,
            usage,
            aq: 0,
            two_pass: false,
            sb128: true,
            tile_columns_log2: 0,
            tile_rows_log2: 0,
        });
        assert!(
            f.len > 50,
            "suspiciously small SB128 stream ({} bytes)",
            f.len
        );
        n += 1;
        allintra_arms += (usage == 2) as u32;
        good_arms += (usage == 0) as u32;
        select_seen += f.tx_select as u32;
        let q = f.base_qindex;
        bands[if q <= 20 {
            0
        } else if q <= 60 {
            1
        } else if q <= 120 {
            2
        } else {
            3
        }] += 1;
        nonzero_lf += (f.filter_level != [0; 4]) as u32;
        cdef_gated += f.cdef_gated as u32;
        cdef_applied += f.cdef_applied as u32;
        lr_gated += f.lr_gated as u32;
        lr_applied += f.lr_applied as u32;
        if w > 128 || h > 128 {
            multi_sb_arms += 1;
        }
    };
    for &(w, h) in &sizes {
        for &(bd, ss, mono) in &combos {
            // ALLINTRA baseline: cq {2,6,36} PROBED 2026-07-14 on this
            // content/sizes to land in qindex bands 0 (cq2), 1 (cq6), 3
            // (cq36); cq 20 targets band 2 (61-120) — an initial {6,36}-only
            // grid landed 20/120 arms in band 1 and 100/120 in band 3, zero
            // in bands 0/2, so this widens the low end + adds a mid point.
            for &cq in &[2i32, 6, 20, 36] {
                run(w, h, bd, ss, mono, cq, false, false, 2);
            }
            // ALLINTRA + CDEF: exercises cdef_transmitted[4]'s 4-way index.
            run(w, h, bd, ss, mono, 36, true, false, 2);
            // ALLINTRA + restoration: exercises lr_corners_in_sb at mib=32.
            run(w, h, bd, ss, mono, 36, false, true, 2);
            // ALLINTRA + both, chained (matches the main gate's LR+CDEF arms).
            run(w, h, bd, ss, mono, 36, true, true, 2);
            // GOOD, retained as a smaller cross-check.
            run(w, h, bd, ss, mono, 36, false, false, 0);
        }
    }
    assert_eq!(n, 4 * 5 * 8, "SB128 arm count");
    println!(
        "SB128 coverage: n={n} allintra={allintra_arms} good={good_arms} \
         multi_sb={multi_sb_arms} bands={bands:?} select_seen={select_seen} \
         nonzero_lf={nonzero_lf} cdef_gated={cdef_gated} cdef_applied={cdef_applied} \
         lr_gated={lr_gated} lr_applied={lr_applied}"
    );
    assert_eq!(allintra_arms, 4 * 5 * 7, "ALLINTRA SB128 arm count");
    assert_eq!(good_arms, 4 * 5, "GOOD SB128 arm count");
    assert_eq!(multi_sb_arms, 2 * 5 * 8, "multi-SB (>1 128 SB) arm count");
    assert!(select_seen > 0, "no TX_MODE_SELECT SB128 stream decoded");
    assert!(
        bands.iter().all(|&b| b > 0),
        "SB128 band coverage {bands:?}"
    );
    assert!(
        cdef_gated >= 1,
        "SB128 CDEF-gated population empty ({cdef_gated})"
    );
    assert!(
        cdef_applied >= 1,
        "SB128 CDEF-applied population empty ({cdef_applied})"
    );
    assert!(
        lr_gated >= 1,
        "SB128 LR-gated population empty ({lr_gated})"
    );
    assert!(
        lr_applied >= 1,
        "SB128 LR-applied population empty ({lr_applied})"
    );
}

#[test]
fn multi_tile_streams_decode_byte_identical_to_c() {
    // MULTI-TILE (`--tile-columns=<log2> --tile-rows=<log2>`) gate: real
    // libaom-encoded streams with more than one tile decode byte-identical
    // to the C decoder. Drops the single-tile restriction in `frame.rs` and
    // exercises the multi-tile-specific code paths: per-tile context resets
    // (`TileKf::start_tile` — above/left/CfL/LR-refs/delta-lf-carry/
    // current_base_qindex all restart at each tile, not just once per
    // frame), TILE-relative `up_available`/`left_available` (a block at a
    // tile's own top/left edge has no available neighbour even when the
    // tile sits interior to the frame — `TileKf::neighbours` /
    // `decode_block`'s up_uv/left_uv / `intra_avail`'s tile_col_end/
    // tile_row_end all had to switch from frame-relative `> 0` to
    // tile-relative `> tile.mi_row/col_start`), the `tile_size_bytes`-
    // prefixed per-tile length parsing (`split_tiles`, mirroring
    // `get_tile_buffers`/`get_tile_buffer`), and each tile's FRESH
    // `KfFrameContext` (CDF adaptation does not carry across tiles). Sizes:
    // 256x256 (SB-exact 4x4 grid @ 64px — every tile a full superblock) and
    // 200x152 (non-SB-aligned — the last tile row/col in each axis is a
    // genuinely partial superblock, `AOMMIN`-clamped to the frame edge).
    // ALLINTRA (usage=2) is the primary arm per the zenavif/avifenc
    // still-image product path; GOOD (usage=0) is retained as a smaller
    // cross-check. Column-only and row-only tiling (one axis > 1, the other
    // exactly 1) are swept alongside full 2D tiling (both axes > 1
    // together). A dedicated 4-tile-column stress (256x64, SB-exact) and an
    // SB128 + multi-tile cross-check (384x384, mib_size=32) close out the
    // sweep — the tile-bounds derivation reads `mib_size_log2` off the
    // parsed header, not a hardcoded 64x64 assumption.
    let sizes = [(256usize, 256usize), (200, 152)];
    let combos = [
        (8i32, (1i32, 1i32), false),
        (8, (0, 0), false),
        (10, (1, 1), false),
        (10, (0, 0), false),
        (8, (1, 1), true), // monochrome
    ];
    let mut n = 0u32;
    let mut allintra_arms = 0u32;
    let mut good_arms = 0u32;
    let mut select_seen = 0u32;
    let mut bands = [0u32; 4];
    let mut nonzero_lf = 0u32;
    let mut cdef_gated = 0u32;
    let mut cdef_applied = 0u32;
    let mut lr_gated = 0u32;
    let mut lr_applied = 0u32;
    let mut seg_gated = 0u32;
    let mut multi_row_arms = 0u32; // tile_rows > 1
    let mut multi_col_arms = 0u32; // tile_cols > 1
    let mut both_axes_arms = 0u32; // tile_rows > 1 AND tile_cols > 1
    let mut sb128_tiles_arms = 0u32;
    #[allow(clippy::too_many_arguments)]
    let mut run = |w: usize,
                   h: usize,
                   bd: i32,
                   ss: (i32, i32),
                   mono: bool,
                   cq: i32,
                   cdef: bool,
                   restoration: bool,
                   usage: u32,
                   aq: u32,
                   two_pass: bool,
                   sb128: bool,
                   tcl: i32,
                   trl: i32| {
        let f = run_config(&Cfg {
            w,
            h,
            bd,
            mono,
            ss,
            cq,
            cdef,
            restoration,
            usage,
            aq,
            two_pass,
            sb128,
            tile_columns_log2: tcl,
            tile_rows_log2: trl,
        });
        assert!(
            f.len > 50,
            "suspiciously small multi-tile stream ({} bytes)",
            f.len
        );
        assert!(
            f.tile_cols > 1 || f.tile_rows > 1,
            "tile_columns_log2={tcl} tile_rows_log2={trl} at {w}x{h} produced only \
             1x1 tiles (size too small for this tile grid -- widen the fixture)"
        );
        n += 1;
        allintra_arms += (usage == 2) as u32;
        good_arms += (usage == 0) as u32;
        select_seen += f.tx_select as u32;
        let q = f.base_qindex;
        bands[if q <= 20 {
            0
        } else if q <= 60 {
            1
        } else if q <= 120 {
            2
        } else {
            3
        }] += 1;
        nonzero_lf += (f.filter_level != [0; 4]) as u32;
        cdef_gated += f.cdef_gated as u32;
        cdef_applied += f.cdef_applied as u32;
        lr_gated += f.lr_gated as u32;
        lr_applied += f.lr_applied as u32;
        seg_gated += f.seg_enabled as u32;
        multi_row_arms += (f.tile_rows > 1) as u32;
        multi_col_arms += (f.tile_cols > 1) as u32;
        both_axes_arms += (f.tile_rows > 1 && f.tile_cols > 1) as u32;
        sb128_tiles_arms += sb128 as u32;
    };
    for &(w, h) in &sizes {
        for &(bd, ss, mono) in &combos {
            // ALLINTRA baseline: full 2D tiling (both axes > 1).
            for &cq in &[2i32, 6, 20, 36] {
                run(
                    w, h, bd, ss, mono, cq, false, false, 2, 0, false, false, 1, 1,
                );
            }
            // ALLINTRA + CDEF / restoration / both, chained with 2D tiling.
            run(
                w, h, bd, ss, mono, 36, true, false, 2, 0, false, false, 1, 1,
            );
            run(
                w, h, bd, ss, mono, 36, false, true, 2, 0, false, false, 1, 1,
            );
            run(w, h, bd, ss, mono, 36, true, true, 2, 0, false, false, 1, 1);
            // Column-only and row-only tiling (each axis independently).
            run(
                w, h, bd, ss, mono, 36, false, false, 2, 0, false, false, 1, 0,
            );
            run(
                w, h, bd, ss, mono, 36, false, false, 2, 0, false, false, 0, 1,
            );
            // GOOD usage, retained as a smaller cross-check.
            run(
                w, h, bd, ss, mono, 36, false, false, 0, 0, false, false, 1, 1,
            );
            // SEGMENTATION (--aq-mode>0 --passes=2), chained with tiling.
            run(
                w, h, bd, ss, mono, 36, false, false, 2, 1, true, false, 1, 1,
            );
        }
    }
    // 4-way tile-column stress (SB-exact fit avoids a partial last tile):
    // 256px wide / 64px SB = 4 SB columns, tile_columns_log2=2 -> 4 tiles.
    for &(bd, ss, mono) in &combos {
        run(
            256, 64, bd, ss, mono, 6, false, false, 2, 0, false, false, 2, 0,
        );
    }
    // SB128 + multi-tile cross-check: the tile-bounds derivation must read
    // mib_size_log2 (32 mi/SB at sb128) off the parsed header, not assume
    // 64x64. 384x384 = 3x3 SBs @ 128px.
    for &(bd, ss, mono) in &combos {
        run(
            384, 384, bd, ss, mono, 6, false, false, 2, 0, false, true, 1, 1,
        );
    }
    assert_eq!(n, 2 * 5 * 11 + 5 + 5, "multi-tile arm count");
    println!(
        "multi-tile coverage: n={n} allintra={allintra_arms} good={good_arms} \
         multi_row={multi_row_arms} multi_col={multi_col_arms} both_axes={both_axes_arms} \
         sb128_tiles={sb128_tiles_arms} bands={bands:?} select_seen={select_seen} \
         nonzero_lf={nonzero_lf} cdef_gated={cdef_gated} cdef_applied={cdef_applied} \
         lr_gated={lr_gated} lr_applied={lr_applied} seg_gated={seg_gated}"
    );
    assert!(
        select_seen > 0,
        "no TX_MODE_SELECT multi-tile stream decoded"
    );
    assert!(
        bands.iter().all(|&b| b > 0),
        "multi-tile band coverage {bands:?}"
    );
    assert!(multi_row_arms >= 1, "no tile_rows>1 stream decoded");
    assert!(multi_col_arms >= 1, "no tile_cols>1 stream decoded");
    assert!(
        both_axes_arms >= 1,
        "no BOTH-axes (2D tiling) stream decoded"
    );
    assert!(sb128_tiles_arms >= 1, "no SB128+multi-tile stream decoded");
    assert!(cdef_gated >= 1, "multi-tile CDEF-gated population empty");
    assert!(
        cdef_applied >= 1,
        "multi-tile CDEF-applied population empty"
    );
    assert!(lr_gated >= 1, "multi-tile LR-gated population empty");
    assert!(lr_applied >= 1, "multi-tile LR-applied population empty");
    assert!(seg_gated >= 1, "multi-tile segmentation population empty");
    assert_eq!(
        allintra_arms,
        2 * 5 * 10 + 5 + 5,
        "ALLINTRA multi-tile arm count"
    );
    assert_eq!(good_arms, 2 * 5, "GOOD multi-tile arm count");
}

#[test]
fn high_q_deblocked_stream_decodes_byte_identical() {
    // The old envelope boundary, now INSIDE the envelope: this config
    // (100x76 4:2:0 cq 52 on the deterministic run_config content) picks
    // filter_level [4, 0, 0, 6] — a genuinely deblocked stream (luma vert +
    // chroma V) whose output must match the C decoder byte-for-byte.
    // (64x64 cq 60 — the old rejection-test config — actually picks level 0
    // on its content; its old Err-arm never fired.)
    let f = run_config(&Cfg {
        w: 100,
        h: 76,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 52,
        cdef: false,
        restoration: false,
        usage: 0,
        aq: 0,
        two_pass: false,
        sb128: false,
        tile_columns_log2: 0,
        tile_rows_log2: 0,
    });
    println!(
        "deblocked companion: {} bytes, filter_level = {:?}",
        f.len, f.filter_level
    );
    // This companion pins REAL deblocking: if the encoder ever stops picking
    // nonzero levels here the assertion below flags the lost coverage.
    assert_ne!(f.filter_level, [0; 4], "cq 52 no longer picks deblocking");
}

#[test]
fn deblocked_422_chroma_nonzero_levels_byte_identical() {
    // The exact config a past session flagged as the "rejection arm"
    // (96x80 bd10 cq44, chroma levels u=16 v=17 — both NONZERO). It is NOT
    // out of envelope: it decodes BYTE-IDENTICAL to the C decoder. This pins
    // the single stream the old rejection was built around; the broad
    // coverage lives in `deblocked_422_chroma_byte_identical_to_c`.
    let cfg = Cfg {
        w: 96,
        h: 80,
        bd: 10,
        mono: false,
        ss: (1, 0),
        cq: 44,
        cdef: false,
        restoration: false,
        usage: 0,
        aq: 0,
        two_pass: false,
        sb128: false,
        tile_columns_log2: 0,
        tile_rows_log2: 0,
    };
    // run_config asserts full-frame Y/U/V byte-identity vs the C decoder.
    let f = run_config(&cfg);
    println!(
        "422 nonzero-chroma arm: filter_level = {:?}",
        f.filter_level
    );
    assert!(
        f.filter_level[2] != 0 && f.filter_level[3] != 0,
        "expected NONZERO U and V chroma deblock levels on this arm, got {:?}",
        f.filter_level
    );
}

// ---- 4:2:2 CHROMA DEBLOCK GATE ------------------------------------------
//
// 4:2:2 (subsampling_x=1, subsampling_y=0) chroma deblocking decodes
// BYTE-IDENTICAL to the REAL C decoder. Past sessions rejected this
// combination believing libaom's chroma path reads
// `max_txsize_rect_lookup[BLOCK_INVALID]` out of bounds "for tall blocks" —
// VERIFIED FALSE. `av1_ss_size_lookup[bsize][1][0]` does mark the portrait
// (height>width) luma sizes BLOCK_INVALID, but those bsizes are never
// chroma-reference blocks in a conformant 4:2:2 stream, so the chroma loop
// filter never queries them (measured: zero OOB hits across a wide
// encode+decode sweep of the real libaom decoder). The wide/square chroma
// tx sizes that DO occur deblock byte-exact.
//
// This gate is deliberately ANTI-VACUOUS. For each of a grid of real 4:2:2
// KEY streams (8- and 10-bit, several sizes INCLUDING non-8-multiple widths
// so the odd chroma half-width edge is exercised, cq chosen so the encoder
// picks nonzero chroma filter levels) it asserts, via `run_config`, that the
// full decode is byte-identical to C on Y/U/V. Then it independently proves
// the chroma deblock stage is (a) ACTIVE — nonzero U/V filter levels — and
// (b) EFFECTIVE — recomposing the decode WITHOUT the deblock stage yields
// DIFFERENT chroma pixels — and that the chroma actually carries AC content
// (non-flat plane). Floors require >= 8 such streams across both bit depths.
#[test]
fn deblocked_422_chroma_byte_identical_to_c() {
    // (bd, w, h, cq). Widths 130 and 100 are non-8-multiple -> chroma widths
    // 65 (ODD) and 50, exercising the half-width chroma edge stepping. cq
    // chosen from the probed levels: bd10 reaches nonzero chroma levels from
    // ~cq32 up; bd8 needs the high-cq arms (~cq56-60).
    let grid: &[(i32, usize, usize, i32)] = &[
        // 10-bit — chroma deblock active across most of the range.
        (10, 96, 80, 40),
        (10, 96, 80, 56),
        (10, 128, 128, 44),
        (10, 128, 128, 60),
        (10, 130, 96, 44), // non-8-multiple width (chroma 65, odd)
        (10, 130, 96, 56),
        (10, 100, 76, 44), // non-8-multiple width (chroma 50)
        (10, 100, 76, 60),
        (10, 256, 192, 52),
        (10, 66, 64, 48), // small + non-8-multiple width (chroma 33, odd)
        // 8-bit — needs the high-cq arms for nonzero chroma levels.
        (8, 96, 80, 60),
        (8, 128, 128, 60),
        (8, 100, 76, 60), // non-8-multiple width
        (8, 256, 192, 60),
        // 12-bit — closes the 12-bit 4:2:2 chroma-deblock gap (seq profile 2).
        (12, 96, 80, 44),
        (12, 96, 80, 60),
        (12, 128, 128, 52),
        (12, 130, 96, 56), // non-8-multiple width (chroma 65, odd)
        (12, 100, 76, 60), // non-8-multiple width (chroma 50)
        (12, 256, 192, 52),
    ];

    let mut chroma_active = 0u32;
    let mut chroma_changed = 0u32;
    let mut bd8_active = 0u32;
    let mut bd10_active = 0u32;
    let mut bd12_active = 0u32;
    let mut odd_halfwidth_active = 0u32;

    for &(bd, w, h, cq) in grid {
        let cfg = Cfg {
            w,
            h,
            bd,
            mono: false,
            ss: (1, 0),
            cq,
            cdef: false,
            restoration: false,
            usage: 0,
            aq: 0,
            two_pass: false,
            sb128: false,
            tile_columns_log2: 0,
            tile_rows_log2: 0,
        };
        // run_config asserts subsampling == (1,0) AND full-frame Y/U/V
        // byte-identity vs the C decoder (with cdef/restoration off, the
        // chroma difference vs undeblocked is purely reconstruction+deblock).
        let f = run_config(&cfg);

        let u_active = f.filter_level[2] != 0;
        let v_active = f.filter_level[3] != 0;
        if u_active || v_active {
            chroma_active += 1;
            match bd {
                8 => bd8_active += 1,
                10 => bd10_active += 1,
                _ => bd12_active += 1,
            }
            // ODD chroma half-width (non-8-multiple, w/2 odd) exercised.
            if (w >> 1) & 1 == 1 {
                odd_halfwidth_active += 1;
            }
        }

        // EFFECTIVENESS + AC-content: recompose without the deblock stage and
        // confirm the chroma pixels genuinely differ, and that the chroma
        // plane is non-flat (real AC content, not a trivial constant field).
        let (mut t, tcfg, header) = decode_frame_obus_prefilter(&encode_stream(&cfg)).unwrap();
        assert_eq!(
            (tcfg.subsampling_x, tcfg.subsampling_y),
            (1, 0),
            "gate must be 4:2:2"
        );
        let pre_u = t.recon_u.clone();
        let pre_v = t.recon_v.clone();
        let distinct_u = {
            let mut s = pre_u.clone();
            s.sort_unstable();
            s.dedup();
            s.len()
        };
        assert!(
            distinct_u > 8,
            "chroma U flat (no AC content) {w}x{h} bd{bd} cq{cq}: {distinct_u} distinct"
        );
        if header.loopfilter.filter_level != [0, 0] {
            apply_deblock(&mut t, &tcfg, &header);
        }
        let this_changed = t.recon_u != pre_u || t.recon_v != pre_v;
        if this_changed {
            chroma_changed += 1;
        }
        println!(
            "422gate {w}x{h} bd{bd} cq{cq}: lvl={:?} u_act={u_active} v_act={v_active} \
             chroma_pixels_changed={this_changed} distinct_u={distinct_u}",
            f.filter_level
        );
    }

    println!(
        "422gate: chroma_active={chroma_active} chroma_changed={chroma_changed} \
         bd8_active={bd8_active} bd10_active={bd10_active} bd12_active={bd12_active} \
         odd_halfwidth_active={odd_halfwidth_active}"
    );
    // ANTI-VACUOUS floors: the byte-identity above is only meaningful if
    // chroma deblock actually fired on a real population of streams.
    assert!(
        chroma_active >= 8,
        "expected >=8 chroma-deblock-active 4:2:2 streams, got {chroma_active}"
    );
    assert!(
        chroma_changed >= 8,
        "expected >=8 streams whose chroma pixels the deblock actually changed, got {chroma_changed}"
    );
    assert!(bd8_active >= 1, "no 8-bit 4:2:2 chroma-deblock stream");
    assert!(bd10_active >= 1, "no 10-bit 4:2:2 chroma-deblock stream");
    assert!(bd12_active >= 1, "no 12-bit 4:2:2 chroma-deblock stream");
    assert!(
        odd_halfwidth_active >= 1,
        "no chroma-deblock stream with an ODD chroma half-width (non-8-mult width)"
    );
}

// ---- 12-BIT COMPOSITION GATE: deblock + CDEF + LR at bit depth 12 ----
//
// The main `real_bitstreams_decode_byte_identical_to_c` sweep runs bit depths
// 8 and 10 only; 12-bit (seq profile 2) rides along in the QM/palette/4:2:2
// gates but its POST-FILTER pipeline is not floored there — the 4:2:2 gate
// runs cdef/lr OFF, and the single bd12 palette arm (4:2:0) doesn't assert the
// filters actually fired. Filter kernels are bit-depth-sensitive (deblock
// thresholds and CDEF clamps scale with bd; the WHT/inverse-transform ranges
// widen), so a 12-bit-only filter bug would slip both. This gate closes that:
// real cpu-used=0 KEY streams at bd12 across 4:2:0 + 4:4:4 + monochrome, with
// `--enable-cdef=1 --enable-restoration=1`, decoded BYTE-IDENTICAL to the REAL
// C decoder (via `run_config`) — then anti-vacuous floors prove deblock, CDEF,
// and loop restoration each genuinely fired (and changed pixels) at bd12.
#[test]
fn bd12_composition_decodes_byte_identical_to_c() {
    let sizes = [(64usize, 64usize), (96, 80), (100, 76)];
    // (ss, mono): 4:2:0, 4:4:4, monochrome — the three chroma layouts the
    // bd8/bd10 main sweep floors but bd12 did not.
    let combos = [((1i32, 1i32), false), ((0, 0), false), ((1, 1), true)];
    let mut n = 0u32;
    let mut max_len = 0usize;
    let mut ss420 = 0u32;
    let mut ss444 = 0u32;
    let mut mono_arms = 0u32;
    let mut nonzero_luma_lf = 0u32;
    let mut nonzero_chroma_lf = 0u32;
    let mut cdef_gated = 0u32;
    let mut cdef_applied = 0u32;
    let mut lr_gated = 0u32;
    let mut lr_applied = 0u32;
    let mut lr_wiener = 0usize;
    let mut lr_sgrproj = 0usize;
    for &(w, h) in &sizes {
        for &(ss, mono) in &combos {
            // cq grid: bd12 (like bd10) picks nonzero deblock levels from low
            // cq; the speed-0 CDEF + restoration searches pick real strengths /
            // RU kernels at the moderate/high-cq arms. cdef+lr both enabled.
            for &cq in &[16i32, 36, 52] {
                let f = run_config(&Cfg {
                    w,
                    h,
                    bd: 12,
                    mono,
                    ss,
                    cq,
                    cdef: true,
                    restoration: true,
                    usage: 0,
                    aq: 0,
                    two_pass: false,
                    sb128: false,
                    tile_columns_log2: 0,
                    tile_rows_log2: 0,
                });
                // No per-arm size floor: a small-but-byte-identical stream is
                // valid coverage (aggressive-q monochrome legitimately codes to
                // ~30-40 bytes). Track the max to catch a degenerate all-empty
                // regression instead.
                max_len = max_len.max(f.len);
                n += 1;
                ss420 += (ss == (1, 1) && !mono) as u32;
                ss444 += (ss == (0, 0)) as u32;
                mono_arms += mono as u32;
                nonzero_luma_lf += (f.filter_level[0] != 0 || f.filter_level[1] != 0) as u32;
                nonzero_chroma_lf += (f.filter_level[2] != 0 || f.filter_level[3] != 0) as u32;
                cdef_gated += f.cdef_gated as u32;
                cdef_applied += f.cdef_applied as u32;
                lr_gated += f.lr_gated as u32;
                lr_applied += f.lr_applied as u32;
                lr_wiener += f.lr_units.0;
                lr_sgrproj += f.lr_units.1;
            }
        }
    }
    assert_eq!(n, (sizes.len() * combos.len() * 3) as u32, "bd12 arm count");
    println!(
        "bd12 composition: n={n} max_len={max_len} ss420={ss420} ss444={ss444} mono={mono_arms} \
         nonzero_luma_lf={nonzero_luma_lf} nonzero_chroma_lf={nonzero_chroma_lf} \
         cdef_gated={cdef_gated} cdef_applied={cdef_applied} \
         lr_gated={lr_gated} lr_applied={lr_applied} wiener={lr_wiener} sgrproj={lr_sgrproj}"
    );
    // All three chroma layouts exercised at bd12.
    assert!(
        ss420 > 0 && ss444 > 0 && mono_arms > 0,
        "bd12 layout coverage gap"
    );
    // A degenerate all-empty encode would give tiny streams everywhere; a real
    // content-bearing arm must exist (PROBED 2026-07-15: max 324 bytes).
    assert!(
        max_len >= 150,
        "bd12 streams all suspiciously small (max {max_len})"
    );
    // Anti-vacuous floors (PROBED 2026-07-15, cpu-used=0 on this content): the
    // byte-identity vs C in run_config is only meaningful if the filters ran.
    // Observed: 22 nonzero-luma-LF, 16 nonzero-chroma-LF, 11 CDEF gated+applied,
    // 2 LR gated+applied (both sgrproj). Floors keep each population from
    // silently collapsing (LR is sparse at bd12 on this content — >=1 is the
    // honest guarantee; the palette bd12 arm adds more LR coverage).
    assert!(
        nonzero_luma_lf >= 10,
        "bd12 luma-deblock population collapsed ({nonzero_luma_lf})"
    );
    assert!(
        nonzero_chroma_lf >= 5,
        "bd12 chroma-deblock population collapsed ({nonzero_chroma_lf})"
    );
    assert!(
        cdef_gated >= 6,
        "bd12 CDEF-gated population too small ({cdef_gated})"
    );
    assert!(
        cdef_applied >= 6,
        "bd12 CDEF-applied population too small ({cdef_applied})"
    );
    assert!(lr_gated >= 1, "no bd12 stream carried LR syntax");
    assert!(lr_applied >= 1, "no bd12 stream had LR change pixels");
}

#[test]
fn cdef_stream_decodes_byte_identical_and_filters() {
    // The old envelope boundary, now INSIDE the envelope: a REAL stream
    // encoded with `--enable-cdef=1` decodes byte-identical to the C decoder
    // (asserted inside run_config) AND the CDEF stage genuinely changes
    // pixels (probed 2026-07-14: this config picks nonzero strengths).
    let f = run_config(&Cfg {
        w: 96,
        h: 80,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cdef: true,
        restoration: false,
        usage: 0,
        aq: 0,
        two_pass: false,
        sb128: false,
        tile_columns_log2: 0,
        tile_rows_log2: 0,
    });
    println!(
        "cdef companion: {} bytes, gated={} applied={}",
        f.len, f.cdef_gated, f.cdef_applied
    );
    assert!(f.cdef_gated, "cq 36 stream lost its CDEF syntax");
    assert!(f.cdef_applied, "cq 36 CDEF no longer changes any pixel");
}

#[test]
fn garbage_input_errors_cleanly() {
    assert!(decode_frame_obus(&[0u8; 4]).is_err());
    assert!(decode_frame_obus(&[]).is_err());
}

// ---- QM gate: --enable-qm=1 streams with forced non-flat matrices ----
//
// REAL `aom_codec_av1_cx` streams encoded with AV1E_SET_ENABLE_QM=1 and
// AV1E_SET_QM_MIN==AV1E_SET_QM_MAX==L (which forces qmatrix_level_{y,u,v}==L for
// every plane — the level formulas clamp into [min,max]). With L in 0..=14 the
// matrix is genuinely non-flat, so each 2-D-transform coefficient is
// dequantized with a per-position inverse-QM weight (`av1_get_iqmatrix` /
// `aom_decode::qm`, folded into `dequant_txb`). Decoding these byte-identically
// vs the REAL C decoder proves the QM dequant path is bit-exact — and would
// FAIL if the decoder ignored QM, since C applies the non-flat weights.
#[test]
fn qm_streams_decode_byte_identical_to_c() {
    // (bd, ss, mono): 4:2:0 + 4:4:4 + monochrome at 8 and 10 bit. (4:2:2 is
    // avoided — its chroma deblock is a separate out-of-envelope axis.)
    let combos = [
        (8i32, (1i32, 1i32), false), // 4:2:0
        (8, (0, 0), false),          // 4:4:4
        (10, (1, 1), false),         // 4:2:0, 10-bit
        (10, (0, 0), false),         // 4:4:4, 10-bit
        (8, (1, 1), true),           // monochrome
    ];
    // Sizes incl. a non-8-multiple crop (100x76) to exercise frame-edge txbs.
    let sizes = [(64usize, 64usize), (96, 80), (100, 76)];
    // Forced QM levels: steep (0), mid (5, 8), and near-flat-but-not-flat (12).
    // All < NUM_QM_LEVELS-1 (15) so the matrix genuinely weights coefficients.
    let qm_levels = [0i32, 5, 8, 12];

    let mut n = 0u32;
    let mut steep_levels = 0u32; // count of L <= 5 arms (strong QM effect)
    let mut min_level_seen = 15i32;
    let mut max_level_seen = 0i32;

    for (si, &(w, h)) in sizes.iter().enumerate() {
        for &(bd, ss, mono) in &combos {
            // Rotate the forced level across the grid so every level is hit at
            // several sizes / formats.
            let lvl = qm_levels[(si + n as usize) % qm_levels.len()];
            // ALL_INTRA usage on some arms (the zenavif/avifenc still mode; a
            // different QM-level formula, still clamped to the forced level).
            let usage = if (n & 1) == 0 { 0u32 } else { 2 };
            let cq = 32; // moderate: non-lossless, real AC coefficients present.

            let (cw, ch) = if mono {
                (0, 0)
            } else {
                ((w + ss.0 as usize) >> ss.0, (h + ss.1 as usize) >> ss.1)
            };
            let seed = ((w as u64) << 40)
                ^ ((h as u64) << 24)
                ^ ((bd as u64) << 12)
                ^ ((lvl as u64) << 4)
                ^ usage as u64;
            let y = gen_plane(w, h, bd, seed ^ 0x1111, false);
            let u = gen_plane(cw, ch, bd, seed ^ 0x2222, true);
            let v = gen_plane(cw, ch, bd, seed ^ 0x3333, true);

            // REAL encoder bytes with QM forced to level `lvl` (cpu-used=0).
            let bytes = c::ref_encode_av1_kf_qm(
                &y, &u, &v, w, h, bd, mono, ss.0, ss.1, cq, 0, /*cdef=*/ false,
                /*restoration=*/ false, usage, /*aq=*/ 0, /*two_pass=*/ false, lvl,
                lvl,
            );

            // Rust decode (hard-errors outside the envelope — that FAILS here).
            let rust = decode_frame_obus(&bytes).unwrap_or_else(|e| {
                panic!(
                    "decode_frame_obus rejected QM stream {w}x{h} bd{bd} mono={mono} ss={ss:?} L={lvl}: {e}"
                )
            });

            // Anti-vacuous facts from OUR OWN parse: the stream really uses QM
            // and the level is non-flat (else this test proves nothing — a flat
            // matrix is the pre-QM path).
            let (_t, _tcfg, header) = decode_frame_obus_prefilter(&bytes).unwrap();
            assert!(
                header.quant.using_qmatrix,
                "QM stream {w}x{h} L={lvl}: using_qmatrix flag not set"
            );
            let ql = [
                header.quant.qmatrix_level_y,
                header.quant.qmatrix_level_u,
                header.quant.qmatrix_level_v,
            ];
            for &q in &ql {
                assert!(
                    (0..=14).contains(&q),
                    "QM stream {w}x{h} L={lvl}: qm level {q} is flat/out of range (vacuous)"
                );
            }
            assert!(rust.base_qindex > 0, "QM gate must be non-lossless");

            // Gold oracle: the REAL C decoder on the same bytes.
            let cref = c::ref_decode_av1_kf(&bytes, w, h);
            assert_eq!(cref.info[0], bd);
            assert_eq!(cref.info[1] != 0, mono);
            assert_eq!(
                rust.y, cref.y,
                "LUMA mismatch QM {w}x{h} bd{bd} mono={mono} ss={ss:?} L={lvl}"
            );
            if mono {
                assert!(rust.u.is_empty() && rust.v.is_empty());
            } else {
                assert_eq!(
                    rust.u, cref.u,
                    "U mismatch QM {w}x{h} bd{bd} ss={ss:?} L={lvl}"
                );
                assert_eq!(
                    rust.v, cref.v,
                    "V mismatch QM {w}x{h} bd{bd} ss={ss:?} L={lvl}"
                );
            }

            // Coefficients present: the reconstruction is not a constant plane
            // (a DC-only / all-skip frame would give QM nothing to weight).
            assert!(
                rust.y.iter().any(|&p| p != rust.y[0]),
                "QM stream {w}x{h} L={lvl}: luma is constant (no AC coefficients?)"
            );

            if ql[0] <= 5 {
                steep_levels += 1;
            }
            min_level_seen = min_level_seen.min(ql[0]);
            max_level_seen = max_level_seen.max(ql[0]);
            n += 1;
        }
    }

    // Coverage floors: the grid actually ran and exercised steep
    // (strong-effect) levels across a spread. (The rigorous "the matrix is
    // genuinely non-flat and effectful" proof is the qm module's own unit test,
    // `aom_decode::qm::tests`; here the byte-identity vs C on non-flat-level
    // streams is itself anti-vacuous — a decoder that ignored QM would mismatch
    // C at every AC coefficient sitting on a non-flat weight.)
    assert!(n >= 15, "QM gate ran too few streams ({n})");
    assert!(
        steep_levels >= 3,
        "QM gate never exercised a steep (L<=5) matrix ({steep_levels})"
    );
    assert!(
        min_level_seen <= 5 && max_level_seen >= 8,
        "QM level spread too narrow"
    );
    eprintln!(
        "qm gate: {n} streams byte-identical, qm levels {min_level_seen}..={max_level_seen}, \
         steep arms {steep_levels}"
    );
}

// ---- PALETTE gate: --enable-palette=1 --enable-intrabc=0 streams ----
//
// gen_plane above is explicitly photographic (smooth gradients + noise) to
// keep the encoder's screen-content heuristics from EVER picking palette in
// the other arms — this section is the deliberate opposite: synthetic
// few-colour, sharp-edged content (the kind palette actually targets:
// rendered UI, icons, text) so `--enable-palette=1` genuinely exercises the
// palette coding path instead of silently coding everything through ordinary
// intra prediction. `--enable-intrabc` stays 0 throughout for THIS palette
// section (intrabc is in the decode envelope and has its own coverage in the
// intrabc_{monochrome,colour}_streams_decode_byte_identical_to_c gates below).

/// Screen-content-like synthetic content: `TILE`-pixel super-tiles, each a
/// simple stripe pattern over a small (2..=5) local colour set. The local set
/// is MOSTLY drawn from one shared per-plane global set (3..=6 colours) so
/// neighbouring tiles often share exact colour values — this is what
/// exercises the palette neighbour CACHE (get_palette_cache's true-positive
/// path), not just palette selection in isolation.
fn gen_plane_screen_content(w: usize, h: usize, bd: i32, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed | 1);
    let maxv = (1i64 << bd) - 1;
    let mut p = vec![0u16; w * h];
    const TILE: usize = 24;
    let global_n = 3 + (rng.next() % 4) as usize; // 3..=6
    let global_colors: Vec<i64> = (0..global_n)
        .map(|_| (rng.next() % (maxv as u64 + 1)) as i64)
        .collect();
    let mut ty = 0usize;
    while ty < h {
        let mut tx = 0usize;
        while tx < w {
            let n_local = 2 + (rng.next() % 4) as usize; // 2..=5
            let local: Vec<i64> = (0..n_local)
                .map(|_| {
                    if !rng.next().is_multiple_of(4) {
                        global_colors[(rng.next() as usize) % global_colors.len()]
                    } else {
                        (rng.next() % (maxv as u64 + 1)) as i64
                    }
                })
                .collect();
            let period = 3 + (rng.next() % 4) as usize;
            for y in ty..(ty + TILE).min(h) {
                for x in tx..(tx + TILE).min(w) {
                    let stripe = ((x + y) / period) % local.len();
                    p[y * w + x] = local[stripe] as u16;
                }
            }
            tx += TILE;
        }
        ty += TILE;
    }
    p
}

struct PalCfg {
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss: (i32, i32),
    cq: i32,
    usage: u32,
}

#[test]
fn palette_streams_decode_byte_identical_to_c() {
    let cfgs = [
        // usage=2 (ALLINTRA) is the PRIMARY product path (zenavif stills) —
        // swept across the cq range including low-q (screen-content AVIF at
        // aggressive settings is exactly where palette earns its keep).
        PalCfg {
            w: 128,
            h: 128,
            bd: 8,
            mono: false,
            ss: (1, 1),
            cq: 12,
            usage: 2,
        },
        PalCfg {
            w: 128,
            h: 128,
            bd: 8,
            mono: false,
            ss: (1, 1),
            cq: 28,
            usage: 2,
        },
        PalCfg {
            w: 128,
            h: 128,
            bd: 8,
            mono: false,
            ss: (1, 1),
            cq: 44,
            usage: 2,
        },
        PalCfg {
            w: 128,
            h: 128,
            bd: 8,
            mono: false,
            ss: (1, 1),
            cq: 60,
            usage: 2,
        },
        PalCfg {
            w: 128,
            h: 128,
            bd: 8,
            mono: false,
            ss: (1, 1),
            cq: 63,
            usage: 2,
        },
        PalCfg {
            w: 128,
            h: 128,
            bd: 10,
            mono: false,
            ss: (1, 1),
            cq: 30,
            usage: 2,
        },
        PalCfg {
            w: 128,
            h: 128,
            bd: 12,
            mono: false,
            ss: (1, 1),
            cq: 30,
            usage: 2,
        },
        PalCfg {
            w: 96,
            h: 96,
            bd: 8,
            mono: false,
            ss: (0, 0),
            cq: 36,
            usage: 2,
        }, // 4:4:4
        // NOTE: 4:2:2 is not swept in THIS palette gate (palette's 4:4:4
        // coverage is the arm just above; 4:2:0 dominates the rest). 4:2:2
        // chroma deblock itself is byte-exact and covered by
        // deblocked_422_chroma_byte_identical_to_c.
        PalCfg {
            w: 96,
            h: 96,
            bd: 8,
            mono: false,
            ss: (1, 1),
            cq: 24,
            usage: 2,
        },
        PalCfg {
            w: 128,
            h: 128,
            bd: 8,
            mono: true,
            ss: (1, 1),
            cq: 36,
            usage: 2,
        },
        // non-multiple-of-SB / non-multiple-of-8: exercises the palette
        // colour-map's frame-edge tail-copy (rows/cols < plane_width/height).
        PalCfg {
            w: 100,
            h: 84,
            bd: 8,
            mono: false,
            ss: (1, 1),
            cq: 30,
            usage: 2,
        },
        PalCfg {
            w: 100,
            h: 84,
            bd: 8,
            mono: false,
            ss: (1, 1),
            cq: 60,
            usage: 2,
        },
        // usage=0 (GOOD) retained per the sweep discipline.
        PalCfg {
            w: 128,
            h: 128,
            bd: 8,
            mono: false,
            ss: (1, 1),
            cq: 20,
            usage: 0,
        },
        PalCfg {
            w: 128,
            h: 128,
            bd: 8,
            mono: false,
            ss: (1, 1),
            cq: 50,
            usage: 0,
        },
        PalCfg {
            w: 128,
            h: 128,
            bd: 8,
            mono: false,
            ss: (1, 1),
            cq: 63,
            usage: 0,
        },
    ];
    let mut total_arms = 0usize;
    let mut palette_arms = 0usize;
    let mut allintra_arms = 0usize;
    let mut allintra_palette_arms = 0usize;
    for c in &cfgs {
        total_arms += 1;
        if c.usage == 2 {
            allintra_arms += 1;
        }
        let (cw, ch) = if c.mono {
            (0, 0)
        } else {
            (
                (c.w + c.ss.0 as usize) >> c.ss.0,
                (c.h + c.ss.1 as usize) >> c.ss.1,
            )
        };
        let seed = ((c.w as u64) << 32)
            ^ ((c.h as u64) << 24)
            ^ ((c.cq as u64) << 12)
            ^ ((c.bd as u64) << 4)
            ^ c.usage as u64;
        let y = gen_plane_screen_content(c.w, c.h, c.bd, seed ^ 0x1111);
        let (u, v) = if c.mono {
            (Vec::new(), Vec::new())
        } else {
            (
                gen_plane_screen_content(cw, ch, c.bd, seed ^ 0x2222),
                gen_plane_screen_content(cw, ch, c.bd, seed ^ 0x3333),
            )
        };
        let bytes = c::ref_encode_av1_kf_screen_content(
            &y, &u, &v, c.w, c.h, c.bd, c.mono, c.ss.0, c.ss.1, c.cq, /* cpu_used */ 0,
            /* enable_cdef */ true, /* enable_restoration */ true, c.usage,
            /* aq_mode */ 0, /* two_pass */ false, /* enable_palette */ true,
            /* enable_intrabc */ false,
        );

        // Byte-identity: our decode vs the REAL C decoder, same bytes.
        let rust = decode_frame_obus(&bytes).unwrap_or_else(|e| {
            panic!(
                "decode_frame_obus rejected palette stream {}x{} bd{} mono={} ss={:?} cq={} usage={}: {e}",
                c.w, c.h, c.bd, c.mono, c.ss, c.cq, c.usage
            )
        });
        let cref = c::ref_decode_av1_kf(&bytes, c.w, c.h);
        assert_eq!(
            rust.y, cref.y,
            "LUMA mismatch palette {}x{} bd{} cq={} usage={}",
            c.w, c.h, c.bd, c.cq, c.usage
        );
        if !c.mono {
            assert_eq!(
                rust.u, cref.u,
                "U mismatch palette {}x{} bd{} cq={} usage={}",
                c.w, c.h, c.bd, c.cq, c.usage
            );
            assert_eq!(
                rust.v, cref.v,
                "V mismatch palette {}x{} bd{} cq={} usage={}",
                c.w, c.h, c.bd, c.cq, c.usage
            );
        }

        // Palette-usage introspection: a second decode via the prefilter entry
        // point, which exposes per-block DecodedBlockKf.info.palette_size (not
        // part of FrameDecode's public surface).
        let (t, _tcfg, _header) = decode_frame_obus_prefilter(&bytes).unwrap();
        let pal_blocks = t
            .blocks
            .iter()
            .filter(|b| b.info.palette_size[0] > 0 || b.info.palette_size[1] > 0)
            .count();
        println!(
            "palette arm {}x{} bd{} mono={} ss={:?} cq={} usage={}: {} bytes, {} blocks, {} palette blocks",
            c.w,
            c.h,
            c.bd,
            c.mono,
            c.ss,
            c.cq,
            c.usage,
            bytes.len(),
            t.blocks.len(),
            pal_blocks
        );
        if pal_blocks > 0 {
            palette_arms += 1;
            if c.usage == 2 {
                allintra_palette_arms += 1;
            }
        }
    }
    println!(
        "palette gate summary: {total_arms} total arms ({allintra_arms} ALL_INTRA), \
         {palette_arms} carrying >=1 palette block ({allintra_palette_arms} of those ALL_INTRA)"
    );
    assert!(
        palette_arms > 0,
        "NO arm actually exercised palette — screen-content generator or \
         --enable-palette=1 wiring is broken"
    );
    assert!(
        allintra_palette_arms > 0,
        "NO ALL_INTRA arm exercised palette (usage=2 is the primary zenavif path)"
    );
}

// ---- INTRABC gate: --enable-intrabc=1 monochrome KEY frames ----
//
// Intra block copy is the OTHER `allow_screen_content_tools` tool. Monochrome
// intrabc is in the decode envelope: the block vector is read at MV_SUBPEL_NONE
// (full-pel luma), predicted via av1_find_mv_refs(INTRA_FRAME) + av1_find_ref_dv
// and validated by av1_is_dv_valid, then luma reconstruction is an integer block
// copy from the already-decoded region. Verified byte-identical to the C
// decoder here. Colour intrabc is still rejected in frame.rs pending chroma
// reconstruction (luma-derived DV can land at chroma half-pel), so this sweep is
// monochrome only. Uses gen_plane_screen_content (repeated stripe super-tiles)
// so `--enable-intrabc=1` genuinely finds block matches.

#[test]
fn intrabc_monochrome_streams_decode_byte_identical_to_c() {
    struct IbcCfg {
        w: usize,
        h: usize,
        bd: i32,
        cq: i32,
        usage: u32,
    }
    let cfgs = [
        IbcCfg {
            w: 128,
            h: 128,
            bd: 8,
            cq: 12,
            usage: 2,
        },
        IbcCfg {
            w: 128,
            h: 128,
            bd: 8,
            cq: 32,
            usage: 2,
        },
        IbcCfg {
            w: 128,
            h: 128,
            bd: 8,
            cq: 55,
            usage: 2,
        },
        IbcCfg {
            w: 128,
            h: 128,
            bd: 8,
            cq: 63,
            usage: 2,
        },
        IbcCfg {
            w: 256,
            h: 128,
            bd: 10,
            cq: 24,
            usage: 2,
        },
        IbcCfg {
            w: 160,
            h: 96,
            bd: 8,
            cq: 40,
            usage: 0,
        },
        IbcCfg {
            w: 256,
            h: 128,
            bd: 8,
            cq: 24,
            usage: 2,
        },
    ];
    let mut total_arms = 0usize;
    let mut intrabc_arms = 0usize;
    for c in &cfgs {
        total_arms += 1;
        let seed = ((c.w as u64) << 32)
            ^ ((c.h as u64) << 24)
            ^ ((c.cq as u64) << 12)
            ^ ((c.bd as u64) << 4)
            ^ c.usage as u64;
        let y = gen_plane_screen_content(c.w, c.h, c.bd, seed ^ 0x1a1b);
        let empty: Vec<u16> = Vec::new();
        let bytes = c::ref_encode_av1_kf_screen_content(
            &y, &empty, &empty, c.w, c.h, c.bd, /* mono */ true, /* ss_x */ 1,
            /* ss_y */ 1, c.cq, /* cpu_used */ 0, /* enable_cdef */ true,
            /* enable_restoration */ true, c.usage, /* aq_mode */ 0,
            /* two_pass */ false, /* enable_palette */ false,
            /* enable_intrabc */ true,
        );

        // Byte-identity: our decode vs the REAL C decoder, same bytes.
        let rust = decode_frame_obus(&bytes).unwrap_or_else(|e| {
            panic!(
                "decode_frame_obus rejected intrabc mono stream {}x{} bd{} cq={} usage={}: {e}",
                c.w, c.h, c.bd, c.cq, c.usage
            )
        });
        let cref = c::ref_decode_av1_kf(&bytes, c.w, c.h);
        assert_eq!(
            rust.y, cref.y,
            "LUMA mismatch intrabc mono {}x{} bd{} cq={} usage={}",
            c.w, c.h, c.bd, c.cq, c.usage
        );

        // intrabc-usage introspection via the prefilter entry point
        // (DecodedBlockKf.info.use_intrabc is not on FrameDecode's surface).
        let (t, _tcfg, _header) = decode_frame_obus_prefilter(&bytes).unwrap();
        let ibc_blocks = t.blocks.iter().filter(|b| b.info.use_intrabc != 0).count();
        println!(
            "intrabc arm {}x{} bd{} cq={} usage={}: {} bytes, {} blocks, {} intrabc blocks",
            c.w,
            c.h,
            c.bd,
            c.cq,
            c.usage,
            bytes.len(),
            t.blocks.len(),
            ibc_blocks
        );
        if ibc_blocks > 0 {
            intrabc_arms += 1;
        }
    }
    println!("intrabc gate summary: {total_arms} arms, {intrabc_arms} carrying >=1 intrabc block");
    assert!(
        intrabc_arms > 0,
        "NO arm actually exercised intrabc — screen-content generator or \
         --enable-intrabc=1 wiring is broken"
    );
}

/// Colour intra-block-copy: REAL-C-encoded screen-content KEY streams with
/// `--enable-intrabc=1` across 4:2:0, 4:4:4 and 4:2:2 must decode byte-identical
/// to the REAL C decoder on ALL THREE planes. This exercises chroma intrabc: the
/// luma-DV-derived chroma block copy (integer for 4:4:4, and half-pel via the
/// 2-tap intrabc bilinear when the luma-pel DV is odd on a subsampled axis) plus
/// the co-located-luma chroma tx-type.
#[test]
fn intrabc_colour_streams_decode_byte_identical_to_c() {
    struct IbcCfg {
        w: usize,
        h: usize,
        bd: i32,
        cq: i32,
        usage: u32,
        ss_x: i32,
        ss_y: i32,
    }
    let cfgs = [
        // 4:2:0 — chroma can land at half-pel on either axis.
        IbcCfg {
            w: 256,
            h: 128,
            bd: 8,
            cq: 24,
            usage: 2,
            ss_x: 1,
            ss_y: 1,
        },
        IbcCfg {
            w: 160,
            h: 96,
            bd: 8,
            cq: 40,
            usage: 0,
            ss_x: 1,
            ss_y: 1,
        },
        IbcCfg {
            w: 256,
            h: 128,
            bd: 8,
            cq: 32,
            usage: 0,
            ss_x: 1,
            ss_y: 1,
        },
        IbcCfg {
            w: 256,
            h: 128,
            bd: 10,
            cq: 24,
            usage: 2,
            ss_x: 1,
            ss_y: 1,
        },
        // 4:4:4 — chroma DV is always integer (a plain copy).
        IbcCfg {
            w: 256,
            h: 128,
            bd: 8,
            cq: 24,
            usage: 2,
            ss_x: 0,
            ss_y: 0,
        },
        IbcCfg {
            w: 160,
            h: 96,
            bd: 8,
            cq: 40,
            usage: 0,
            ss_x: 0,
            ss_y: 0,
        },
        // 4:2:2 — horizontal axis can be half-pel, vertical always integer.
        IbcCfg {
            w: 256,
            h: 128,
            bd: 8,
            cq: 24,
            usage: 2,
            ss_x: 1,
            ss_y: 0,
        },
    ];
    let mut total_arms = 0usize;
    let mut intrabc_arms = 0usize;
    let mut halfpel_blocks = 0usize;
    for c in &cfgs {
        total_arms += 1;
        let seed = ((c.w as u64) << 32)
            ^ ((c.h as u64) << 24)
            ^ ((c.cq as u64) << 12)
            ^ ((c.bd as u64) << 6)
            ^ ((c.usage as u64) << 3)
            ^ ((c.ss_x as u64) << 1)
            ^ c.ss_y as u64;
        let y = gen_plane_screen_content(c.w, c.h, c.bd, seed ^ 0x1a1b);
        let (cw, ch) = ((c.w >> c.ss_x), (c.h >> c.ss_y));
        let u = gen_plane_screen_content(cw, ch, c.bd, seed ^ 0x2c2d);
        let v = gen_plane_screen_content(cw, ch, c.bd, seed ^ 0x3e3f);
        let bytes = c::ref_encode_av1_kf_screen_content(
            &y, &u, &v, c.w, c.h, c.bd, /* mono */ false, c.ss_x, c.ss_y, c.cq,
            /* cpu_used */ 0, /* enable_cdef */ true, /* enable_restoration */ true,
            c.usage, /* aq_mode */ 0, /* two_pass */ false,
            /* enable_palette */ false, /* enable_intrabc */ true,
        );

        let rust = decode_frame_obus(&bytes).unwrap_or_else(|e| {
            panic!(
                "decode_frame_obus rejected colour intrabc {}x{} bd{} ss=({},{}) cq={} usage={}: {e}",
                c.w, c.h, c.bd, c.ss_x, c.ss_y, c.cq, c.usage
            )
        });
        let cref = c::ref_decode_av1_kf(&bytes, c.w, c.h);
        let ctx = format!(
            "colour intrabc {}x{} bd{} ss=({},{}) cq={} usage={}",
            c.w, c.h, c.bd, c.ss_x, c.ss_y, c.cq, c.usage
        );
        assert_eq!(rust.y, cref.y, "LUMA mismatch {ctx}");
        assert_eq!(rust.u, cref.u, "U mismatch {ctx}");
        assert_eq!(rust.v, cref.v, "V mismatch {ctx}");

        let (t, _tcfg, _header) = decode_frame_obus_prefilter(&bytes).unwrap();
        let ibc_blocks = t.blocks.iter().filter(|b| b.info.use_intrabc != 0).count();
        // Half-pel chroma occurs when the integer-luma-pel DV is odd on a
        // subsampled axis (mv_q4 & 15 == 8) — exercising the 2-tap intrabc
        // bilinear rather than a plain copy. dv is a multiple of 8, so odd D
        // <=> (dv & 8) != 0.
        let hp = t
            .blocks
            .iter()
            .filter(|b| {
                b.info.use_intrabc != 0
                    && ((c.ss_x == 1 && (b.info.dv_col & 8) != 0)
                        || (c.ss_y == 1 && (b.info.dv_row & 8) != 0))
            })
            .count();
        halfpel_blocks += hp;
        println!(
            "{ctx}: {} bytes, {} blocks, {} intrabc blocks ({} half-pel chroma)",
            bytes.len(),
            t.blocks.len(),
            ibc_blocks,
            hp
        );
        if ibc_blocks > 0 {
            intrabc_arms += 1;
        }
    }
    println!(
        "colour intrabc gate summary: {total_arms} arms, {intrabc_arms} carrying >=1 intrabc block, {halfpel_blocks} half-pel chroma blocks"
    );
    assert!(
        intrabc_arms > 0,
        "NO colour arm actually exercised intrabc — generator or wiring is broken"
    );
    assert!(
        halfpel_blocks > 0,
        "no half-pel chroma intrabc block exercised — the 2-tap intrabc convolve \
         path is untested; add a config whose DVs land at odd luma offsets"
    );
}

// ---- CODED-LOSSLESS gate: --lossless=1 streams ----
//
// Coded-lossless is in the decode envelope. AV1E_SET_LOSSLESS=1 forces
// base_qindex 0 (coded_lossless), which flips every block's transform to forced
// TX_4X4 + the 4x4 Walsh-Hadamard (WHT) with the qindex-0 dequant, narrows
// is_cfl_allowed to BLOCK_4X4, and gates the header's loop-filter / CDEF /
// restoration / tx-mode reads off (frame.rs does a two-phase parse: probe ->
// compute coded_lossless -> re-parse). The decoder reconstructs via
// aom_transform::av1_highbd_iwht4x4_add instead of the DCT/ADST inverse.
//
// Photographic content (gradients + noise) guarantees real residual, so the WHT
// runs on plenty of non-skip 4x4 txbs. TWO independent correctness checks per
// arm: (1) byte-identity vs the REAL C decoder (aom_codec_av1_dx), and (2)
// because the stream is truly lossless, the decoded planes must equal the
// ORIGINAL source pixels exactly.

#[test]
fn lossless_streams_decode_byte_identical_to_c() {
    struct LlCfg {
        w: usize,
        h: usize,
        bd: i32,
        usage: u32,
        ss_x: i32,
        ss_y: i32,
        mono: bool,
    }
    let cfgs = [
        // 8-bit 4:2:0, SB-multiple, ALL_INTRA
        LlCfg {
            w: 64,
            h: 64,
            bd: 8,
            usage: 2,
            ss_x: 1,
            ss_y: 1,
            mono: false,
        },
        // 8-bit 4:2:0, non-SB-multiple crop, GOOD
        LlCfg {
            w: 96,
            h: 80,
            bd: 8,
            usage: 0,
            ss_x: 1,
            ss_y: 1,
            mono: false,
        },
        // 8-bit 4:4:4, ALL_INTRA
        LlCfg {
            w: 64,
            h: 64,
            bd: 8,
            usage: 2,
            ss_x: 0,
            ss_y: 0,
            mono: false,
        },
        // 8-bit 4:4:4, non-8-multiple crop (100x76 cropped from the 8px mi grid), GOOD
        LlCfg {
            w: 100,
            h: 76,
            bd: 8,
            usage: 0,
            ss_x: 0,
            ss_y: 0,
            mono: false,
        },
        // 8-bit monochrome, GOOD
        LlCfg {
            w: 128,
            h: 96,
            bd: 8,
            usage: 0,
            ss_x: 1,
            ss_y: 1,
            mono: true,
        },
        // 10-bit 4:2:0, ALL_INTRA
        LlCfg {
            w: 64,
            h: 64,
            bd: 10,
            usage: 2,
            ss_x: 1,
            ss_y: 1,
            mono: false,
        },
        // 10-bit 4:4:4, non-SB-multiple crop, GOOD
        LlCfg {
            w: 96,
            h: 80,
            bd: 10,
            usage: 0,
            ss_x: 0,
            ss_y: 0,
            mono: false,
        },
        // 10-bit monochrome, ALL_INTRA
        LlCfg {
            w: 96,
            h: 96,
            bd: 10,
            usage: 2,
            ss_x: 1,
            ss_y: 1,
            mono: true,
        },
    ];

    let mut total_arms = 0usize;
    let mut wht_luma_txbs_total = 0usize;
    let mut wht_uv_txbs_total = 0usize;
    for c in &cfgs {
        total_arms += 1;
        let seed = ((c.w as u64) << 32)
            ^ ((c.h as u64) << 24)
            ^ ((c.bd as u64) << 12)
            ^ ((c.usage as u64) << 6)
            ^ ((c.ss_x as u64) << 2)
            ^ ((c.ss_y as u64) << 1)
            ^ c.mono as u64;
        let y = gen_plane(c.w, c.h, c.bd, seed ^ 0x1111, false);
        let (cw, ch) = if c.mono {
            (0, 0)
        } else {
            (c.w >> c.ss_x, c.h >> c.ss_y)
        };
        let u = if c.mono {
            Vec::new()
        } else {
            gen_plane(cw, ch, c.bd, seed ^ 0x2222, true)
        };
        let v = if c.mono {
            Vec::new()
        } else {
            gen_plane(cw, ch, c.bd, seed ^ 0x3333, true)
        };

        let bytes = c::ref_encode_av1_kf_lossless(
            &y, &u, &v, c.w, c.h, c.bd, c.mono, c.ss_x, c.ss_y, /* cpu_used */ 0, c.usage,
            /* two_pass */ false,
        );

        let ctx = format!(
            "lossless {}x{} bd{} ss=({},{}) mono={} usage={}",
            c.w, c.h, c.bd, c.ss_x, c.ss_y, c.mono, c.usage
        );

        // (1) byte-identity vs the REAL C decoder.
        let rust = decode_frame_obus(&bytes)
            .unwrap_or_else(|e| panic!("decode_frame_obus rejected {ctx}: {e}"));
        let cref = c::ref_decode_av1_kf(&bytes, c.w, c.h);
        assert_eq!(rust.y, cref.y, "LUMA mismatch vs C {ctx}");
        if !c.mono {
            assert_eq!(rust.u, cref.u, "U mismatch vs C {ctx}");
            assert_eq!(rust.v, cref.v, "V mismatch vs C {ctx}");
        }

        // (2) truly lossless: decoded == original source pixels.
        assert_eq!(rust.y, y, "LUMA not lossless vs source {ctx}");
        if !c.mono {
            assert_eq!(rust.u, u, "U not lossless vs source {ctx}");
            assert_eq!(rust.v, v, "V not lossless vs source {ctx}");
        }

        // Anti-vacuous coverage: every block is forced TX_4X4, and the WHT must
        // actually run (non-skip 4x4 txbs with eob>0) — not a trivial all-skip
        // frame that would pass without exercising the WHT.
        let (t, _tcfg, _header) = decode_frame_obus_prefilter(&bytes).unwrap();
        assert!(
            t.blocks.iter().all(|b| b.tx_size == 0),
            "{ctx}: a block escaped the lossless TX_4X4 preempt (tx_size != TX_4X4)"
        );
        let wht_luma: usize = t
            .blocks
            .iter()
            .map(|b| b.txbs.iter().filter(|(eob, _)| *eob > 0).count())
            .sum();
        let wht_uv: usize = t
            .blocks
            .iter()
            .map(|b| b.txbs_uv.iter().filter(|(eob, _)| *eob > 0).count())
            .sum();
        wht_luma_txbs_total += wht_luma;
        wht_uv_txbs_total += wht_uv;
        println!(
            "{ctx}: {} bytes, {} blocks, {} luma WHT txbs, {} chroma WHT txbs",
            bytes.len(),
            t.blocks.len(),
            wht_luma,
            wht_uv
        );
    }
    println!(
        "lossless gate summary: {total_arms} arms, {wht_luma_txbs_total} luma WHT txbs, \
         {wht_uv_txbs_total} chroma WHT txbs"
    );
    assert!(
        wht_luma_txbs_total > 0,
        "NO luma WHT txb exercised — the lossless generator or WHT wiring is broken"
    );
    assert!(
        wht_uv_txbs_total > 0,
        "NO chroma WHT txb exercised — the colour lossless arms did not run the WHT"
    );
}
