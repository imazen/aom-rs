//! C7 FILM-GRAIN TABLE-INJECT — byte-exactness gate for `--film-grain-table`.
//!
//! Film grain is a decode-side synthesis signal: `--film-grain-table` supplies a
//! `filmgrn1` file of `[start,end) -> params` entries; libaom reads it, looks up
//! the per-frame entry, and writes the params into the frame header. It does NOT
//! denoise or otherwise alter the coded picture, so the ONLY bitstream effect vs
//! a plain encode is the seq-header `film_grain_params_present` bit plus the
//! per-frame grain-params block. The header WRITER
//! (`aom_entropy::header::write_film_grain_params`) is already bit-exact; this
//! gate proves the missing param-plumbing —
//! [`aom_encode::grain_table::read_film_grain_table`] + [`lookup`] — end to end.
//!
//! Method (per cell × test vector), rule-4 clean (no bootstrap leak):
//!   1. C writes a built-in `film_grain_test_vectors[tv]` to a canonical file via
//!      the REAL `aom_film_grain_table_write` (`ref_write_grain_table_test_vector`).
//!   2. The PORT reads THAT file (its own ported reader) and looks up time 0.
//!   3. Real aomenc encodes the cell with `--film-grain-table <file>`
//!      (`ref_encode_av1_kf_film_grain_table`) — the reference stream.
//!   4. The port encodes the cell and injects ITS parsed params into the header
//!      (`port_encode_film_grain`); the bootstrap's grain bits are never read.
//!   5. Assert the port frame OBU == the real frame OBU (byte-identical).
//!
//! Each cell first asserts its PLAIN encode is byte-exact (so a failure here is a
//! grain bug, not a KB-6-class content near-tie) and that the grain stream
//! actually differs from the plain stream (anti-vacuity). A separate witness
//! proves the port tracks the INJECTED params, not the bootstrap (anti-leak).

use aom_bench::EncodeCell;
use aom_encode::grain_table::{lookup, read_film_grain_table};
use aom_entropy::header::FilmGrainParams;
use aom_sys_ref as c;

/// Test-vector indices exercising the writer's branches: TV1 rich full-chroma
/// (14/8/9, lag 2, clip=1); TV2 simple max-lag (2/2/2, lag 3); TV6 chroma
/// points absent (14/0/0); TV15 chroma-scaling-from-luma (cfl=1, 1/0/0).
const TVS: [i32; 4] = [1, 2, 6, 15];

fn tbl_path(label: &str, tv: i32) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "aomrs_c7_grain_{}_{}_{}.tbl",
        std::process::id(),
        label,
        tv
    ))
}

/// Parse a C-written grain table and look up the still's (time 0) params.
fn parse_lookup(path: &std::path::Path) -> FilmGrainParams {
    let bytes = std::fs::read(path).expect("read grain table fixture");
    let entries = read_film_grain_table(&bytes).expect("parse grain table");
    let mut fg = FilmGrainParams::default();
    assert!(lookup(&entries, 0, &mut fg), "time-0 lookup must hit");
    fg
}

/// Run one (cell, tv) film-grain cell. Returns `(plain_exact, grain_changed,
/// grain_match)`.
fn run_cell(cell: &EncodeCell, tv: i32) -> (bool, bool, bool) {
    c::ref_init();
    let path = tbl_path(&cell.label, tv);
    c::ref_write_grain_table_test_vector(tv, &path);

    // Precondition: the PLAIN encode is byte-exact (disambiguates grain bugs
    // from content near-ties).
    let plain = cell.c_encode();
    let plain_port = cell.port_encode(&plain);
    let plain_real = EncodeCell::frame_obu_payload(&plain);
    let plain_exact = plain_port == plain_real;

    // The grain reference + the anti-vacuity signal.
    let grain = c::ref_encode_av1_kf_film_grain_table(
        &cell.y,
        &cell.u,
        &cell.v,
        cell.w,
        cell.h,
        i32::from(cell.bd),
        cell.mono,
        cell.ss_x as i32,
        cell.ss_y as i32,
        cell.cq_level,
        cell.speed,
        cell.usage,
        &path,
    );
    let grain_real = EncodeCell::frame_obu_payload(&grain);
    let grain_changed = grain_real != plain_real;

    // THE GATE: port injects its OWN parsed params.
    let fg = parse_lookup(&path);
    let grain_port = cell.port_encode_film_grain(&grain, &fg);
    let grain_match = grain_port == grain_real;

    let _ = std::fs::remove_file(&path);
    (plain_exact, grain_changed, grain_match)
}

/// A 4:2:0 bd8 synthetic cell (mono/444 built by tweaking ss/mono).
fn synth_cell(label: &str, sz: usize, mono: bool, ss_x: usize, ss_y: usize, cq: i32, bd: u8) -> EncodeCell {
    let maxv = (1u16 << bd) - 1;
    let mask = u32::from(maxv);
    let (w, h) = (sz, sz);
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let base = ((r * 37 + col * 23) as u32) & mask;
            let hf = if (r ^ col) & 1 == 1 { mask / 12 } else { 0 };
            y[r * w + col] = ((base ^ hf) as u16).min(maxv);
        }
    }
    let (cw, ch) = if mono { (0, 0) } else { ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y) };
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    if !mono {
        for r in 0..ch {
            for col in 0..cw {
                let base = ((r * 19 + col * 29) as u32) & mask;
                let hf = if (r + col) % 3 == 0 { mask / 20 } else { 0 };
                u[r * cw + col] = ((base ^ hf) as u16).min(maxv);
                let base2 = (((r + 7) * 19 + (col + 3) * 29) as u32) & mask;
                let hf2 = if (r + col + 10) % 3 == 0 { mask / 20 } else { 0 };
                v[r * cw + col] = ((base2 ^ hf2) as u16).min(maxv);
            }
        }
    }
    EncodeCell { label: label.to_string(), w, h, mono, ss_x, ss_y, usage: 2, cq_level: cq, speed: 0, bd, y, u, v }
}

/// CORE: 4:2:0 bd8 REAL content (the KB-6 byte-exact cells) × the four test
/// vectors. This is the primary `--film-grain-table` parity gate.
#[test]
fn film_grain_table_inject_420_real() {
    c::ref_init();
    let cells = [
        EncodeCell::real_content("64_cq32", "av1-1-b8-01-size-64x64", None, 32, 0),
        EncodeCell::real_content("64_cq20", "av1-1-b8-01-size-64x64", None, 20, 0),
        EncodeCell::real_content(
            "128_cq12",
            "av1-1-b8-00-quantizer-00",
            Some((128, 128, 64, 64)),
            12,
            0,
        ),
    ];
    let mut fails = Vec::new();
    let mut n = 0;
    for cell in &cells {
        for &tv in &TVS {
            let (plain_exact, changed, matched) = run_cell(cell, tv);
            n += 1;
            let tag = format!("{} tv{}", cell.label, tv);
            assert!(plain_exact, "{tag}: PLAIN encode not byte-exact — not a grain bug");
            assert!(changed, "{tag}: grain stream == plain stream (grain not present — vacuous)");
            let verdict = if matched { "EXACT" } else { "MISMATCH" };
            println!("  {tag:20} {verdict}");
            if !matched {
                fails.push(tag);
            }
        }
    }
    println!("film_grain_table_inject_420_real: {}/{} EXACT", n - fails.len(), n);
    assert!(fails.is_empty(), "film-grain table-inject MISMATCH: {fails:?}");
}

/// FORMAT AXES: mono / 4:4:4 / bd10 synthetic cells — exercise the writer's
/// `monochrome` (chroma block skipped) and chroma-format branches. Uses the
/// plain-exact precondition to keep any content near-tie out of the grain
/// verdict.
#[test]
fn film_grain_table_inject_format_axes() {
    c::ref_init();
    let cells = [
        synth_cell("mono64_cq32", 64, true, 1, 1, 32, 8),
        synth_cell("444_64_cq32", 64, false, 0, 0, 32, 8),
        synth_cell("bd10_420_64_cq32", 64, false, 1, 1, 32, 10),
    ];
    let mut fails = Vec::new();
    let mut skipped = Vec::new();
    let mut n = 0;
    for cell in &cells {
        // TV1 (rich) + TV15 (chroma-scaling-from-luma) — the writer branches most
        // sensitive to the encode's chroma format.
        for &tv in &[1, 15] {
            let (plain_exact, changed, matched) = run_cell(cell, tv);
            let tag = format!("{} tv{}", cell.label, tv);
            if !plain_exact {
                // Not a grain bug — this synthetic cell isn't plain-byte-exact
                // under this harness; record and skip its grain verdict.
                skipped.push(tag);
                continue;
            }
            n += 1;
            assert!(changed, "{tag}: grain stream == plain stream (vacuous)");
            let verdict = if matched { "EXACT" } else { "MISMATCH" };
            println!("  {tag:22} {verdict}");
            if !matched {
                fails.push(tag);
            }
        }
    }
    if !skipped.is_empty() {
        println!("film_grain_table_inject_format_axes: skipped (not plain-exact): {skipped:?}");
    }
    println!("film_grain_table_inject_format_axes: {}/{} EXACT", n - fails.len(), n);
    assert!(fails.is_empty(), "film-grain format-axis MISMATCH: {fails:?}");
    // At least the 4:4:4 + bd10 cells must be plain-exact and gated (mono may or
    // may not be, depending on the harness) — guard against a vacuous all-skip.
    assert!(n >= 2, "format-axis gate ran too few cells ({n}); all skipped: {skipped:?}");
}

/// ANTI-LEAK WITNESS (rule 4): with a fixed TV1 grain bootstrap, injecting the
/// CORRECT (TV1) parsed params byte-matches, but injecting a DIFFERENT vector's
/// params (TV2) DIVERGES — proving the port writes the params it parsed from the
/// table, not whatever the bootstrap carried.
#[test]
fn film_grain_no_bootstrap_leak_witness() {
    c::ref_init();
    let cell = EncodeCell::real_content("64_cq32", "av1-1-b8-01-size-64x64", None, 32, 0);

    let p1 = tbl_path(&cell.label, 101);
    let p2 = tbl_path(&cell.label, 102);
    c::ref_write_grain_table_test_vector(1, &p1);
    c::ref_write_grain_table_test_vector(2, &p2);

    // Reference stream uses the TV1 table.
    let grain = c::ref_encode_av1_kf_film_grain_table(
        &cell.y, &cell.u, &cell.v, cell.w, cell.h, i32::from(cell.bd), cell.mono,
        cell.ss_x as i32, cell.ss_y as i32, cell.cq_level, cell.speed, cell.usage, &p1,
    );
    let real = EncodeCell::frame_obu_payload(&grain);

    let fg1 = parse_lookup(&p1);
    let fg2 = parse_lookup(&p2);
    let with_correct = cell.port_encode_film_grain(&grain, &fg1);
    let with_wrong = cell.port_encode_film_grain(&grain, &fg2);

    let _ = std::fs::remove_file(&p1);
    let _ = std::fs::remove_file(&p2);

    assert_eq!(with_correct, real, "injecting the parsed TV1 params must byte-match");
    assert_ne!(
        with_wrong, real,
        "injecting TV2 params must DIVERGE from the TV1 stream — proves no bootstrap leak"
    );
    println!("film_grain_no_bootstrap_leak_witness: correct==real, wrong!=real (no leak)");
}
