//! Encoder-side film-grain table I/O — port of `aom_dsp/grain_table.c`.
//!
//! The `--film-grain-table` encoder input is an ascii `filmgrn1` file mapping
//! `[start_time, end_time)` ranges to a [`FilmGrainParams`] set. libaom reads it
//! once (`aom_film_grain_table_read`) and, per coded frame, looks up the entry
//! covering the frame's timestamp (`aom_film_grain_table_lookup`) to fill
//! `cm->film_grain_params`, which `write_film_grain_params` then emits into the
//! frame header. Film grain is a decode-side synthesis signal: the `-table`
//! path does NOT denoise or otherwise alter the coded picture, so the only
//! bitstream effect is the seq-header `film_grain_params_present` bit plus the
//! per-frame grain-params block.
//!
//! For a still (single KEY frame at time 0) the entry covering time 0 supplies
//! the params verbatim. The already-bit-exact
//! [`aom_dsp::entropy::header::write_film_grain_params`] does the write; this module
//! is the missing param-plumbing (read + lookup).
//!
//! The context fields of [`FilmGrainParams`] (`monochrome` / `subsampling_x` /
//! `subsampling_y` / `is_inter_frame`) are NOT in the file — mirroring C, where
//! they come from the seq/frame header, not `aom_film_grain_t`. The caller sets
//! them from the encode config after [`lookup`].

use aom_dsp::entropy::header::FilmGrainParams;

/// The 8-byte file magic (`aom_dsp/grain_table.c` `kFileMagic`). C reads a 9th
/// character (the whitespace after the magic); splitting the remainder on ascii
/// whitespace subsumes it.
pub const FILE_MAGIC: &[u8; 8] = b"filmgrn1";

/// One `[start_time, end_time)` → params mapping (`aom_film_grain_table_entry_t`,
/// minus the intrusive linked-list `next`; the port keeps a flat `Vec`).
#[derive(Clone, Debug)]
pub struct GrainTableEntry {
    pub params: FilmGrainParams,
    pub start_time: i64,
    pub end_time: i64,
}

/// Number of AR-luma coeff positions for `ar_coeff_lag` (`aom_dsp/grain_table.c`
/// `n = 2 * lag * (lag + 1)`). Chroma has `n + 1`.
fn num_pos_luma(ar_coeff_lag: i32) -> usize {
    (2 * ar_coeff_lag * (ar_coeff_lag + 1)) as usize
}

struct Toks<'a> {
    inner: core::str::SplitAsciiWhitespace<'a>,
}

impl<'a> Toks<'a> {
    fn next_tok(&mut self, what: &str) -> Result<&'a str, String> {
        self.inner
            .next()
            .ok_or_else(|| format!("film grain table: unexpected EOF reading {what}"))
    }
    fn i64(&mut self, what: &str) -> Result<i64, String> {
        let t = self.next_tok(what)?;
        t.parse::<i64>()
            .map_err(|_| format!("film grain table: bad integer {t:?} for {what}"))
    }
    fn i32(&mut self, what: &str) -> Result<i32, String> {
        let t = self.next_tok(what)?;
        t.parse::<i32>()
            .map_err(|_| format!("film grain table: bad integer {t:?} for {what}"))
    }
    fn expect(&mut self, marker: &str) -> Result<(), String> {
        let t = self.next_tok(marker)?;
        if t == marker {
            Ok(())
        } else {
            Err(format!("film grain table: expected marker {marker:?}, got {t:?}"))
        }
    }
}

/// Read a film-grain table (`aom_film_grain_table_read` + `grain_table_entry_read`).
///
/// Faithful to C's field order and the `update_parameters`-gated body; only the
/// grain fields C reads are populated (context fields stay default). Returns
/// `Err` on any malformed input instead of panicking (untrusted file).
pub fn read_film_grain_table(bytes: &[u8]) -> Result<Vec<GrainTableEntry>, String> {
    if bytes.len() < 8 || &bytes[..8] != FILE_MAGIC {
        return Err("film grain table: invalid or missing file magic".to_string());
    }
    let text = core::str::from_utf8(&bytes[8..])
        .map_err(|_| "film grain table: body is not valid UTF-8".to_string())?;
    let mut toks = Toks {
        inner: text.split_ascii_whitespace(),
    };
    let mut entries = Vec::new();
    loop {
        // `E` marks each entry; EOF here is the normal terminator.
        let marker = match toks.inner.next() {
            None => break,
            Some(m) => m,
        };
        if marker != "E" {
            return Err(format!("film grain table: expected entry marker 'E', got {marker:?}"));
        }
        let start_time = toks.i64("start_time")?;
        let end_time = toks.i64("end_time")?;
        let mut params = FilmGrainParams::default();
        params.apply_grain = toks.i32("apply_grain")? != 0;
        params.random_seed = toks.i32("random_seed")?;
        params.update_parameters = toks.i32("update_parameters")? != 0;
        if params.update_parameters {
            toks.expect("p")?;
            params.ar_coeff_lag = toks.i32("ar_coeff_lag")?;
            params.ar_coeff_shift = toks.i32("ar_coeff_shift")?;
            params.grain_scale_shift = toks.i32("grain_scale_shift")?;
            params.scaling_shift = toks.i32("scaling_shift")?;
            params.chroma_scaling_from_luma = toks.i32("chroma_scaling_from_luma")? != 0;
            params.overlap_flag = toks.i32("overlap_flag")? != 0;
            params.cb_mult = toks.i32("cb_mult")?;
            params.cb_luma_mult = toks.i32("cb_luma_mult")?;
            params.cb_offset = toks.i32("cb_offset")?;
            params.cr_mult = toks.i32("cr_mult")?;
            params.cr_luma_mult = toks.i32("cr_luma_mult")?;
            params.cr_offset = toks.i32("cr_offset")?;

            toks.expect("sY")?;
            params.num_y_points = toks.i32("num_y_points")?;
            read_points(&mut toks, params.num_y_points, &mut params.scaling_points_y, "y")?;
            toks.expect("sCb")?;
            params.num_cb_points = toks.i32("num_cb_points")?;
            read_points(&mut toks, params.num_cb_points, &mut params.scaling_points_cb, "cb")?;
            toks.expect("sCr")?;
            params.num_cr_points = toks.i32("num_cr_points")?;
            read_points(&mut toks, params.num_cr_points, &mut params.scaling_points_cr, "cr")?;

            let n = num_pos_luma(params.ar_coeff_lag);
            if n > params.ar_coeffs_y.len() {
                return Err(format!("film grain table: ar_coeff_lag {} out of range", params.ar_coeff_lag));
            }
            toks.expect("cY")?;
            for i in 0..n {
                params.ar_coeffs_y[i] = toks.i32("ar_coeffs_y")?;
            }
            toks.expect("cCb")?;
            for i in 0..=n {
                params.ar_coeffs_cb[i] = toks.i32("ar_coeffs_cb")?;
            }
            toks.expect("cCr")?;
            for i in 0..=n {
                params.ar_coeffs_cr[i] = toks.i32("ar_coeffs_cr")?;
            }
        }
        entries.push(GrainTableEntry { params, start_time, end_time });
    }
    Ok(entries)
}

fn read_points(
    toks: &mut Toks<'_>,
    num: i32,
    points: &mut [[i32; 2]],
    plane: &str,
) -> Result<(), String> {
    if num < 0 || num as usize > points.len() {
        return Err(format!("film grain table: num_{plane}_points {num} out of range"));
    }
    for i in 0..num as usize {
        points[i][0] = toks.i32("scaling_point_x")?;
        points[i][1] = toks.i32("scaling_point_y")?;
    }
    Ok(())
}

/// Look up the grain params for `time_stamp` (`aom_film_grain_table_lookup`,
/// `erase=0` — the encoder read path; the erase/split branches are table
/// management, unused for a lone still). On a hit `out` is overwritten with the
/// entry's params; C preserves the caller's running seed for `time_stamp != 0`
/// (irrelevant to a still at time 0, but mirrored). Returns whether an entry
/// covered the time. Context fields (`monochrome`/`subsampling_*`) are left as
/// the entry's defaults; the caller re-applies them from the encode config,
/// exactly as C derives them from the seq/frame header rather than the table.
pub fn lookup(entries: &[GrainTableEntry], time_stamp: i64, out: &mut FilmGrainParams) -> bool {
    let prev_seed = out.random_seed;
    for e in entries {
        if time_stamp >= e.start_time && time_stamp < e.end_time {
            *out = e.params.clone();
            if time_stamp != 0 {
                out.random_seed = prev_seed;
            }
            return true;
        }
    }
    false
}

/// Serialize a film-grain table (`aom_film_grain_table_write` +
/// `grain_table_entry_write`), byte-for-byte in C's `fprintf` shape. Used to
/// synthesize table fixtures and as the inverse for a read/write round-trip
/// check; the encoder itself never writes a table file (it writes the params
/// straight into the header).
pub fn write_film_grain_table(entries: &[GrainTableEntry]) -> Vec<u8> {
    use core::fmt::Write as _;
    let mut s = String::new();
    // Magic (8 bytes) then a newline (aom_film_grain_table_write emits "\n").
    for e in entries {
        let p = &e.params;
        let _ = write!(
            s,
            "E {} {} {} {} {}\n",
            e.start_time, e.end_time, p.apply_grain as i32, p.random_seed, p.update_parameters as i32
        );
        if p.update_parameters {
            let _ = write!(
                s,
                "\tp {} {} {} {} {} {} {} {} {} {} {} {}\n",
                p.ar_coeff_lag,
                p.ar_coeff_shift,
                p.grain_scale_shift,
                p.scaling_shift,
                p.chroma_scaling_from_luma as i32,
                p.overlap_flag as i32,
                p.cb_mult,
                p.cb_luma_mult,
                p.cb_offset,
                p.cr_mult,
                p.cr_luma_mult,
                p.cr_offset
            );
            let _ = write!(s, "\tsY {} ", p.num_y_points);
            for pt in &p.scaling_points_y[..p.num_y_points as usize] {
                let _ = write!(s, " {} {}", pt[0], pt[1]);
            }
            let _ = write!(s, "\n\tsCb {}", p.num_cb_points);
            for pt in &p.scaling_points_cb[..p.num_cb_points as usize] {
                let _ = write!(s, " {} {}", pt[0], pt[1]);
            }
            let _ = write!(s, "\n\tsCr {}", p.num_cr_points);
            for pt in &p.scaling_points_cr[..p.num_cr_points as usize] {
                let _ = write!(s, " {} {}", pt[0], pt[1]);
            }
            let n = num_pos_luma(p.ar_coeff_lag);
            s.push_str("\n\tcY");
            for &c in &p.ar_coeffs_y[..n] {
                let _ = write!(s, " {}", c);
            }
            s.push_str("\n\tcCb");
            for &c in &p.ar_coeffs_cb[..=n] {
                let _ = write!(s, " {}", c);
            }
            s.push_str("\n\tcCr");
            for &c in &p.ar_coeffs_cr[..=n] {
                let _ = write!(s, " {}", c);
            }
            s.push('\n');
        }
    }
    let mut out = Vec::with_capacity(9 + s.len());
    out.extend_from_slice(FILE_MAGIC);
    out.push(b'\n');
    out.extend_from_slice(s.as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rich() -> FilmGrainParams {
        // Shape of film_grain_test_vectors[0] (lag 2, 14/8/9 points) — authored
        // here only to exercise the writer/reader round-trip; the e2e gate uses
        // C-written fixtures from the real built-in vectors.
        let mut p = FilmGrainParams::default();
        p.apply_grain = true;
        p.update_parameters = true;
        p.random_seed = 45231;
        p.num_y_points = 3;
        p.scaling_points_y[0] = [16, 0];
        p.scaling_points_y[1] = [64, 128];
        p.scaling_points_y[2] = [128, 200];
        p.num_cb_points = 2;
        p.scaling_points_cb[0] = [16, 0];
        p.scaling_points_cb[1] = [120, 100];
        p.num_cr_points = 2;
        p.scaling_points_cr[0] = [16, 0];
        p.scaling_points_cr[1] = [120, 100];
        p.scaling_shift = 11;
        p.ar_coeff_lag = 2;
        for i in 0..num_pos_luma(2) {
            p.ar_coeffs_y[i] = (i as i32) - 6;
        }
        for i in 0..=num_pos_luma(2) {
            p.ar_coeffs_cb[i] = (i as i32) - 6;
            p.ar_coeffs_cr[i] = 6 - (i as i32);
        }
        p.ar_coeff_shift = 8;
        p.grain_scale_shift = 0;
        p.cb_mult = 247;
        p.cb_luma_mult = 192;
        p.cb_offset = 18;
        p.cr_mult = 229;
        p.cr_luma_mult = 192;
        p.cr_offset = 54;
        p.overlap_flag = false;
        p
    }

    #[test]
    fn write_read_roundtrip() {
        let entries = vec![GrainTableEntry {
            params: sample_rich(),
            start_time: 0,
            end_time: i64::MAX,
        }];
        let bytes = write_film_grain_table(&entries);
        assert_eq!(&bytes[..8], FILE_MAGIC);
        let back = read_film_grain_table(&bytes).expect("reparse");
        // Re-serializing the parsed entries must reproduce the exact bytes
        // (read ∘ write is identity on every field the format carries).
        assert_eq!(bytes, write_film_grain_table(&back));
    }

    #[test]
    fn lookup_time0_uses_entry_seed() {
        let entries = vec![GrainTableEntry {
            params: sample_rich(),
            start_time: 0,
            end_time: i64::MAX,
        }];
        let mut out = FilmGrainParams::default();
        out.random_seed = 999; // running seed — must be IGNORED at time 0
        assert!(lookup(&entries, 0, &mut out));
        assert_eq!(out.random_seed, 45231);
        assert_eq!(out.num_y_points, 3);
    }

    #[test]
    fn lookup_miss_out_of_range() {
        let entries = vec![GrainTableEntry {
            params: sample_rich(),
            start_time: 10,
            end_time: 20,
        }];
        let mut out = FilmGrainParams::default();
        assert!(!lookup(&entries, 0, &mut out));
        assert!(!lookup(&entries, 20, &mut out));
        assert!(lookup(&entries, 10, &mut out));
    }

    #[test]
    fn apply_grain_off_entry() {
        // apply_grain=0, update_parameters=0 → only the E line, no param body.
        let mut p = FilmGrainParams::default();
        p.apply_grain = false;
        p.update_parameters = false;
        p.random_seed = 7;
        let entries = vec![GrainTableEntry { params: p, start_time: 0, end_time: 100 }];
        let bytes = write_film_grain_table(&entries);
        let back = read_film_grain_table(&bytes).expect("reparse");
        assert_eq!(back.len(), 1);
        assert!(!back[0].params.apply_grain);
        assert_eq!(bytes, write_film_grain_table(&back));
    }
}
