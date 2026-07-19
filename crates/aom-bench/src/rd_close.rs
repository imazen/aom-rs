//! RD-CLOSENESS harness — the shared validation gate for the stills-parity
//! bulk-port wave.
//!
//! The byte-exact gates (`encoder_gate_*`) prove the port IS libaom for the
//! configurations they cover. Bulk-ported features land in a weaker but still
//! honest state first: **RD-close** — same knobs, same input, and the port's
//! encode must match the real aomenc encode in BOTH compressed size and
//! perceptual quality within tight bands, without yet being bit-identical.
//! A feature graduates from RD-close to byte-exact later (PARITY.md tracks
//! the ledger; rows move from section B to section A).
//!
//! What one cell does:
//! 1. Encode the SAME source pixels with the port (feature enabled) and with
//!    real aomenc / `ref_encode_av1_kf*` (same knobs). The caller supplies
//!    both full temporal-unit byte streams — see [`splice_frame_obu`] for
//!    wrapping a port-produced frame OBU payload into a decodable stream.
//! 2. Decode BOTH streams with the PORT decoder (bit-exact vs C across the
//!    conformance corpus — the proven half of the project), so decoder error
//!    cannot differ between the two sides.
//! 3. Score each reconstruction against the ORIGINAL source with
//!    **zensim** (single-threaded for determinism; 100 = identical, higher
//!    is better), after converting source + both recons with the SAME
//!    YUV→RGB transform ([`yuv_to_rgb8`]) so the colorimetry approximation
//!    cancels in the delta.
//! 4. Report per cell: `size_port`, `size_c`, `size_delta_pct`,
//!    `zensim_port`, `zensim_c`, `zensim_drop` — and a BIT-IDENTICAL fast
//!    path: byte-equal streams are recorded as `EXACT` (components that
//!    happen to come out exact get credited as such).
//!
//! Acceptance bands ([`RdBands::default`]): `|size_delta| <= 5%` AND
//! `zensim_drop <= 0.5` (port may be *better* — negative drop always
//! passes). Chosen from the first real data on the proven envelope
//! (byte-exact cells report 0/0; a cq32→cq63 quality jump on real 64×64
//! content moves zensim by tens of points and size by >40%, so the bands
//! discriminate near-ties from real regressions with wide margin — see
//! `tests/rd_close_harness.rs`). Tighten per family if a feature's first
//! measured deltas come in far under band; never widen without user sign-off
//! (that's a test relaxation).
//!
//! CAVEAT the bulk agents must respect: the port's stock encode
//! ([`crate::EncodeCell::port_encode`]) bootstraps frame-header FIELDS
//! (qindex mapping, tile limits, …) from the C stream. That is fine for
//! RD-closeness — but the FEATURE UNDER TEST must not leak through the
//! bootstrap. A CDEF-search port must derive its own strengths, an LR-search
//! port its own RU params, etc.; copying those header fields from the C
//! stream would fake parity. Derive the feature's decisions in the port and
//! write them yourself.
//!
//! Typical use (see `tests/rd_close_harness.rs` for the runnable version):
//!
//! ```ignore
//! let cell = EncodeCell::real_content("cdef_64_cq32", "av1-1-b8-01-size-64x64", None, 32, 0);
//! let c_tu = c::ref_encode_av1_kf(.., /*enable_cdef=*/ true, ..);
//! let port_payload = my_cdef_enabled_port_encode(&cell, &c_tu);
//! let port_tu = rd_close::splice_frame_obu(&c_tu, &port_payload);
//! results.push(rd_close::compare_cell(&cell.label, &cell, &port_tu, &c_tu));
//! // ... more cells ...
//! rd_close::assert_rd_close(&results, &RdBands::default());
//! ```

use aom_decode::frame::FrameDecode;
use aom_dsp::entropy::leb128;
use aom_dsp::entropy::obu::read_obu_header;
use zensim::{RgbSlice, Zensim, ZensimProfile};

use crate::EncodeCell;

/// OBU_FRAME (combined frame header + tile group), the payload the port's
/// encode pipeline produces.
const OBU_FRAME: u32 = 6;

// ---------------------------------------------------------------------------
// Acceptance bands
// ---------------------------------------------------------------------------

/// Acceptance bands for the bulk-port RD-closeness gate.
#[derive(Debug, Clone, Copy)]
pub struct RdBands {
    /// Maximum |size delta| in percent of the C stream size.
    pub max_size_delta_pct: f64,
    /// Maximum zensim drop (`zensim_c - zensim_port`; positive = port worse).
    /// A port that scores BETTER than C (negative drop) always passes this
    /// axis — the band only guards quality REGRESSIONS vs C.
    pub max_zensim_drop: f64,
}

impl Default for RdBands {
    /// The documented bulk-port gate: |size| <= 5%, zensim drop <= 0.5.
    fn default() -> Self {
        RdBands {
            max_size_delta_pct: 5.0,
            max_zensim_drop: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-cell result
// ---------------------------------------------------------------------------

/// One RD-closeness comparison cell.
#[derive(Debug, Clone)]
pub struct RdCellResult {
    pub label: String,
    /// Full temporal-unit byte size of the port stream.
    pub size_port: usize,
    /// Full temporal-unit byte size of the real aomenc stream.
    pub size_c: usize,
    /// `(size_port - size_c) / size_c * 100`.
    pub size_delta_pct: f64,
    /// zensim(source, port recon) — 100 = identical, higher is better.
    pub zensim_port: f64,
    /// zensim(source, C recon).
    pub zensim_c: f64,
    /// `zensim_c - zensim_port`; positive = the port's recon is worse.
    pub zensim_drop: f64,
    /// The two streams were byte-identical (the fast path: recorded EXACT,
    /// deltas are structurally zero).
    pub bit_identical: bool,
}

impl RdCellResult {
    /// Whether this cell passes the given bands (bit-identical always does).
    pub fn within(&self, bands: &RdBands) -> bool {
        self.bit_identical
            || (self.size_delta_pct.abs() <= bands.max_size_delta_pct
                && self.zensim_drop <= bands.max_zensim_drop)
    }

    /// Verdict string for the report table.
    pub fn verdict(&self, bands: &RdBands) -> &'static str {
        if self.bit_identical {
            "EXACT"
        } else if self.within(bands) {
            "CLOSE"
        } else {
            "FAIL"
        }
    }
}

// ---------------------------------------------------------------------------
// Stream plumbing
// ---------------------------------------------------------------------------

/// Rewrap a port-produced frame OBU **payload** into a full decodable
/// temporal unit by splicing it into `reference_stream` (a real aomenc
/// output) in place of the reference's own OBU_FRAME payload. Every other
/// OBU (temporal delimiter, sequence header, …) is copied byte-for-byte, so
/// a size comparison between the spliced stream and the reference isolates
/// the frame OBU delta.
pub fn splice_frame_obu(reference_stream: &[u8], frame_obu_payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(reference_stream.len() + frame_obu_payload.len());
    let mut pos = 0usize;
    let mut spliced = false;
    while pos < reference_stream.len() {
        let hdr = read_obu_header(&reference_stream[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        assert!(hdr.obu_has_size_field, "shim streams always carry sizes");
        let (size, size_bytes) =
            leb128::uleb_decode(&reference_stream[after_header..]).expect("valid leb128 OBU size");
        let payload_end = after_header + size_bytes + size as usize;
        if hdr.obu_type == OBU_FRAME {
            assert!(!spliced, "reference stream has more than one frame OBU");
            spliced = true;
            // Original OBU header byte(s) + re-encoded size + the new payload.
            out.extend_from_slice(&reference_stream[pos..after_header]);
            let size_enc = leb128::uleb_encode(frame_obu_payload.len() as u64, 8)
                .expect("frame payload size encodes in <= 8 leb128 bytes");
            out.extend_from_slice(&size_enc);
            out.extend_from_slice(frame_obu_payload);
        } else {
            out.extend_from_slice(&reference_stream[pos..payload_end]);
        }
        pos = payload_end;
    }
    assert!(spliced, "reference stream has no frame OBU");
    out
}

// ---------------------------------------------------------------------------
// YUV -> RGB (shared by source and both recons so the transform cancels)
// ---------------------------------------------------------------------------

/// Planar YUV (u16 at any bit depth, tight strides) → interleaved 8-bit RGB.
///
/// BT.601 limited-range integer conversion with nearest-neighbour chroma
/// upsampling; bit depths above 8 are rounded down to 8 bits first. This is
/// deliberately a FIXED, simple transform: zensim consumes RGB, and as long
/// as the source and both reconstructions go through the SAME transform the
/// colorimetry approximation cancels out of the port-vs-C delta. Monochrome
/// replicates luma to all three channels.
#[allow(clippy::too_many_arguments)]
pub fn yuv_to_rgb8(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    bd: u8,
) -> Vec<[u8; 3]> {
    assert!(bd >= 8, "bit depth below 8 is not a thing in AV1");
    assert_eq!(y.len(), w * h, "luma plane must be tight w*h");
    let cw = if mono { 0 } else { (w + ss_x) >> ss_x };
    let shift = u32::from(bd) - 8;
    let round = if shift > 0 { 1u32 << (shift - 1) } else { 0 };
    let to8 = |px: u16| -> i32 { (((u32::from(px) + round) >> shift).min(255)) as i32 };
    let clip = |x: i32| -> u8 { x.clamp(0, 255) as u8 };
    let mut out = vec![[0u8; 3]; w * h];
    for r in 0..h {
        for col in 0..w {
            let c = to8(y[r * w + col]) - 16;
            let (d, e) = if mono {
                (0, 0)
            } else {
                let ci = (r >> ss_y) * cw + (col >> ss_x);
                (to8(u[ci]) - 128, to8(v[ci]) - 128)
            };
            out[r * w + col] = [
                clip((298 * c + 409 * e + 128) >> 8),
                clip((298 * c - 100 * d - 208 * e + 128) >> 8),
                clip((298 * c + 516 * d + 128) >> 8),
            ];
        }
    }
    out
}

fn rgb_of_decode(f: &FrameDecode) -> Vec<[u8; 3]> {
    yuv_to_rgb8(
        &f.y,
        &f.u,
        &f.v,
        f.width,
        f.height,
        f.monochrome,
        f.subsampling_x,
        f.subsampling_y,
        f.bit_depth as u8,
    )
}

// ---------------------------------------------------------------------------
// The comparison core
// ---------------------------------------------------------------------------

/// Decode a full temporal unit with the PORT decoder (panics with the cell
/// label if the port rejects it — a feature whose streams the port decoder
/// cannot decode is not RD-closeable yet; extend the decoder envelope first).
pub fn port_decode_tu(label: &str, tu: &[u8]) -> FrameDecode {
    aom_decode::frame::decode_frame_obus(tu)
        .unwrap_or_else(|e| panic!("{label}: port decoder rejected the stream: {e}"))
}

/// Compare a port stream against the C reference stream for the same source.
///
/// `src` carries the ORIGINAL pixels (and geometry) both encoders consumed;
/// `port_tu` / `c_tu` are full temporal units (see [`splice_frame_obu`]).
/// Both are decoded with the port decoder and scored with zensim against the
/// source. Byte-identical streams take the fast path (decode+score once).
pub fn compare_cell(label: &str, src: &EncodeCell, port_tu: &[u8], c_tu: &[u8]) -> RdCellResult {
    let bit_identical = port_tu == c_tu;

    let c_dec = port_decode_tu(&format!("{label}/c"), c_tu);
    assert_eq!(
        (c_dec.width, c_dec.height, c_dec.monochrome),
        (src.w, src.h, src.mono),
        "{label}: C stream geometry differs from the source cell"
    );
    let src_rgb = yuv_to_rgb8(
        &src.y, &src.u, &src.v, src.w, src.h, src.mono, src.ss_x, src.ss_y, src.bd,
    );
    let src_slice = RgbSlice::new(&src_rgb, src.w, src.h);
    // Single-threaded zensim: deterministic scores (no parallel-reduction
    // float-order variance) — these numbers land in gates and in PARITY.md.
    let z = Zensim::new(ZensimProfile::latest()).with_parallel(false);

    let c_rgb = rgb_of_decode(&c_dec);
    let zensim_c = z
        .compute(
            &src_slice,
            &RgbSlice::new(&c_rgb, c_dec.width, c_dec.height),
        )
        .unwrap_or_else(|e| panic!("{label}: zensim on the C recon failed: {e:?}"))
        .score();

    let zensim_port = if bit_identical {
        zensim_c
    } else {
        let p_dec = port_decode_tu(&format!("{label}/port"), port_tu);
        assert_eq!(
            (p_dec.width, p_dec.height, p_dec.monochrome),
            (src.w, src.h, src.mono),
            "{label}: port stream geometry differs from the source cell"
        );
        let p_rgb = rgb_of_decode(&p_dec);
        z.compute(
            &src_slice,
            &RgbSlice::new(&p_rgb, p_dec.width, p_dec.height),
        )
        .unwrap_or_else(|e| panic!("{label}: zensim on the port recon failed: {e:?}"))
        .score()
    };

    let size_port = port_tu.len();
    let size_c = c_tu.len();
    RdCellResult {
        label: label.to_string(),
        size_port,
        size_c,
        size_delta_pct: (size_port as f64 - size_c as f64) / size_c as f64 * 100.0,
        zensim_port,
        zensim_c,
        zensim_drop: zensim_c - zensim_port,
        bit_identical,
    }
}

/// The zero-knob convenience: run one cell through the port's STOCK encode
/// pipeline ([`EncodeCell::port_encode`], header bootstrapped from the C
/// stream) and compare. Bulk agents with new knobs build their own streams
/// and call [`compare_cell`] directly.
pub fn run_stock_cell(cell: &EncodeCell) -> RdCellResult {
    aom_sys_ref::ref_init();
    let c_tu = cell.c_encode();
    assert!(!c_tu.is_empty(), "{}: C encode failed", cell.label);
    let port_payload = cell.port_encode(&c_tu);
    let port_tu = splice_frame_obu(&c_tu, &port_payload);
    compare_cell(&cell.label, cell, &port_tu, &c_tu)
}

// ---------------------------------------------------------------------------
// Reporting + the gate assert
// ---------------------------------------------------------------------------

/// Render the per-cell report table (markdown-ish, aligned).
pub fn render_table(results: &[RdCellResult], bands: &RdBands) -> String {
    let mut s = String::new();
    let label_w = results
        .iter()
        .map(|r| r.label.len())
        .chain(std::iter::once("cell".len()))
        .max()
        .unwrap_or(4);
    s.push_str(&format!(
        "{:<label_w$} | size_port |    size_c | Δsize%   | zensim_port | zensim_c | Δzensim  | verdict\n",
        "cell"
    ));
    s.push_str(&format!(
        "{:-<label_w$}-+-----------+-----------+----------+-------------+----------+----------+--------\n",
        ""
    ));
    for r in results {
        s.push_str(&format!(
            "{:<label_w$} | {:>9} | {:>9} | {:>+7.2}% | {:>11.3} | {:>8.3} | {:>+8.3} | {}\n",
            r.label,
            r.size_port,
            r.size_c,
            r.size_delta_pct,
            r.zensim_port,
            r.zensim_c,
            r.zensim_drop,
            r.verdict(bands),
        ));
    }
    s.push_str(&format!(
        "bands: |Δsize| <= {:.1}%  AND  Δzensim <= {:.2}  (EXACT = byte-identical fast path)\n",
        bands.max_size_delta_pct, bands.max_zensim_drop
    ));
    s
}

/// Print the table and assert every cell is within the bands. The table is
/// embedded in the panic message too, so a failing CI log always shows the
/// full per-cell picture.
pub fn assert_rd_close(results: &[RdCellResult], bands: &RdBands) {
    assert!(!results.is_empty(), "assert_rd_close: no cells were run");
    let table = render_table(results, bands);
    println!("{table}");
    let failing: Vec<&RdCellResult> = results.iter().filter(|r| !r.within(bands)).collect();
    assert!(
        failing.is_empty(),
        "{} of {} RD-closeness cells out of band:\n{}",
        failing.len(),
        results.len(),
        table
    );
}
