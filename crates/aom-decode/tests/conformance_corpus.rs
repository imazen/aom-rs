//! GATE-1 CONFORMANCE: the official AV1 decode-conformance vectors.
//!
//! Everything else in the decoder track validates against streams *we* asked
//! the C encoder to make (`real_bitstream.rs` and friends). Those prove the
//! port matches C on the encoder's own choices — but the port and that encoder
//! can share blind spots. This gate closes that: it decodes the standardized
//! AV1 conformance vectors (libaom v3.14.1's own `test-data.sha1` set, hosted
//! at `storage.googleapis.com/aom-test-data`) — bitstreams produced
//! INDEPENDENTLY of our tooling — and asserts three things per frame:
//!
//!   1. the port's decoded planes are byte-identical to the REAL C decoder
//!      (`aom_codec_av1_dx`, via [`aom_sys_ref::ref_decode_av1_kf`]) on the
//!      same bytes — the ground-truth gate;
//!   2. the C decoder's own planes reproduce the shipped golden per-frame MD5
//!      (`conformance/data/<name>.ivf.md5`), computed in libaom's exact
//!      `md5_helper.h::Add(aom_image_t*)` layout — this validates our MD5
//!      layout AND that the corpus bytes are intact;
//!   3. the port's planes reproduce that same golden MD5 — closing the loop
//!      independently of (1).
//!
//! # Scope — single-frame / all-intra / single-shown-KEY (caller-visible)
//!
//! The current decoder is KEY-frame, intra-only (no inter / motion comp). The
//! in-scope frame set is declared EXPLICITLY in [`CORPUS`] below (a
//! caller-visible manifest, not a silent in-test early-return): all-intra
//! vectors decode every frame; the `00-quantizer` vectors are `KEY,INTER` so
//! only frame 0 (the KEY frame) is in scope; the second (INTER) frame is
//! excluded by the manifest. Frame types were verified from the bitstream
//! headers (all listed in-scope frames are KEY, `reduced_still_picture=0`,
//! `show_existing_frame=0`, `frame_size_override=0`).
//!
//! Vectors exercised here (as fetched): the 39-frame 8-bit all-intra vector
//! (SB128 + CDEF + LR), six 10-bit `00-quantizer` vectors (SB128 + CDEF + LR +
//! high bit depth, frame 0 only), and the 8-bit intra-only intrabc "extreme
//! DV" vector (SB64 + intrabc, 1920x1080). These co-exercise SB128, 10-bit,
//! CDEF, loop restoration, and intrabc on streams our encoder never produced.
//!
//! # Corpus presence
//!
//! The `.ivf` bytes are gitignored (repo policy: no large binaries in git). The
//! corpus lives in `conformance/data/` and is fetched with
//! `python3 xtask/conformance.py --fetch --scope intra` (override the dir with
//! `AOM_CONFORMANCE_DIR`). If NOTHING is present the test FAILS LOUD (never a
//! silent pass) with the fetch command — per the no-graceful-skip rule.

use aom_decode::frame::{FrameDecode, decode_frame_obus};
use aom_sys_ref as c;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// RFC 1321 MD5 (safe Rust; self-tested below against known vectors). Used only
// to reproduce libaom's shipped golden per-frame hashes.
// ---------------------------------------------------------------------------
mod md5 {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    /// Streaming MD5 so we never materialize a whole-frame byte buffer.
    pub struct Md5 {
        a: [u32; 4],
        buf: [u8; 64],
        buf_len: usize,
        total: u64,
    }
    impl Md5 {
        pub fn new() -> Self {
            Md5 {
                a: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
                buf: [0; 64],
                buf_len: 0,
                total: 0,
            }
        }
        pub fn update(&mut self, mut data: &[u8]) {
            self.total = self.total.wrapping_add(data.len() as u64);
            if self.buf_len > 0 {
                let need = 64 - self.buf_len;
                let take = need.min(data.len());
                self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
                self.buf_len += take;
                data = &data[take..];
                if self.buf_len == 64 {
                    let block = self.buf;
                    self.process(&block);
                    self.buf_len = 0;
                }
            }
            while data.len() >= 64 {
                let mut block = [0u8; 64];
                block.copy_from_slice(&data[..64]);
                self.process(&block);
                data = &data[64..];
            }
            if !data.is_empty() {
                self.buf[..data.len()].copy_from_slice(data);
                self.buf_len = data.len();
            }
        }
        pub fn finish(mut self) -> String {
            let bitlen = self.total.wrapping_mul(8);
            let mut pad = vec![0x80u8];
            while (self.total.wrapping_add(pad.len() as u64)) % 64 != 56 {
                pad.push(0);
            }
            pad.extend_from_slice(&bitlen.to_le_bytes());
            self.update(&pad);
            debug_assert_eq!(self.buf_len, 0);
            let mut out = String::with_capacity(32);
            for word in self.a {
                for byte in word.to_le_bytes() {
                    out.push_str(&format!("{byte:02x}"));
                }
            }
            out
        }
        fn process(&mut self, chunk: &[u8; 64]) {
            let mut m = [0u32; 16];
            for i in 0..16 {
                m[i] = u32::from_le_bytes([
                    chunk[i * 4],
                    chunk[i * 4 + 1],
                    chunk[i * 4 + 2],
                    chunk[i * 4 + 3],
                ]);
            }
            let [mut a, mut b, mut c, mut d] = self.a;
            for i in 0..64 {
                let (f, g) = match i {
                    0..=15 => ((b & c) | (!b & d), i),
                    16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                    32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | !d), (7 * i) % 16),
                };
                let f = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
                a = d;
                d = c;
                c = b;
                b = b.wrapping_add(f.rotate_left(S[i]));
            }
            self.a[0] = self.a[0].wrapping_add(a);
            self.a[1] = self.a[1].wrapping_add(b);
            self.a[2] = self.a[2].wrapping_add(c);
            self.a[3] = self.a[3].wrapping_add(d);
        }
    }

    pub fn hex(data: &[u8]) -> String {
        let mut m = Md5::new();
        m.update(data);
        m.finish()
    }
}

/// libaom `md5_helper.h::Add(aom_image_t*)`: hash each plane's cropped rows,
/// 2 bytes/sample little-endian at high bit depth. Chroma dims round UP
/// (`(d + shift) >> shift`). Planes are tightly packed (stride == width) here,
/// which is byte-identical to libaom hashing `w` bytes per strided row.
fn image_md5(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    ss_x: usize,
    ss_y: usize,
    monochrome: bool,
) -> String {
    let mut m = md5::Md5::new();
    let hi = bd > 8;
    let push_plane = |m: &mut md5::Md5, plane: &[u16], pw: usize, ph: usize| {
        assert_eq!(plane.len(), pw * ph, "plane size mismatch");
        let mut row = Vec::with_capacity(pw * if hi { 2 } else { 1 });
        for r in 0..ph {
            row.clear();
            for &s in &plane[r * pw..r * pw + pw] {
                if hi {
                    row.extend_from_slice(&s.to_le_bytes());
                } else {
                    row.push(s as u8);
                }
            }
            m.update(&row);
        }
    };
    push_plane(&mut m, y, w, h);
    if monochrome {
        // libaom emits a decoded monochrome frame as I420 with the two chroma
        // planes filled with the neutral value 1<<(bd-1) at 4:2:0, and the
        // shipped golden .md5 (named "...-WxH-NNNN.i420") hashes exactly that
        // Y + neutral-U + neutral-V image. The C decode shim returns EMPTY
        // chroma for monochrome, so synthesise the same neutral planes here to
        // reproduce the golden (the port likewise carries no chroma, so this
        // affects only the golden-integrity check, not the port-vs-C compare).
        let (cw, ch) = ((w + 1) >> 1, (h + 1) >> 1);
        let neutral = vec![1u16 << (bd - 1); cw * ch];
        push_plane(&mut m, &neutral, cw, ch);
        push_plane(&mut m, &neutral, cw, ch);
    } else {
        let cw = (w + ss_x) >> ss_x;
        let ch = (h + ss_y) >> ss_y;
        push_plane(&mut m, u, cw, ch);
        push_plane(&mut m, v, cw, ch);
    }
    m.finish()
}

/// Split an IVF container into per-frame temporal-unit payloads (raw OBU bytes).
fn ivf_temporal_units(data: &[u8]) -> Vec<Vec<u8>> {
    assert!(
        data.len() >= 32 && &data[0..4] == b"DKIF",
        "not an IVF file"
    );
    let hdr_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut off = hdr_len;
    let mut tus = Vec::new();
    while off + 12 <= data.len() {
        let sz =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize;
        off += 12; // 4-byte size + 8-byte timestamp
        assert!(off + sz <= data.len(), "IVF frame runs past end of file");
        tus.push(data[off..off + sz].to_vec());
        off += sz;
    }
    tus
}

/// Parse `<name>.ivf.md5`: one `"<md5hex>  <name>-WxH-NNNN.i420"` line per
/// decoded frame, in frame order.
fn parse_golden(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .next()
                .expect("golden md5 line empty")
                .to_ascii_lowercase()
        })
        .collect()
}

/// Which frames of a vector are in scope (caller-visible; never a runtime skip).
#[derive(Clone, Copy)]
enum Scope {
    /// Every frame is KEY/intra — decode them all.
    AllIntra,
    /// First `n` frames are KEY; the rest need inter tooling — excluded here.
    FirstKey(usize),
    /// The whole vector is out of the current envelope (inter / cross-frame CDF
    /// carry); documented reason, contributes zero in-scope frames.
    OutOfEnvelope(&'static str),
}

/// Caller-visible family classification. Every vector name a fetch can produce
/// is classified EXPLICITLY here (verified from the bitstream frame types).
/// A present vector whose family is not listed FAILS LOUD — new families must
/// be classified deliberately, never silently skipped.
fn scope_for(name: &str) -> Option<Scope> {
    if name.contains("-02-allintra") {
        Some(Scope::AllIntra) // every frame KEY (SB128 + CDEF + LR)
    } else if name.contains("-16-intra_only") {
        Some(Scope::AllIntra) // KEY frames using intrabc
    } else if name.contains("-00-quantizer-") {
        Some(Scope::FirstKey(1)) // KEY,INTER — frame 0 only
    } else if name.contains("-01-size-") {
        Some(Scope::FirstKey(1)) // KEY,INTER — frame 0 only
    } else if name.contains("-23-film_grain") {
        Some(Scope::FirstKey(1)) // KEY,INTER — frame 0 (film grain synthesised on output)
    } else if name.contains("-24-monochrome") {
        Some(Scope::FirstKey(1)) // KEY,INTER — frame 0 (monochrome: Y plane only)
    } else if name.contains("-04-cdfupdate") {
        // The stream USES cross-frame CDF adaptation (frames >=1 carry state the
        // stateless single-frame decoder can't reproduce), but frame 0 is an
        // ordinary KEY frame decoded from default CDFs — single-frame in scope.
        Some(Scope::FirstKey(1))
    } else if name.contains("-05-mv") || name.contains("-06-mfmv") || name.contains("-22-svc") {
        Some(Scope::OutOfEnvelope(
            "inter: motion compensation / multi-ref",
        ))
    } else {
        None
    }
}

/// IVF display dimensions (header bytes 12..16, little-endian). Verified to
/// match the coded frame size across the whole `01-size` family.
fn ivf_hdr_dims(data: &[u8]) -> (usize, usize) {
    (
        u16::from_le_bytes([data[12], data[13]]) as usize,
        u16::from_le_bytes([data[14], data[15]]) as usize,
    )
}

fn corpus_dir() -> PathBuf {
    if let Ok(d) = std::env::var("AOM_CONFORMANCE_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("conformance")
        .join("data")
}

#[test]
fn md5_self_test() {
    // Anchor the MD5 implementation against RFC 1321 test vectors before it is
    // trusted to reproduce the shipped goldens.
    assert_eq!(md5::hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(md5::hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    assert_eq!(
        md5::hex(b"The quick brown fox jumps over the lazy dog"),
        "9e107d9d372bb6826bd81d3542a419d6"
    );
    // Cross a 64-byte block boundary (56..64 padding edge) and a multi-block msg.
    assert_eq!(
        md5::hex(
            b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"
        ),
        "57edf4a22be3c955ac49da2e2107b67a"
    );
}

/// Explicit regression for the two decode bugs that only surface on coding
/// blocks LARGER than 64x64: (1) the txb 64x64-chunk plane-interleave order in
/// `TileKf` reconstruction (`decode_token_recon_block`, decodeframe.c:929-962),
/// and (2) per-64x64-unit CDEF strength stamping in `apply_cdef` (a >64 block
/// shares one MB_MODE_INFO across all its 64x64 units, so every covered unit
/// gets the strength — cdef.c:304). Blocks >64x64 are chosen only at the
/// aggressive high end of the quantizer range, so `quantizer-62`/`quantizer-63`
/// are the vectors that exercise them; both are covered at bd8 AND bd10 because
/// the bugs are bit-depth-independent. The main corpus gate above already
/// asserts these, but naming the exact failure envelope pins the regression so
/// it cannot silently vanish if the corpus enumeration changes.
#[test]
fn high_qindex_gt64_partition_byte_identical_to_c() {
    let dir = corpus_dir();
    let vectors = [
        "av1-1-b8-00-quantizer-62",
        "av1-1-b8-00-quantizer-63",
        "av1-1-b10-00-quantizer-62",
        "av1-1-b10-00-quantizer-63",
    ];
    for name in vectors {
        let path = dir.join(format!("{name}.ivf"));
        // Fail LOUD if the vector is absent: CI provisions the intra scope
        // (which includes 00-quantizer) deterministically, so a missing file is
        // a provisioning bug, never a reason to skip the assertion silently.
        let ivf = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "{name}: conformance vector missing at {path:?} ({e}); fetch via \
                 `python3 xtask/conformance.py --fetch --scope intra`"
            )
        });
        let (w, h) = ivf_hdr_dims(&ivf);
        let tus = ivf_temporal_units(&ivf);
        // `00-quantizer` streams are KEY,INTER; only the first (KEY) frame is in
        // the current intra decoder envelope.
        let tu = &tus[0];
        let cref = c::ref_decode_av1_kf(tu, w, h);
        let rust = decode_frame_obus(tu)
            .unwrap_or_else(|e| panic!("{name}: port rejected the KEY frame: {e}"));
        assert_eq!(rust.y, cref.y, "{name} ({w}x{h}): luma differs from C oracle");
        assert_eq!(rust.u, cref.u, "{name} ({w}x{h}): U differs from C oracle");
        assert_eq!(rust.v, cref.v, "{name} ({w}x{h}): V differs from C oracle");
    }
}

/// Enumerate the `av1-1-*.ivf` conformance vectors present in `dir`, sorted.
fn present_vectors(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter_map(|n| {
                n.strip_prefix("")
                    .and_then(|n| n.strip_suffix(".ivf"))
                    .filter(|n| n.starts_with("av1-1-"))
                    .map(str::to_owned)
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    names.sort();
    names
}

#[test]
fn conformance_single_frame_intra_byte_identical_to_c_and_golden() {
    let dir = corpus_dir();
    let names = present_vectors(&dir);

    // Presence is caller-visible: FAIL LOUD if nothing is fetched (never a
    // silent pass). Which subset is present is a corpus-management fact; the
    // decoder SCOPE (KEY-only) is fixed per-family in `scope_for`.
    assert!(
        !names.is_empty(),
        "no conformance vectors found in {}\n\
         fetch them first:  python3 xtask/conformance.py --fetch --scope intra\n\
         (or point AOM_CONFORMANCE_DIR at a populated corpus dir)",
        dir.display()
    );

    let mut report = String::new();
    let mut failures: Vec<String> = Vec::new();
    let mut frames_checked = 0usize;
    let mut vectors_in_scope = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    // Coverage witnesses (anti-vacuous: the corpus must genuinely exercise
    // high bit depth, CDEF, loop restoration, intrabc, odd frame sizes,
    // monochrome, and film-grain synthesis).
    let (mut saw_bd10, mut saw_cdef, mut saw_lr, mut saw_intrabc) = (false, false, false, false);
    // Stronger CDEF witness: a frame that actually READ a per-SB strength literal
    // (`cdef_bits > 0`), not merely a `cdef_bits == 0` single-strength frame (which
    // reads zero bits). This is the anti-vacuity proof that the intra KEY-frame
    // `read_cdef` path (`decode_block` -> `read_mb_modes_kf_fc` ->
    // `read_mb_modes_kf_prefix` -> `read_cdef`) was exercised with a LIVE read: a
    // `cdef_bits > 0` KEY frame decoded byte-identical to C means the per-SB read is
    // consuming the exact bits C does (an omitted read would desync every following
    // symbol). Kept distinct from `saw_cdef` so a corpus of only `cdef_bits == 0`
    // CDEF frames can never satisfy it.
    let mut saw_cdef_bits_gt0 = false;
    let mut saw_odd_size = false;
    let (mut saw_monochrome, mut saw_film_grain) = (false, false);

    for name in &names {
        let scope = match scope_for(name) {
            Some(s) => s,
            None => {
                // Unknown family present in the corpus dir — do NOT guess a
                // scope. Force explicit classification.
                failures.push(format!(
                    "{name}: unclassified conformance family — add it to scope_for()"
                ));
                continue;
            }
        };
        if let Scope::OutOfEnvelope(reason) = scope {
            skipped.push(format!("{name} ({reason})"));
            continue;
        }
        vectors_in_scope += 1;

        let ivf = std::fs::read(dir.join(format!("{name}.ivf"))).unwrap();
        let golden_txt = std::fs::read_to_string(dir.join(format!("{name}.ivf.md5"))).unwrap();
        let golden = parse_golden(&golden_txt);
        let tus = ivf_temporal_units(&ivf);
        assert_eq!(
            tus.len(),
            golden.len(),
            "{name}: {} temporal units but {} golden md5 lines",
            tus.len(),
            golden.len()
        );
        let (w, h) = ivf_hdr_dims(&ivf);

        let in_scope: Vec<usize> = match scope {
            Scope::AllIntra => (0..tus.len()).collect(),
            Scope::FirstKey(n) => (0..n.min(tus.len())).collect(),
            Scope::OutOfEnvelope(_) => unreachable!(),
        };

        for &i in &in_scope {
            let tu = &tus[i];
            let where_ = format!("{name} frame {i}");

            // (2) C reference decode -> derive (bd, mono, ss) and validate the
            // golden layout + corpus integrity from the ground-truth decoder.
            let cref = c::ref_decode_av1_kf(tu, w, h);
            let bd = cref.info[0];
            let mono = cref.info[1] != 0;
            let (ss_x, ss_y) = (cref.info[2] as usize, cref.info[3] as usize);
            let md5_c = image_md5(&cref.y, &cref.u, &cref.v, w, h, bd, ss_x, ss_y, mono);
            if md5_c != golden[i] {
                failures.push(format!(
                    "{where_}: C-decoder MD5 {md5_c} != golden {} (corpus/layout mismatch)",
                    golden[i]
                ));
                continue;
            }

            // (1) Port decode -> byte-identical to C.
            let rust: FrameDecode = match decode_frame_obus(tu) {
                Ok(fd) => fd,
                Err(e) => {
                    failures.push(format!("{where_}: port rejected in-scope frame: {e}"));
                    continue;
                }
            };
            let mut frame_ok = true;
            if rust.y != cref.y {
                let n = rust
                    .y
                    .iter()
                    .zip(&cref.y)
                    .take_while(|(a, b)| a == b)
                    .count();
                let (x, yy) = (n % w, n / w);
                failures.push(format!(
                    "{where_} ({w}x{h} bd{bd}): LUMA differs at pixel {n} (x={x}, y={yy}) port={} c={}",
                    rust.y.get(n).copied().unwrap_or(0),
                    cref.y.get(n).copied().unwrap_or(0)
                ));
                frame_ok = false;
            }
            if !mono && rust.u != cref.u {
                let n = rust
                    .u
                    .iter()
                    .zip(&cref.u)
                    .take_while(|(a, b)| a == b)
                    .count();
                failures.push(format!(
                    "{where_} ({w}x{h} bd{bd}): U differs at chroma sample {n}"
                ));
                frame_ok = false;
            }
            if !mono && rust.v != cref.v {
                let n = rust
                    .v
                    .iter()
                    .zip(&cref.v)
                    .take_while(|(a, b)| a == b)
                    .count();
                failures.push(format!(
                    "{where_} ({w}x{h} bd{bd}): V differs at chroma sample {n}"
                ));
                frame_ok = false;
            }
            if !frame_ok {
                continue;
            }

            // (3) Port planes reproduce the golden MD5 independently.
            let md5_r = image_md5(
                &rust.y,
                &rust.u,
                &rust.v,
                w,
                h,
                bd,
                ss_x,
                ss_y,
                rust.monochrome,
            );
            if md5_r != golden[i] {
                failures.push(format!(
                    "{where_}: port MD5 {md5_r} != golden {}",
                    golden[i]
                ));
                continue;
            }

            // Coverage witnesses from the port's own parse.
            saw_bd10 |= rust.bit_depth >= 10;
            saw_cdef |= rust.cdef_bits != 0
                || rust.cdef_strengths[0] != 0
                || rust.cdef_uv_strengths[0] != 0;
            saw_cdef_bits_gt0 |= rust.cdef_bits > 0;
            saw_lr |= rust.lr_frame_restoration_type.iter().any(|&t| t != 0);
            saw_intrabc |= name.contains("intrabc");
            saw_odd_size |= w % 8 != 0 || h % 8 != 0;
            saw_monochrome |= rust.monochrome;
            // Name-based (like intrabc): the golden for the -23-film_grain frame
            // is the grain-APPLIED md5 -- verified distinct from `aomdec
            // --skip-film-grain` (9fc2c754.. vs d10c5738.. on b8 frame 0) -- so
            // the byte-match above IS the functional grain gate; this witness
            // just proves the vector was reached, not silently absent.
            saw_film_grain |= name.contains("film_grain");

            frames_checked += 1;
            report.push_str(&format!("  OK  {where_}  ({w}x{h} bd{bd})\n"));
        }
    }

    eprintln!(
        "conformance: {} in-scope vectors, {} frames byte-identical (port==C==golden); {} out-of-envelope vectors skipped{}",
        vectors_in_scope,
        frames_checked,
        skipped.len(),
        if skipped.is_empty() {
            String::new()
        } else {
            format!(" [{}]", skipped.join(", "))
        }
    );
    eprint!("{report}");

    assert!(
        failures.is_empty(),
        "conformance gate: {} frame(s) FAILED:\n{}",
        failures.len(),
        failures.join("\n")
    );

    // Anti-vacuous floors: prove the run actually exercised the tools the
    // corpus carries. These fire only when the relevant vectors are present.
    assert!(frames_checked > 0, "no in-scope frames were checked");
    let has = |sub: &str| names.iter().any(|n| n.contains(sub));
    if has("-b10-") {
        assert!(
            saw_bd10,
            "10-bit vectors present but no bd>=10 frame verified"
        );
    }
    if has("intrabc") {
        assert!(saw_intrabc, "intrabc vector present but not verified");
    }
    if has("-02-allintra") || has("-00-quantizer") {
        assert!(
            saw_cdef,
            "CDEF-carrying vectors present but no CDEF frame verified"
        );
        // Prove the per-SB CDEF strength READ (intra `read_cdef`) was exercised —
        // not just a `cdef_bits == 0` frame. Without this, the corpus could pass
        // while the intra KEY path never consumed a per-SB strength literal, which
        // is exactly the latent-desync blind spot the weaker `saw_cdef` left open.
        assert!(
            saw_cdef_bits_gt0,
            "CDEF-carrying vectors present but no `cdef_bits > 0` KEY frame was \
             decoded byte-identical — the intra per-SB CDEF strength read \
             (read_mb_modes_kf_fc -> read_mb_modes_kf_prefix -> read_cdef) is \
             UNPROVEN; a cdef_bits>0 KEY frame must be exercised"
        );
        assert!(
            saw_lr,
            "LR-carrying vectors present but no LR frame verified"
        );
    }
    if has("-01-size-") {
        assert!(
            saw_odd_size,
            "size vectors present but no non-8-multiple frame verified"
        );
    }
    if has("-24-monochrome") {
        assert!(
            saw_monochrome,
            "monochrome vector present but no monochrome frame verified"
        );
    }
    if has("film_grain") {
        assert!(saw_film_grain, "film-grain vector present but not verified");
    }
}
