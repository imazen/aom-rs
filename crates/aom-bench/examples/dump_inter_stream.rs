//! Dump the "simplest inter config" (INTER-ENCODE-ROADMAP.md §3) 2-frame
//! `[KEY, P]` `aomenc` stream to an IVF file, so the instrumented libaom
//! decoder (`/root/aom-inspect/examples/inspect`, CONFIG_INSPECTION=1) can be
//! pointed at it to read C's OWN per-block partition / mode / MV / ref / skip
//! decisions for the P frame.
//!
//! This is the inter-ENCODE track's ground-truth tool: before porting the RD
//! loop we must know exactly which modes `aomenc` picks for the target frame,
//! rather than inferring them. Pairs with the decode-both localizer
//! (`inter_localize`), which compares PIXELS; this reads the C encoder's
//! DECISIONS out of the coded bytes.
//!
//! ```text
//! cargo run --profile test-fast -p zenav1-aom-bench --example dump_inter_stream -- \
//!     --out /tmp/zero_mv --dx 0 --dy 0 --cq 60 --w 64 --h 64 --speed 0
//! /root/aom-inspect/examples/inspect -bs -ts -m -r -mm /tmp/zero_mv.ivf
//! ```

use aom_bench::{EncodeCell, MultiFrameEncodeCell};

/// OBU_TEMPORAL_DELIMITER — starts each temporal unit in the low-overhead
/// bitstream the encoder shim emits.
const OBU_TEMPORAL_DELIMITER: u32 = 2;

fn obu_spans(bytes: &[u8]) -> Vec<(u32, usize, usize)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let start = pos;
        let b0 = bytes[pos];
        let obu_type = u32::from((b0 >> 3) & 0xF);
        let extension = (b0 >> 2) & 1;
        let has_size = (b0 >> 1) & 1;
        let mut p = pos + 1 + usize::from(extension == 1);
        assert_eq!(has_size, 1, "shim always sets obu_has_size_field");
        // leb128 size
        let mut size = 0u64;
        let mut shift = 0;
        loop {
            let b = bytes[p];
            size |= u64::from(b & 0x7f) << shift;
            p += 1;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        let end = p + size as usize;
        out.push((obu_type, start, end));
        pos = end;
    }
    out
}

/// Split the OBU stream into temporal units: a new TU starts at each
/// OBU_TEMPORAL_DELIMITER.
fn temporal_units(bytes: &[u8]) -> Vec<&[u8]> {
    let spans = obu_spans(bytes);
    let mut starts: Vec<usize> = spans
        .iter()
        .filter(|(t, _, _)| *t == OBU_TEMPORAL_DELIMITER)
        .map(|(_, s, _)| *s)
        .collect();
    if starts.first() != Some(&0) {
        starts.insert(0, 0);
    }
    let mut out = Vec::new();
    for (i, &s) in starts.iter().enumerate() {
        let e = starts.get(i + 1).copied().unwrap_or(bytes.len());
        if e > s {
            out.push(&bytes[s..e]);
        }
    }
    out
}

fn write_ivf(path: &str, w: usize, h: usize, tus: &[&[u8]]) {
    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes()); // version
    ivf.extend_from_slice(&32u16.to_le_bytes()); // header length
    ivf.extend_from_slice(b"AV01"); // fourcc
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&30u32.to_le_bytes()); // timebase denominator (rate)
    ivf.extend_from_slice(&1u32.to_le_bytes()); // timebase numerator (scale)
    ivf.extend_from_slice(&(tus.len() as u32).to_le_bytes()); // frame count
    ivf.extend_from_slice(&0u32.to_le_bytes()); // unused
    for (i, tu) in tus.iter().enumerate() {
        ivf.extend_from_slice(&(tu.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&(i as u64).to_le_bytes()); // pts
        ivf.extend_from_slice(tu);
    }
    std::fs::write(path, &ivf).unwrap_or_else(|e| panic!("write {path}: {e}"));
    eprintln!("wrote {} ({} bytes, {} TUs)", path, ivf.len(), tus.len());
}

fn base(label: &str, w: usize, h: usize, mono: bool, cq: i32, speed: i32) -> EncodeCell {
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
        speed,
        bd: 8,
        y,
        u,
        v,
    }
}

fn arg(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find(|p| p[0] == key)
        .map(|p| p[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = arg(&args, "--out", "/tmp/inter_stream");
    let dx: i32 = arg(&args, "--dx", "0").parse().unwrap();
    let dy: i32 = arg(&args, "--dy", "0").parse().unwrap();
    let cq: i32 = arg(&args, "--cq", "60").parse().unwrap();
    let w: usize = arg(&args, "--w", "64").parse().unwrap();
    let h: usize = arg(&args, "--h", "64").parse().unwrap();
    let speed: i32 = arg(&args, "--speed", "0").parse().unwrap();
    let mono: bool = arg(&args, "--mono", "false").parse().unwrap();
    let cdef: bool = arg(&args, "--cdef", "false").parse().unwrap();
    let lr: bool = arg(&args, "--lr", "false").parse().unwrap();

    let cell = MultiFrameEncodeCell::translational(
        &base("dump", w, h, mono, cq, speed),
        dx,
        dy,
    );
    let stream = cell.c_encode_inter(cdef, lr);
    eprintln!(
        "cfg: {w}x{h} mono={mono} cq={cq} speed={speed} dx={dx} dy={dy} cdef={cdef} lr={lr}"
    );
    eprintln!("stream: {} bytes", stream.len());
    for (t, s, e) in obu_spans(&stream) {
        eprintln!("  OBU type={t:<2} [{s:>5}..{e:>5}) len={}", e - s);
    }
    let tus = temporal_units(&stream);
    for (i, tu) in tus.iter().enumerate() {
        eprintln!("  TU{i}: {} bytes", tu.len());
    }
    std::fs::write(format!("{out}.obu"), &stream).unwrap();
    write_ivf(&format!("{out}.ivf"), w, h, &tus);
}
