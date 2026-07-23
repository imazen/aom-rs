//! Extract per-frame AV1 temporal units from animated AVIF files.
//!
//! Usage: avif-extract <avif-file> <output-dir>
//!
//! Writes, per input `<stem>.avif`:
//!   <out>/<stem>/frame_<i>.obu        — color-track sample i (one temporal unit)
//!   <out>/<stem>/alpha_<i>.obu        — alpha-track sample i (if present)
//!   <out>/<stem>/color.obu            — all color samples concatenated (decode-order stream)
//!   <out>/<stem>/alpha.obu            — all alpha samples concatenated (if present)
//!   <out>/<stem>/manifest.txt         — frame count, per-sample sizes, durations

use enough::Unstoppable;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <avif-file-or-dir> <output-dir>", args[0]);
        std::process::exit(1);
    }
    let input = Path::new(&args[1]);
    let out_root = Path::new(&args[2]);

    let files: Vec<_> = if input.is_dir() {
        let mut v: Vec<_> = fs::read_dir(input)
            .expect("read input dir")
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("avif"))
            .collect();
        v.sort();
        v
    } else {
        vec![input.to_path_buf()]
    };

    for path in &files {
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let data = fs::read(path).expect("read avif");
        let config = zenavif_parse::DecodeConfig::default().lenient(true);
        let parser =
            match zenavif_parse::AvifParser::from_owned_with_config(data, &config, &Unstoppable) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("SKIP {stem}: parse error: {e}");
                    continue;
                }
            };
        let Some(info) = parser.animation_info() else {
            eprintln!("SKIP {stem}: not animated");
            continue;
        };
        let dir = out_root.join(&stem);
        fs::create_dir_all(&dir).expect("mkdir");
        let mut manifest = format!(
            "frames={} loop_count={} has_alpha={} timescale={}\n",
            info.frame_count, info.loop_count, info.has_alpha, info.timescale
        );
        let mut color_cat: Vec<u8> = Vec::new();
        let mut alpha_cat: Vec<u8> = Vec::new();
        for i in 0..info.frame_count {
            let fr = parser.frame(i).expect("frame");
            fs::write(dir.join(format!("frame_{i}.obu")), &fr.data).expect("write");
            color_cat.extend_from_slice(&fr.data);
            let alen = fr.alpha_data.as_ref().map(|d| d.len()).unwrap_or(0);
            if let Some(ad) = &fr.alpha_data {
                fs::write(dir.join(format!("alpha_{i}.obu")), ad).expect("write");
                alpha_cat.extend_from_slice(ad);
            }
            manifest.push_str(&format!(
                "frame {i}: color={} alpha={} duration_ms={}\n",
                fr.data.len(),
                alen,
                fr.duration_ms
            ));
        }
        fs::write(dir.join("color.obu"), &color_cat).expect("write");
        if !alpha_cat.is_empty() {
            fs::write(dir.join("alpha.obu"), &alpha_cat).expect("write");
        }
        fs::write(dir.join("manifest.txt"), &manifest).expect("write");
        eprintln!(
            "OK {stem}: {} frames, color {} B{}",
            info.frame_count,
            color_cat.len(),
            if alpha_cat.is_empty() {
                String::new()
            } else {
                format!(", alpha {} B", alpha_cat.len())
            }
        );
    }
}
