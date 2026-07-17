//! Build a photographic mosaic y4m from the 25 distinct gb82 576x576 lossless
//! photos, tiled row-major (repeating to fill) and cropped to the target
//! WxH. Real high-frequency photographic content everywhere (only thin tile
//! seams) — a far more representative decode workload than a smooth upscale
//! of a single small image.
//!
//! Output: single-frame YUV4MPEG2 (C420jpeg, full-range JPEG BT.601 RGB->YUV),
//! 4:2:0 chroma by 2x2 box average. Feed to aomenc --i420 (or via y4m).
//!
//! Usage: mk_mosaic_y4m <tiles_dir> <out.y4m> <width> <height>

use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn clamp_u8(v: f32) -> u8 {
    let r = (v + 0.5).floor();
    if r < 0.0 {
        0
    } else if r > 255.0 {
        255
    } else {
        r as u8
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: {} <tiles_dir> <out.y4m> <width> <height>", args[0]);
        std::process::exit(2);
    }
    let tiles_dir = PathBuf::from(&args[1]);
    let out_path = PathBuf::from(&args[2]);
    let w: usize = args[3].parse().expect("width");
    let h: usize = args[4].parse().expect("height");
    assert!(w % 2 == 0 && h % 2 == 0, "width/height must be even for 4:2:0");

    // Deterministic tile order (sorted by filename).
    let mut tile_paths: Vec<PathBuf> = fs::read_dir(&tiles_dir)
        .expect("read tiles_dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "png").unwrap_or(false))
        .collect();
    tile_paths.sort();
    assert!(!tile_paths.is_empty(), "no PNG tiles found in {:?}", tiles_dir);

    // Load all tiles as RGB8.
    let tiles: Vec<image::RgbImage> = tile_paths
        .iter()
        .map(|p| image::open(p).expect("open tile").to_rgb8())
        .collect();
    let tw = tiles[0].width() as usize;
    let th = tiles[0].height() as usize;
    for t in &tiles {
        assert_eq!(t.width() as usize, tw, "all tiles must share width");
        assert_eq!(t.height() as usize, th, "all tiles must share height");
    }
    eprintln!(
        "mk_mosaic_y4m: {} tiles of {}x{}, target {}x{}",
        tiles.len(),
        tw,
        th,
        w,
        h
    );

    let ncols = w.div_ceil(tw);
    let ntiles = tiles.len();

    // Full-res Y and (pre-subsample) Cb/Cr, from the mosaic sampled at each
    // output pixel.
    let mut y_plane = vec![0u8; w * h];
    let mut cb_full = vec![0f32; w * h];
    let mut cr_full = vec![0f32; w * h];

    for oy in 0..h {
        let tile_row = oy / th;
        let in_y = oy % th;
        for ox in 0..w {
            let tile_col = ox / tw;
            let in_x = ox % tw;
            // Row-major tiling with repetition; a small rotation per row keeps
            // adjacent rows from being identical tiles (more content variety).
            let idx = (tile_row * ncols + tile_col + tile_row * 3) % ntiles;
            let px = tiles[idx].get_pixel(in_x as u32, in_y as u32);
            let r = px[0] as f32;
            let g = px[1] as f32;
            let b = px[2] as f32;
            // JPEG / full-range BT.601.
            let yv = 0.299 * r + 0.587 * g + 0.114 * b;
            let cb = -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0;
            let cr = 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0;
            y_plane[oy * w + ox] = clamp_u8(yv);
            cb_full[oy * w + ox] = cb;
            cr_full[oy * w + ox] = cr;
        }
    }

    // 4:2:0 chroma by 2x2 box average.
    let cw = w / 2;
    let ch = h / 2;
    let mut u_plane = vec![0u8; cw * ch];
    let mut v_plane = vec![0u8; cw * ch];
    for cy in 0..ch {
        for cx in 0..cw {
            let x0 = cx * 2;
            let y0 = cy * 2;
            let mut su = 0f32;
            let mut sv = 0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = (y0 + dy) * w + (x0 + dx);
                    su += cb_full[i];
                    sv += cr_full[i];
                }
            }
            u_plane[cy * cw + cx] = clamp_u8(su / 4.0);
            v_plane[cy * cw + cx] = clamp_u8(sv / 4.0);
        }
    }

    let mut f = std::io::BufWriter::new(fs::File::create(&out_path).expect("create out"));
    // F25:1 framerate, progressive, 1:1 PAR, C420jpeg chroma siting.
    write!(f, "YUV4MPEG2 W{} H{} F25:1 Ip A1:1 C420jpeg\n", w, h).unwrap();
    write!(f, "FRAME\n").unwrap();
    f.write_all(&y_plane).unwrap();
    f.write_all(&u_plane).unwrap();
    f.write_all(&v_plane).unwrap();
    f.flush().unwrap();
    eprintln!("wrote {:?} ({}x{} 4:2:0 8-bit, 1 frame)", out_path, w, h);
}
