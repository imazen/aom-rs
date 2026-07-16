//! Profile driver for callgrind/perf: run ONE Gate-3 cell's port or C side
//! in a loop, so instruction-count profiles rank the hot kernels.
//!
//! Usage: gate3_profile <enc|dec> <port|c> <cell-label> <iters>
//!   e.g. gate3_profile enc port enc_s0_128_cq32 3
//!        gate3_profile dec port dec_352x288_q32 20
//!
//! Cell labels are the ones `aom_bench::{encode_cells, decode_cells}` define.
//! The first iteration byte-verifies port-vs-C output (an invalid cell must
//! never be profiled); subsequent iterations are the pure loop, so use
//! `--toggle-collect` on the loop functions or just subtract the setup cost
//! (it is one C encode/decode + one port encode/decode).

use aom_bench::{decode_cells, encode_cells};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: gate3_profile <enc|dec> <port|c> <cell-label> <iters>");
        eprintln!("encode cells:");
        for c in encode_cells() {
            eprintln!("  {}", c.label);
        }
        eprintln!("decode cells:");
        for c in decode_cells() {
            eprintln!("  {}", c.label);
        }
        std::process::exit(2);
    }
    let (kind, side, label, iters) = (&args[1], &args[2], &args[3], &args[4]);
    let iters: usize = iters.parse().expect("iters must be a number");

    match kind.as_str() {
        "enc" => {
            let cell = encode_cells()
                .into_iter()
                .find(|c| &c.label == label)
                .unwrap_or_else(|| panic!("unknown encode cell {label}"));
            let bootstrap = cell.assert_byte_exact();
            let mut sink = 0usize;
            for _ in 0..iters {
                match side.as_str() {
                    "port" => sink = sink.wrapping_add(cell.port_encode(&bootstrap).len()),
                    "c" => sink = sink.wrapping_add(cell.c_encode().len()),
                    other => panic!("side must be port|c, got {other}"),
                }
            }
            eprintln!("{label} {side} x{iters}: sink={sink}");
        }
        "dec" => {
            let cell = decode_cells()
                .into_iter()
                .find(|c| &c.label == label)
                .unwrap_or_else(|| panic!("unknown decode cell {label}"));
            cell.assert_byte_exact();
            let mut sink = 0usize;
            for _ in 0..iters {
                match side.as_str() {
                    "port" => sink = sink.wrapping_add(cell.port_decode().y.len()),
                    "c" => sink = sink.wrapping_add(cell.c_decode().y.len()),
                    other => panic!("side must be port|c, got {other}"),
                }
            }
            eprintln!("{label} {side} x{iters}: sink={sink}");
        }
        other => panic!("kind must be enc|dec, got {other}"),
    }
}
