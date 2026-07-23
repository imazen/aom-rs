//! STABLE-PATH DECODER FUZZ SWEEP (no nightly / cargo-fuzz required).
//!
//! Complements `fuzz_regression.rs` (which REPLAYS the committed crash POCs): this
//! adds a seeded structured-random MUTATION sweep over the committed seeds — the
//! discovery mechanism that finds NEW escaping panics a replay-only gate cannot.
//!
//! The `crates/aom-decode/fuzz/` targets need a nightly toolchain + `cargo-fuzz`
//! + libFuzzer. This test gives the same robustness contract a home on **stable**
//! `cargo test`, so CI enforces it on every platform without nightly:
//!
//!   for ANY input, `decode_frame_obus` / `decode_frames` return `Ok` / `Err(String)`
//!   — never a panic (unwrap / expect / out-of-bounds slice / `assert!` / debug
//!   arithmetic overflow) and never an unbounded allocation.
//!
//! Two parts, both on stable:
//!   1. REPLAY — every committed `fuzz/regression/*` reproducer (minimized POCs of
//!      fixed crashes) and every `fuzz/seeds/**` seed is run through both public
//!      OBU entry points; a panic fails the test. This is the
//!      `tests/fuzz_regression.rs` gate that keeps fixed crashes fixed.
//!   2. STRUCTURED-RANDOM SWEEP — a seeded XorShift mutates the committed seeds
//!      (bit flips, truncation, length-field corruption, header splicing, insert /
//!      delete) and asserts no entry panics. Self-contained (mutates only committed
//!      seeds — no external corpus), so it always runs in CI. The frame-dimension
//!      DoS ceiling (`frame.rs`) keeps a mutated giant-dimension header from OOMing
//!      this in-process sweep.
//!
//! Every distinct panic found in the sweep is collected (deduped by message), its
//! reproducer written to `$FUZZ_CRASH_DIR` (default
//! `/root/fuzz-corpus/aom-rs/stable-crashes/`), and the test fails with the full
//! list — RED while any panic exists, GREEN once fixed.
//!
//! Knobs (env): `FUZZ_SMOKE_ITERS` sweep iterations (default 60000),
//! `FUZZ_SMOKE_SEED` PRNG seed, `FUZZ_CRASH_DIR`.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

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
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

fn fuzz_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/aom-decode
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz")
}

/// Recursively load every regular file under `sub` (skipping dotfiles).
fn load_tree(sub: &str) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let root = fuzz_root().join(sub);
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut ents: Vec<_> = rd.filter_map(|e| e.ok()).collect();
        ents.sort_by_key(|e| e.file_name());
        for e in ents {
            let p = e.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if let Ok(bytes) = std::fs::read(&p) {
                out.push((name.to_string(), bytes));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn panic_msg(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Run one input through both public decode entries under `catch_unwind`.
/// Returns the panic message of the first entry that panics, else `None`.
fn probe(data: &[u8]) -> Option<String> {
    if let Err(p) = catch_unwind(AssertUnwindSafe(|| {
        let _ = aom_decode::frame::decode_frame_obus(data);
    })) {
        return Some(format!("[decode_frame_obus] {}", panic_msg(&p)));
    }
    if let Err(p) = catch_unwind(AssertUnwindSafe(|| {
        let _ = aom_decode::frame::decode_frames(data);
    })) {
        return Some(format!("[decode_frames] {}", panic_msg(&p)));
    }
    None
}

fn crash_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("FUZZ_CRASH_DIR")
            .unwrap_or_else(|_| "/root/fuzz-corpus/aom-rs/stable-crashes".to_string()),
    )
}

fn mutate(rng: &mut Rng, base: &[u8], others: &[Vec<u8>]) -> Vec<u8> {
    let mut v = base.to_vec();
    match rng.below(7) {
        0 => {
            // bit flips (1..=6)
            for _ in 0..1 + rng.below(6) {
                if v.is_empty() {
                    break;
                }
                let i = rng.below(v.len());
                v[i] ^= 1u8 << rng.below(8);
            }
        }
        1 => {
            // truncate to a random prefix
            if !v.is_empty() {
                let keep = rng.below(v.len() + 1);
                v.truncate(keep);
            }
        }
        2 => {
            // length-field corruption: hammer the early bytes (OBU header + size
            // leb128 + frame-size fields all live in the first ~48 bytes).
            let span = v.len().min(48);
            for _ in 0..1 + rng.below(6) {
                if span == 0 {
                    break;
                }
                let i = rng.below(span);
                v[i] = (rng.next() & 0xff) as u8;
            }
        }
        3 => {
            // splice: overwrite a random middle run with another seed's bytes
            if !others.is_empty() && !v.is_empty() {
                let src = &others[rng.below(others.len())];
                if !src.is_empty() {
                    let at = rng.below(v.len());
                    let take = 1 + rng.below(src.len());
                    let so = rng.below(src.len());
                    for k in 0..take {
                        if at + k >= v.len() || so + k >= src.len() {
                            break;
                        }
                        v[at + k] = src[so + k];
                    }
                }
            }
        }
        4 => {
            // concatenate two seeds (multi-OBU / multi-frame splicing)
            if !others.is_empty() {
                v.extend_from_slice(&others[rng.below(others.len())]);
            }
        }
        5 => {
            // random byte insertion
            for _ in 0..1 + rng.below(8) {
                let at = rng.below(v.len() + 1);
                v.insert(at, (rng.next() & 0xff) as u8);
            }
        }
        _ => {
            // random byte deletion
            for _ in 0..1 + rng.below(8) {
                if v.is_empty() {
                    break;
                }
                let at = rng.below(v.len());
                v.remove(at);
            }
        }
    }
    // Cap mutated size so splice/concat chains cannot balloon (real in-scope
    // temporal units are tiny); keeps the sweep fast.
    v.truncate(8192);
    v
}

#[test]
fn decoder_sweep_never_panics_on_mutated_input() {
    let regressions = load_tree("regression");
    let seeds = load_tree("seeds");
    assert!(
        !seeds.is_empty(),
        "no fuzz seeds found under {} — the seed corpus must be committed",
        fuzz_root().join("seeds").display()
    );

    // Silence the default panic hook during replay+sweep so a would-be panic
    // (caught by probe) does not flood stderr; restore before the final report.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    // ---- part 1: replay every committed reproducer + seed (hard gate) -------
    let mut replay_failures: Vec<(String, String)> = Vec::new();
    for (name, bytes) in regressions.iter().chain(seeds.iter()) {
        if let Some(msg) = probe(bytes) {
            replay_failures.push((name.clone(), msg));
        }
    }

    // ---- part 2: structured-random sweep over the seeds ---------------------
    let iters: u64 = std::env::var("FUZZ_SMOKE_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60_000);
    let seed: u64 = std::env::var("FUZZ_SMOKE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let mut rng = Rng(seed | 1);

    let corpus: Vec<Vec<u8>> = seeds
        .iter()
        .chain(regressions.iter())
        .map(|(_, b)| b.clone())
        .collect();

    let mut distinct: std::collections::BTreeMap<String, Vec<u8>> =
        std::collections::BTreeMap::new();
    for _ in 0..iters {
        let base = &corpus[rng.below(corpus.len())];
        let input = mutate(&mut rng, base, &corpus);
        if let Some(msg) = probe(&input) {
            distinct.entry(msg).or_insert(input);
        }
    }

    std::panic::set_hook(prev_hook);

    // ---- report -------------------------------------------------------------
    if replay_failures.is_empty() && distinct.is_empty() {
        return; // GREEN: no panic escaped any public entry.
    }

    let dir = crash_dir();
    let _ = std::fs::create_dir_all(&dir);
    let mut report = String::new();
    if !replay_failures.is_empty() {
        report.push_str(&format!(
            "\n{} committed seed/regression input(s) PANIC (must never happen):\n",
            replay_failures.len()
        ));
        for (name, msg) in &replay_failures {
            report.push_str(&format!("  - {name}: {msg}\n"));
        }
    }
    if !distinct.is_empty() {
        report.push_str(&format!(
            "\n{} DISTINCT panic(s) found by the structured-random sweep:\n",
            distinct.len()
        ));
        for (i, (msg, input)) in distinct.iter().enumerate() {
            let path = dir.join(format!("stable-crash-{i:03}.obu"));
            let _ = std::fs::write(&path, input);
            report.push_str(&format!(
                "  - {msg}\n      ({} bytes) reproducer: {}\n",
                input.len(),
                path.display()
            ));
        }
    }
    panic!("{report}\nseed={seed:#x} iters={iters}");
}
