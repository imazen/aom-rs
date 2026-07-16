//! Gate-3 baseline: port vs REAL C oracle, paired per cell via zenbench's
//! interleaved rounds. The `c_oracle` bench is each group's baseline, so the
//! port row's "95% CI vs base" column IS the Gate-3 ratio (with confidence)
//! for that cell. Target: port <= 1.20x C.
//!
//! Every cell byte-verifies port-vs-C output in setup before any timing.
//!
//! Usage:
//!   cargo bench -p aom-bench --bench gate3                      # full run
//!   cargo bench -p aom-bench --bench gate3 -- --group=dec       # decode only
//!   AOM_BENCH_SMOKE=1 cargo bench -p aom-bench --bench gate3    # smoke (numbers meaningless)
//!
//! WALL-CLOCK DISCIPLINE: the committed Gate-3 baseline must be measured on a
//! QUIET box (zenbench's resource gate also flags noisy rounds). Smoke runs
//! exist only to prove the harness executes end-to-end.

use std::time::Duration;

use aom_bench::{decode_cells, encode_cells};
use zenbench::prelude::*;

fn smoke() -> bool {
    std::env::var_os("AOM_BENCH_SMOKE").is_some_and(|v| v != "0")
}

/// Group tuning. Encode calls run 10ms..multi-second each, so cap rounds and
/// give the group a generous per-group budget; decode calls are ms-scale.
fn tune(g: &mut BenchGroup, heavy: bool) {
    if smoke() {
        g.config()
            .min_rounds(2)
            .max_rounds(3)
            .warmup_time(Duration::from_millis(50))
            .max_time(Duration::from_secs(2))
            .max_wall_time(Duration::from_secs(60));
    } else if heavy {
        g.config()
            .min_rounds(8)
            .max_rounds(60)
            .warmup_time(Duration::from_millis(500))
            .max_time(Duration::from_secs(45))
            .max_wall_time(Duration::from_secs(300));
    } else {
        g.config()
            .min_rounds(10)
            .max_rounds(120)
            .warmup_time(Duration::from_millis(300))
            .max_time(Duration::from_secs(20))
            .max_wall_time(Duration::from_secs(120));
    }
}

fn bench_decode(suite: &mut Suite) {
    for cell in decode_cells() {
        cell.assert_byte_exact();
        let pixels = (cell.w * cell.h) as u64;
        let label = cell.label.clone();
        suite.group(label, |g| {
            g.throughput(Throughput::Elements(pixels));
            g.throughput_unit("px");
            let c1 = std::sync::Arc::new(cell);
            let c2 = std::sync::Arc::clone(&c1);
            g.bench("c_oracle", move |b| {
                b.iter(|| c1.c_decode());
            });
            g.bench("port", move |b| {
                b.iter(|| c2.port_decode());
            });
            g.baseline("c_oracle");
            tune(g, false);
        });
    }
}

fn bench_encode(suite: &mut Suite) {
    for cell in encode_cells() {
        let bootstrap = cell.assert_byte_exact();
        let pixels = (cell.w * cell.h) as u64;
        let label = cell.label.clone();
        suite.group(label, |g| {
            g.throughput(Throughput::Elements(pixels));
            g.throughput_unit("px");
            let c1 = std::sync::Arc::new(cell);
            let c2 = std::sync::Arc::clone(&c1);
            g.bench("c_oracle", move |b| {
                b.iter(|| c1.c_encode());
            });
            g.bench("port", move |b| {
                b.iter(|| c2.port_encode(&bootstrap));
            });
            g.baseline("c_oracle");
            tune(g, true);
        });
    }
}

fn main() {
    let group_filter: Option<String> = std::env::args()
        .find_map(|a| a.strip_prefix("--group=").map(String::from));
    // Smoke mode exists to prove the harness runs; its numbers are discarded,
    // so skip the resource gate's per-round waits. Real runs keep the default
    // gate (which flags noisy rounds — the Gate-3 baseline must be quiet-box).
    let mut gate = GateConfig::default();
    if smoke() {
        gate.enabled = false;
    }
    let result = zenbench::run_gated(gate, |suite: &mut Suite| {
        if let Some(f) = group_filter {
            suite.set_group_filter(f);
        }
        bench_decode(suite);
        bench_encode(suite);
    });
    zenbench::postprocess_result(&result);
}
