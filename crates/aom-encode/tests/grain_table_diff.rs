//! C7 grain-estimator — DIFFERENTIAL gate for film-grain TABLE serialization.
//!
//! For each of libaom's 16 built-in `film_grain_test_vectors`, the REAL
//! `aom_film_grain_table_write` serializes it to a canonical `filmgrn1` file;
//! the Rust port then READS those bytes ([`read_film_grain_table`]) and
//! RE-SERIALIZES them ([`write_film_grain_table`]). The re-serialization must be
//! BYTE-IDENTICAL to C's output — proving the port's writer reproduces C's exact
//! `fprintf` shape AND the reader is field-faithful (a dropped/mis-read field
//! would perturb the rewrite). The vectors span the full parameter space:
//! `ar_coeff_lag` 0..3, apply_grain on/off, `update_parameters` on/off, and
//! varied Y/Cb/Cr scaling-point counts.

use aom_encode::grain_table::{read_film_grain_table, write_film_grain_table};
use aom_sys_ref as c;

#[test]
fn grain_table_write_matches_c() {
    let dir = std::env::temp_dir();
    let mut updated = 0; // vectors that carry a full param body (update_parameters=1)
    for idx in 1..=16i32 {
        let path = dir.join(format!("aomrs_grain_tv_{}_{}.tbl", std::process::id(), idx));
        c::ref_write_grain_table_test_vector(idx, &path);
        let c_bytes = std::fs::read(&path).expect("read C-written grain table");
        let _ = std::fs::remove_file(&path);

        let entries = read_film_grain_table(&c_bytes)
            .unwrap_or_else(|e| panic!("idx {idx}: port failed to read C grain table: {e}"));
        let port_bytes = write_film_grain_table(&entries);

        assert_eq!(
            port_bytes, c_bytes,
            "idx {idx}: port re-serialization differs from C aom_film_grain_table_write\n\
             --- C ---\n{}\n--- port ---\n{}",
            String::from_utf8_lossy(&c_bytes),
            String::from_utf8_lossy(&port_bytes)
        );
        updated += entries.iter().filter(|e| e.params.update_parameters).count();
    }
    // Anti-vacuity: most vectors carry a full param body (not just the E line).
    assert!(updated >= 12, "too few update_parameters vectors exercised ({updated})");
    println!("grain_table_write_diff: 16 built-in vectors byte-identical (read∘write == C aom_film_grain_table_write)");
}
