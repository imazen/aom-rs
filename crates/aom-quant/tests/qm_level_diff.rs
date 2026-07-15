//! Differential: the Rust qindex -> QM-level mappings
//! [`aom_quant::aom_get_qmlevel`] and [`aom_quant::aom_get_qmlevel_allintra`] vs
//! the real libaom `static inline` formulas (via the shim). These set
//! `qmatrix_level_{y,u,v}` in `av1_set_quantizer`; byte-exact level selection is
//! a prerequisite for QM-on encode byte-match.

use aom_quant::{aom_get_qmlevel, aom_get_qmlevel_allintra};
use aom_sys_ref as c;

#[test]
fn qmlevel_matches_c_over_full_qindex_range() {
    // The allintra default range is [4, 10]; also sweep wider + degenerate
    // ranges to exercise the interpolation and the allintra clamp.
    let ranges = [(4, 10), (0, 15), (2, 10), (5, 5), (0, 0), (10, 15)];
    let mut checked = 0usize;
    for &(first, last) in &ranges {
        for qindex in 0..=255i32 {
            assert_eq!(
                aom_get_qmlevel(qindex, first, last),
                c::ref_get_qmlevel(qindex, first, last),
                "aom_get_qmlevel mismatch at qindex={qindex}, range=[{first},{last}]"
            );
            assert_eq!(
                aom_get_qmlevel_allintra(qindex, first, last),
                c::ref_get_qmlevel_allintra(qindex, first, last),
                "aom_get_qmlevel_allintra mismatch at qindex={qindex}, range=[{first},{last}]"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, ranges.len() * 256);
}
