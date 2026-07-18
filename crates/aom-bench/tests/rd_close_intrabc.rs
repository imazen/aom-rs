//! IntraBC (intra-block-copy) DV-search gate — screen-content stills (PARITY C3).
//!
//! Both sides encode genuine screen content (crops of the decoded
//! `intra_only-intrabc-extreme-dv` conformance vector) with
//! `--enable-palette=0 --enable-intrabc=1`. The port runs the ported DV search:
//! the source-frame hash + the NSTEP diamond + the mesh (`rd_pick_intrabc_mode_sb`),
//! the skip-arm RD (`predict_skip_txfm` regime), and the intrabc pack
//! (use_intrabc flag + DV + skip).
//!
//! **Status: PINNED (honest).** The port codes intrabc only in the skip-arm
//! regime (luma `predict_skip_txfm` fires AND the chroma match is exact), the
//! subset where C forces `skip_txfm=1` and BYPASSES the inter var-tx coeff arm.
//! Real screen content codes the MAJORITY of its intrabc blocks via the
//! COEFF arm (a nonzero, quantized residual) and as NON-SQUARE shapes — e.g.
//! this cell's C encode uses 49 intrabc blocks, of which ~39 are coeff-arm and
//! ~42 are non-square (4x8 / 8x4 / 16x4). The inter var-tx coeff arm (the
//! `av1_pick_recursive_tx_size_type_yrd` quadtree + `prune_tx_2D`/
//! `ml_predict_tx_split` NN prunes + the var-tx pack) is NOT yet ported, so the
//! port codes those blocks as intra and the frame diverges.
//!
//! This gate therefore (1) asserts the content is anti-vacuous — real aomenc
//! genuinely codes intrabc blocks here (the DV search + wiring is exercised on
//! live screen content, not a config that never fires), and (2) PINS the
//! divergence self-promotingly: when the coeff arm lands and a cell byte-matches,
//! the pin fails → promote it into `BYTE_EXACT_CELLS`. It reports C's
//! skip/coeff/square split per cell for provenance.

use aom_bench::EncodeCell;
use aom_bench::ToggleKnobs;

const VEC: &str = "av1-1-b8-16-intra_only-intrabc-extreme-dv";
/// `(label, w, h, off_x, off_y, cq)` — crops whose C re-encode codes intrabc
/// blocks (found by the `intrabc_content_probe` sweep).
// One cell keeps this pin runtime bounded: the scalar per-leaf mesh search on a
// 196² frame is slow (a Gate-3 perf item, not correctness). The 480x180 cq48
// crop is the richest — 49 C intrabc blocks incl. 10 skip + 39 coeff.
const INTRABC_CROPS: &[(&str, usize, usize, usize, usize, i32)] =
    &[("scc_480x180_196_cq48", 196, 196, 480, 180, 48)];

/// C's intrabc-block census for a decoded stream: `(total_intrabc, skip, coeff,
/// non_square)`.
fn intrabc_census(stream: &[u8]) -> (usize, usize, usize, usize) {
    let (t, _, _) = aom_decode::frame::decode_frame_obus_prefilter(stream)
        .expect("decode of the C intrabc stream failed");
    // block_size_wide/high[BLOCK_SIZES_ALL].
    const BW: [usize; 22] = [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
    const BH: [usize; 22] = [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];
    let (mut n, mut skip, mut coeff, mut nonsq) = (0, 0, 0, 0);
    for b in t.blocks.iter().filter(|b| b.info.use_intrabc != 0) {
        n += 1;
        if b.info.skip != 0 {
            skip += 1;
        } else {
            coeff += 1;
        }
        if BW[b.bsize] != BH[b.bsize] {
            nonsq += 1;
        }
    }
    (n, skip, coeff, nonsq)
}

#[test]
fn intrabc_dv_search_pinned() {
    let cells: Vec<EncodeCell> = INTRABC_CROPS
        .iter()
        .map(|&(label, w, h, ox, oy, cq)| {
            EncodeCell::real_content(label, VEC, Some((w, h, ox, oy)), cq, 0)
        })
        .collect();

    // A cell whose C encode is byte-matched by the port (would fail the pin →
    // promote). Empty until the intrabc coeff arm lands.
    const BYTE_EXACT_CELLS: &[&str] = &[];

    let mut any_intrabc = false;
    eprintln!("=== intrabc DV-search census (C, --enable-intrabc=1) ===");
    for cell in &cells {
        let c_on = cell.c_encode_screen(false, true);
        assert!(!c_on.is_empty(), "{}: C encode failed", cell.label);
        let (n, skip, coeff, nonsq) = intrabc_census(&c_on);
        eprintln!(
            "  {}: C intrabc blocks={n} (skip={skip} coeff={coeff} non_square={nonsq})",
            cell.label
        );
        // Anti-vacuous: this crop genuinely exercises intrabc in the reference.
        assert!(
            n > 0,
            "{}: real aomenc coded NO intrabc block — the gate would be vacuous, \
             re-pick the crop (intrabc_content_probe)",
            cell.label
        );
        if n > 0 {
            any_intrabc = true;
        }

        // Run the port's intrabc encode. It reaches byte-parity only when every
        // C intrabc block is skip-arm + square (the coeff arm being unported);
        // on real content it diverges (pinned).
        let port_on = cell.port_encode_with(
            &c_on,
            &ToggleKnobs {
                enable_intrabc: true,
                ..Default::default()
            },
        );
        let c_frame = EncodeCell::frame_obu_payload(&c_on);
        let matched = port_on == c_frame;
        if BYTE_EXACT_CELLS.contains(&cell.label.as_str()) {
            assert!(
                matched,
                "{}: expected BYTE-IDENTICAL vs real aomenc but diverged \
                 (port={}B c={}B)",
                cell.label,
                port_on.len(),
                c_frame.len()
            );
        } else {
            // PIN: the port must still DIVERGE here (coeff arm unported). A
            // MATCH means the coeff arm landed — fail so the cell gets promoted.
            assert!(
                !matched,
                "{}: port now BYTE-MATCHES real aomenc on intrabc content — \
                 promote it into BYTE_EXACT_CELLS",
                cell.label
            );
        }
    }
    assert!(any_intrabc, "no cell exercised intrabc");
}
