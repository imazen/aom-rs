//! IntraBC (intra-block-copy) DV-search gate — screen-content stills (PARITY C3).
//!
//! Both sides encode genuine screen content (crops of the decoded
//! `intra_only-intrabc-extreme-dv` conformance vector) with
//! `--enable-palette=0 --enable-intrabc=1`. The port runs the ported DV search:
//! the source-frame hash + the NSTEP diamond + the mesh (`rd_pick_intrabc_mode_sb`),
//! the skip-arm RD (`predict_skip_txfm` regime), and the intrabc pack
//! (use_intrabc flag + DV + skip).
//!
//! **Status: PINNED (honest).** The inter var-tx COEFF arm IS now wired
//! end-to-end — `av1_pick_recursive_tx_size_type_yrd` + both NN prunes, the
//! chroma `av1_txfm_uvrd` inter arm, the `av1_encode_sb` var-tx re-encode, and
//! the `write_tx_size_vartx` + inter-ext-tx pack — so the port evaluates and
//! codes intrabc outside the `predict_skip_txfm` regime. It is NOT yet
//! byte-exact: this cell's C encode uses 49 intrabc blocks (~39 coeff-arm, ~24
//! non-square) and the port currently emits 1907B against C's 1891B, byte-
//! identical for the first 646 bytes. The remaining delta is what this pin
//! holds. The gate prints the size delta, the first differing byte, and (when
//! the port's stream still decodes) the first block whose mode-info differs.
//!
//! This gate therefore (1) asserts the content is anti-vacuous — real aomenc
//! genuinely codes intrabc blocks here (the DV search + wiring is exercised on
//! live screen content, not a config that never fires), and (2) PINS the
//! divergence self-promotingly: when a cell byte-matches, the pin fails →
//! promote it into `BYTE_EXACT_CELLS`. It reports C's skip/coeff/square split
//! per cell for provenance.

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
            // Provenance for the pin: the size delta + first divergence is the
            // KB-6-style signature that says WHICH WAY the port is wrong
            // (fewer bytes => it is making cheaper RD decisions than C).
            let first_diff = port_on
                .iter()
                .zip(c_frame.iter())
                .position(|(a, b)| a != b)
                .map_or_else(
                    || port_on.len().min(c_frame.len()),
                    |i| i,
                );
            eprintln!(
                "    PINNED: port {}B vs c {}B (delta {:+}), first differing byte {} of {}",
                port_on.len(),
                c_frame.len(),
                port_on.len() as i64 - c_frame.len() as i64,
                first_diff,
                c_frame.len()
            );
            // REGRESSION FLOOR (tightens as roots land; never relaxed).
            // Every byte before `first_diff` is byte-identical to real aomenc,
            // so `first_diff` only moves FORWARD as the port gets more correct.
            // Landed roots and where they put it:
            //   646  — coeff arm wired end-to-end (2026-07-19)
            //   1038 — unclamped DV tile bounds (`434b865d`)
            //   1120 — the four intra-RD roots: the intrabc rescue on the
            //          intra-budget miss, `skip_ctx` from the DV grid, the
            //          missing `intrabc_cost[0]` on every intra leaf, and
            //          `cfl_store_block` for inter/intrabc non-chroma-ref
            //          blocks.
            // A DROP below the floor means a landed root was undone — that is a
            // regression, not a near-tie. (This is a strengthening, not a
            // weakening: the pin above still requires divergence, and the
            // correct end state is full byte-identity, which fails the pin and
            // promotes the cell.)
            const FIRST_DIFF_FLOOR: usize = 1120;
            assert!(
                first_diff >= FIRST_DIFF_FLOOR,
                "{}: first differing byte REGRESSED to {} (floor {}) — a landed \
                 intrabc/intra-RD root was undone; see KB-15",
                cell.label,
                first_diff,
                FIRST_DIFF_FLOOR
            );
            // Decode BOTH streams and report the first block whose coded
            // mode-info differs — the actionable half of the pin. Everything
            // before it is byte-exact, so the divergence is AT this block.
            let port_stream = aom_bench::rd_close::splice_frame_obu(&c_on, &port_on);
            // Best-effort: the port's stream diverges mid-tile, so the decoder
            // may desync and panic (e.g. an intrabc DV that fails validity is
            // usually a desync artifact, not a genuinely bad DV). The pin's
            // assertion must not depend on this diagnostic succeeding.
            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let decoded = std::panic::catch_unwind(|| {
                aom_decode::frame::decode_frame_obus_prefilter(&port_stream)
            });
            std::panic::set_hook(prev_hook);
            match decoded.unwrap_or_else(|_| Err(Default::default())) {
                Ok((tp, _, _)) => {
                    let (tc, _, _) = aom_decode::frame::decode_frame_obus_prefilter(&c_on)
                        .expect("C stream decodes");
                    let (nb_p, nb_c) = (tp.blocks.len(), tc.blocks.len());
                    eprintln!(
                        "    decoded blocks: port {nb_p} vs c {nb_c}; port intrabc census {:?}",
                        intrabc_census(&port_stream)
                    );
                    let mut reported = false;
                    for (bp, bc) in tp.blocks.iter().zip(tc.blocks.iter()) {
                        let key_p = (bp.mi_row, bp.mi_col, bp.bsize, bp.info.use_intrabc,
                                     bp.info.skip, bp.tx_size);
                        let key_c = (bc.mi_row, bc.mi_col, bc.bsize, bc.info.use_intrabc,
                                     bc.info.skip, bc.tx_size);
                        if key_p != key_c {
                            eprintln!(
                                "    FIRST DIVERGENT BLOCK\n      port (mi {},{}) bsize={} \
                                 intrabc={} skip={} tx={}\n      c    (mi {},{}) bsize={} \
                                 intrabc={} skip={} tx={}",
                                key_p.0, key_p.1, key_p.2, key_p.3, key_p.4, key_p.5,
                                key_c.0, key_c.1, key_c.2, key_c.3, key_c.4, key_c.5,
                            );
                            reported = true;
                            break;
                        }
                    }
                    if !reported {
                        eprintln!(
                            "    every common block's mode-info MATCHES — the divergence is \
                             in coefficient bytes, not in the mode/partition decisions"
                        );
                    }
                }
                Err(_) => eprintln!(
                    "    (port stream does not decode past the divergence —                      expected once the bitstream desyncs)"
                ),
            }
            // PIN: the port must still DIVERGE here. A MATCH means the coeff
            // arm is complete — fail so the cell gets promoted.
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
