//! KB-4 diagnostic (report-only): feed the EXACT C-captured residual + txb ctx
//! for the divergent txb — bd10 cq12(qindex=48) ramp, luma leaf mi(14,12)
//! BLOCK_16X8 TX_4X4 blk(1,1) — into the port's real `search_tx_type_intra` and
//! print the winner's (tt, eob, rate, dist, rd), to compare against the C
//! instrumented harness dump:
//!   C: DCT_DCT tt0 eob=1 rate=4520 dist=496 rd=120173  (LOSES)
//!      tt1/2/3/9 eob=0     rate=380  dist=896 rd=119454  (C picks eob=0)
//! The port codes eob=1 there (localizer). This isolates which quantity flips.

use aom_encode::BlockContext;
use aom_encode::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use aom_encode::real_costs::derive_real_costs;
use aom_encode::speed_features::SpeedFeatures;
use aom_encode::tx_search::{TxTypeSearchInputs, search_tx_type_intra};
use aom_dsp::entropy::partition::KfFrameContext;
use aom_dsp::quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_dsp::txb::get_txb_ctx;

#[test]
fn kb4_txb_tie_probe_bd10_cq12() {
    let bd = 10u8;
    let qindex = 48usize;
    // C [KB4-RESDIFF]: 4x4 row-major residual for blk(1,1) in the BLOCK_16X8 ctx.
    let residual: [i16; 16] = [
        2, -10, -12, -3, //
        -10, -12, -3, 1, //
        -12, -3, 1, -10, //
        -3, 1, -10, 4,
    ];
    let tx_size = 0usize; // TX_4X4
    let bsize = 5usize; // BLOCK_16X8 (leaf plane_bsize)

    // Quantizer rows (verify dc_q=170, ac_q=195 vs C).
    let mut quants = Quants::zeroed();
    let mut deq = Dequants::zeroed();
    av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
    let rows = set_q_index(&quants, &deq, qindex, 0);
    let dc_q = rows.dequant[0];
    let ac_q = rows.dequant[1];

    // rdmult (verify 6421 vs C).
    let rdmult = av1_compute_rd_mult_based_on_qindex(
        bd,
        FrameUpdateType::Kf,
        qindex as i32,
        TuneMetric::Psnr,
        EncMode::Allintra,
    );

    // Real cost tables at this qindex (NOT random).
    let kf = KfFrameContext::default_for_qindex(qindex as i32);
    let real = derive_real_costs(&kf, true);
    let coeff_tables = real.coeff_costs_y.tables(tx_size);

    // Neighbour ctx: find above[0]/left[0] that reproduce C's (txb_skip_ctx=2,
    // dc_sign_ctx=1) for a TX_4X4 luma txb in a BLOCK_16X8.
    let mut a0 = 0i8;
    let mut l0 = 0i8;
    let mut found = false;
    'o: for a in 0..=255u16 {
        for l in 0..=255u16 {
            let (skip, sign) = get_txb_ctx(bsize, tx_size, 0, &[a as i8], &[l as i8]);
            if skip == 2 && sign == 1 {
                a0 = a as i8;
                l0 = l as i8;
                found = true;
                break 'o;
            }
        }
    }
    assert!(found, "no above/left reproduces ctx (2,1)");
    let above = vec![a0; 32];
    let left = vec![l0; 32];
    let (skip_v, sign_v) = get_txb_ctx(bsize, tx_size, 0, &above, &left);
    let bctx = BlockContext {
        plane_bsize: bsize,
        plane: 0,
        above: &above,
        left: &left,
    };

    // pred=mid, src=pred+residual (small residual => no clamp).
    let pred = vec![512u16; 16];
    let src: Vec<u16> = (0..16)
        .map(|i| (512i32 + residual[i] as i32) as u16)
        .collect();

    let sf = SpeedFeatures::set_allintra(0, false, false);
    let pol = sf.tx_type_search_policy(false, 0);

    eprintln!(
        "\n=== KB4 txb-tie probe: bd{bd} qindex{qindex} TX_4X4 mi(14,12) blk(1,1) ===\n\
         port: dc_q={dc_q} ac_q={ac_q} rdmult={rdmult}  ctx(skip,sign)=({skip_v},{sign_v})\n\
         C ref: dc_q=170 ac_q=195 rdmult=6421 ctx=(2,1);  \
         C winner=eob0 rd=119454;  C DCT_DCT eob1 rate=4520 dist=496 rd=120173"
    );

    for &reduced in &[false, true] {
        let inp = TxTypeSearchInputs {
            residual: &residual,
            src: &src,
            src_off: 0,
            src_stride: 4,
            pred: &pred,
            tx_size,
            plane: 0,
            uv_mode: 0,
            mode: 3, // leaf y_mode = 3
            use_filter_intra: false,
            filter_intra_mode: 0,
            lossless: false,
            reduced_tx_set_used: reduced,
            bd,
            rows: &rows,
            bctx: &bctx,
            rdmult,
            coeff_costs: &coeff_tables,
            tx_type_costs: &real.tx_type_costs_y,
            // Interior TX_4X4 txb: the visible area is the full 4x4.
            visible_cols: 4,
            visible_rows: 4,
            qm_level: None,
            predict_skip_zero_blk_rate: 0,
        };
        let win = search_tx_type_intra(&inp, &pol, i64::MAX).expect("winner");
        eprintln!(
            "  reduced_tx_set_used={reduced}: WINNER tt={} eob={} rate={} dist={} sse={} rd={} evaluated_mask={:#06x}",
            win.best_tx_type, win.best_eob, win.rate, win.dist, win.sse, win.rd, win.evaluated_mask
        );
    }
}
