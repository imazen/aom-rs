//! Differential harness for `dr_predict_high` — the highbd directional predictor
//! dispatch (`highbd_dr_predictor`): route by angle to z1/z2/z3 or V/H at the
//! cardinals. Against C libaom v3.14.1 (`shim_hbd_dr_predict`, reusing the public
//! av1_highbd_dr_prediction_z* + the V/H predictor dispatch), swept over every
//! valid angle × all 19 tx sizes × upsample settings × bitdepths {8,10,12}.

use aom_intra::dir::{get_dx, get_dy};
use aom_intra::dr_predict_high;
use aom_sys_ref as c;

const TX_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TX_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
const PAD: usize = 16;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Is `angle` a valid directional prediction angle (non-degenerate derivative)?
fn valid_angle(angle: i32) -> bool {
    if angle == 90 || angle == 180 {
        return true; // V / H cardinals
    }
    if angle > 0 && angle < 90 {
        get_dx(angle) != 0
    } else if angle > 90 && angle < 180 {
        get_dx(angle) != 0 && get_dy(angle) != 0
    } else if angle > 180 && angle < 270 {
        get_dy(angle) != 0
    } else {
        false
    }
}

#[test]
fn dr_predict_high_matches_c() {
    let mut rng = Rng(0xc0ff_eedd_1234_9990);
    let mut checks = 0u64;
    let mut saw_v = false;
    let mut saw_h = false;
    let mut saw_z2 = false;
    for &bd in &[8i32, 10, 12] {
        for tx_size in 0..19usize {
            let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
            let n = PAD + 2 * (txw + txh) + 16;
            for up in 0..2i32 {
                for angle in 1..270i32 {
                    if !valid_angle(angle) {
                        continue;
                    }
                    let above: Vec<u16> = (0..n).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
                    let left: Vec<u16> = (0..n).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();

                    let mut got = vec![0u16; txw * txh];
                    dr_predict_high(&mut got, txw, tx_size, &above, &left, PAD, up, up, angle, bd);
                    let want =
                        c::ref_hbd_dr_predict(tx_size, txw, txh, &above, &left, PAD, up, up, angle, bd);

                    assert_eq!(
                        got, want,
                        "dr_predict_high divergence tx_size={tx_size} ({txw}x{txh}) angle={angle} up={up} bd={bd}"
                    );
                    checks += 1;
                    saw_v |= angle == 90;
                    saw_h |= angle == 180;
                    saw_z2 |= angle > 90 && angle < 180;
                }
            }
        }
    }
    assert!(saw_v && saw_h && saw_z2, "test missed a dispatch zone (V/H/z2)");
    assert!(checks > 5000, "expected many checks, got {checks}");
}
