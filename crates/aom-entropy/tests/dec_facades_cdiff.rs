//! Direct C differentials for the tx-size context facades — the real
//! `get_tx_size_context` / `set_txfm_ctxs` / `tx_size_from_tx_mode` /
//! `depth_to_tx_size` static inlines (via dec_shim.c MACROBLOCKD facades) vs
//! the aom-entropy ports. These were previously roundtrip-covered only (both
//! sides shared the Rust facade, so a shared misread was invisible).

use aom_entropy::partition::{
    depth_to_tx_size, get_tx_size_context, set_txfm_ctxs, tx_size_from_tx_mode, TxMode,
};
use aom_sys_ref as c;

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
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const BLOCK_SIZES_ALL: usize = 22;

#[test]
fn get_tx_size_context_matches_c() {
    let mut rng = Rng(0xDEC0_FACE_0000_0001);
    // txfm-context bytes from the reachable alphabet (set_txfm_ctxs writes tx
    // dims 4..64 or block dims up to 128; TXFM_CTX_INIT=64) plus raw randoms.
    let bytes: [u8; 8] = [4, 8, 16, 32, 64, 128, 12, 200];
    let mut cases = 0u64;
    for bsize in 0..BLOCK_SIZES_ALL {
        for &at in &bytes {
            for &lt in &bytes {
                for avail in 0..4u32 {
                    let (has_above, has_left) = (avail & 1 != 0, avail & 2 != 0);
                    // Neighbour inter-ness x a random neighbour bsize.
                    for nbr in 0..4u32 {
                        let ab = rng.below(22) as usize;
                        let lb = rng.below(22) as usize;
                        let above_inter = nbr & 1 != 0;
                        let left_inter = nbr & 2 != 0;
                        let r = get_tx_size_context(
                            bsize,
                            at,
                            lt,
                            has_above,
                            has_left,
                            if above_inter { Some(ab) } else { None },
                            if left_inter { Some(lb) } else { None },
                        );
                        let cr = c::ref_get_tx_size_context(
                            bsize,
                            at,
                            lt,
                            has_above,
                            has_left,
                            ab,
                            above_inter,
                            lb,
                            left_inter,
                        );
                        assert_eq!(
                            r as i32, cr,
                            "bsize={bsize} at={at} lt={lt} avail=({has_above},{has_left}) \
                             above=({above_inter},{ab}) left=({left_inter},{lb})"
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 22 * 8 * 8 * 4 * 4);
}

#[test]
fn set_txfm_ctxs_matches_c() {
    let mut cases = 0u64;
    for tx_size in 0..19 {
        for n4_w in [1usize, 2, 3, 4, 8, 16, 32] {
            for n4_h in [1usize, 2, 3, 4, 8, 16, 32] {
                for skip in [false, true] {
                    let mut a_r = [0xAAu8; 33];
                    let mut l_r = [0x55u8; 33];
                    let mut a_c = [0xAAu8; 33];
                    let mut l_c = [0x55u8; 33];
                    set_txfm_ctxs(&mut a_r, &mut l_r, tx_size, n4_w, n4_h, skip);
                    c::ref_set_txfm_ctxs(tx_size, n4_w, n4_h, skip, &mut a_c, &mut l_c);
                    assert_eq!(
                        a_r, a_c,
                        "above tx={tx_size} n4=({n4_w},{n4_h}) skip={skip}"
                    );
                    assert_eq!(l_r, l_c, "left tx={tx_size} n4=({n4_w},{n4_h}) skip={skip}");
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 19 * 7 * 7 * 2);
}

#[test]
fn tx_size_maps_match_c() {
    // tx_size_from_tx_mode over every (bsize, mode) — including the dead-under-
    // conformance ONLY_4X4 rect arm the port carries verbatim.
    for bsize in 0..BLOCK_SIZES_ALL {
        for (mode, tm) in [
            (0, TxMode::Only4x4),
            (1, TxMode::Largest),
            (2, TxMode::Select),
        ] {
            assert_eq!(
                tx_size_from_tx_mode(bsize, tm),
                c::ref_tx_size_from_tx_mode(bsize, mode),
                "tx_size_from_tx_mode bsize={bsize} mode={mode}"
            );
        }
        for depth in 0..=2 {
            assert_eq!(
                depth_to_tx_size(depth, bsize),
                c::ref_depth_to_tx_size(depth, bsize),
                "depth_to_tx_size depth={depth} bsize={bsize}"
            );
        }
    }
}
