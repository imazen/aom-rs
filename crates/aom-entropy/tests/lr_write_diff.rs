//! Loop-restoration RU-params WRITE syntax vs C: random unit-parameter
//! sequences are written by BOTH the REAL C arithmetic writer
//! (`ref_lr_units_roundtrip` — EXPORTED `aom_write_primitive_refsubexpfin` +
//! `aom_write_symbol` over the REAL default LR CDFs) AND this port's
//! `aom_entropy::lr::write_lr_unit`; the two bitstreams must be
//! BYTE-IDENTICAL, the writer-side adapted CDFs must equal the C reader's
//! final CDFs (writer and reader adapt in lockstep), and the port's own
//! reader must roundtrip the port's stream back to the written intent.
//! The syntax bit counters (`count_wiener_bits` / `count_sgrproj_bits` /
//! `count_primitive_refsubexpfin`) are diffed against the REAL EXPORTED
//! `aom_count_primitive_refsubexpfin` over both the coded LR ranges and the
//! unit sequences themselves.

use aom_entropy::enc::OdEcEnc;
use aom_entropy::lr::{
    self, LrRefState, LrUnitInfo, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE,
    RESTORE_WIENER, SGR_PARAMS_R, SGRPROJ_PRJ_MAX0, SGRPROJ_PRJ_MAX1, SGRPROJ_PRJ_MIN0,
    SGRPROJ_PRJ_MIN1, SgrprojInfoLr, WIENER_WIN, WIENER_WIN_CHROMA, WienerInfoLr,
};
use aom_entropy::partition::KfFrameContext;
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
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u64) as i32
    }
}

/// Generate one encoder-legal unit intent in the shim's LRU packing
/// (mirrors lr_read_diff.rs `gen_unit`).
fn gen_unit(rng: &mut Rng, plane: usize, frame_rtype: u8) -> [i32; c::LRU_WORDS] {
    let mut u = [0i32; c::LRU_WORDS];
    u[0] = plane as i32;
    u[1] = frame_rtype as i32;
    let rtype = match frame_rtype {
        RESTORE_SWITCHABLE => rng.below(3) as u8,
        t => {
            if rng.below(3) == 0 {
                RESTORE_NONE
            } else {
                t
            }
        }
    };
    u[2] = rtype as i32;
    match rtype {
        RESTORE_WIENER => {
            for d in 0..2 {
                u[3 + 3 * d] = if plane > 0 { 0 } else { rng.range(-5, 10) };
                u[4 + 3 * d] = rng.range(-23, 8);
                u[5 + 3 * d] = rng.range(-17, 46);
            }
        }
        RESTORE_SGRPROJ => {
            let ep = rng.below(16) as i32;
            u[9] = ep;
            let r = SGR_PARAMS_R[ep as usize];
            if r[0] == 0 {
                u[10] = 0;
                u[11] = rng.range(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
            } else if r[1] == 0 {
                u[10] = rng.range(SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0);
                u[11] = (128 - u[10]).clamp(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
            } else {
                u[10] = rng.range(SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0);
                u[11] = rng.range(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
            }
        }
        _ => {}
    }
    u
}

/// The intent words of one unit as `LrUnitInfo` (centre tap + mirror derived
/// exactly as the search's `finalize_sym_filter` constraints produce them).
fn unit_info_from_intent(u: &[i32]) -> LrUnitInfo {
    let mut info = LrUnitInfo {
        restoration_type: u[2] as u8,
        wiener: WienerInfoLr {
            vfilter: [0; 8],
            hfilter: [0; 8],
        },
        sgrproj: SgrprojInfoLr { ep: 0, xqd: [0; 2] },
    };
    if info.restoration_type == RESTORE_WIENER {
        for d in 0..2 {
            let t = &u[3 + 3 * d..6 + 3 * d];
            let f = if d == 0 {
                &mut info.wiener.vfilter
            } else {
                &mut info.wiener.hfilter
            };
            f[0] = t[0] as i16;
            f[1] = t[1] as i16;
            f[2] = t[2] as i16;
            f[3] = (-2 * (t[0] + t[1] + t[2])) as i16;
            f[4] = t[2] as i16;
            f[5] = t[1] as i16;
            f[6] = t[0] as i16;
            f[7] = 0;
        }
    } else if info.restoration_type == RESTORE_SGRPROJ {
        info.sgrproj = SgrprojInfoLr {
            ep: u[9],
            xqd: [u[10], u[11]],
        };
    }
    info
}

#[test]
fn lr_unit_params_write_matches_c_bytes() {
    let mut rng = Rng(0xD1FF_BEEF_CAFE_1234);
    for case in 0..400 {
        let mut frame_rtype = [RESTORE_NONE; 3];
        loop {
            for t in frame_rtype.iter_mut() {
                *t = rng.below(4) as u8;
            }
            if frame_rtype.iter().any(|&t| t != RESTORE_NONE) {
                break;
            }
        }
        let n_units = 1 + rng.below(40) as usize;
        let mut units = Vec::with_capacity(n_units * c::LRU_WORDS);
        for _ in 0..n_units {
            let plane = loop {
                let p = rng.below(3) as usize;
                if frame_rtype[p] != RESTORE_NONE {
                    break p;
                }
            };
            units.extend_from_slice(&gen_unit(&mut rng, plane, frame_rtype[plane]));
        }

        // C writer (the REAL arithmetic writer over REAL default LR CDFs).
        let (c_stream, _readback, c_cdfs) = c::ref_lr_units_roundtrip(&units);

        // Port writer over the same defaults.
        let fc = KfFrameContext::default_for_qindex(60);
        let mut sw = fc.switchable_restore;
        let mut wn = fc.wiener_restore;
        let mut sg = fc.sgrproj_restore;
        let mut refs = LrRefState::default();
        let mut enc = OdEcEnc::new();
        for i in 0..n_units {
            let u = &units[i * c::LRU_WORDS..(i + 1) * c::LRU_WORDS];
            let info = unit_info_from_intent(u);
            lr::write_lr_unit(
                &mut enc, &info, u[1] as u8, u[0] as usize, &mut refs, &mut sw, &mut wn, &mut sg,
                /* allow_update_cdf= */ true,
            );
        }
        let our_stream = enc.done().to_vec();
        assert_eq!(
            our_stream, c_stream,
            "case {case}: written bitstream differs from the REAL C writer \
             ({} units, frame_rtype {frame_rtype:?})",
            n_units
        );
        // Writer-side adapted CDFs equal the C reader's final CDFs (the C
        // writer/reader adapt in lockstep; ours must too).
        assert_eq!(sw.as_slice(), &c_cdfs[0..4], "case {case}: switchable CDF");
        assert_eq!(wn.as_slice(), &c_cdfs[4..7], "case {case}: wiener CDF");
        assert_eq!(sg.as_slice(), &c_cdfs[7..10], "case {case}: sgrproj CDF");

        // Pure-Rust roundtrip: our reader recovers the written intent from
        // our own bytes.
        let mut dsw = fc.switchable_restore;
        let mut dwn = fc.wiener_restore;
        let mut dsg = fc.sgrproj_restore;
        let mut drefs = LrRefState::default();
        let mut dec = aom_entropy::dec::OdEcDec::new(&our_stream);
        for i in 0..n_units {
            let u = &units[i * c::LRU_WORDS..(i + 1) * c::LRU_WORDS];
            let want = unit_info_from_intent(u);
            let got = lr::read_lr_unit(
                &mut dec, u[1] as u8, u[0] as usize, &mut drefs, &mut dsw, &mut dwn, &mut dsg,
            );
            assert_eq!(
                got.restoration_type, want.restoration_type,
                "case {case} unit {i}: roundtrip type"
            );
            match got.restoration_type {
                RESTORE_WIENER => {
                    assert_eq!(got.wiener, want.wiener, "case {case} unit {i}: wiener taps")
                }
                RESTORE_SGRPROJ => assert_eq!(
                    got.sgrproj, want.sgrproj,
                    "case {case} unit {i}: sgrproj params"
                ),
                _ => {}
            }
        }
    }
}

/// The port's `count_primitive_refsubexpfin` == the REAL EXPORTED
/// `aom_count_primitive_refsubexpfin` across every `(n, k)` pair the LR
/// syntax uses, exhaustively over `(ref, v)`.
#[test]
fn lr_count_primitive_matches_c_exhaustive() {
    // (n, k) pairs: wiener taps 0/1/2, sgrproj xqd 0/1.
    for &(n, k) in &[(16u16, 1u16), (32, 2), (64, 3), (128, 4)] {
        for r in 0..n {
            for v in 0..n {
                assert_eq!(
                    lr::count_primitive_refsubexpfin(n, k, r, v),
                    c::ref_count_primitive_refsubexpfin(n, k, r, v),
                    "count mismatch at n={n} k={k} ref={r} v={v}"
                );
            }
        }
    }
}

/// `count_wiener_bits` / `count_sgrproj_bits` equal the exact number of bits
/// their writer counterparts code, for randomized params + references — the
/// property pickrst.c's RD costing relies on. Verified structurally: the
/// count formulas call the same primitive counter just diffed against C.
#[test]
fn lr_count_bits_match_write_composition() {
    let mut rng = Rng(0x0C0FFEE0_5EED_77);
    for _ in 0..2000 {
        // Random wiener pair (full 7-tap and 5-tap chroma windows).
        for &win in &[WIENER_WIN, WIENER_WIN_CHROMA] {
            let mut mk = |rng: &mut Rng| {
                let t0 = if win == WIENER_WIN_CHROMA {
                    0
                } else {
                    rng.range(-5, 10)
                };
                let (t1, t2) = (rng.range(-23, 8), rng.range(-17, 46));
                let mut f = [0i16; 8];
                f[0] = t0 as i16;
                f[1] = t1 as i16;
                f[2] = t2 as i16;
                f[3] = (-2 * (t0 + t1 + t2)) as i16;
                f[4] = f[2];
                f[5] = f[1];
                f[6] = f[0];
                f
            };
            let w = WienerInfoLr {
                vfilter: mk(&mut rng),
                hfilter: mk(&mut rng),
            };
            let r = WienerInfoLr {
                vfilter: mk(&mut rng),
                hfilter: mk(&mut rng),
            };
            let bits = lr::count_wiener_bits(win, &w, &r);
            // Reference composition against the C primitive counter.
            let mut want = 0;
            for (f, rf) in [(&w.vfilter, &r.vfilter), (&w.hfilter, &r.hfilter)] {
                if win == WIENER_WIN {
                    want += c::ref_count_primitive_refsubexpfin(
                        16,
                        1,
                        (rf[0] as i32 + 5) as u16,
                        (f[0] as i32 + 5) as u16,
                    );
                }
                want += c::ref_count_primitive_refsubexpfin(
                    32,
                    2,
                    (rf[1] as i32 + 23) as u16,
                    (f[1] as i32 + 23) as u16,
                );
                want += c::ref_count_primitive_refsubexpfin(
                    64,
                    3,
                    (rf[2] as i32 + 17) as u16,
                    (f[2] as i32 + 17) as u16,
                );
            }
            assert_eq!(bits, want, "count_wiener_bits win={win}");
        }
        // Random sgrproj pair over every ep class (r0+r1 / r1-only / r0-only).
        let ep = rng.below(16) as i32;
        let rad = SGR_PARAMS_R[ep as usize];
        let mk_xqd = |rng: &mut Rng, rad: [i32; 2]| {
            if rad[0] == 0 {
                [0, rng.range(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1)]
            } else if rad[1] == 0 {
                let x0 = rng.range(SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0);
                [x0, (128 - x0).clamp(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1)]
            } else {
                [
                    rng.range(SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0),
                    rng.range(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1),
                ]
            }
        };
        let s = SgrprojInfoLr {
            ep,
            xqd: mk_xqd(&mut rng, rad),
        };
        // The reference may carry any ep's params (per-tile chaining).
        let ref_ep = rng.below(16) as i32;
        let r = SgrprojInfoLr {
            ep: ref_ep,
            xqd: mk_xqd(&mut rng, SGR_PARAMS_R[ref_ep as usize]),
        };
        let bits = lr::count_sgrproj_bits(&s, &r);
        let mut want = 4; // SGRPROJ_PARAMS_BITS
        if rad[0] > 0 {
            want += c::ref_count_primitive_refsubexpfin(
                128,
                4,
                (r.xqd[0] + 96) as u16,
                (s.xqd[0] + 96) as u16,
            );
        }
        if rad[1] > 0 {
            want += c::ref_count_primitive_refsubexpfin(
                128,
                4,
                (r.xqd[1] + 32) as u16,
                (s.xqd[1] + 32) as u16,
            );
        }
        assert_eq!(bits, want, "count_sgrproj_bits ep={ep} ref_ep={ref_ep}");
    }
}
