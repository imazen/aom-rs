//! Daala range *decoder* (`od_ec_dec`), bit-exact port of libaom v3.14.1
//! `aom_dsp/entdec.c`.
//!
//! PERF (Gate 3, task #37): the window is widened from C's 32-bit
//! `od_ec_window` to **64 bits**, refilled with a single 8-byte big-endian
//! load (the dav1d/rav1d msac `ctx_refill` technique — see rav1d-safe
//! `src/msac.rs`). This is *decision-invariant* with respect to the 32-bit
//! formulation:
//!
//! * every symbol decision compares `dif` against `vw = v << (W - 16)`, whose
//!   low `W - 16` bits are zero, so `dif >= vw ⟺ (dif >> (W - 16)) >= v` —
//!   only the top 16 consumed stream bits ever influence a decision, for ANY
//!   window width `W`;
//! * bytes are consumed in the same order; the wider window merely holds more
//!   of them in flight (`cnt` bookkeeping is the same `s = W - 9 - (cnt+15)`
//!   recurrence with `W = 64`);
//! * past the buffer end both formulations consume all-ones filler bits (the
//!   `dif` init/normalize maintain ones below the consumed region) and stop
//!   refilling via the same `cnt = OD_EC_LOTS_OF_BITS` latch.
//!
//! Validated by `entropy_diff.rs` against the REAL C `od_ec_dec` (random op
//! sequences, plus truncated-buffer sequences that decode far past the end of
//! the buffer). The win: refills happen ~half as often as with a 32-bit window,
//! and each common-case refill is one branchless 64-bit load instead of an
//! up-to-3-iteration per-byte loop. The `#[cold]`/never-inline refill wrapper
//! and the check-free `decode_cdf_q15` iterator scan from the prior entropy
//! codegen chunk are preserved.

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;
const OD_EC_WINDOW_SIZE: i32 = 64;
const OD_EC_LOTS_OF_BITS: i32 = 0x4000;

#[inline]
fn od_ilog_nz(x: u32) -> i32 {
    debug_assert!(x != 0);
    (32 - x.leading_zeros()) as i32
}

/// The entropy decoder context (borrows the input buffer).
pub struct OdEcDec<'a> {
    buf: &'a [u8],
    bptr: usize,
    end: usize,
    tell_offs: i32,
    dif: u64,
    rng: u16,
    cnt: i32,
    /// `aom_reader.allow_update_cdf` (`aom_dsp/bitreader.h`): when false the
    /// symbol reader ([`crate::entropy::read_symbol`]) skips the post-decode `update_cdf`
    /// adaptation step, leaving every CDF at its loaded/initial value for the
    /// whole tile. The decoder sets this to `!features.disable_cdf_update`
    /// (`av1/decoder/decodeframe.c`: `r->allow_update_cdf = allow_update_cdf`,
    /// where `allow_update_cdf = !large_scale && !disable_cdf_update`). Defaults
    /// to `true` so the adapting path is byte-identical and overhead-free unless
    /// a caller explicitly disables it.
    pub allow_update_cdf: bool,
}

impl<'a> OdEcDec<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        let mut d = OdEcDec {
            buf,
            bptr: 0,
            end: buf.len(),
            tell_offs: 10 - (OD_EC_WINDOW_SIZE - 8),
            dif: (1u64 << (OD_EC_WINDOW_SIZE - 1)) - 1,
            rng: 0x8000,
            cnt: -15,
            allow_update_cdf: true,
        };
        d.refill();
        d
    }

    /// `od_ec_dec_refill`
    ///
    /// PERF (Gate 3, task #37): `#[cold]` + never-inline, the rav1d msac
    /// `ctx_refill` structure. Refill count is bounded by the STREAM BYTES
    /// (~len/2.5 calls per tile), not by symbol count, so it is inherently
    /// rare next to the symbol functions. Inlined, LLVM auto-vectorized this
    /// <=3-iteration byte loop into every symbol decoder (measured: ~30-instr
    /// setup with xmm moves per refill, plus a 5-push prologue on EVERY
    /// `decode_cdf_q15`/`decode_bool_q15` call from the register pressure).
    /// Out-of-line, the symbol functions compile to the same compact shape as
    /// C's `od_ec_decode_cdf_q15`. Byte-exact: identical arithmetic, verified
    /// by `entropy_diff.rs` against the real C decoder.
    ///
    /// Widened to the 64-bit window (task #37): with ≥ 8 readable bytes the
    /// whole refill is one masked big-endian load; otherwise the per-byte tail
    /// loop (identical to C) handles the end-of-buffer `OD_EC_LOTS_OF_BITS`
    /// latch. Invariants at entry (`new` or a `cnt < 0` normalize): `cnt >= -15`
    /// ⇒ the deficit `s = 64 - 9 - (cnt + 15) = 40 - cnt` is in `[41, 55]`, so
    /// the byte count `k = s/8 + 1` is at most 7 and one 8-byte load cannot
    /// reach the buffer end.
    #[cold]
    #[inline(never)]
    fn refill(&mut self) {
        let mut dif = self.dif;
        let mut cnt = self.cnt;
        let mut bptr = self.bptr;
        let end = self.end;
        let mut s = OD_EC_WINDOW_SIZE - 9 - (cnt + 15);
        if s >= 0 && end - bptr >= 8 {
            debug_assert!(s <= 55, "cnt >= -15 at every refill site");
            let raw = u64::from_be_bytes(self.buf[bptr..bptr + 8].try_into().unwrap());
            let k = (s as usize >> 3) + 1; // bytes consumed, 1..=7
            // Keep the top k bytes of `raw`; byte i (MSB-first) belongs at
            // shift `s - 8*i`, i.e. the whole masked value shifts down by
            // `56 - s` (byte 0 sits at bits 56..64 of `raw`).
            let masked = raw & (!0u64 << (64 - 8 * k));
            dif ^= masked >> (56 - s) as u32;
            cnt += 8 * k as i32;
            bptr += k;
        // s -= 8*k would now be negative; bptr < end is guaranteed (k <= 7).
        } else {
            while s >= 0 && bptr < end {
                dif ^= (self.buf[bptr] as u64) << s;
                cnt += 8;
                s -= 8;
                bptr += 1;
            }
            if bptr >= end {
                self.tell_offs += OD_EC_LOTS_OF_BITS - cnt;
                cnt = OD_EC_LOTS_OF_BITS;
            }
        }
        self.dif = dif;
        self.cnt = cnt;
        self.bptr = bptr;
    }

    /// `od_ec_dec_normalize`
    #[inline]
    fn normalize(&mut self, dif: u64, rng: u32, ret: i32) -> i32 {
        let d = 16 - od_ilog_nz(rng);
        self.cnt -= d;
        self.dif = (dif.wrapping_add(1) << d).wrapping_sub(1);
        self.rng = (rng << d) as u16;
        if self.cnt < 0 {
            self.refill();
        }
        ret
    }

    /// `od_ec_decode_bool_q15`
    pub fn decode_bool_q15(&mut self, f: u32) -> i32 {
        let mut dif = self.dif;
        let r = self.rng as u32;
        let mut v = ((r >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
        v += EC_MIN_PROB;
        let vw = (v as u64) << (OD_EC_WINDOW_SIZE - 16);
        let mut ret = 1;
        let mut r_new = v;
        if dif >= vw {
            r_new = r - v;
            dif -= vw;
            ret = 0;
        }
        self.normalize(dif, r_new, ret)
    }

    /// `od_ec_decode_cdf_q15`
    pub fn decode_cdf_q15(&mut self, icdf: &[u16], nsyms: i32) -> i32 {
        let mut dif = self.dif;
        let r = self.rng as u32;
        let n = nsyms - 1;
        let c = (dif >> (OD_EC_WINDOW_SIZE - 16)) as u32;
        let mut v = r;
        let mut u = r;
        let mut ret = 0i32;
        // The same serial scan as C (entdec.c), in iterator form so the
        // per-entry access carries no bounds check (measured +3 instr/iter as
        // an indexed loop). Identical read/compare sequence: entry i is read
        // with weight `n - i`, and the loop breaks when `c >= v`. A valid AV1
        // (i)cdf ends in 0 (update_cdf never touches the trailing entry), so
        // by `i == nsyms - 1` we have `v == 0 <= c` and the break always
        // fires within the slice, exactly like C.
        for (i, &e) in icdf[..nsyms as usize].iter().enumerate() {
            u = v;
            ret = i as i32;
            v = (((r >> 8) * (e as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
                + EC_MIN_PROB * (n - ret) as u32;
            if c >= v {
                break;
            }
        }
        let r_new = u - v;
        dif -= (v as u64) << (OD_EC_WINDOW_SIZE - 16);
        self.normalize(dif, r_new, ret)
    }
}
