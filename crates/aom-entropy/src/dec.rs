//! Daala range *decoder* (`od_ec_dec`), bit-exact port of libaom v3.14.1
//! `aom_dsp/entdec.c`. `od_ec_window` is 32-bit (`entcode.h`).

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;
const OD_EC_WINDOW_SIZE: i32 = 32;
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
    dif: u32,
    rng: u16,
    cnt: i32,
    /// `aom_reader.allow_update_cdf` (`aom_dsp/bitreader.h`): when false the
    /// symbol reader ([`crate::read_symbol`]) skips the post-decode `update_cdf`
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
            dif: (1u32 << (OD_EC_WINDOW_SIZE - 1)) - 1,
            rng: 0x8000,
            cnt: -15,
            allow_update_cdf: true,
        };
        d.refill();
        d
    }

    /// `od_ec_dec_refill`
    fn refill(&mut self) {
        let mut dif = self.dif;
        let mut cnt = self.cnt;
        let mut bptr = self.bptr;
        let end = self.end;
        let mut s = OD_EC_WINDOW_SIZE - 9 - (cnt + 15);
        while s >= 0 && bptr < end {
            dif ^= (self.buf[bptr] as u32) << s;
            cnt += 8;
            s -= 8;
            bptr += 1;
        }
        if bptr >= end {
            self.tell_offs += OD_EC_LOTS_OF_BITS - cnt;
            cnt = OD_EC_LOTS_OF_BITS;
        }
        self.dif = dif;
        self.cnt = cnt;
        self.bptr = bptr;
    }

    /// `od_ec_dec_normalize`
    fn normalize(&mut self, dif: u32, rng: u32, ret: i32) -> i32 {
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
        let vw = v << (OD_EC_WINDOW_SIZE - 16);
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
        let c = dif >> (OD_EC_WINDOW_SIZE - 16);
        let mut v = r;
        let mut u;
        let mut ret = -1i32;
        loop {
            u = v;
            ret += 1;
            v = (((r >> 8) * (icdf[ret as usize] as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
                + EC_MIN_PROB * (n - ret) as u32;
            if c >= v {
                break;
            }
        }
        let r_new = u - v;
        dif -= v << (OD_EC_WINDOW_SIZE - 16);
        self.normalize(dif, r_new, ret)
    }
}
