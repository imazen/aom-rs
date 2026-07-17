//! Daala range *encoder* (`od_ec_enc`), bit-exact port of libaom v3.14.1
//! `aom_dsp/entenc.c` (+ `entcode.h` constants). Byte-identical output is the
//! contract. Little-endian target (matches the HW matrix); the 8-byte flush
//! store uses big-endian byte order (`HToBE64`).

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;
const CDF_PROB_TOP: u32 = 1 << 15;

#[inline]
fn od_ilog_nz(x: u32) -> i32 {
    // 1 + get_msb(x) = bit length of x (x != 0)
    debug_assert!(x != 0);
    (32 - x.leading_zeros()) as i32
}

/// `propagate_carry_bwd`: add 1 at `offs` and ripple the carry backwards.
#[inline]
fn propagate_carry_bwd(buf: &mut [u8], mut offs: usize) {
    loop {
        let sum = buf[offs] as u16 + 1;
        buf[offs] = sum as u8;
        if sum >> 8 == 0 {
            break;
        }
        offs = offs.wrapping_sub(1);
    }
}

/// The entropy encoder context.
pub struct OdEcEnc {
    buf: Vec<u8>,
    offs: u32,
    low: u64,
    rng: u16,
    cnt: i32,
    error: bool,
    /// `aom_writer.allow_update_cdf` (bitwriter.h): `aom_write_symbol`
    /// adapts the CDF iff this is set — C's write_modes clears it when the
    /// frame header codes `disable_cdf_update` (`--cdf-update-mode=0`).
    /// Default TRUE (the adapting form, byte-identical to the pre-flag
    /// behavior); mirrors `OdEcDec::allow_update_cdf` on the read side.
    pub allow_update_cdf: bool,
}

impl Default for OdEcEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl OdEcEnc {
    pub fn new() -> Self {
        let mut e = OdEcEnc {
            buf: Vec::new(),
            offs: 0,
            low: 0,
            rng: 0x8000,
            cnt: -9,
            error: false,
            allow_update_cdf: true,
        };
        e.reset();
        e
    }

    pub fn reset(&mut self) {
        self.offs = 0;
        self.low = 0;
        self.rng = 0x8000;
        self.cnt = -9; // crosses zero after one byte + one carry bit
        self.error = false;
        self.buf.clear();
    }

    #[inline]
    fn ensure(&mut self, need: u32) {
        if (self.buf.len() as u32) < need {
            self.buf.resize(need as usize, 0);
        }
    }

    /// `od_ec_enc_normalize`
    fn normalize(&mut self, mut low: u64, rng: u32) {
        if self.error {
            return;
        }
        let mut c = self.cnt;
        let d = 16 - od_ilog_nz(rng);
        let mut s = c + d;
        if s >= 40 {
            self.ensure(self.offs + 8);
            let offs = self.offs as usize;
            let num_bytes_ready = ((s >> 3) + 1) as u32;
            c += 24 - (num_bytes_ready << 3) as i32;
            let output = low >> c;
            low &= (1u64 << c) - 1;
            let mask = 1u64 << (num_bytes_ready << 3);
            let carry = output & mask;
            let out_val = output & (mask - 1);
            let value = out_val << ((8 - num_bytes_ready) << 3);
            self.buf[offs..offs + 8].copy_from_slice(&value.to_be_bytes());
            if carry != 0 {
                propagate_carry_bwd(&mut self.buf, offs - 1);
            }
            self.offs = offs as u32 + num_bytes_ready;
            s = c + d - 24;
        }
        self.low = low << d;
        self.rng = (rng << d) as u16;
        self.cnt = s;
    }

    /// `od_ec_encode_q15`
    fn encode_q15(&mut self, fl: u32, fh: u32, s: i32, nsyms: i32) {
        let mut l = self.low;
        let mut r = self.rng as u32;
        let n = nsyms - 1;
        if fl < CDF_PROB_TOP {
            let u = (((r >> 8) * (fl >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
                + EC_MIN_PROB * (n - (s - 1)) as u32;
            let v = (((r >> 8) * (fh >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
                + EC_MIN_PROB * (n - s) as u32;
            l += (r - u) as u64;
            r = u - v;
        } else {
            r -= (((r >> 8) * (fh >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
                + EC_MIN_PROB * (n - s) as u32;
        }
        self.normalize(l, r);
    }

    /// `od_ec_encode_bool_q15`
    pub fn encode_bool_q15(&mut self, val: i32, f: u32) {
        let mut l = self.low;
        let r = self.rng as u32;
        let mut v = ((r >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
        v += EC_MIN_PROB;
        if val != 0 {
            l += (r - v) as u64;
        }
        let r_new = if val != 0 { v } else { r - v };
        self.normalize(l, r_new);
    }

    /// `od_ec_encode_cdf_q15`. `icdf` is the inverse CDF (Q15), `icdf[nsyms-1]==0`.
    pub fn encode_cdf_q15(&mut self, s: i32, icdf: &[u16], nsyms: i32) {
        let fl = if s > 0 {
            icdf[(s - 1) as usize] as u32
        } else {
            CDF_PROB_TOP
        };
        let fh = icdf[s as usize] as u32;
        self.encode_q15(fl, fh, s, nsyms);
    }

    /// `od_ec_enc_done`: flush and return the final byte buffer.
    pub fn done(&mut self) -> &[u8] {
        let l = self.low;
        let mut c = self.cnt;
        let mut s = 10 + c;
        let m: u64 = 0x3FFF;
        let mut e = ((l + m) & !m) | (m + 1);
        let mut offs = self.offs;
        let s_bits = (s + 7) >> 3;
        let b = s_bits.max(0) as u32;
        self.ensure(offs + b + 8);
        if s > 0 {
            let mut n: u64 = (1u64 << (c + 16)) - 1;
            loop {
                let val = (e >> (c + 16)) as u16;
                self.buf[offs as usize] = (val & 0x00FF) as u8;
                if val & 0x0100 != 0 {
                    propagate_carry_bwd(&mut self.buf, offs as usize - 1);
                }
                offs += 1;
                e &= n;
                s -= 8;
                c -= 8;
                n >>= 8;
                if s <= 0 {
                    break;
                }
            }
        }
        &self.buf[..offs as usize]
    }
}
