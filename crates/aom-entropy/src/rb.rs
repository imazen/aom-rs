//! `aom_read_bit_buffer` (libaom `aom_dsp/bitreader_buffer.c`): the byte-aligned,
//! MSB-first bit reader used for the uncompressed headers (sequence / frame / tile
//! group / OBU) — the exact inverse of [`crate::wb::WriteBitBuffer`]. Distinct from
//! the `od_ec` arithmetic coder used for coefficients. Reading past the end returns
//! `0` and sets `error` (mirroring libaom's error handler).

/// A borrowed MSB-first bit reader.
#[derive(Clone, Debug)]
pub struct ReadBitBuffer<'a> {
    buf: &'a [u8],
    bit_offset: usize,
    /// Set once a read runs past the end of the buffer.
    pub error: bool,
}

impl<'a> ReadBitBuffer<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            bit_offset: 0,
            error: false,
        }
    }

    /// `aom_rb_read_bit`: one bit at the current MSB-first position.
    pub fn read_bit(&mut self) -> u32 {
        let off = self.bit_offset;
        let p = off >> 3;
        if p < self.buf.len() {
            let q = 7 - (off & 7);
            self.bit_offset = off + 1;
            ((self.buf[p] >> q) & 1) as u32
        } else {
            self.error = true;
            0
        }
    }

    /// `aom_rb_read_literal`: `bits` MSB-first bits into a signed int (`bits <= 31`).
    pub fn read_literal(&mut self, bits: u32) -> i32 {
        let mut v = 0i32;
        for _ in 0..bits {
            v = (v << 1) | self.read_bit() as i32;
        }
        v
    }

    /// `aom_rb_read_unsigned_literal`: `bits` MSB-first bits into a u32 (`bits <= 32`).
    pub fn read_unsigned_literal(&mut self, bits: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..bits {
            v = (v << 1) | self.read_bit();
        }
        v
    }

    /// `aom_rb_read_inv_signed_literal`: inverse of
    /// [`crate::wb::WriteBitBuffer::write_inv_signed_literal`] — read `bits+1` bits and
    /// sign-extend from bit `bits`.
    pub fn read_inv_signed_literal(&mut self, bits: u32) -> i32 {
        let nbits = 32 - bits - 1;
        let value = self.read_unsigned_literal(bits + 1) << nbits;
        (value as i32) >> nbits
    }

    /// `aom_rb_read_uvlc`: inverse of [`crate::wb::WriteBitBuffer::write_uvlc`] — the
    /// Exp-Golomb unsigned variable-length code. Returns `u32::MAX` on 32+ leading zeros.
    pub fn read_uvlc(&mut self) -> u32 {
        let mut leading_zeros = 0u32;
        while self.read_bit() == 0 {
            leading_zeros += 1;
            if leading_zeros >= 32 {
                return u32::MAX;
            }
        }
        let base = (1u32 << leading_zeros) - 1;
        base + self.read_unsigned_literal(leading_zeros)
    }

    /// Current bit position.
    pub fn bit_position(&self) -> usize {
        self.bit_offset
    }

    /// `aom_rb_bytes_read` (rounds up to whole bytes).
    pub fn bytes_read(&self) -> usize {
        self.bit_offset.div_ceil(8)
    }

    /// Whether the read cursor sits on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_offset.is_multiple_of(8)
    }

    /// Advance to the next byte boundary (consumes the padding/stop bits).
    pub fn byte_align(&mut self) {
        while !self.is_byte_aligned() {
            self.read_bit();
        }
    }
}

/// `get_msb`: floor(log2(n)) for n > 0.
fn msb32(n: u32) -> u32 {
    31 - n.leading_zeros()
}

/// `inv_recenter_nonneg` (`aom_dsp/recenter.h`): inverse of `recenter_nonneg`.
fn inv_recenter_nonneg(r: u16, v: u16) -> u16 {
    if v > (r << 1) {
        v
    } else if v & 1 == 0 {
        (v >> 1) + r
    } else {
        r - ((v + 1) >> 1)
    }
}

/// `inv_recenter_finite_nonneg` (`aom_dsp/recenter.h`): inverse of `recenter_finite_nonneg`.
fn inv_recenter_finite_nonneg(n: u16, r: u16, v: u16) -> u16 {
    if (r << 1) <= n {
        inv_recenter_nonneg(r, v)
    } else {
        n - 1 - inv_recenter_nonneg(n - 1 - r, v)
    }
}

impl ReadBitBuffer<'_> {
    /// `read_primitive_quniform`: inverse of `wb_write_primitive_quniform` — a
    /// truncated-uniform value in `[0, n)`.
    fn read_primitive_quniform(&mut self, n: u16) -> u16 {
        if n <= 1 {
            return 0;
        }
        let l = msb32(n as u32) + 1;
        let m = (1u32 << l) - n as u32;
        let v = self.read_unsigned_literal(l - 1);
        if v < m {
            v as u16
        } else {
            ((v << 1) - m + self.read_bit()) as u16
        }
    }

    /// `read_primitive_subexpfin`: inverse of `wb_write_primitive_subexpfin`.
    fn read_primitive_subexpfin(&mut self, n: u16, k: u16) -> u16 {
        let (n, k) = (n as i32, k as i32);
        let mut i = 0i32;
        let mut mk = 0i32;
        loop {
            let b = if i != 0 { k + i - 1 } else { k };
            let a = 1i32 << b;
            if n <= mk + 3 * a {
                return self.read_primitive_quniform((n - mk) as u16) + mk as u16;
            }
            if self.read_bit() != 0 {
                i += 1;
                mk += a;
            } else {
                return (self.read_unsigned_literal(b as u32) as i32 + mk) as u16;
            }
        }
    }

    /// `read_primitive_refsubexpfin`: inverse of `wb_write_primitive_refsubexpfin` —
    /// subexp-coded relative to `ref` after recentering into `[0, n)`.
    fn read_primitive_refsubexpfin(&mut self, n: u16, k: u16, ref_: u16) -> u16 {
        let v = self.read_primitive_subexpfin(n, k);
        inv_recenter_finite_nonneg(n, ref_, v)
    }

    /// `aom_rb_read_signed_primitive_refsubexpfin` — inverse of
    /// [`crate::wb::WriteBitBuffer::write_signed_primitive_refsubexpfin`]: a signed value
    /// in `[-(n-1), n-1]` subexp-coded relative to `ref` (global-motion parameters).
    pub fn read_signed_primitive_refsubexpfin(&mut self, n: u16, k: u16, ref_: i16) -> i16 {
        let ref_u = (ref_ as i32 + n as i32 - 1) as u16;
        let scaled_n = (n << 1) - 1;
        let v = self.read_primitive_refsubexpfin(scaled_n, k, ref_u);
        (v as i32 - (n as i32 - 1)) as i16
    }
}
