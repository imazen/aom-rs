//! `aom_write_bit_buffer` (libaom `aom_dsp/bitwriter_buffer.c`): the byte-aligned,
//! MSB-first bit writer used for the uncompressed headers (sequence / frame / tile
//! group / OBU). Distinct from the `od_ec` arithmetic coder used for coefficients.
//! Byte-identical output to C libaom.

/// A growable MSB-first bit buffer.
#[derive(Clone, Debug, Default)]
pub struct WriteBitBuffer {
    buf: Vec<u8>,
    bit_offset: usize,
    /// A single saved append position (bits), the port's analogue of libaom's
    /// `*saved_wb = *wb` in `write_tile_info` (`av1/encoder/bitstream.c`),
    /// recorded just before the `context_update_tile_id` +
    /// `tile_size_bytes_minus_1` placeholders so a later [`Self::overwrite_literal`]
    /// can patch them once the real tile sizes / largest-tile id are known.
    saved_pos: Option<usize>,
}

impl WriteBitBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `aom_wb_write_bit`: append one bit at the current MSB-first position.
    pub fn write_bit(&mut self, bit: u32) {
        let off = self.bit_offset;
        let p = off / 8;
        let q = 7 - off % 8;
        if p >= self.buf.len() {
            self.buf.push(0);
        }
        if q == 7 {
            // First bit of a fresh byte: zero it and set.
            self.buf[p] = (bit << q) as u8;
        } else {
            self.buf[p] &= !(1u8 << q);
            self.buf[p] |= (bit << q) as u8;
        }
        self.bit_offset = off + 1;
    }

    /// `aom_wb_write_literal`: `bits` MSB-first bits of `data` (signed source, `bits <= 31`).
    pub fn write_literal(&mut self, data: i32, bits: u32) {
        for bit in (0..bits).rev() {
            self.write_bit(((data >> bit) & 1) as u32);
        }
    }

    /// `aom_wb_write_unsigned_literal` (`bits <= 32`).
    pub fn write_unsigned_literal(&mut self, data: u32, bits: u32) {
        for bit in (0..bits).rev() {
            self.write_bit((data >> bit) & 1);
        }
    }

    /// `aom_wb_write_inv_signed_literal`: an extra sign bit (`write_literal(data, bits+1)`).
    pub fn write_inv_signed_literal(&mut self, data: i32, bits: u32) {
        self.write_literal(data, bits + 1);
    }

    /// `add_trailing_bits` (`av1/encoder/bitstream.c`): the byte-alignment stop
    /// bit that closes an uncompressed header — `0x80` when already aligned, else
    /// a single `1` bit (the remaining bits of the byte are already 0).
    pub fn add_trailing_bits(&mut self) {
        if self.is_byte_aligned() {
            self.write_literal(0x80, 8);
        } else {
            self.write_bit(1);
        }
    }

    /// `aom_wb_is_byte_aligned`.
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_offset.is_multiple_of(8)
    }

    /// Pad with zero bits to the next byte boundary (the `mod > 0 =>
    /// write_literal(0, CHAR_BIT - mod)` idiom in `write_ext_tile_info`). No-op
    /// when already aligned.
    pub fn byte_align_zeros(&mut self) {
        let m = self.bit_offset % 8;
        if m > 0 {
            self.write_literal(0, 8 - m as u32);
        }
    }

    /// `aom_wb_bytes_written` (rounds up to whole bytes).
    pub fn bytes_written(&self) -> usize {
        self.bit_offset / 8 + usize::from(!self.bit_offset.is_multiple_of(8))
    }

    /// The written bytes (`bytes_written()`-long).
    pub fn bytes(&self) -> &[u8] {
        &self.buf[..self.bytes_written()]
    }

    /// The current append position, in bits (libaom `aom_wb`'s `bit_offset`).
    pub fn bit_position(&self) -> usize {
        self.bit_offset
    }

    /// Record the current append position as the single saved position — the
    /// port's `*saved_wb = *wb` (libaom `write_tile_info`,
    /// `av1/encoder/bitstream.c`). Read it back with [`Self::saved_position`] and
    /// patch the placeholder(s) that follow with [`Self::overwrite_literal`].
    pub fn mark_saved_position(&mut self) {
        self.saved_pos = Some(self.bit_offset);
    }

    /// The position recorded by the most recent [`Self::mark_saved_position`].
    pub fn saved_position(&self) -> Option<usize> {
        self.saved_pos
    }

    /// `aom_wb_overwrite_literal` (libaom `aom_dsp/bitwriter_buffer.c`): overwrite
    /// `bits` MSB-first bits of `data` at an EARLIER, already-written bit position
    /// `bit_pos`, read-modify-writing each target bit (never zero-filling) and
    /// leaving the append cursor untouched. Unlike libaom's `overwrite_bit`,
    /// which advances a SEPARATE `saved_wb` copy, this port has a single buffer,
    /// so the caller passes each field's explicit `bit_pos` (the next field lives
    /// at `bit_pos + bits`). `bit_pos + bits` must be <= the current append
    /// position (the placeholder must already have been written).
    pub fn overwrite_literal(&mut self, bit_pos: usize, data: i32, bits: u32) {
        debug_assert!(
            bit_pos + bits as usize <= self.bit_offset,
            "overwrite_literal targets bits past the append cursor"
        );
        for i in 0..bits {
            let off = bit_pos + i as usize;
            let bit = ((data >> (bits - 1 - i)) & 1) as u8;
            let p = off / 8;
            let q = 7 - off % 8;
            self.buf[p] &= !(1u8 << q);
            self.buf[p] |= bit << q;
        }
    }

    /// `aom_wb_write_uvlc`: unsigned variable-length (Exp-Golomb) code — `msb(v+1)`
    /// zero bits then `v+1` in `msb(v+1)+1` bits. `v` must not be `u32::MAX`.
    pub fn write_uvlc(&mut self, v: u32) {
        let v = v + 1;
        let leading_zeroes = msb32(v);
        self.write_literal(0, leading_zeroes);
        self.write_unsigned_literal(v, leading_zeroes + 1);
    }

    /// `wb_write_primitive_quniform` (`bitwriter_buffer.c`): a truncated-uniform
    /// value `v in [0, n)`.
    fn write_primitive_quniform(&mut self, n: u16, v: u16) {
        if n <= 1 {
            return;
        }
        let l = msb32(n as u32) + 1;
        let m = (1u32 << l) - n as u32;
        let v = v as u32;
        if v < m {
            self.write_literal(v as i32, l - 1);
        } else {
            self.write_literal((m + ((v - m) >> 1)) as i32, l - 1);
            self.write_bit((v - m) & 1);
        }
    }

    /// `wb_write_primitive_subexpfin`: the finite subexponential code.
    fn write_primitive_subexpfin(&mut self, n: u16, k: u16, v: u16) {
        let (n, k, v) = (n as i32, k as i32, v as i32);
        let mut i = 0i32;
        let mut mk = 0i32;
        loop {
            let b = if i != 0 { k + i - 1 } else { k };
            let a = 1i32 << b;
            if n <= mk + 3 * a {
                self.write_primitive_quniform((n - mk) as u16, (v - mk) as u16);
                break;
            }
            let t = v >= mk + a;
            self.write_bit(t as u32);
            if t {
                i += 1;
                mk += a;
            } else {
                self.write_literal(v - mk, b as u32);
                break;
            }
        }
    }

    /// `wb_write_primitive_refsubexpfin`: `v` subexp-coded relative to `ref` after
    /// recentering into `[0, n)`.
    fn write_primitive_refsubexpfin(&mut self, n: u16, k: u16, ref_: u16, v: u16) {
        self.write_primitive_subexpfin(n, k, recenter_finite_nonneg(n, ref_, v));
    }

    /// `aom_wb_write_signed_primitive_refsubexpfin`: subexp-with-final coding of a
    /// signed `v` in `[-(n-1), n-1]` relative to a reference `ref` in the same range
    /// (used for the global-motion model parameters).
    pub fn write_signed_primitive_refsubexpfin(&mut self, n: u16, k: u16, ref_: i16, v: i16) {
        let ref_u = (ref_ as i32 + n as i32 - 1) as u16;
        let v_u = (v as i32 + n as i32 - 1) as u16;
        let scaled_n = (n << 1) - 1;
        self.write_primitive_refsubexpfin(scaled_n, k, ref_u, v_u);
    }
}

/// `get_msb` on a 32-bit value: `floor(log2(n))` for `n > 0`.
fn msb32(n: u32) -> u32 {
    31 - n.leading_zeros()
}

/// `recenter_nonneg` (`aom_dsp/recenter.h`).
fn recenter_nonneg(r: u16, v: u16) -> u16 {
    if v > (r << 1) {
        v
    } else if v >= r {
        (v - r) << 1
    } else {
        ((r - v) << 1) - 1
    }
}

/// `recenter_finite_nonneg` (`aom_dsp/recenter.h`): recenter `v in [0, n-1]` around
/// a reference `r` in the same range.
fn recenter_finite_nonneg(n: u16, r: u16, v: u16) -> u16 {
    if (r << 1) <= n {
        recenter_nonneg(r, v)
    } else {
        recenter_nonneg(n - 1 - r, n - 1 - v)
    }
}
