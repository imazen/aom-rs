//! `aom_write_bit_buffer` (libaom `aom_dsp/bitwriter_buffer.c`): the byte-aligned,
//! MSB-first bit writer used for the uncompressed headers (sequence / frame / tile
//! group / OBU). Distinct from the `od_ec` arithmetic coder used for coefficients.
//! Byte-identical output to C libaom.

/// A growable MSB-first bit buffer.
#[derive(Clone, Debug, Default)]
pub struct WriteBitBuffer {
    buf: Vec<u8>,
    bit_offset: usize,
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

    /// `aom_wb_is_byte_aligned`.
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_offset % 8 == 0
    }

    /// `aom_wb_bytes_written` (rounds up to whole bytes).
    pub fn bytes_written(&self) -> usize {
        self.bit_offset / 8 + usize::from(self.bit_offset % 8 > 0)
    }

    /// The written bytes (`bytes_written()`-long).
    pub fn bytes(&self) -> &[u8] {
        &self.buf[..self.bytes_written()]
    }
}
