//! Hadamard transform + SATD, bit-exact port of libaom v3.14.1 `aom_dsp/avg.c`.
//! Used for SATD-based RD cost in the encoder. Internal passes are int16
//! (wrapping), matching the C dynamic-range contract; the SSE2/AVX2 output
//! transposes are replicated.

#[inline]
fn hadamard_col8(src: &[i16], off: usize, stride: usize) -> [i16; 8] {
    let s = |k: usize| src[off + k * stride];
    let b0 = s(0).wrapping_add(s(1));
    let b1 = s(0).wrapping_sub(s(1));
    let b2 = s(2).wrapping_add(s(3));
    let b3 = s(2).wrapping_sub(s(3));
    let b4 = s(4).wrapping_add(s(5));
    let b5 = s(4).wrapping_sub(s(5));
    let b6 = s(6).wrapping_add(s(7));
    let b7 = s(6).wrapping_sub(s(7));
    let c0 = b0.wrapping_add(b2);
    let c1 = b1.wrapping_add(b3);
    let c2 = b0.wrapping_sub(b2);
    let c3 = b1.wrapping_sub(b3);
    let c4 = b4.wrapping_add(b6);
    let c5 = b5.wrapping_add(b7);
    let c6 = b4.wrapping_sub(b6);
    let c7 = b5.wrapping_sub(b7);
    let mut o = [0i16; 8];
    o[0] = c0.wrapping_add(c4);
    o[7] = c1.wrapping_add(c5);
    o[3] = c2.wrapping_add(c6);
    o[4] = c3.wrapping_add(c7);
    o[2] = c0.wrapping_sub(c4);
    o[6] = c1.wrapping_sub(c5);
    o[1] = c2.wrapping_sub(c6);
    o[5] = c3.wrapping_sub(c7);
    o
}

#[inline]
fn hadamard_col4(src: &[i16], off: usize, stride: usize) -> [i16; 4] {
    let s = |k: usize| src[off + k * stride] as i32;
    let b0 = ((s(0) + s(1)) >> 1) as i16;
    let b1 = ((s(0) - s(1)) >> 1) as i16;
    let b2 = ((s(2) + s(3)) >> 1) as i16;
    let b3 = ((s(2) - s(3)) >> 1) as i16;
    [
        b0.wrapping_add(b2),
        b1.wrapping_add(b3),
        b0.wrapping_sub(b2),
        b1.wrapping_sub(b3),
    ]
}

/// `aom_hadamard_4x4_c`. `src` row stride is `src_stride`. Returns 16 coeffs.
pub fn hadamard_4x4(src: &[i16], src_stride: usize) -> [i32; 16] {
    let mut buffer = [0i16; 16];
    for idx in 0..4 {
        let col = hadamard_col4(src, idx, src_stride);
        buffer[idx * 4..idx * 4 + 4].copy_from_slice(&col);
    }
    let mut buffer2 = [0i16; 16];
    for idx in 0..4 {
        let col = hadamard_col4(&buffer, idx, 4);
        buffer2[idx * 4..idx * 4 + 4].copy_from_slice(&col);
    }
    let mut coeff = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            coeff[i * 4 + j] = buffer2[j * 4 + i] as i32;
        }
    }
    coeff
}

/// `aom_hadamard_8x8_c`. Returns 64 coeffs.
pub fn hadamard_8x8(src: &[i16], src_stride: usize) -> [i32; 64] {
    let mut buffer = [0i16; 64];
    for idx in 0..8 {
        let col = hadamard_col8(src, idx, src_stride);
        buffer[idx * 8..idx * 8 + 8].copy_from_slice(&col);
    }
    let mut buffer2 = [0i16; 64];
    for idx in 0..8 {
        let col = hadamard_col8(&buffer, idx, 8);
        buffer2[idx * 8..idx * 8 + 8].copy_from_slice(&col);
    }
    let mut coeff = [0i32; 64];
    for i in 0..8 {
        for j in 0..8 {
            coeff[i * 8 + j] = buffer2[j * 8 + i] as i32;
        }
    }
    coeff
}

/// `aom_hadamard_16x16_c`. Returns 256 coeffs.
pub fn hadamard_16x16(src: &[i16], src_stride: usize) -> [i32; 256] {
    let mut coeff = [0i32; 256];
    for idx in 0..4 {
        let off = (idx >> 1) * 8 * src_stride + (idx & 1) * 8;
        let sub = hadamard_8x8(&src[off..], src_stride);
        coeff[idx * 64..idx * 64 + 64].copy_from_slice(&sub);
    }
    for idx in 0..64 {
        let a0 = coeff[idx];
        let a1 = coeff[idx + 64];
        let a2 = coeff[idx + 128];
        let a3 = coeff[idx + 192];
        let b0 = (a0.wrapping_add(a1)) >> 1;
        let b1 = (a0.wrapping_sub(a1)) >> 1;
        let b2 = (a2.wrapping_add(a3)) >> 1;
        let b3 = (a2.wrapping_sub(a3)) >> 1;
        coeff[idx] = b0.wrapping_add(b2);
        coeff[idx + 64] = b1.wrapping_add(b3);
        coeff[idx + 128] = b0.wrapping_sub(b2);
        coeff[idx + 192] = b1.wrapping_sub(b3);
    }
    // Swap columns [4..8) and [8..12) of each row (AVX2 output order).
    for i in 0..16 {
        for j in 0..4 {
            coeff.swap(i * 16 + 4 + j, i * 16 + 8 + j);
        }
    }
    coeff
}

/// `aom_satd_c`: sum of absolute coefficients.
pub fn satd(coeff: &[i32]) -> i32 {
    let mut s: i32 = 0;
    for &c in coeff {
        s = s.wrapping_add(c.abs());
    }
    s
}
