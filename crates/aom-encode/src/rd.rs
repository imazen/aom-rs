//! Rate-distortion model primitives (libaom `av1/encoder/rd.c`). Fixed-point,
//! bit-exact — the model the encoder's fast RD search uses to estimate a
//! transform block's rate and distortion from its variance and quantizer step.

// Generated from av1/encoder/rd.c model_rd_norm tables (104 entries each).
const RATE_TAB_Q10: [i32; 104] = [
    65536, 6086, 5574, 5275, 5063, 4899, 4764, 4651, 4553, 4389, 4255, 4142, 4044, 3958, 3881, 3811,
    3748, 3635, 3538, 3453, 3376, 3307, 3244, 3186, 3133, 3037, 2952, 2877, 2809, 2747, 2690, 2638,
    2589, 2501, 2423, 2353, 2290, 2232, 2179, 2130, 2084, 2001, 1928, 1862, 1802, 1748, 1698, 1651,
    1608, 1530, 1460, 1398, 1342, 1290, 1243, 1199, 1159, 1086, 1021, 963, 911, 864, 821, 781, 745,
    680, 623, 574, 530, 490, 455, 424, 395, 345, 304, 269, 239, 213, 190, 171, 154, 126, 104, 87, 73,
    61, 52, 44, 38, 28, 21, 16, 12, 10, 8, 6, 5, 3, 2, 1, 1, 1, 0, 0,
];
const DIST_TAB_Q10: [i32; 104] = [
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 21, 24, 26, 29,
    31, 34, 36, 39, 44, 49, 54, 59, 64, 69, 73, 78, 88, 97, 106, 115, 124, 133, 142, 151, 167, 184,
    200, 215, 231, 245, 260, 274, 301, 327, 351, 375, 397, 418, 439, 458, 495, 528, 559, 587, 613,
    637, 659, 680, 717, 749, 777, 801, 823, 842, 859, 874, 899, 919, 936, 949, 960, 969, 977, 983,
    994, 1001, 1006, 1010, 1013, 1015, 1017, 1018, 1020, 1022, 1022, 1023, 1023, 1023, 1024,
];
const XSQ_IQ_Q10: [i32; 104] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 112, 128, 144, 160, 176, 192,
    208, 224, 256, 288, 320, 352, 384, 416, 448, 480, 544, 608, 672, 736, 800, 864, 928, 992, 1120,
    1248, 1376, 1504, 1632, 1760, 1888, 2016, 2272, 2528, 2784, 3040, 3296, 3552, 3808, 4064, 4576,
    5088, 5600, 6112, 6624, 7136, 7648, 8160, 9184, 10208, 11232, 12256, 13280, 14304, 15328, 16352,
    18400, 20448, 22496, 24544, 26592, 28640, 30688, 32736, 36832, 40928, 45024, 49120, 53216, 57312,
    61408, 65504, 73696, 81888, 90080, 98272, 106464, 114656, 122848, 131040, 147424, 163808, 180192,
    196576, 212960, 229344, 245728,
];

const AV1_PROB_COST_SHIFT: u32 = 9;

/// `model_rd_norm`: 4-MSB-sampled interpolation of the Laplacian rate/dist tables.
/// Returns `(r_q10, d_q10)`.
fn model_rd_norm(xsq_q10: i32) -> (i32, i32) {
    let tmp = (xsq_q10 >> 2) + 8;
    let k = (tmp as u32).ilog2() as i32 - 3; // get_msb(tmp) - 3
    let xq = ((k << 3) + ((tmp >> k) & 0x7)) as usize;
    let a_q10 = ((xsq_q10 - XSQ_IQ_Q10[xq]) << 10) >> (2 + k);
    let b_q10 = (1 << 10) - a_q10;
    let r_q10 = (RATE_TAB_Q10[xq] * b_q10 + RATE_TAB_Q10[xq + 1] * a_q10) >> 10;
    let d_q10 = (DIST_TAB_Q10[xq] * b_q10 + DIST_TAB_Q10[xq + 1] * a_q10) >> 10;
    (r_q10, d_q10)
}

/// `av1_model_rd_from_var_lapndz`: model the `(rate, dist)` of a Laplacian source
/// of variance `var`, block area `1<<n_log2` pixels, uniformly quantized with
/// step `qstep`. Fixed-point, bit-exact vs C.
pub fn model_rd_from_var_lapndz(var: i64, n_log2: u32, qstep: u32) -> (i32, i64) {
    if var == 0 {
        return (0, 0);
    }
    const MAX_XSQ_Q10: u64 = 245727;
    let num = ((qstep as u64 * qstep as u64) << (n_log2 + 10)) + (var as u64 >> 1);
    let xsq_q10 = (num / var as u64).min(MAX_XSQ_Q10) as i32;
    let (r_q10, d_q10) = model_rd_norm(xsq_q10);
    // ROUND_POWER_OF_TWO(r_q10 << n_log2, 10 - AV1_PROB_COST_SHIFT = 1).
    let shift = 10 - AV1_PROB_COST_SHIFT;
    let rate = ((((r_q10 as i64) << n_log2) + (1 << (shift - 1))) >> shift) as i32;
    let dist = (var * d_q10 as i64 + 512) >> 10;
    (rate, dist)
}
