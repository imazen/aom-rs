//! `av1_cost_tokens_from_cdf` + `av1_cost_symbol` (libaom `av1/encoder/cost.{c,h}`):
//! derive the per-symbol integer RD costs (in `1<<9`-per-bit units) that feed
//! `cost_coeffs_txb` from a frame's adaptive CDFs. Byte-exact vs C.

/// `av1_prob_cost[128]` = round(-log2(i/256) << 9) for i = 128..255.
#[rustfmt::skip]
const AV1_PROB_COST: [u16; 128] = [
    512, 506, 501, 495, 489, 484, 478, 473, 467, 462, 456, 451, 446, 441, 435,
    430, 425, 420, 415, 410, 405, 400, 395, 390, 385, 380, 375, 371, 366, 361,
    356, 352, 347, 343, 338, 333, 329, 324, 320, 316, 311, 307, 302, 298, 294,
    289, 285, 281, 277, 273, 268, 264, 260, 256, 252, 248, 244, 240, 236, 232,
    228, 224, 220, 216, 212, 209, 205, 201, 197, 194, 190, 186, 182, 179, 175,
    171, 168, 164, 161, 157, 153, 150, 146, 143, 139, 136, 132, 129, 125, 122,
    119, 115, 112, 109, 105, 102, 99,  95,  92,  89,  86,  82,  79,  76,  73,
    70,  66,  63,  60,  57,  54,  51,  48,  45,  42,  38,  35,  32,  29,  26,
    23,  20,  18,  15,  12,  9,   6,   3,
];

const CDF_PROB_BITS: u32 = 15;
const CDF_PROB_TOP: i32 = 1 << CDF_PROB_BITS; // 32768
const EC_MIN_PROB: i32 = 4;
const PROB_COST_SHIFT: i32 = 9;

/// `get_prob(num, den)`: branchless `clamp((num*256 + den/2)/den, 1, 255)`.
#[inline]
fn get_prob(num: u32, den: u32) -> i32 {
    let p = (((num as u64) * 256 + (den as u64 >> 1)) / den as u64) as i32;
    // p | ((255 - p) >> 23) | (p == 0), then truncated to u8.
    let clipped = p | ((255 - p) >> 23) | i32::from(p == 0);
    (clipped as u8) as i32
}

/// `av1_cost_symbol`: cost of a symbol with Q15 probability `p15`.
#[inline]
pub fn cost_symbol(p15: i32) -> i32 {
    let p15 = p15.clamp(1, CDF_PROB_TOP - 1);
    let shift = CDF_PROB_BITS as i32 - 1 - (31 - (p15 as u32).leading_zeros()) as i32; // get_msb
    let prob = get_prob((p15 << shift) as u32, CDF_PROB_TOP as u32);
    AV1_PROB_COST[(prob - 128) as usize] as i32 + (shift << PROB_COST_SHIFT)
}

/// `av1_cost_tokens_from_cdf`: fill `costs` with the per-symbol RD costs derived
/// from the `nsymbs`-symbol inverse-CDF `cdf` (`cdf[nsymbs-1] == 0`). If
/// `inv_map` is given, `costs[inv_map[i]]` is written instead of `costs[i]`.
pub fn cost_tokens_from_cdf(costs: &mut [i32], cdf: &[u16], inv_map: Option<&[i32]>) {
    let mut prev_cdf: i32 = 0;
    let mut i = 0usize;
    loop {
        let icdf = CDF_PROB_TOP - cdf[i] as i32; // AOM_ICDF(cdf[i])
        let mut p15 = icdf - prev_cdf;
        if p15 < EC_MIN_PROB {
            p15 = EC_MIN_PROB;
        }
        prev_cdf = icdf;
        let c = cost_symbol(p15);
        match inv_map {
            Some(m) => costs[m[i] as usize] = c,
            None => costs[i] = c,
        }
        if cdf[i] == 0 {
            // AOM_ICDF(CDF_PROB_TOP) == 0
            break;
        }
        i += 1;
    }
}
