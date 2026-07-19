//! IntraBC (intra block copy) SEARCH machinery — the screen-content stills
//! DV search + skip-arm RD (`rd_pick_intrabc_mode_sb`, rdopt.c:3427):
//! - the source-frame hash table (`av1/encoder/hash_motion.c` + the CRC-32C
//!   calculator, `hash.c`) + block hash query + `av1_intrabc_hash_search`
//!   (mcomp.c:1908);
//! - the DV signalling cost table (`av1_build_nmv_cost_table` /
//!   `av1_fill_dv_costs`, encodemv.c / rd.c:708) + mv_cost / mv_err_cost /
//!   mvsad_err_cost forms;
//! - the full-pel search (`av1_full_pixel_search`, mcomp.c:1768): the NSTEP
//!   `full_pixel_diamond` + `full_pixel_exhaustive` mesh (the pixel search
//!   ALWAYS runs at `intrabc_search_level 0`);
//! - `predict_skip_txfm` (tx_search.c:183) + the skip-arm RD.
//!
//! **Coeff-arm scope (honest, KB-14):** this offers an intrabc candidate ONLY
//! in the skip regime (luma `predict_skip_txfm` fires AND chroma is an exact
//! match), where `av1_txfm_search` forces `skip_txfm=1` and BYPASSES the inter
//! var-tx coeff arm — that arm (`av1_pick_recursive_tx_size_type_yrd` quadtree
//! + `prune_tx_2D` / `ml_predict_tx_split` + the var-tx pack) is NOT ported, so
//! real screen content (which codes most intrabc blocks via the coeff arm) is
//! PINNED, not byte-exact. See `rd_close_intrabc` + PARITY C3.
//!
//! Faithfulness notes:
//! - CRC input serialization: the C feeds `uint32_t[4]` buffers to CRC32C as
//!   raw bytes (acknowledged arch-dependent, bug aomedia:433531610); the
//!   oracle build is x86-64 little-endian, so this port serializes with
//!   `to_le_bytes` — bit-identical to the reference behaviour.
//! - The hash table is bucket-keyed by `hash_value1` (block-size tag + CRC
//!   low 16 bits); buckets keep INSERTION order (the C `aom_vector`) capped
//!   at 256 entries — candidate iteration order is part of the tie-breaking
//!   semantics (`<` keeps the earliest best).
//! - `av1_get_crc32c_value` dispatches to a SSE4.2 hardware version in the
//!   reference build; CRC-32C is a bit-exact function of the input either
//!   way, so the table-driven software port is equivalent.
//!
//! Pixel planes are this port's u16-at-any-depth convention; the `bd > 8`
//! flag selects the C hbd hash arms (xor-fold) exactly as
//! `is_cur_buf_hbd(xd)` does.

use std::collections::HashMap;

/// `kSrcBits` (hash_motion.c).
const K_SRC_BITS: u32 = 16;
/// `kMaxCandidatesPerHashBucket`.
const K_MAX_CANDIDATES_PER_BUCKET: usize = 256;
/// `AOM_BUFFER_SIZE_FOR_BLOCK_HASH` (hash.h).
const BLOCK_HASH_BUF: usize = 4096;

// ---------------------------------------------------------------------------
// CRC-32C (hash.c — iSCSI polynomial, table-driven software version)
// ---------------------------------------------------------------------------

/// `CRC32C` + `av1_crc32c_calculator_init` (hash.c): the 8x256 slicing table.
pub struct Crc32c {
    table: [[u32; 256]; 8],
}

impl Crc32c {
    pub fn new() -> Self {
        const POLY: u32 = 0x82f63b78;
        let mut table = [[0u32; 256]; 8];
        for n in 0..256u32 {
            let mut crc = n;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ POLY
                } else {
                    crc >> 1
                };
            }
            table[0][n as usize] = crc;
        }
        for n in 0..256usize {
            let mut crc = table[0][n];
            for k in 1..8 {
                crc = table[0][(crc & 0xff) as usize] ^ (crc >> 8);
                table[k][n] = crc;
            }
        }
        Crc32c { table }
    }

    /// `av1_get_crc32c_value_c` (hash.c): little-endian 8-byte slicing.
    pub fn value(&self, buf: &[u8]) -> u32 {
        let t = &self.table;
        let mut crc: u64 = 0xffffffff;
        let mut i = 0usize;
        // The C aligns to 8; our buffers are 16-byte Vec-backed slices whose
        // base alignment Rust doesn't pin, but the RESULT is
        // alignment-independent (the prefix loop consumes bytes identically).
        while i < buf.len() && (buf.len() - i) >= 8 {
            let chunk = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap());
            crc ^= chunk;
            crc = u64::from(t[7][(crc & 0xff) as usize])
                ^ u64::from(t[6][((crc >> 8) & 0xff) as usize])
                ^ u64::from(t[5][((crc >> 16) & 0xff) as usize])
                ^ u64::from(t[4][((crc >> 24) & 0xff) as usize])
                ^ u64::from(t[3][((crc >> 32) & 0xff) as usize])
                ^ u64::from(t[2][((crc >> 40) & 0xff) as usize])
                ^ u64::from(t[1][((crc >> 48) & 0xff) as usize])
                ^ u64::from(t[0][(crc >> 56) as usize]);
            i += 8;
        }
        while i < buf.len() {
            crc = u64::from(t[0][((crc ^ u64::from(buf[i])) & 0xff) as usize]) ^ (crc >> 8);
            i += 1;
        }
        (crc as u32) ^ 0xffffffff
    }
}

impl Default for Crc32c {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// hash table build (hash_motion.c)
// ---------------------------------------------------------------------------

/// `block_hash` (hash_motion.h): a candidate block position + its full CRC.
#[derive(Clone, Copy, Debug)]
pub struct BlockHash {
    pub x: i16,
    pub y: i16,
    pub hash_value2: u32,
}

/// `hash_table` + `IntraBCHashInfo`: bucket map keyed by `hash_value1`
/// (`(crc & 0xffff) + (size_index << 16)`), insertion-ordered buckets capped
/// at 256 candidates.
pub struct IntrabcHashTable {
    pub buckets: HashMap<u32, Vec<BlockHash>>,
    pub crc: Crc32c,
}

/// `hash_block_size_to_index` (hash_motion.c).
fn hash_block_size_to_index(block_size: usize) -> i32 {
    match block_size {
        4 => 0,
        8 => 1,
        16 => 2,
        32 => 3,
        64 => 4,
        128 => 5,
        _ => -1,
    }
}

/// `get_identity_hash_value` (lbd 2x2 base hash).
#[inline]
fn identity_hash(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (u32::from(a) << 24) + (u32::from(b) << 16) + (u32::from(c) << 8) + u32::from(d)
}

/// `get_xor_hash_value_hbd` (hbd 2x2 base hash).
#[inline]
fn xor_hash_hbd(a: u16, b: u16, c: u16, d: u16) -> u32 {
    let mut result = (u32::from(a & 0x00ff) << 24)
        + (u32::from(b & 0x00ff) << 16)
        + (u32::from(c & 0x00ff) << 8)
        + u32::from(d & 0x00ff);
    result ^= (u32::from(a & 0xff00) << 16)
        + (u32::from(b & 0xff00) << 8)
        + u32::from(c & 0xff00)
        + (u32::from(d & 0xff00) >> 8);
    result
}

/// `av1_generate_block_2x2_hash_value` over this port's u16 planes: the 2x2
/// base layer, one hash per (x, y) with `x < w-1, y < h-1` (stored at
/// `y * width + x`; the +1 borders are never read downstream).
fn generate_block_2x2_hash(
    src: &[u16],
    off: usize,
    stride: usize,
    width: usize,
    height: usize,
    is_hbd: bool,
    out: &mut [u32],
) {
    let x_end = width - 1;
    let y_end = height - 1;
    for y in 0..y_end {
        for x in 0..x_end {
            let p = off + y * stride + x;
            let (a, b, c, d) = (src[p], src[p + 1], src[p + stride], src[p + stride + 1]);
            out[y * width + x] = if is_hbd {
                xor_hash_hbd(a, b, c, d)
            } else {
                identity_hash(a as u8, b as u8, c as u8, d as u8)
            };
        }
    }
}

/// `av1_generate_block_hash_value`: compose 4 half-size hashes into the
/// `block_size` layer via CRC-32C (LE-serialized `u32[4]`).
fn generate_block_hash_layer(
    crc: &Crc32c,
    width: usize,
    height: usize,
    block_size: usize,
    src_hash: &[u32],
    dst_hash: &mut [u32],
) {
    // C: x_end = width - block_size + 1 (int; negative -> the loops skip).
    let x_end = (width + 1).saturating_sub(block_size);
    let y_end = (height + 1).saturating_sub(block_size);
    let src_size = block_size >> 1;
    let mut bytes = [0u8; 16];
    for y in 0..y_end {
        for x in 0..x_end {
            let pos = y * width + x;
            let p = [
                src_hash[pos],
                src_hash[pos + src_size],
                src_hash[pos + src_size * width],
                src_hash[pos + src_size * width + src_size],
            ];
            for (i, v) in p.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            dst_hash[pos] = crc.value(&bytes);
        }
    }
}

/// `av1_add_to_hash_map_by_row_with_precal_data`: the hierarchical
/// (coarse-to-fine, no-two-adjacent) exploration that inserts every candidate
/// block position for one `block_size` layer.
fn add_to_hash_map_by_row(
    buckets: &mut HashMap<u32, Vec<BlockHash>>,
    pic_hash: &[u32],
    pic_width: usize,
    pic_height: usize,
    block_size: usize,
) {
    // C: int x_end/y_end go negative for layers larger than the frame.
    let x_end = (pic_width + 1).saturating_sub(block_size);
    let y_end = (pic_height + 1).saturating_sub(block_size);
    if x_end == 0 || y_end == 0 {
        return;
    }
    let add_value = (hash_block_size_to_index(block_size) as u32) << K_SRC_BITS;
    let crc_mask = (1u32 << K_SRC_BITS) - 1;
    let mut step = block_size;
    let mut x_offset = 0usize;
    let mut y_offset = 0usize;

    while step > 1 {
        let mut x_pos = x_offset;
        while x_pos < x_end {
            let mut y_pos = y_offset;
            while y_pos < y_end {
                let pos = y_pos * pic_width + x_pos;
                let hash_value1 = (pic_hash[pos] & crc_mask) + add_value;
                let bucket = buckets.entry(hash_value1).or_default();
                if bucket.len() < K_MAX_CANDIDATES_PER_BUCKET {
                    bucket.push(BlockHash {
                        x: x_pos as i16,
                        y: y_pos as i16,
                        hash_value2: pic_hash[pos],
                    });
                }
                y_pos += step;
            }
            x_pos += step;
        }
        // The offset/step state machine (hash_motion.c:318-338).
        if x_offset == 0 && y_offset == 0 {
            x_offset = step / 2;
        } else if x_offset == step / 2 && y_offset == 0 {
            x_offset = 0;
            y_offset = step / 2;
        } else if x_offset == 0 && y_offset == step / 2 {
            x_offset = step / 2;
        } else {
            debug_assert!(x_offset == step / 2 && y_offset == step / 2);
            step /= 2;
            x_offset = step / 2;
            y_offset = 0;
        }
    }
}

/// The `encodeframe.c:2199-2255` intrabc hash-table build: the 2x2 base layer,
/// then sizes 4..=`max_size` ping-pong composed and (from `min_alloc_size`=4,
/// `mi_alloc_bsize` BLOCK_4X4 in this envelope) inserted into the table.
/// `src` is the SOURCE luma plane (`cpi->source`), crop `width x height`.
#[allow(clippy::too_many_arguments)]
pub fn build_intrabc_hash_table(
    src: &[u16],
    off: usize,
    stride: usize,
    width: usize,
    height: usize,
    is_hbd: bool,
    sb_px: usize,
) -> IntrabcHashTable {
    let crc = Crc32c::new();
    let mut buckets = HashMap::new();
    let mut buf0 = vec![0u32; width * height];
    let mut buf1 = vec![0u32; width * height];

    generate_block_2x2_hash(src, off, stride, width, height, is_hbd, &mut buf0);
    let max_size = 64.min(sb_px);
    let min_alloc_size = 4usize; // block_size_wide[BLOCK_4X4]

    let mut size = 4usize;
    let mut src_is_0 = true;
    while size <= max_size {
        {
            let (s, d): (&[u32], &mut [u32]) = if src_is_0 {
                (&buf0, &mut buf1)
            } else {
                (&buf1, &mut buf0)
            };
            generate_block_hash_layer(&crc, width, height, size, s, d);
        }
        let d: &[u32] = if src_is_0 { &buf1 } else { &buf0 };
        if size >= min_alloc_size {
            add_to_hash_map_by_row(&mut buckets, d, width, height, size);
        }
        size *= 2;
        src_is_0 = !src_is_0;
    }
    IntrabcHashTable { buckets, crc }
}

/// `av1_get_block_hash_value`: the query block's `(hash_value1, hash_value2)`
/// — 2x2 base + iterated CRC composition over the block only.
pub fn get_block_hash_value(
    table: &IntrabcHashTable,
    src: &[u16],
    off: usize,
    stride: usize,
    block_size: usize,
    is_hbd: bool,
) -> (u32, u32) {
    let add_value = (hash_block_size_to_index(block_size) as u32) << K_SRC_BITS;
    let crc_mask = (1u32 << K_SRC_BITS) - 1;
    let mut buf = [vec![0u32; BLOCK_HASH_BUF], vec![0u32; BLOCK_HASH_BUF]];

    // 2x2 sub-block hashes.
    let mut sub_block_in_width = block_size >> 1;
    for y in (0..block_size).step_by(2) {
        for x in (0..block_size).step_by(2) {
            let pos = (y >> 1) * sub_block_in_width + (x >> 1);
            let p = off + y * stride + x;
            let (a, b, c, d) = (src[p], src[p + 1], src[p + stride], src[p + stride + 1]);
            buf[0][pos] = if is_hbd {
                xor_hash_hbd(a, b, c, d)
            } else {
                identity_hash(a as u8, b as u8, c as u8, d as u8)
            };
        }
    }

    let mut src_sub_block_in_width = sub_block_in_width;
    sub_block_in_width >>= 1;
    let mut src_idx = 0usize;
    let mut dst_idx = 1usize;
    let mut bytes = [0u8; 16];

    let mut sub_width = 4usize;
    while sub_width <= block_size {
        dst_idx = src_idx ^ 1;
        let (lo, hi) = buf.split_at_mut(1);
        let (s, d): (&[u32], &mut [u32]) = if src_idx == 0 {
            (&lo[0], &mut hi[0])
        } else {
            (&hi[0], &mut lo[0])
        };
        let mut dst_pos = 0usize;
        for y in 0..sub_block_in_width {
            for x in 0..sub_block_in_width {
                let src_pos = (y << 1) * src_sub_block_in_width + (x << 1);
                let p = [
                    s[src_pos],
                    s[src_pos + 1],
                    s[src_pos + src_sub_block_in_width],
                    s[src_pos + src_sub_block_in_width + 1],
                ];
                for (i, v) in p.iter().enumerate() {
                    bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
                d[dst_pos] = table.crc.value(&bytes);
                dst_pos += 1;
            }
        }
        src_sub_block_in_width = sub_block_in_width;
        sub_block_in_width >>= 1;
        sub_width *= 2;
        src_idx ^= 1;
    }

    let h = buf[dst_idx][0];
    ((h & crc_mask) + add_value, h)
}

// ---------------------------------------------------------------------------
// DV signalling costs (av1_fill_dv_costs -> av1_build_nmv_cost_table)
// ---------------------------------------------------------------------------

/// `MV_MAX` / `MV_VALS` (entropymv.h): 14-bit magnitudes.
pub const MV_MAX: i32 = (1 << 14) - 1;
pub const MV_VALS: usize = (MV_MAX as usize) * 2 + 1;
const MV_CLASSES: usize = 11;
const MV_OFFSET_BITS: usize = 10;
const CLASS0_BITS: usize = 1;
const CLASS0_SIZE: usize = 1 << CLASS0_BITS;
const MV_FP_SIZE: usize = 4;
/// `MvSubpelPrecision` (entropymv.h:95-97): NONE=-1, LOW_PRECISION=0,
/// HIGH_PRECISION=1. The DV case builds at NONE; inter builds at LOW/HIGH.
pub const MV_SUBPEL_NONE: i32 = -1;
pub const MV_SUBPEL_LOW: i32 = 0;
pub const MV_SUBPEL_HIGH: i32 = 1;

/// `IntraBCMVCosts` (block.h): the joint-type costs + the per-component
/// magnitude cost tables, index-centred at `MV_MAX` (`dv_costs[c][MV_MAX + v]`
/// = C's `dv_costs[c][v]` through the centred pointer).
pub struct DvCosts {
    pub joint_mv: [i32; 4],
    pub dv_costs: [Vec<i32>; 2],
}

/// `av1_cost_tokens_from_cdf` on a padded row (delegates to aom-txb's port).
fn cost_tokens(out: &mut [i32], cdf: &[u16]) {
    aom_dsp::txb::cost_tokens_from_cdf(out, cdf, None);
}

/// `av1_build_nmv_component_cost_table` (encodemv.c:124) at
/// `MV_SUBPEL_NONE` precision — the DV case (`av1_fill_dv_costs`). `mvcost`
/// is the centred view (`[MV_MAX + v]`). The component CDF blob layout is
/// the port's 69-u16 packing (aom-entropy partition.rs:452-461):
/// sign 0..3, classes 3..15, class0 15..18, bits[10] 18..48,
/// class0_fp[2] 48..58, fp 58..63, class0_hp 63..66, hp 66..69.
fn build_nmv_component_cost_table(mvcost: &mut [i32], comp_cdf: &[u16; 69], precision: i32) {
    let mut sign_cost = [0i32; 2];
    let mut class_cost = [0i32; MV_CLASSES];
    let mut class0_cost = [0i32; CLASS0_SIZE];
    let mut bits_cost = [[0i32; 2]; MV_OFFSET_BITS];
    // Fractional (fp) + high-precision (hp) cost arrays: zero at MV_SUBPEL_NONE
    // (the C `= { 0 }` initializers), filled from the component CDFs at higher
    // precision — matching the precision gates in
    // `av1_build_nmv_component_cost_table` (encodemv.c:145-152).
    let mut class0_fp_cost = [[0i32; MV_FP_SIZE]; CLASS0_SIZE];
    let mut fp_cost = [0i32; MV_FP_SIZE];
    let mut class0_hp_cost = [0i32; 2];
    let mut hp_cost = [0i32; 2];

    cost_tokens(&mut sign_cost, &comp_cdf[0..3]);
    cost_tokens(&mut class_cost, &comp_cdf[3..15]);
    cost_tokens(&mut class0_cost, &comp_cdf[15..18]);
    for i in 0..MV_OFFSET_BITS {
        cost_tokens(&mut bits_cost[i], &comp_cdf[18 + i * 3..18 + i * 3 + 3]);
    }
    // precision > MV_SUBPEL_NONE (-1): the fractional-pel bit costs.
    if precision > MV_SUBPEL_NONE {
        for i in 0..CLASS0_SIZE {
            cost_tokens(&mut class0_fp_cost[i], &comp_cdf[48 + i * 5..48 + i * 5 + 5]);
        }
        cost_tokens(&mut fp_cost, &comp_cdf[58..63]);
    }
    // precision > MV_SUBPEL_LOW (0): the high-precision bit costs.
    if precision > MV_SUBPEL_LOW {
        cost_tokens(&mut class0_hp_cost, &comp_cdf[63..66]);
        cost_tokens(&mut hp_cost, &comp_cdf[66..69]);
    }

    let c = MV_MAX as usize; // centre index

    let mut cost_swap = [0i32; MV_OFFSET_BITS];
    let negate_sign = sign_cost[1] - sign_cost[0];
    for i in 1..MV_OFFSET_BITS {
        cost_swap[i] = bits_cost[i - 1][1];
        if i > CLASS0_BITS {
            cost_swap[i] -= class_cost[i - CLASS0_BITS];
        }
    }

    // Seed the fractional costs (fp/hp are zero at MV_SUBPEL_NONE, so this
    // seeds sign_cost[0] over the first 2*MV_FP_SIZE magnitudes).
    for o in 0..MV_FP_SIZE {
        for hp in 0..2usize {
            let v = 2 * o + hp + 1;
            mvcost[c + v] = fp_cost[o] + hp_cost[hp] + sign_cost[0];
        }
    }

    mvcost[c] = 0;
    // Per-exponent fill from the previous exponents.
    for i in 0..MV_OFFSET_BITS {
        let exponent = (2 * MV_FP_SIZE) << i;
        let class = if i >= CLASS0_BITS {
            class_cost[i - CLASS0_BITS + 1]
        } else {
            0
        };
        let mut mantissa = 0usize;
        for j in 0..=i {
            while mantissa < ((2 * MV_FP_SIZE) << j) {
                let cost = mvcost[c + mantissa + 1] + class + cost_swap[j];
                let v = exponent + mantissa + 1;
                mvcost[c + v] = cost;
                mvcost[c - v] = cost + negate_sign;
                mantissa += 1;
            }
            cost_swap[j] += bits_cost[i][0];
        }
    }

    // The last-exponent special case (buffer-overrun guard in C).
    {
        let exponent = (2 * MV_FP_SIZE) << MV_OFFSET_BITS;
        let class = class_cost[MV_CLASSES - 1];
        let mut mantissa = 0usize;
        for j in 0..MV_OFFSET_BITS {
            while mantissa < ((2 * MV_FP_SIZE) << j) {
                let cost = mvcost[c + mantissa + 1] + class + cost_swap[j];
                let v = exponent + mantissa + 1;
                mvcost[c + v] = cost;
                mvcost[c - v] = cost + negate_sign;
                mantissa += 1;
            }
        }
        let cost_swap_hi = bits_cost[MV_OFFSET_BITS - 1][1] - class_cost[MV_CLASSES - 2];
        while mantissa < exponent - 1 {
            let cost = mvcost[c + mantissa + 1] + class + cost_swap_hi;
            let v = exponent + mantissa + 1;
            mvcost[c + v] = cost;
            mvcost[c - v] = cost + negate_sign;
            mantissa += 1;
        }
    }

    // Class-0 vectors overwrite the placeholders.
    for i in 0..CLASS0_SIZE {
        let top = i * 2 * MV_FP_SIZE;
        for o in 0..MV_FP_SIZE {
            let cost = class0_fp_cost[i][o] + class_cost[0] + class0_cost[i];
            for hp in 0..2usize {
                let v = top + 2 * o + hp + 1;
                mvcost[c + v] = cost + class0_hp_cost[hp] + sign_cost[0];
                let neg = cost + class0_hp_cost[hp] + sign_cost[1];
                mvcost[c - v] = neg;
            }
        }
    }
}

/// `av1_fill_dv_costs` (rd.c:708): joint costs + both component tables from
/// the live ndvc CDFs, at `MV_SUBPEL_NONE`.
pub fn fill_dv_costs(
    ndvc_joints: &[u16],
    ndvc_comp0: &[u16; 69],
    ndvc_comp1: &[u16; 69],
) -> DvCosts {
    let mut joint_mv = [0i32; 4];
    cost_tokens(&mut joint_mv, ndvc_joints);
    let mut c0 = vec![0i32; MV_VALS];
    let mut c1 = vec![0i32; MV_VALS];
    build_nmv_component_cost_table(&mut c0, ndvc_comp0, MV_SUBPEL_NONE);
    build_nmv_component_cost_table(&mut c1, ndvc_comp1, MV_SUBPEL_NONE);
    DvCosts {
        joint_mv,
        dv_costs: [c0, c1],
    }
}

/// `av1_build_nmv_cost_table` (encodemv.c:294): the inter MV cost tables built
/// from the frame's live `nmv_context` at a given subpel `precision`
/// ([`MV_SUBPEL_NONE`]/[`MV_SUBPEL_LOW`]/[`MV_SUBPEL_HIGH`]). Same shape as
/// [`DvCosts`] (joint costs + the two centred component magnitude tables). The
/// DV case ([`fill_dv_costs`]) is exactly this at `MV_SUBPEL_NONE`; the inter
/// motion search (`x->mv_costs`) uses `LOW` (integer/low-precision frames) or
/// `HIGH` (`allow_high_precision_mv`). `nmv_comp{0,1}` are the port's 69-u16
/// component packing (partition.rs); `nmv_joints` is the 5-u16 joints CDF.
pub fn fill_nmv_costs(
    precision: i32,
    nmv_joints: &[u16],
    nmv_comp0: &[u16; 69],
    nmv_comp1: &[u16; 69],
) -> DvCosts {
    let mut joint_mv = [0i32; 4];
    cost_tokens(&mut joint_mv, nmv_joints);
    let mut c0 = vec![0i32; MV_VALS];
    let mut c1 = vec![0i32; MV_VALS];
    build_nmv_component_cost_table(&mut c0, nmv_comp0, precision);
    build_nmv_component_cost_table(&mut c1, nmv_comp1, precision);
    DvCosts {
        joint_mv,
        dv_costs: [c0, c1],
    }
}

/// `mv_cost` (mcomp.c:329): `joint + comp0[row] + comp1[col]` on the centred
/// tables.
#[inline]
pub fn mv_cost(diff_row: i32, diff_col: i32, dv: &DvCosts) -> i32 {
    let joint = aom_dsp::entropy::partition::get_mv_joint(diff_row, diff_col) as usize;
    dv.joint_mv[joint]
        + dv.dv_costs[0][(MV_MAX + diff_row) as usize]
        + dv.dv_costs[1][(MV_MAX + diff_col) as usize]
}

/// `av1_mv_bit_cost` (mcomp.c:334): the RD-search DV rate —
/// `ROUND_POWER_OF_TWO(mv_cost(diff * 8) * weight, 7)` with
/// `MV_COST_WEIGHT_SUB` = 120 at the intrabc call site (rdopt.c:3606).
#[inline]
pub fn mv_bit_cost_sub(dv_row: i32, dv_col: i32, ref_row: i32, ref_col: i32, dv: &DvCosts) -> i32 {
    const MV_COST_WEIGHT_SUB: i64 = 120;
    let c = i64::from(mv_cost(dv_row - ref_row, dv_col - ref_col, dv));
    ((c * MV_COST_WEIGHT_SUB + 64) >> 7) as i32
}

/// `mv_err_cost` (mcomp.c:341), `MV_COST_ENTROPY` arm: the full-pel search's
/// variance-metric MV cost. `error_per_bit` = `AOMMAX(rdmult >> 6, 1)`
/// (av1_set_error_per_bit); the shift is `RDDIV_BITS(7) +
/// AV1_PROB_COST_SHIFT(9) - RD_EPB_SHIFT(6) + PIXEL_TRANSFORM_ERROR_SCALE(4)`
/// = 14, 64-bit rounded.
#[inline]
pub fn mv_err_cost(
    diff_row_subpel: i32,
    diff_col_subpel: i32,
    dv: &DvCosts,
    error_per_bit: i32,
) -> i32 {
    let c = i64::from(mv_cost(diff_row_subpel, diff_col_subpel, dv)) * i64::from(error_per_bit);
    ((c + (1 << 13)) >> 14) as i32
}

/// `mvsad_err_cost` (mcomp.c:372), `MV_COST_ENTROPY` arm: the SAD-metric MV
/// cost over FULL-PEL diffs promoted to subpel (`GET_MV_SUBPEL` = `*8`);
/// `sad_per_bit` from `av1_set_sad_per_bit`.
#[inline]
pub fn mvsad_err_cost(
    full_diff_row: i32,
    full_diff_col: i32,
    dv: &DvCosts,
    sad_per_bit: i32,
) -> i32 {
    let c = u64::from(mv_cost(full_diff_row * 8, full_diff_col * 8, dv) as u32)
        * u64::from(sad_per_bit as u32);
    ((c + (1 << 8)) >> 9) as i32
}

// ---------------------------------------------------------------------------
// pixel metrics (aom_dsp variance/sad over u16 planes)
// ---------------------------------------------------------------------------

/// `aom_variance{w}x{h}` (variance.c): returns the variance
/// (`sse - sum^2 / (w*h)`) — the hash search's `get_mvpred_var_cost` metric.
/// Arithmetic on u16 pixel values matches C's lowbd path at bd 8 and the
/// highbd path above (both accumulate the same integers).
pub fn variance_wxh(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut sum: i64 = 0;
    let mut sse: u64 = 0;
    for r in 0..h {
        for c in 0..w {
            let d = i64::from(src[src_off + r * src_stride + c])
                - i64::from(refb[ref_off + r * ref_stride + c]);
            sum += d;
            sse += (d * d) as u64;
        }
    }
    (sse - ((sum * sum) as u64) / (w as u64 * h as u64)) as u32
}

/// `aom_sad{w}x{h}`: the diamond search's SAD metric.
pub fn sad_wxh(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut sad: u64 = 0;
    for r in 0..h {
        for c in 0..w {
            let d = i64::from(src[src_off + r * src_stride + c])
                - i64::from(refb[ref_off + r * ref_stride + c]);
            sad += d.unsigned_abs();
        }
    }
    sad as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CRC-32C known-answer test (iSCSI polynomial): "123456789" ->
    /// 0xE3069283 (the canonical check value).
    #[test]
    fn crc32c_known_answer() {
        let crc = Crc32c::new();
        assert_eq!(crc.value(b"123456789"), 0xE3069283);
        // 16-byte (u32[4]) shape the hash layers feed.
        assert_eq!(crc.value(&[0u8; 16]), crc.value(&[0u8; 16]));
        assert_ne!(crc.value(&[0u8; 16]), crc.value(&[1u8; 16]));
    }

    /// `intrabc_predict_chroma` must match the (bit-exact vs C) DECODER's
    /// `intrabc_chroma_predict` for every full-pel DV and subsampling. The
    /// decoder (aom-decode/src/lib.rs, conformance-bit-exact) derives the
    /// chroma ref position + subpel as `mvq4 = dv << (1 - ss); off = mvq4 >> 4;
    /// subpel = mvq4 & 15`, then the same 2-tap copy/h-half/v-half/bilinear
    /// interpolation. Our encoder-side predictor derives subpel INTERNALLY
    /// (`(dv>>3>>ss)`, `((dv>>3)&1)*8`); this transcribes the decoder's exact
    /// reference and asserts byte-identity across DVs / ss / bit depth — the
    /// HANDOFF-flagged "diff-test chroma predict vs the decoder" hazard.
    #[test]
    fn intrabc_chroma_predict_matches_decoder() {
        let stride = 48usize;
        let mut recon = vec![0u16; stride * 48];
        let mut s = 0x1234_5678u32;
        for p in recon.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *p = ((s >> 13) & 0x3ff) as u16;
        }
        let (cw, ch) = (8usize, 8usize);
        let block_off = 20 * stride + 20;
        for &(ss_x, ss_y) in &[(1usize, 1usize), (1, 0), (0, 0)] {
            for &bd in &[8i32, 10, 12] {
                for kr in -6..=0i32 {
                    for kc in -6..=0i32 {
                        let (dv_row, dv_col) = (kr * 8, kc * 8); // full-pel luma DV
                        // ---- decoder reference derivation (transcribed) ----
                        let mvq4_row = dv_row << (1 - ss_y as i32);
                        let mvq4_col = dv_col << (1 - ss_x as i32);
                        let ref_off = (block_off as i64
                            + (mvq4_row >> 4) as i64 * stride as i64
                            + (mvq4_col >> 4) as i64)
                            as usize;
                        let (subpel_x, subpel_y) = (mvq4_col & 15, mvq4_row & 15);
                        let max = (1i32 << bd) - 1;
                        let clip = |v: i32| v.clamp(0, max) as u16;
                        let mut want = vec![0u16; cw * ch];
                        for r in 0..ch {
                            let so = ref_off + r * stride;
                            for c in 0..cw {
                                let a00 = recon[so + c] as i32;
                                want[r * cw + c] = match (subpel_x != 0, subpel_y != 0) {
                                    (false, false) => a00 as u16,
                                    (true, false) => {
                                        clip((a00 + recon[so + c + 1] as i32 + 1) >> 1)
                                    }
                                    (false, true) => {
                                        clip((a00 + recon[so + c + stride] as i32 + 1) >> 1)
                                    }
                                    (true, true) => clip(
                                        (a00 + recon[so + c + 1] as i32
                                            + recon[so + c + stride] as i32
                                            + recon[so + c + stride + 1] as i32
                                            + 2)
                                            >> 2,
                                    ),
                                };
                            }
                        }
                        // ---- encoder predictor ----
                        let mut got = vec![0u16; cw * ch];
                        intrabc_predict_chroma(
                            &recon, block_off, stride, dv_row, dv_col, ss_x, ss_y, &mut got, cw,
                            cw, ch, bd,
                        );
                        assert_eq!(
                            got, want,
                            "chroma predict mismatch ss=({ss_x},{ss_y}) bd={bd} dv=({dv_row},{dv_col})"
                        );
                    }
                }
            }
        }
    }

    /// The hierarchical exploration must visit every candidate exactly once
    /// (the C doc-comment's 8x8/block-4 example: 25 candidates).
    #[test]
    fn exploration_visits_all_once() {
        let mut buckets: HashMap<u32, Vec<BlockHash>> = HashMap::new();
        // 8x8 picture, block 4 -> x_end = y_end = 5 -> 25 candidates. Use a
        // constant hash so they all land in one bucket.
        let pic_hash = vec![7u32; 8 * 8];
        add_to_hash_map_by_row(&mut buckets, &pic_hash, 8, 8, 4);
        let bucket = buckets.values().next().unwrap();
        assert_eq!(buckets.len(), 1);
        assert_eq!(bucket.len(), 25);
        let mut seen = std::collections::HashSet::new();
        for b in bucket {
            assert!(seen.insert((b.x, b.y)), "duplicate visit ({},{})", b.x, b.y);
            assert!(b.x <= 4 && b.y <= 4);
        }
    }

    /// The NSTEP site config must match C's `init_motion_compensation_nstep`
    /// (mcomp.c:479): 15 stages, `radius = max(radius*1.5+0.5, radius+1)`
    /// truncated (frozen at stage 12), `tan = radius` (8 pts) for `radius <= 5`
    /// else `(int)(0.41*radius)` (12 pts), and the exact 13-site MV table.
    #[test]
    fn nstep_config_matches_c() {
        // Radii regenerated by the C growth rule.
        let mut radii = [0i32; 15];
        let mut r = 1i32;
        for (i, slot) in radii.iter_mut().enumerate() {
            *slot = r;
            if i < 12 {
                r = ((r as f64 * 1.5 + 0.5) as i32).max(r + 1);
            }
        }
        assert_eq!(radii, NSTEP_RADII);
        assert_eq!(NSTEP_RADII, [1, 2, 3, 5, 8, 12, 18, 27, 41, 62, 93, 140, 210, 210, 210]);

        // Stage 0 (R=1 <= 5): 8 pts, tan = 1; the 4 axis + 4 diagonal sites.
        let (s0, p0) = nstep_stage_sites(1);
        assert_eq!(p0, 8);
        assert_eq!(&s0[..9], &[(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]);
        // Stage 4 (R=8 > 5): 12 pts, tan = (int)(0.41*8) = 3.
        let (s4, p4) = nstep_stage_sites(8);
        assert_eq!(p4, 12);
        assert_eq!(s4[1], (-8, 0));
        assert_eq!(s4[5], (-8, -3));
        assert_eq!(s4[7], (-3, 8));
        assert_eq!(s4[12], (-3, -8));
        // tan truncation spot-checks.
        assert_eq!(nstep_stage_sites(12).0[5], (-12, -4));
        assert_eq!(nstep_stage_sites(210).0[5], (-210, -86));
    }

    /// `av1_init_search_range` (mcomp.c:263) via the frame-level derivation:
    /// `size=max(16,max(w,h)); sr=0; while ((size<<sr) < 1023) sr++;
    /// min(sr,9)`.
    fn init_search_range(size: i32) -> usize {
        let size = size.max(16);
        let mut sr = 0usize;
        while (size << sr) < 1023 {
            sr += 1;
        }
        sr.min(MAX_MVSEARCH_STEPS - 2)
    }

    #[test]
    fn mv_step_param_matches_c() {
        assert_eq!(init_search_range(64), 4);
        assert_eq!(init_search_range(128), 3);
        assert_eq!(init_search_range(256), 2);
        assert_eq!(init_search_range(512), 1);
        assert_eq!(init_search_range(1280), 0);
    }

    /// The full-pel diamond must find an exact repeat located near `dv_ref`
    /// (SAD 0), returning the full-pel MV of the repeat with variance 0.
    #[test]
    fn diamond_finds_exact_repeat() {
        // A 64-wide recon with an 8x8 block at (0,0) copied to (20, 4)
        // (row=4, col=20 in pixels — i.e. the block at pixel (row 4, col 20)).
        let stride = 128usize;
        // SRC has the block at `cur`; RECON has the identical block ONLY at
        // `matpos` (its position at `cur` differs), so SAD is 0 only at the
        // repeat MV, forcing the diamond to move there.
        let mut src = vec![0u16; stride * 64];
        let mut recon = vec![0u16; stride * 64];
        let mut s = 0x9e3779b9u32;
        let cur = (20usize, 20usize);
        let matpos = (4usize, 20usize);
        let mut blk = [0u16; 64];
        for v in blk.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((s >> 16) & 0xff) as u16;
        }
        for r in 0..8 {
            for c in 0..8 {
                src[(cur.0 + r) * stride + cur.1 + c] = blk[r * 8 + c];
                recon[(matpos.0 + r) * stride + matpos.1 + c] = blk[r * 8 + c];
                // recon@cur is a DIFFERENT pattern (nonzero) so SAD@start != 0.
                recon[(cur.0 + r) * stride + cur.1 + c] = 255 - blk[r * 8 + c];
            }
        }
        let dv = DvCosts {
            joint_mv: [0; 4],
            dv_costs: [vec![0; MV_VALS], vec![0; MV_VALS]],
        };
        let off = cur.0 * stride + cur.1;
        let sr = FullPelSearch {
            src: &src,
            src_off: off,
            refb: &recon,
            ref_off: off,
            src_stride: stride,
            ref_stride: stride,
            w: 8,
            h: 8,
            limits: FullMvLimits {
                col_min: -20,
                col_max: 20,
                row_min: -20,
                row_max: 20,
            },
            dv: &dv,
            ref_row_sub: 0,
            ref_col_sub: 0,
            full_ref_row: 0,
            full_ref_col: 0,
            error_per_bit: 1,
            sad_per_bit: 1,
        };
        // start at (0,0); the exact match is at fullmv (row=-16, col=0).
        let (sme, r, c) = full_pixel_diamond(&sr, 0, 0, 4);
        assert_eq!((r, c), (-16, 0), "diamond must land on the exact-repeat MV");
        // variance at the match is 0 (+ mv cost, which is 0 with zero tables).
        assert_eq!(sme, 0);
    }

    /// Identical source blocks must produce identical (hash1, hash2) through
    /// both the table build and the query path (the match invariant the
    /// search relies on).
    #[test]
    fn build_and_query_hashes_agree() {
        // 32x32 source with an exact 8x8 repeat at (0,0) and (16, 8).
        let w = 32usize;
        let mut src = vec![0u16; w * w];
        for y in 0..8 {
            for x in 0..8 {
                let v = (y * 8 + x) as u16;
                src[y * w + x] = v;
                src[(y + 8) * w + (x + 16)] = v;
            }
        }
        let table = build_intrabc_hash_table(&src, 0, w, w, w, false, 64);
        let (h1a, h2a) = get_block_hash_value(&table, &src, 0, w, 8, false);
        let (h1b, h2b) = get_block_hash_value(&table, &src, 8 * w + 16, w, 8, false);
        assert_eq!((h1a, h2a), (h1b, h2b), "identical blocks must hash equal");
        // And the table must contain candidates for that bucket, including
        // both repeat positions.
        let bucket = table.buckets.get(&h1a).expect("bucket exists");
        let mut found = [false; 2];
        for b in bucket.iter().filter(|b| b.hash_value2 == h2a) {
            if (b.x, b.y) == (0, 0) {
                found[0] = true;
            }
            if (b.x, b.y) == (16, 8) {
                found[1] = true;
            }
        }
        assert!(found[0] && found[1], "both repeat positions in the table");
    }
}

// ---------------------------------------------------------------------------
// chunk 3b/3c: the intrabc leaf search (rd_pick_intrabc_mode_sb, rdopt.c:3427)
// ---------------------------------------------------------------------------

use crate::encode_intra::TrellisOptType;
use aom_dsp::entropy::dv_ref::{DvNbr, DvTileBounds, find_dv_ref_mvs, find_ref_dv, is_dv_valid};

/// `default_txfm_partition_cdf` (entropymode.c) — the var-tx split-flag CDF
/// defaults (relocated from the decoder's TileKf per its own FORK NOTE; the
/// encoder needs the same table for the pack-side var-tx symbols + the
/// frame-init `txfm_partition_cost` fill, rd.c:110).
pub const DEFAULT_TXFM_PARTITION_CDF: [[u16; 3]; 21] = [
    [4187, 0, 0],
    [8922, 0, 0],
    [11921, 0, 0],
    [8453, 0, 0],
    [14572, 0, 0],
    [20635, 0, 0],
    [13977, 0, 0],
    [21881, 0, 0],
    [21763, 0, 0],
    [5589, 0, 0],
    [12764, 0, 0],
    [21487, 0, 0],
    [6219, 0, 0],
    [13460, 0, 0],
    [18544, 0, 0],
    [4753, 0, 0],
    [11222, 0, 0],
    [18368, 0, 0],
    [4603, 0, 0],
    [10367, 0, 0],
    [16680, 0, 0],
];

/// The `txfm_partition_cost` slice of `av1_fill_mode_rates` (rd.c:108-111):
/// per-context 2-symbol costs from the frame-init var-tx split CDF.
pub fn fill_txfm_partition_costs(cdf: &[[u16; 3]; 21]) -> [[i32; 2]; 21] {
    let mut out = [[0i32; 2]; 21];
    for (o, row) in out.iter_mut().zip(cdf.iter()) {
        aom_dsp::txb::cost_tokens_from_cdf(o, row, None);
    }
    out
}

/// Full-pel luma intrabc prediction: copy `w x h` from the RECON plane at the
/// DV offset (regions are disjoint by DV validity — the source is fully
/// reconstructed before this block).
pub fn intrabc_predict_luma(
    recon: &[u16],
    block_off: usize,
    stride: usize,
    dv_row_px: i32,
    dv_col_px: i32,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    let src_off =
        (block_off as i64 + i64::from(dv_row_px) * stride as i64 + i64::from(dv_col_px)) as usize;
    for r in 0..h {
        let s = src_off + r * stride;
        dst[r * dst_stride..r * dst_stride + w].copy_from_slice(&recon[s..s + w]);
    }
}

/// Chroma intrabc prediction (the decoder's `intrabc_chroma_predict` mirror):
/// the subsampled DV lands at full- or half-pel per axis; half-pel is the
/// 2-tap {64,64} average (bit-identical closed form of the intrabc bilinear
/// convolve at FILTER_BITS=7).
#[allow(clippy::too_many_arguments)]
pub fn intrabc_predict_chroma(
    recon: &[u16],
    block_off: usize,
    stride: usize,
    dv_row: i32, // 1/8 luma pel
    dv_col: i32,
    ss_x: usize,
    ss_y: usize,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    bd: i32,
) {
    // av1_dv_offset / dec convention: chroma subpel = (dv >> ss) & 7 with the
    // integer part floor-divided. dv is a multiple of 8 (full luma pel), so
    // the chroma position is dv/8 >> ss with a half-pel remainder when the
    // luma DV is odd in the subsampled axis.
    let px_row = dv_row >> 3; // luma px
    let px_col = dv_col >> 3;
    let c_row = px_row >> ss_y;
    let c_col = px_col >> ss_x;
    let subpel_y = if ss_y == 1 { (px_row & 1) * 8 } else { 0 };
    let subpel_x = if ss_x == 1 { (px_col & 1) * 8 } else { 0 };
    let src_off = (block_off as i64 + i64::from(c_row) * stride as i64 + i64::from(c_col)) as usize;
    let max = (1i32 << bd) - 1;
    let clip = |v: i32| v.clamp(0, max) as u16;
    match (subpel_x != 0, subpel_y != 0) {
        (false, false) => {
            for r in 0..h {
                let s = src_off + r * stride;
                dst[r * dst_stride..r * dst_stride + w].copy_from_slice(&recon[s..s + w]);
            }
        }
        (true, false) => {
            for r in 0..h {
                let s = src_off + r * stride;
                for c in 0..w {
                    let a = recon[s + c] as i32;
                    let b = recon[s + c + 1] as i32;
                    dst[r * dst_stride + c] = clip((a + b + 1) >> 1);
                }
            }
        }
        (false, true) => {
            for r in 0..h {
                let s = src_off + r * stride;
                for c in 0..w {
                    let a = recon[s + c] as i32;
                    let b = recon[s + c + stride] as i32;
                    dst[r * dst_stride + c] = clip((a + b + 1) >> 1);
                }
            }
        }
        (true, true) => {
            for r in 0..h {
                let s = src_off + r * stride;
                for c in 0..w {
                    let a00 = recon[s + c] as i32;
                    let a01 = recon[s + c + 1] as i32;
                    let a10 = recon[s + c + stride] as i32;
                    let a11 = recon[s + c + stride + 1] as i32;
                    dst[r * dst_stride + c] = clip((a00 + a01 + a10 + a11 + 2) >> 2);
                }
            }
        }
    }
}

/// One committed block's DV projection for the search-side mi grid (the
/// `DvNbr` source; also carries `skip_txfm` for `av1_get_skip_txfm_context`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DvCell {
    pub bsize: u8,
    pub mode: u8,
    pub use_intrabc: bool,
    pub skip_txfm: bool,
    /// The block's DV (1/8 pel), meaningful when `use_intrabc`.
    pub dv_row: i16,
    pub dv_col: i16,
}

impl DvCell {
    pub fn to_nbr(self) -> DvNbr {
        DvNbr {
            bsize: self.bsize as usize,
            // KEY frame: intra/intrabc candidates carry INTRA_FRAME/NONE.
            ref_frame0: 0,  // INTRA_FRAME
            ref_frame1: -1, // NONE_FRAME
            use_intrabc: self.use_intrabc,
            mode: self.mode as i32,
            mv0_row: i32::from(self.dv_row),
            mv0_col: i32::from(self.dv_col),
            mv1_row: 0,
            mv1_col: 0,
        }
    }
}

/// `FULLPEL_MV` limits (mcomp.h `FullMvLimits`).
#[derive(Clone, Copy, Debug)]
pub struct FullMvLimits {
    pub col_min: i32,
    pub col_max: i32,
    pub row_min: i32,
    pub row_max: i32,
}

/// `av1_set_mv_search_range` (mcomp.c:233): intersect the caller's limits
/// with the ±MAX_FULL_PEL_VAL window around the (subpel) ref MV.
pub fn set_mv_search_range(lim: &mut FullMvLimits, ref_row: i32, ref_col: i32) {
    // VERIFIED vs reference/libaom (mcomp_structs.h:19,22): MAX_MVSEARCH_STEPS
    // = 11, so MAX_FULL_PEL_VAL = (1 << (MAX_MVSEARCH_STEPS - 1)) - 1 = 1023.
    const MAX_FULL_PEL_VAL: i32 = (1 << 10) - 1;
    // mv.h: MV_LOW = -(1 << 14), MV_UPP = (1 << 14).
    const MV_LOW: i32 = -(1 << 14);
    const MV_UPP: i32 = 1 << 14;
    let mut col_min = ((ref_col + 7) >> 3) - MAX_FULL_PEL_VAL;
    let mut row_min = ((ref_row + 7) >> 3) - MAX_FULL_PEL_VAL;
    let mut col_max = (ref_col >> 3) + MAX_FULL_PEL_VAL;
    let mut row_max = (ref_row >> 3) + MAX_FULL_PEL_VAL;
    col_min = col_min.max((MV_LOW >> 3) + 1);
    row_min = row_min.max((MV_LOW >> 3) + 1);
    col_max = col_max.min((MV_UPP >> 3) - 1);
    row_max = row_max.min((MV_UPP >> 3) - 1);
    lim.col_min = lim.col_min.max(col_min);
    lim.col_max = lim.col_max.min(col_max);
    lim.row_min = lim.row_min.max(row_min);
    lim.row_max = lim.row_max.min(row_max);
    lim.col_max = lim.col_min.max(lim.col_max);
    lim.row_max = lim.row_min.max(lim.row_max);
}

// ---------------------------------------------------------------------------
// Full-pel motion search: NSTEP diamond + mesh (mcomp.c). At intrabc speed 0
// the search method is NSTEP and the pixel search ALWAYS runs after the hash
// (rdopt.c:3570 `intrabc_search_level == 0`).
// ---------------------------------------------------------------------------

/// `MAX_MVSEARCH_STEPS` (mcomp_structs.h:19).
const MAX_MVSEARCH_STEPS: usize = 11;
/// The NSTEP config's per-stage radius (`init_motion_compensation_nstep`,
/// mcomp.c:479, level 0 — 15 stages, `radius = max(radius*1.5+0.5, radius+1)`
/// truncated, frozen at stage 12).
const NSTEP_RADII: [i32; 15] = [1, 2, 3, 5, 8, 12, 18, 27, 41, 62, 93, 140, 210, 210, 210];

/// One NSTEP stage's search sites (`search_site_mvs[13]`, mcomp.c:493): center
/// + 4 axis + up to 8 tangents. `radius <= 5` → 8 pts (tan = radius), else 12
/// pts (`tan = (int)(0.41*radius)`). Returns `(sites[(row,col); 13],
/// num_search_pts)`.
fn nstep_stage_sites(radius: i32) -> ([(i32, i32); 13], usize) {
    let (tan, pts) = if radius <= 5 {
        (radius, 8usize)
    } else {
        (((0.41f64 * radius as f64) as i32).max(1), 12usize)
    };
    let s = [
        (0, 0),
        (-radius, 0),
        (radius, 0),
        (0, -radius),
        (0, radius),
        (-radius, -tan),
        (radius, tan),
        (-tan, radius),
        (tan, -radius),
        (-radius, tan),
        (radius, -tan),
        (tan, radius),
        (-tan, -radius),
    ];
    (s, pts)
}

/// The frame-invariant inputs of one full-pel search direction (`src`, `ref =
/// recon`, the DV cost tables, the mv limits). Offsets are block-origin; a
/// full-pel MV `(row, col)` addresses `recon[ref_off + row*stride + col]`.
struct FullPelSearch<'a> {
    src: &'a [u16],
    src_off: usize,
    refb: &'a [u16],
    ref_off: usize,
    /// Source-plane stride (current frame). Equal to `ref_stride` for intrabc
    /// (src == ref == current frame); distinct for inter (ref = a reference
    /// frame recon of possibly different stride).
    src_stride: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
    limits: FullMvLimits,
    dv: &'a DvCosts,
    /// `dv_ref` in subpel (1/8-pel, `& 7 == 0`) and full-pel.
    ref_row_sub: i32,
    ref_col_sub: i32,
    full_ref_row: i32,
    full_ref_col: i32,
    error_per_bit: i32,
    sad_per_bit: i32,
}

impl FullPelSearch<'_> {
    #[inline]
    fn in_range(&self, r: i32, c: i32) -> bool {
        c >= self.limits.col_min
            && c <= self.limits.col_max
            && r >= self.limits.row_min
            && r <= self.limits.row_max
    }
    #[inline]
    fn clamp(&self, r: i32, c: i32) -> (i32, i32) {
        (
            r.clamp(self.limits.row_min, self.limits.row_max),
            c.clamp(self.limits.col_min, self.limits.col_max),
        )
    }
    #[inline]
    fn ref_at(&self, r: i32, c: i32) -> usize {
        (self.ref_off as i64 + i64::from(r) * self.ref_stride as i64 + i64::from(c)) as usize
    }
    #[inline]
    fn sad(&self, r: i32, c: i32) -> u32 {
        sad_wxh(
            self.src,
            self.src_off,
            self.src_stride,
            self.refb,
            self.ref_at(r, c),
            self.ref_stride,
            self.w,
            self.h,
        )
    }
    #[inline]
    fn var(&self, r: i32, c: i32) -> u32 {
        variance_wxh(
            self.src,
            self.src_off,
            self.src_stride,
            self.refb,
            self.ref_at(r, c),
            self.ref_stride,
            self.w,
            self.h,
        )
    }
    /// `mvsad_err_cost_` — the SAD-metric MV cost (`sad_per_bit`).
    #[inline]
    fn sad_mv_cost(&self, r: i32, c: i32) -> u32 {
        mvsad_err_cost(r - self.full_ref_row, c - self.full_ref_col, self.dv, self.sad_per_bit) as u32
    }
    /// `get_mvpred_var_cost` (mcomp.c:644) = variance + `mv_err_cost_`
    /// (`error_per_bit`). The subpel diff is `fullmv*8 - dv_ref_subpel`.
    #[inline]
    fn var_cost(&self, r: i32, c: i32) -> i64 {
        i64::from(self.var(r, c))
            + i64::from(mv_err_cost(
                r * 8 - self.ref_row_sub,
                c * 8 - self.ref_col_sub,
                self.dv,
                self.error_per_bit,
            ))
    }
}

/// `diamond_search_sad` (mcomp.c:1311, single-ref). Walks the NSTEP stages
/// coarse→fine from `search_step`, SAD-metric; returns `(best_sad, best_row,
/// best_col, num00)`. `start_sad` = `mvsad_err_cost(start) + sad(start)`.
fn diamond_search_sad(
    s: &FullPelSearch,
    start_row: i32,
    start_col: i32,
    start_sad: u32,
    search_step: usize,
) -> (u32, i32, i32, i32) {
    let tot_steps = NSTEP_RADII.len() - search_step;
    let mut best_row = start_row;
    let mut best_col = start_col;
    let mut bestsad = start_sad;
    let mut is_off_center = false;
    let mut num_center_steps = 0i32;

    let mut step = tot_steps as i32 - 1;
    while step >= 0 {
        let radius = NSTEP_RADII[step as usize];
        let (site, num_searches) = nstep_stage_sites(radius);
        let mut best_site = 0usize;

        // Trap: whole ±radius cross inside limits (mcomp.c:1400-1405) — then the
        // per-site range check is skipped (all sites are in range).
        let all_in = best_row + site[1].0 >= s.limits.row_min
            && best_row + site[2].0 <= s.limits.row_max
            && best_col + site[3].1 >= s.limits.col_min
            && best_col + site[4].1 <= s.limits.col_max;

        for idx in 1..=num_searches {
            let (dr, dc) = site[idx];
            let (this_r, this_c) = (best_row + dr, best_col + dc);
            if all_in || s.in_range(this_r, this_c) {
                let sad = s.sad(this_r, this_c);
                // update_best_site (mcomp.c:1295): two-stage `<` test.
                if sad < bestsad {
                    let thissad = sad + s.sad_mv_cost(this_r, this_c);
                    if thissad < bestsad {
                        bestsad = thissad;
                        best_site = idx;
                    }
                }
            }
        }

        // UPDATE_SEARCH_STEP (mcomp.c:1315).
        if best_site != 0 {
            best_row += site[best_site].0;
            best_col += site[best_site].1;
            is_off_center = true;
        }
        if !is_off_center {
            num_center_steps += 1;
        }
        if best_site == 0 && step > 2 {
            let mut next = NSTEP_RADII[(step - 1) as usize];
            while next == NSTEP_RADII[step as usize] && step > 2 {
                num_center_steps += 1;
                step -= 1;
                next = NSTEP_RADII[(step - 1) as usize];
            }
        }
        step -= 1;
    }
    (bestsad, best_row, best_col, num_center_steps)
}

/// `full_pixel_diamond` (mcomp.c:1481, single-ref, no cost_list / second_best).
/// Returns `(bestsme_var_cost, best_row, best_col)`.
fn full_pixel_diamond(s: &FullPelSearch, start_row: i32, start_col: i32, step_param: usize) -> (i64, i32, i32) {
    let (start_row, start_col) = s.clamp(start_row, start_col);
    let start_sad = s.sad_mv_cost(start_row, start_col) + s.sad(start_row, start_col);

    let (_sad0, mut best_row, mut best_col, n0) =
        diamond_search_sad(s, start_row, start_col, start_sad, step_param);
    let mut bestsme = s.var_cost(best_row, best_col);

    let further_steps = NSTEP_RADII.len() as i32 - 1 - step_param as i32;
    let mut n = n0;
    while n < further_steps {
        n += 1;
        let (_sad, tr, tc, num00) =
            diamond_search_sad(s, start_row, start_col, start_sad, step_param + n as usize);
        let thissme = s.var_cost(tr, tc);
        if thissme < bestsme {
            bestsme = thissme;
            best_row = tr;
            best_col = tc;
        }
        if num00 != 0 {
            n += num00;
        }
    }
    (bestsme, best_row, best_col)
}

/// `exhaustive_mesh_search` (mcomp.c:1542) for `step == interval`; SAD-metric.
/// Returns `(best_sad, best_row, best_col)`.
fn exhaustive_mesh_search(
    s: &FullPelSearch,
    start_row: i32,
    start_col: i32,
    range: i32,
    step: i32,
) -> (u32, i32, i32) {
    let col_step = if step > 1 { step } else { 4 };
    let (start_row, start_col) = s.clamp(start_row, start_col);
    let mut best_row = start_row;
    let mut best_col = start_col;
    let mut best_sad = s.sad(start_row, start_col) + s.sad_mv_cost(start_row, start_col);

    let sr = (-range).max(s.limits.row_min - start_row);
    let sc = (-range).max(s.limits.col_min - start_col);
    let er = range.min(s.limits.row_max - start_row);
    let ec = range.min(s.limits.col_max - start_col);

    let mut r = sr;
    while r <= er {
        let mut c = sc;
        while c <= ec {
            let hi = if step > 1 { 1 } else { (ec - c).min(4) };
            // step==1: process up to 4 columns; step>1: single position.
            let n = if step > 1 { 1 } else { hi };
            for i in 0..n {
                let (mr, mc) = (start_row + r, start_col + c + i);
                let sad = s.sad(mr, mc);
                // update_mvs_and_sad (mcomp.c:846): if sad >= best return; else
                // add mvsad cost; keep strict-min.
                if sad < best_sad {
                    let cost = sad + s.sad_mv_cost(mr, mc);
                    if cost < best_sad {
                        best_sad = cost;
                        best_row = mr;
                        best_col = mc;
                    }
                }
            }
            c += col_step;
        }
        r += step;
    }
    (best_sad, best_row, best_col)
}

/// `full_pixel_exhaustive` (mcomp.c:1615) for the intrabc mesh pattern
/// (`{range:256, interval:1}` at speed 0, `fine_search_interval == 0`): a
/// single range-256 interval-1 pass; returns the variance cost + MV.
fn full_pixel_exhaustive(s: &FullPelSearch, start_row: i32, start_col: i32) -> (i64, i32, i32) {
    const K_MIN_RANGE: i32 = 7;
    const K_MAX_RANGE: i32 = 256;
    const K_MIN_INTERVAL: i32 = 1;
    let interval = 1i32;
    let mut range = 256i32;
    let mut best_row = start_row;
    let mut best_col = start_col;
    if range < K_MIN_RANGE || range > K_MAX_RANGE || interval < K_MIN_INTERVAL || interval > range {
        return (i64::MAX, start_row, start_col);
    }
    let baseline_interval_divisor = range / interval;
    range = range.max((5 * best_row.abs().max(best_col.abs())) / 4);
    range = range.min(K_MAX_RANGE);
    let interval = interval.max(range / baseline_interval_divisor);
    // fine_search_interval == 0 → no interval clamp.
    let (best_sad, r, c) = exhaustive_mesh_search(s, best_row, best_col, range, interval);
    best_row = r;
    best_col = c;
    // interval == 1 → the progressive-search loop is skipped (single pass).
    let bestsme = if best_sad < u32::MAX {
        s.var_cost(best_row, best_col)
    } else {
        i64::MAX
    };
    (bestsme, best_row, best_col)
}

/// `GET_MV_RAWPEL` / `get_fullmv_from_mv` (mv.h:28,79): round a 1/8-pel subpel
/// component to the nearest full-pel — `((x) + 3 + (x >= 0)) >> 3` (arithmetic
/// shift). Inter ref MVs carry subpel bits (unlike integer intrabc DVs), so the
/// full-pel search start + the `mvsad_err_cost` full-pel ref must ROUND, not
/// truncate.
#[inline]
pub fn get_fullmv_from_mv(subpel: i32) -> i32 {
    (subpel + 3 + (subpel >= 0) as i32) >> 3
}

/// Inter single-ref full-pixel motion search — the intrabc NSTEP diamond
/// retargeted to a reference frame: separate src/ref plane strides and the
/// subpel-precision inter MV cost tables ([`fill_nmv_costs`]) in place of the
/// integer-pel DV tables. Mirrors `av1_single_motion_search` →
/// `av1_full_pixel_search` (mcomp.c:1768) for the SIMPLE_TRANSLATION speed-0
/// path with the mesh disabled: `start_mv = get_fullmv_from_mv(ref_mv)`, run the
/// NSTEP diamond, return its variance cost (`get_mvpred_var_cost` = variance +
/// `mv_err_cost_`).
///
/// The follow-on exhaustive mesh (`av1_full_pixel_search` runs it when
/// `var > force_mesh_thresh >> (10 - log2(w*h))`) is NOT run here — it needs the
/// inter mesh pattern (`mv_sf->mesh_patterns`), distinct from the intrabc one
/// the port's [`full_pixel_exhaustive`] hard-codes; a follow-up. `cost_list` /
/// `second_best_mv` (subpel-handoff outputs) are likewise not produced yet.
///
/// Returns `(var_cost, best_row, best_col)` in full-pel units.
#[allow(clippy::too_many_arguments)]
pub fn full_pixel_search_inter(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
    ref_row_sub: i32,
    ref_col_sub: i32,
    dv: &DvCosts,
    error_per_bit: i32,
    sad_per_bit: i32,
    limits: FullMvLimits,
    step_param: usize,
) -> (i64, i32, i32) {
    let full_ref_row = get_fullmv_from_mv(ref_row_sub);
    let full_ref_col = get_fullmv_from_mv(ref_col_sub);
    let s = FullPelSearch {
        src,
        src_off,
        refb,
        ref_off,
        src_stride,
        ref_stride,
        w,
        h,
        limits,
        dv,
        ref_row_sub,
        ref_col_sub,
        full_ref_row,
        full_ref_col,
        error_per_bit,
        sad_per_bit,
    };
    full_pixel_diamond(&s, full_ref_row, full_ref_col, step_param)
}

// ---------------------------------------------------------------------------
// predict_skip_txfm (tx_search.c:183) — the early-skip check that fires for a
// (near-)zero residual, forcing skip_txfm=1 (bypassing the coeff arm).
// ---------------------------------------------------------------------------

/// `skip_pred_threshold[3][BLOCK_SIZES_ALL]` (tx_search.c:50).
const SKIP_PRED_THRESHOLD: [[u32; 22]; 3] = [
    [
        64, 64, 64, 70, 60, 60, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 64, 64, 70, 70, 68, 68,
    ],
    [
        88, 88, 88, 86, 87, 87, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 88, 88, 86, 86, 68, 68,
    ],
    [
        90, 93, 93, 90, 93, 93, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 90, 90, 90, 90, 74, 74,
    ],
];
/// `max_predict_sf_tx_size[BLOCK_SIZES_ALL]` (tx_search.c:69): TX_SIZE indices.
const MAX_PREDICT_SF_TX_SIZE: [usize; 22] = [
    0, 5, 6, 1, 7, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 13, 14, 1, 1, 2, 2,
];

/// `predict_skip_txfm` (tx_search.c:183) at `skip_txfm_level == 1` (DEFAULT_EVAL
/// speed 0). `residual` is the block's src−pred (i16, row-major, stride `bw`).
/// Returns true iff the block is predicted to skip (all coeffs below the
/// per-bd threshold). `sse` is the residual SSE (`av1_pixel_diff_dist`).
fn predict_skip_txfm(
    residual: &[i16],
    bw: usize,
    bh: usize,
    bsize: usize,
    sse: i64,
    qindex: i32,
    bd: i32,
    reduced_tx_set: bool,
) -> bool {
    let dc_q = i64::from(aom_dsp::quant::av1_dc_quant_qtx(qindex, 0, bd as u8));
    let mse = sse / (bw as i64) / (bh as i64);
    let normalized_dc_q = dc_q >> 3;
    let mse_thresh = normalized_dc_q * normalized_dc_q / 8;
    if mse > mse_thresh {
        return false;
    }
    // The fwd-DCT max-coeff check (skip_txfm_level 1 continues here).
    let max_tx = MAX_PREDICT_SF_TX_SIZE[bsize];
    let tx_w = crate::tx_search::TXS_W[max_tx];
    let tx_h = crate::tx_search::TXS_H[max_tx];
    let bd_idx = if bd == 8 { 0 } else if bd == 10 { 1 } else { 2 };
    let max_qcoef_thresh = u64::from(SKIP_PRED_THRESHOLD[bd_idx][bsize]);
    let ac_q = i64::from(aom_dsp::quant::av1_ac_quant_qtx(qindex, 0, bd as u8));
    let dc_thresh = max_qcoef_thresh * dc_q as u64;
    let ac_thresh = max_qcoef_thresh * ac_q as u64;
    let n_coeff = tx_w * tx_h;
    let _ = reduced_tx_set;
    let mut coeff = vec![0i32; n_coeff];
    let mut sub = vec![0i16; n_coeff];
    let mut row = 0;
    while row < bh {
        let mut col = 0;
        while col < bw {
            // Extract the max_tx sub-block from the residual (stride bw).
            for r in 0..tx_h {
                for c in 0..tx_w {
                    sub[r * tx_w + c] = residual[(row + r) * bw + (col + c)];
                }
            }
            aom_dsp::transform::txfm2d::av1_fwd_txfm2d(&sub, &mut coeff, tx_w, 0, max_tx);
            let dc_coef = (coeff[0].unsigned_abs() as u64) << 7;
            if dc_coef >= dc_thresh {
                return false;
            }
            for &v in coeff.iter().skip(1) {
                let ac_coef = (v.unsigned_abs() as u64) << 7;
                if ac_coef >= ac_thresh {
                    return false;
                }
            }
            col += tx_w;
        }
        row += tx_h;
    }
    true
}

// ---------------------------------------------------------------------------
// rd_pick_intrabc_mode_sb (rdopt.c:3427) leaf search.
// ---------------------------------------------------------------------------

/// Everything the leaf intrabc search needs. Construct in
/// `partition_pick::leaf_pick_sb_modes` / thread through `rd_pick.rs` step 6
/// (the currently-documented "envelope-excluded no-op" site).
pub struct IntrabcLeafArgs<'a> {
    // Geometry
    pub sb_size: usize,
    pub bsize: usize,
    pub mi_row: i32,
    pub mi_col: i32,
    pub mi_rows: i32,
    pub mi_cols: i32,
    /// Tile bounds in mi units (single-tile: 0..mi_rows/cols).
    pub tile: DvTileBounds,
    pub mib_size_log2: i32,
    pub up_available: bool,
    pub left_available: bool,
    pub is_chroma_ref: bool,
    pub monochrome: bool,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bd: u8,
    /// `mbmi->partition` at this leaf (find_dv_ref_mvs's has_tr input).
    pub partition: usize,
    // Pixels: SOURCE (hash query + residual base). The RECON planes (prediction
    // source) are passed separately to `rd_pick_intrabc_mode_sb` — the caller
    // holds them `&mut` for the intra search, so they can only be reborrowed
    // `&` at the step-6 call site (not stored in these args).
    pub stride: usize,
    pub src_y: &'a [u16],
    pub src_u: &'a [u16],
    pub src_v: &'a [u16],
    /// Block origin offsets (luma; chroma via the caller's chroma_plane_offset).
    pub off_y: usize,
    pub off_uv: usize,
    // Search state
    pub hash: &'a IntrabcHashTable,
    pub dv_costs: &'a DvCosts,
    /// Closure over the search ModeGrid's DvCell state (relative offsets from
    /// (mi_row, mi_col) exactly as `find_dv_ref_mvs` requests them).
    pub dv_grid: &'a dyn Fn(i32, i32) -> DvNbr,
    // RD inputs
    pub rdmult: i32,
    pub qindex: i32,
    pub reduced_tx_set_used: bool,
    /// `x->errorperbit = AOMMAX(rdmult >> 6, 1)` (av1_set_error_per_bit).
    pub error_per_bit: i32,
    /// `x->sadperbit` (`av1_set_sad_per_bit`) — the diamond/mesh SAD metric.
    pub sad_per_bit: i32,
    /// `cpi->mv_search_params.mv_step_param` (`av1_init_search_range(max(w,h))`)
    /// — the NSTEP diamond's coarsest stage.
    pub mv_step_param: usize,
    pub intrabc_cost: &'a [i32; 2],
    pub skip_costs: &'a [[i32; 2]; 3],
    /// `av1_get_skip_txfm_context(xd)` from the DvCell grid's above/left skip
    /// flags (0 when intrabc-off — the pre-existing invariant).
    pub skip_ctx: usize,
    pub txfm_partition_costs: &'a [[i32; 2]; 21],
    // Coeff-arm inputs (HANDOFF: DCT_DCT-only; C searches the inter tx set)
    pub rows_y: &'a aom_dsp::quant::PlaneQuantRows<'a>,
    pub rows_u: &'a aom_dsp::quant::PlaneQuantRows<'a>,
    pub rows_v: &'a aom_dsp::quant::PlaneQuantRows<'a>,
    pub coeff_costs_y: &'a aom_dsp::txb::CoeffCostSet,
    pub coeff_costs_uv: &'a aom_dsp::txb::CoeffCostSet,
    /// Inter tx-type costs (TxTypeCosts.inter — HANDOFF: derive_real_costs
    /// currently fills inter with a DUMMY zero cdf; fill from
    /// `kf.inter_ext_tx` (flatten [4][4][17]) before enabling this search).
    pub tx_type_costs: &'a aom_dsp::txb::TxTypeCosts,
    pub sharpness: i32,
    pub enable_optimize_b: TrellisOptType,
    pub qm_levels: Option<[usize; 3]>,
    /// Entropy neighbour ctx slices at this block (per plane, like the
    /// intra leaf) for the coeff-arm txb_skip/dc_sign contexts.
    pub above_ctx: [&'a [i8]; 3],
    pub left_ctx: [&'a [i8]; 3],
}

/// The intrabc winner (mirrors `best_mbmi` + rd_stats of rdopt.c:3494-3646).
#[derive(Clone, Debug)]
pub struct IntrabcBest {
    /// The winning DV (1/8 pel, full-pel multiples of 8).
    pub dv_row: i32,
    pub dv_col: i32,
    /// The ref DV the mode-rate was computed against — the PACK must write
    /// `diff = dv - dv_ref` (bitstream.c write_intrabc_info uses the STORED
    /// search-time ref stack, mbmi_ext_frame->ref_mv_stack[0].this_mv).
    pub dv_ref_row: i32,
    pub dv_ref_col: i32,
    /// The winning arm: true = skip_txfm (no residual coded).
    pub skip_txfm: bool,
    pub rate: i32,
    pub dist: i64,
    pub rdcost: i64,
    /// COEFF arm only (`!skip_txfm`): `mbmi->inter_tx_size[16]` from the var-tx
    /// quadtree ([`crate::var_tx::VarTxResult::inter_tx_size`]). `[0; 16]` on
    /// the skip arm (dead — no tx-size syntax is written for a skipped inter
    /// block, bitstream.c:1540 `!(is_inter_tx && skip_txfm)`).
    pub inter_tx_size: [usize; 16],
    /// COEFF arm only: `mbmi->tx_size` (the var-tx root size, `inter_tx_size[0]`).
    pub tx_size: usize,
    /// COEFF arm only: the block-local luma `xd->tx_type_map` (stride
    /// `mi_size_wide[bsize]`) the var-tx search wrote — C saves it per winning
    /// DV (`av1_copy_array(best_tx_type_map, xd->tx_type_map, ..)`,
    /// rdopt.c:3619) and restores it after the loop (:3624). Chroma reads it
    /// too, at the subsampling-scaled-back luma position (`av1_get_tx_type`,
    /// blockd.h:1296-1301).
    pub tx_type_map: Vec<u8>,
}

/// `rd_pick_intrabc_mode_sb` (rdopt.c:3427): dv-ref derivation + the
/// two-direction mv-limit loop + hash search (`av1_intrabc_hash_search`) +
/// the NSTEP diamond + mesh (`av1_full_pixel_search`, which ALWAYS runs at
/// intrabc_search_level 0) + DV validity + `av1_txfm_search` RD.
///
/// **Coeff-arm scope (honest):** the block is coded INTER (var-tx quadtree),
/// which the port does NOT yet have. So this offers an intrabc candidate ONLY
/// when C's `predict_skip_txfm` fires (the block quantizes to all-zero — the
/// exact-repeat / near-zero-residual case), where `av1_txfm_search` forces
/// `skip_txfm = 1` and BYPASSES the coeff arm entirely (tx_search.c:3596 →
/// `set_skip_txfm`). For those blocks the skip-arm RD (rate = mode+mv+skip1,
/// dist = sse) is byte-exact. When `predict_skip_txfm` does NOT fire the
/// coeff arm would decide, and this returns no candidate (conservative — see
/// PARITY C3): the frame then keeps the intra winner, which byte-matches only
/// on content where every winning intrabc block is a (near-)perfect match.
///
/// `recon_{y,u,v}` are the reconstruction planes (`xd->cur_buf`, the intrabc
/// prediction source); passed separately (the caller holds them `&mut`).
pub fn rd_pick_intrabc_mode_sb(
    a: &IntrabcLeafArgs,
    recon_y: &[u16],
    recon_u: &[u16],
    recon_v: &[u16],
    best_rd_in: i64,
) -> Option<IntrabcBest> {
    let bw = crate::tx_search::BLK_W_B[a.bsize];
    let bh = crate::tx_search::BLK_H_B[a.bsize];
    // Only the HASH is square-gated (`av1_intrabc_hash_search` returns INT_MAX
    // when block_width != block_height, mcomp.c:1918); the full-pel pixel
    // search runs for every bsize. (Non-square intrabc IS common — real screen
    // content codes many 4x8 / 8x4 / 16x4 intrabc blocks via the diamond.)
    let hash_eligible = bw == bh;

    // --- dv_ref (rdopt.c:3453-3478) ---
    // find_dv_ref_mvs returns (nearest_row, nearest_col, near_row, near_col)
    // — VERIFIED against dv_ref.rs:579's doc + the dv_ref_diff.rs destructure.
    let (mut nearest_r, mut nearest_c, mut near_r, mut near_c) = find_dv_ref_mvs(
        a.mi_row,
        a.mi_col,
        a.bsize,
        a.partition,
        a.up_available,
        a.left_available,
        a.tile,
        a.mi_rows,
        a.mi_cols,
        1 << a.mib_size_log2,
        a.dv_grid,
    );
    // rdopt.c:3465-3471: INVALID_MV (row/col == -32768, mv.h:26) -> 0 before
    // the selection. (The decoder-side twin `assign_and_validate_dv` has no
    // such step — its inputs are already the raw pair; the ENCODER normalizes.)
    const INVALID_MV_ROW_COL: i32 = -32768;
    if nearest_r == INVALID_MV_ROW_COL && nearest_c == INVALID_MV_ROW_COL {
        nearest_r = 0;
        nearest_c = 0;
    }
    if near_r == INVALID_MV_ROW_COL && near_c == INVALID_MV_ROW_COL {
        near_r = 0;
        near_c = 0;
    }
    let (mut ref_r, mut ref_c) = if nearest_r == 0 && nearest_c == 0 {
        (near_r, near_c)
    } else {
        (nearest_r, nearest_c)
    };
    if ref_r == 0 && ref_c == 0 {
        let (rr, rc) = find_ref_dv(a.tile.mi_row_start, 1 << a.mib_size_log2, a.mi_row);
        ref_r = rr;
        ref_c = rc;
    }
    debug_assert_eq!(ref_r & 7, 0);
    debug_assert_eq!(ref_c & 7, 0);

    let sb_row = a.mi_row >> a.mib_size_log2;
    let sb_col = a.mi_col >> a.mib_size_log2;
    let mib = 1i32 << a.mib_size_log2;
    const MI_SIZE: i32 = 4;

    let mut best: Option<IntrabcBest> = None;
    let mut best_rd = best_rd_in;

    // IBC_MOTION_ABOVE=0, IBC_MOTION_LEFT=1; intrabc_search_level=0 (speed 0)
    // searches BOTH (rdopt.c:3510-3512).
    for dir in 0..2 {
        let mut lim = if dir == 0 {
            FullMvLimits {
                col_min: (a.tile.mi_col_start - a.mi_col) * MI_SIZE,
                col_max: (a.tile.mi_col_end - a.mi_col) * MI_SIZE - bw as i32,
                row_min: (a.tile.mi_row_start - a.mi_row) * MI_SIZE,
                row_max: (sb_row * mib - a.mi_row) * MI_SIZE - bh as i32,
            }
        } else {
            let bottom_coded_mi_edge = ((sb_row + 1) * mib).min(a.tile.mi_row_end);
            FullMvLimits {
                col_min: (a.tile.mi_col_start - a.mi_col) * MI_SIZE,
                col_max: (sb_col * mib - a.mi_col) * MI_SIZE - bw as i32,
                row_min: (a.tile.mi_row_start - a.mi_row) * MI_SIZE,
                row_max: (bottom_coded_mi_edge - a.mi_row) * MI_SIZE - bh as i32,
            }
        };
        set_mv_search_range(&mut lim, ref_r, ref_c);
        if lim.col_max < lim.col_min || lim.row_max < lim.row_min {
            continue;
        }

        // --- av1_intrabc_hash_search (mcomp.c:1908) ---
        // bestsme starts INT_MAX; the hash sets it (or leaves it) then the
        // pixel search ALWAYS runs at intrabc_search_level 0 (rdopt.c:3570).
        let mut bestsme: i64 = i64::MAX;
        let mut best_mv: Option<(i32, i32)> = None; // full-pel (row, col)
        // Square blocks only (mcomp.c:1918) query the source-frame hash.
        let (h1, h2) = if hash_eligible {
            get_block_hash_value(a.hash, a.src_y, a.off_y, a.stride, bw, a.bd > 8)
        } else {
            (u32::MAX, 0)
        };
        if let Some(bucket) = a.hash.buckets.get(&h1).filter(|_| hash_eligible) {
            if bucket.len() > 1 {
                let x_pos = a.mi_col * MI_SIZE;
                let y_pos = a.mi_row * MI_SIZE;
                let mut best_hash_cost = i64::MAX;
                for cand in bucket.iter() {
                    if cand.hash_value2 != h2 {
                        continue;
                    }
                    let dv_r = (i32::from(cand.y) - y_pos) * 8;
                    let dv_c = (i32::from(cand.x) - x_pos) * 8;
                    if !is_dv_valid(
                        dv_r, dv_c, a.mi_row, a.mi_col, a.bsize, a.tile, a.mib_size_log2,
                        a.is_chroma_ref, if a.monochrome { 1 } else { 3 }, a.ss_x as i32,
                        a.ss_y as i32,
                    ) {
                        continue;
                    }
                    let fm_r = i32::from(cand.y) - y_pos;
                    let fm_c = i32::from(cand.x) - x_pos;
                    if fm_r < lim.row_min
                        || fm_r > lim.row_max
                        || fm_c < lim.col_min
                        || fm_c > lim.col_max
                    {
                        continue;
                    }
                    // get_mvpred_var_cost: variance(src, recon@mv) + mv_err_cost.
                    let ref_off = (a.off_y as i64
                        + i64::from(fm_r) * a.stride as i64
                        + i64::from(fm_c)) as usize;
                    let var =
                        variance_wxh(a.src_y, a.off_y, a.stride, recon_y, ref_off, a.stride, bw, bh);
                    let cost = i64::from(var)
                        + i64::from(mv_err_cost(dv_r - ref_r, dv_c - ref_c, a.dv_costs, a.error_per_bit));
                    if cost < best_hash_cost {
                        best_hash_cost = cost;
                        best_mv = Some((fm_r, fm_c));
                    }
                }
                if best_mv.is_some() {
                    bestsme = best_hash_cost;
                }
            }
        }

        // --- av1_full_pixel_search (NSTEP diamond + mesh), ALWAYS at level 0.
        let s = FullPelSearch {
            src: a.src_y,
            src_off: a.off_y,
            refb: recon_y,
            ref_off: a.off_y,
            src_stride: a.stride,
            ref_stride: a.stride,
            w: bw,
            h: bh,
            limits: lim,
            dv: a.dv_costs,
            ref_row_sub: ref_r,
            ref_col_sub: ref_c,
            full_ref_row: ref_r >> 3,
            full_ref_col: ref_c >> 3,
            error_per_bit: a.error_per_bit,
            sad_per_bit: a.sad_per_bit,
        };
        {
            let (mut sme, mut pr, mut pc) =
                full_pixel_diamond(&s, ref_r >> 3, ref_c >> 3, a.mv_step_param);
            // Mesh gate (mcomp.c:1827): `var > (force_mesh_thresh >> (10 -
            // (w_log2 + h_log2)))`. force_mesh_thresh = exhaustive_searches_thresh
            // = 1<<20 (screen content). is_intra_mode ⇒ no prune gate.
            let w_log2 = (crate::tx_search::MI_SIZE_WIDE_B[a.bsize]).trailing_zeros() as i32;
            let h_log2 = (crate::tx_search::MI_SIZE_HIGH_B[a.bsize]).trailing_zeros() as i32;
            let exhaustive_thr: i64 = (1i64 << 20) >> (10 - (w_log2 + h_log2));
            if sme > exhaustive_thr {
                let (msme, mr, mc) = full_pixel_exhaustive(&s, pr, pc);
                if msme < sme {
                    sme = msme;
                    pr = mr;
                    pc = mc;
                }
            }
            if sme < bestsme {
                bestsme = sme;
                best_mv = Some((pr, pc));
            }
        }

        let Some((fm_r, fm_c)) = best_mv else {
            continue; // bestsme == INT_MAX
        };
        // Re-validate (rdopt.c:3582-3587): in-range + dv-valid.
        if !s.in_range(fm_r, fm_c) {
            continue;
        }
        let dv_r = fm_r * 8;
        let dv_c = fm_c * 8;
        if !is_dv_valid(
            dv_r, dv_c, a.mi_row, a.mi_col, a.bsize, a.tile, a.mib_size_log2, a.is_chroma_ref,
            if a.monochrome { 1 } else { 3 }, a.ss_x as i32, a.ss_y as i32,
        ) {
            continue;
        }

        // --- av1_txfm_search RD (rdopt.c:3606-3614) ---
        let rate_mv = mv_bit_cost_sub(dv_r, dv_c, ref_r, ref_c, a.dv_costs);
        let rate_mode = a.intrabc_cost[1];

        // Prediction into scratch (luma + chroma from the recon at the DV).
        let mut pred_y = vec![0u16; bw * bh];
        intrabc_predict_luma(recon_y, a.off_y, a.stride, fm_r, fm_c, &mut pred_y, bw, bw, bh);
        let (cw, ch) = (bw >> a.ss_x, bh >> a.ss_y);
        let (mut pred_u, mut pred_v) = (Vec::new(), Vec::new());
        if !a.monochrome && a.is_chroma_ref {
            pred_u = vec![0u16; cw * ch];
            pred_v = vec![0u16; cw * ch];
            intrabc_predict_chroma(
                recon_u, a.off_uv, a.stride, dv_r, dv_c, a.ss_x, a.ss_y, &mut pred_u, cw, cw, ch,
                i32::from(a.bd),
            );
            intrabc_predict_chroma(
                recon_v, a.off_uv, a.stride, dv_r, dv_c, a.ss_x, a.ss_y, &mut pred_v, cw, cw, ch,
                i32::from(a.bd),
            );
        }

        // Residual SSE (luma + chroma), and the luma residual for predict_skip.
        let mut luma_resid = vec![0i16; bw * bh];
        let mut luma_sse: i64 = 0;
        for r in 0..bh {
            for c in 0..bw {
                let d = i32::from(a.src_y[a.off_y + r * a.stride + c]) - i32::from(pred_y[r * bw + c]);
                luma_resid[r * bw + c] = d as i16;
                luma_sse += i64::from(d) * i64::from(d);
            }
        }
        let mut chroma_sse: i64 = 0;
        if !pred_u.is_empty() {
            for r in 0..ch {
                for c in 0..cw {
                    let du =
                        i32::from(a.src_u[a.off_uv + r * a.stride + c]) - i32::from(pred_u[r * cw + c]);
                    let dvv =
                        i32::from(a.src_v[a.off_uv + r * a.stride + c]) - i32::from(pred_v[r * cw + c]);
                    chroma_sse += i64::from(du) * i64::from(du) + i64::from(dvv) * i64::from(dvv);
                }
            }
        }

        // C's av1_txfm_search: `predict_skip_txfm` (LUMA) forces skip_txfm=1 and
        // BYPASSES the coeff arm (tx_search.c:3596). The block is coded as skip
        // only if the chroma ALSO skips (rd_stats.skip = luma_skip && uv_skip).
        // We offer the intrabc candidate only in that exact-skip regime: luma
        // predict_skip fires AND chroma is a perfect match (uv sse 0 ⇒ eob 0 ⇒
        // uv skip). Outside it the coeff arm (unported var-tx) would decide, so
        // we return no candidate (the frame keeps the intra winner). See the fn
        // doc + PARITY C3.
        let luma_skip = predict_skip_txfm(
            &luma_resid,
            bw,
            bh,
            a.bsize,
            luma_sse,
            a.qindex,
            i32::from(a.bd),
            a.reduced_tx_set_used,
        );
        if !luma_skip || chroma_sse != 0 {
            continue;
        }

        // set_skip_txfm (tx_search.c:245): dist = sse = ROUND_POWER_OF_TWO(luma
        // dist, 2*(bd-8)) << 4 (chroma sse is 0 here). rate = mode+mv+skip1.
        let scaled = if a.bd > 8 {
            let sh = 2 * (u32::from(a.bd) - 8);
            (luma_sse + (1 << (sh - 1))) >> sh
        } else {
            luma_sse
        };
        let skip_dist = scaled << 4;
        let skip_rate = rate_mode + rate_mv + a.skip_costs[a.skip_ctx][1];
        let this_rd = crate::rd::rdcost(a.rdmult, skip_rate, skip_dist);

        if this_rd < best_rd {
            best_rd = this_rd;
            best = Some(IntrabcBest {
                dv_row: dv_r,
                dv_col: dv_c,
                dv_ref_row: ref_r,
                dv_ref_col: ref_c,
                skip_txfm: true,
                rate: skip_rate,
                dist: skip_dist,
                rdcost: this_rd,
                inter_tx_size: [0; 16],
                tx_size: 0,
                tx_type_map: Vec::new(),
            });
        }
    }
    best
}
