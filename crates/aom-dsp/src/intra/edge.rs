//! Intra edge filtering + upsampling (libaom `av1/common/reconintra.c` +
//! `av1_dsp` `av1_filter_intra_edge_c` / `av1_upsample_intra_edge_c`): the
//! reference-conditioning applied before the directional predictors for
//! non-cardinal angles. Byte-exact vs C libaom. Lowbd (8-bit).

/// `intra_edge_filter_strength(bs0, bs1, delta, type)` — the edge low-pass
/// strength (0..3) from block size, angle delta from cardinal, and filter type.
pub fn edge_filter_strength(bs0: i32, bs1: i32, delta: i32, ty: i32) -> i32 {
    let d = delta.abs();
    let blk_wh = bs0 + bs1;
    let mut strength = 0;
    if ty == 0 {
        if blk_wh <= 8 {
            if d >= 56 {
                strength = 1;
            }
        } else if blk_wh <= 16 {
            // libaom has separate <=12 and <=16 branches with identical action.
            if d >= 40 {
                strength = 1;
            }
        } else if blk_wh <= 24 {
            if d >= 8 {
                strength = 1;
            }
            if d >= 16 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else if blk_wh <= 32 {
            if d >= 1 {
                strength = 1;
            }
            if d >= 4 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else if d >= 1 {
            strength = 3;
        }
    } else if blk_wh <= 8 {
        if d >= 40 {
            strength = 1;
        }
        if d >= 64 {
            strength = 2;
        }
    } else if blk_wh <= 16 {
        if d >= 20 {
            strength = 1;
        }
        if d >= 48 {
            strength = 2;
        }
    } else if blk_wh <= 24 {
        if d >= 4 {
            strength = 3;
        }
    } else if d >= 1 {
        strength = 3;
    }
    strength
}

/// `av1_use_intra_edge_upsample(bs0, bs1, delta, type)`.
pub fn use_upsample(bs0: i32, bs1: i32, delta: i32, ty: i32) -> i32 {
    let d = delta.abs();
    let blk_wh = bs0 + bs1;
    if d == 0 || d >= 40 {
        return 0;
    }
    if ty != 0 {
        i32::from(blk_wh <= 8)
    } else {
        i32::from(blk_wh <= 16)
    }
}

/// `av1_filter_intra_edge_c`: strength-indexed 5-tap low-pass over `p[0..sz]`,
/// modifying `p[1..sz]` in place. `p[0]` (the corner) is preserved.
pub fn filter_intra_edge(p: &mut [u8], sz: usize, strength: i32) {
    if strength == 0 {
        return;
    }
    const KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    let filt = (strength - 1) as usize;
    let edge: [i32; 129] = {
        let mut e = [0i32; 129];
        e[..sz]
            .iter_mut()
            .zip(p.iter().take(sz))
            .for_each(|(d, &s)| *d = s as i32);
        e
    };
    #[allow(clippy::needless_range_loop)]
    for i in 1..sz {
        let mut s = 0i32;
        for j in 0..5 {
            let k = (i as i32 - 2 + j as i32).clamp(0, sz as i32 - 1) as usize;
            s += edge[k] * KERNEL[filt][j];
        }
        p[i] = ((s + 8) >> 4) as u8;
    }
}

/// `av1_upsample_intra_edge_c`: double the edge. Logical index `i` lives at
/// `buf[off + i]`; the kernel reads `p[-1]` and writes `p[-2]`, `p[2i-1]`,
/// `p[2i]`, i.e. logical `-2 .. 2*sz-2`. `off >= 2`, `sz <= 16`.
pub fn upsample_intra_edge(buf: &mut [u8], off: usize, sz: usize) {
    // in[0]=in[1]=p[-1]; in[i+2]=p[i]; in[sz+2]=p[sz-1].
    let mut inp = [0i32; 19]; // MAX_UPSAMPLE_SZ(16) + 3
    inp[0] = buf[off - 1] as i32;
    inp[1] = buf[off - 1] as i32;
    for i in 0..sz {
        inp[i + 2] = buf[off + i] as i32;
    }
    inp[sz + 2] = buf[off + sz - 1] as i32;
    buf[off - 2] = inp[0] as u8;
    #[allow(clippy::needless_range_loop)]
    for i in 0..sz {
        let s = (-inp[i] + 9 * inp[i + 1] + 9 * inp[i + 2] - inp[i + 3] + 8) >> 4;
        let s = s.clamp(0, 255) as u8;
        buf[(off as i32 + 2 * i as i32 - 1) as usize] = s;
        buf[off + 2 * i] = inp[i + 2] as u8;
    }
}

/// `highbd_filter_intra_edge_corner` (reconintra.c): 3-tap blend of the top-left
/// corner with the first above and left samples, written to both corners.
/// `above`/`left` are `[-1..]` views — index 0 is the corner (`above_row[-1]` /
/// `left_col[-1]`), index 1 the first edge sample (`above_row[0]` / `left_col[0]`).
pub fn filter_corner_high(above: &mut [u16], left: &mut [u16]) {
    // s = left_col[0]*5 + above_row[-1]*6 + above_row[0]*5, rounded >> 4.
    let s = left[1] as i32 * 5 + above[0] as i32 * 6 + above[1] as i32 * 5;
    let s = ((s + 8) >> 4) as u16;
    above[0] = s;
    left[0] = s;
}

/// `av1_highbd_filter_intra_edge_c`: highbd (u16) 5-tap edge low-pass. Same as
/// the lowbd filter but 16-bit (no clip needed — outputs stay in range).
pub fn highbd_filter_intra_edge(p: &mut [u16], sz: usize, strength: i32) {
    if strength == 0 {
        return;
    }
    const KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    let filt = (strength - 1) as usize;
    let mut edge = [0i32; 129];
    edge[..sz]
        .iter_mut()
        .zip(p.iter().take(sz))
        .for_each(|(d, &s)| *d = s as i32);
    #[allow(clippy::needless_range_loop)]
    for i in 1..sz {
        let mut s = 0i32;
        for j in 0..5 {
            let k = (i as i32 - 2 + j as i32).clamp(0, sz as i32 - 1) as usize;
            s += edge[k] * KERNEL[filt][j];
        }
        p[i] = ((s + 8) >> 4) as u16;
    }
}

/// `av1_highbd_upsample_intra_edge_c`: highbd edge doubling, clipping half-sample
/// outputs to `[0, (1<<bd)-1]`. Layout as `upsample_intra_edge` (`off >= 2`).
pub fn highbd_upsample_intra_edge(buf: &mut [u16], off: usize, sz: usize, bd: u8) {
    let max_v = (1i32 << bd) - 1;
    let mut inp = [0i32; 19];
    inp[0] = buf[off - 1] as i32;
    inp[1] = buf[off - 1] as i32;
    for i in 0..sz {
        inp[i + 2] = buf[off + i] as i32;
    }
    inp[sz + 2] = buf[off + sz - 1] as i32;
    buf[off - 2] = inp[0] as u16;
    #[allow(clippy::needless_range_loop)]
    for i in 0..sz {
        let s = (-inp[i] + 9 * inp[i + 1] + 9 * inp[i + 2] - inp[i + 3] + 8) >> 4;
        let s = s.clamp(0, max_v) as u16;
        buf[(off as i32 + 2 * i as i32 - 1) as usize] = s;
        buf[off + 2 * i] = inp[i + 2] as u16;
    }
}
