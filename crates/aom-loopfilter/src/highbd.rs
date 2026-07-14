//! Highbd (10/12-bit) deblocking loop filter, bit-exact port of libaom v3.14.1
//! `aom_dsp/loopfilter.c` highbd path. Mirrors the lowbd module with `u16`
//! pixels, thresholds scaled by `(bd-8)`, and `signed_char_clamp_high`.

#[inline]
fn scc(t: i32, bd: i32) -> i16 {
    let lim = 128i32 << (bd - 8);
    t.clamp(-lim, lim - 1) as i16
}

#[inline]
fn rpo2(v: i32, n: i32) -> u16 {
    ((v + ((1 << n) >> 1)) >> n) as u16
}

#[inline]
fn iabs(a: u16, b: u16) -> i32 {
    (a as i32 - b as i32).abs()
}

#[inline]
fn hev_mask(thresh: u8, p1: u16, p0: u16, q0: u16, q1: u16, bd: i32) -> i16 {
    let t = (thresh as i32) << (bd - 8);
    let mut hev = 0i16;
    hev |= -((iabs(p1, p0) > t) as i16);
    hev |= -((iabs(q1, q0) > t) as i16);
    hev
}

#[inline]
fn filter_mask2(limit: u8, blimit: u8, p1: u16, p0: u16, q0: u16, q1: u16, bd: i32) -> i8 {
    let l = (limit as i32) << (bd - 8);
    let bl = (blimit as i32) << (bd - 8);
    let mut mask = 0i8;
    mask |= -((iabs(p1, p0) > l) as i8);
    mask |= -((iabs(q1, q0) > l) as i8);
    mask |= -((iabs(p0, q0) * 2 + iabs(p1, q1) / 2 > bl) as i8);
    !mask
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn filter_mask(limit: u8, blimit: u8, p3: u16, p2: u16, p1: u16, p0: u16, q0: u16, q1: u16, q2: u16, q3: u16, bd: i32) -> i8 {
    let l = (limit as i32) << (bd - 8);
    let bl = (blimit as i32) << (bd - 8);
    let mut mask = 0i8;
    mask |= -((iabs(p3, p2) > l) as i8);
    mask |= -((iabs(p2, p1) > l) as i8);
    mask |= -((iabs(p1, p0) > l) as i8);
    mask |= -((iabs(q1, q0) > l) as i8);
    mask |= -((iabs(q2, q1) > l) as i8);
    mask |= -((iabs(q3, q2) > l) as i8);
    mask |= -((iabs(p0, q0) * 2 + iabs(p1, q1) / 2 > bl) as i8);
    !mask
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn filter_mask3_chroma(limit: u8, blimit: u8, p2: u16, p1: u16, p0: u16, q0: u16, q1: u16, q2: u16, bd: i32) -> i8 {
    let l = (limit as i32) << (bd - 8);
    let bl = (blimit as i32) << (bd - 8);
    let mut mask = 0i8;
    mask |= -((iabs(p2, p1) > l) as i8);
    mask |= -((iabs(p1, p0) > l) as i8);
    mask |= -((iabs(q1, q0) > l) as i8);
    mask |= -((iabs(q2, q1) > l) as i8);
    mask |= -((iabs(p0, q0) * 2 + iabs(p1, q1) / 2 > bl) as i8);
    !mask
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn flat_mask3_chroma(thresh: u8, p2: u16, p1: u16, p0: u16, q0: u16, q1: u16, q2: u16, bd: i32) -> i8 {
    let t = (thresh as i32) << (bd - 8);
    let mut mask = 0i8;
    mask |= -((iabs(p1, p0) > t) as i8);
    mask |= -((iabs(q1, q0) > t) as i8);
    mask |= -((iabs(p2, p0) > t) as i8);
    mask |= -((iabs(q2, q0) > t) as i8);
    !mask
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn flat_mask4(thresh: u8, p3: u16, p2: u16, p1: u16, p0: u16, q0: u16, q1: u16, q2: u16, q3: u16, bd: i32) -> i8 {
    let t = (thresh as i32) << (bd - 8);
    let mut mask = 0i8;
    mask |= -((iabs(p1, p0) > t) as i8);
    mask |= -((iabs(q1, q0) > t) as i8);
    mask |= -((iabs(p2, p0) > t) as i8);
    mask |= -((iabs(q2, q0) > t) as i8);
    mask |= -((iabs(p3, p0) > t) as i8);
    mask |= -((iabs(q3, q0) > t) as i8);
    !mask
}

#[allow(clippy::too_many_arguments)]
fn filter4(buf: &mut [u16], i1: usize, i0: usize, j0: usize, j1: usize, mask: i8, thresh: u8, bd: i32) {
    let shift = bd - 8;
    let bias = 0x80i32 << shift;
    let (op1, op0, oq0, oq1) = (buf[i1], buf[i0], buf[j0], buf[j1]);
    let ps1 = op1 as i32 - bias;
    let ps0 = op0 as i32 - bias;
    let qs0 = oq0 as i32 - bias;
    let qs1 = oq1 as i32 - bias;
    let hev = hev_mask(thresh, op1, op0, oq0, oq1, bd);

    let mut filter = scc(ps1 - qs1, bd) & hev;
    filter = scc(filter as i32 + 3 * (qs0 - ps0), bd) & (mask as i16);
    let filter1 = scc(filter as i32 + 4, bd) >> 3;
    let filter2 = scc(filter as i32 + 3, bd) >> 3;

    buf[j0] = (scc(qs0 - filter1 as i32, bd) as i32 + bias) as u16;
    buf[i0] = (scc(ps0 + filter2 as i32, bd) as i32 + bias) as u16;
    let f = ((filter1 as i32 + 1) >> 1) as i16 & !hev;
    buf[j1] = (scc(qs1 - f as i32, bd) as i32 + bias) as u16;
    buf[i1] = (scc(ps1 + f as i32, bd) as i32 + bias) as u16;
}

fn filter6(buf: &mut [u16], idx: [usize; 6], mask: i8, thresh: u8, flat: i8, bd: i32) {
    if flat != 0 && mask != 0 {
        let [i2, i1, i0, j0, j1, j2] = idx;
        let (p2, p1, p0) = (buf[i2] as i32, buf[i1] as i32, buf[i0] as i32);
        let (q0, q1, q2) = (buf[j0] as i32, buf[j1] as i32, buf[j2] as i32);
        buf[i1] = rpo2(p2 * 3 + p1 * 2 + p0 * 2 + q0, 3);
        buf[i0] = rpo2(p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1, 3);
        buf[j0] = rpo2(p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2, 3);
        buf[j1] = rpo2(p0 + q0 * 2 + q1 * 2 + q2 * 3, 3);
    } else {
        filter4(buf, idx[1], idx[2], idx[3], idx[4], mask, thresh, bd);
    }
}

fn filter8(buf: &mut [u16], idx: [usize; 8], mask: i8, thresh: u8, flat: i8, bd: i32) {
    if flat != 0 && mask != 0 {
        let [i3, i2, i1, i0, j0, j1, j2, j3] = idx;
        let (p3, p2, p1, p0) = (buf[i3] as i32, buf[i2] as i32, buf[i1] as i32, buf[i0] as i32);
        let (q0, q1, q2, q3) = (buf[j0] as i32, buf[j1] as i32, buf[j2] as i32, buf[j3] as i32);
        buf[i2] = rpo2(p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0, 3);
        buf[i1] = rpo2(p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1, 3);
        buf[i0] = rpo2(p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2, 3);
        buf[j0] = rpo2(p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3, 3);
        buf[j1] = rpo2(p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3, 3);
        buf[j2] = rpo2(p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3, 3);
    } else {
        filter4(buf, idx[2], idx[3], idx[4], idx[5], mask, thresh, bd);
    }
}

fn filter14(buf: &mut [u16], idx: [usize; 14], mask: i8, thresh: u8, flat: i8, flat2: i8, bd: i32) {
    if flat2 != 0 && flat != 0 && mask != 0 {
        let v: Vec<i32> = idx.iter().map(|&i| buf[i] as i32).collect();
        let (p6, p5, p4, p3, p2, p1, p0) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
        let (q0, q1, q2, q3, q4, q5, q6) = (v[7], v[8], v[9], v[10], v[11], v[12], v[13]);
        buf[idx[1]] = rpo2(p6 * 7 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0, 4);
        buf[idx[2]] = rpo2(p6 * 5 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1, 4);
        buf[idx[3]] = rpo2(p6 * 4 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2, 4);
        buf[idx[4]] = rpo2(p6 * 3 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3, 4);
        buf[idx[5]] = rpo2(p6 * 2 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4, 4);
        buf[idx[6]] = rpo2(p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5, 4);
        buf[idx[7]] = rpo2(p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6, 4);
        buf[idx[8]] = rpo2(p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 * 2, 4);
        buf[idx[9]] = rpo2(p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 * 3, 4);
        buf[idx[10]] = rpo2(p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 * 4, 4);
        buf[idx[11]] = rpo2(p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 * 5, 4);
        buf[idx[12]] = rpo2(p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 7, 4);
    } else {
        let idx8 = [idx[3], idx[4], idx[5], idx[6], idx[7], idx[8], idx[9], idx[10]];
        filter8(buf, idx8, mask, thresh, flat, bd);
    }
}

#[inline]
fn idx(center: isize, k: isize, ts: isize) -> usize {
    (center + k * ts) as usize
}

#[allow(clippy::too_many_arguments)]
fn lpf_4(buf: &mut [u16], mut c: isize, ts: isize, step: isize, bl: u8, li: u8, th: u8, bd: i32) {
    for _ in 0..4 {
        let g = |k| buf[idx(c, k, ts)];
        let mask = filter_mask2(li, bl, g(-2), g(-1), g(0), g(1), bd);
        filter4(buf, idx(c, -2, ts), idx(c, -1, ts), idx(c, 0, ts), idx(c, 1, ts), mask, th, bd);
        c += step;
    }
}

#[allow(clippy::too_many_arguments)]
fn lpf_6(buf: &mut [u16], mut c: isize, ts: isize, step: isize, bl: u8, li: u8, th: u8, bd: i32) {
    for _ in 0..4 {
        let g = |k| buf[idx(c, k, ts)];
        let mask = filter_mask3_chroma(li, bl, g(-3), g(-2), g(-1), g(0), g(1), g(2), bd);
        let flat = flat_mask3_chroma(1, g(-3), g(-2), g(-1), g(0), g(1), g(2), bd);
        let ix = [idx(c, -3, ts), idx(c, -2, ts), idx(c, -1, ts), idx(c, 0, ts), idx(c, 1, ts), idx(c, 2, ts)];
        filter6(buf, ix, mask, th, flat, bd);
        c += step;
    }
}

#[allow(clippy::too_many_arguments)]
fn lpf_8(buf: &mut [u16], mut c: isize, ts: isize, step: isize, bl: u8, li: u8, th: u8, bd: i32) {
    for _ in 0..4 {
        let g = |k| buf[idx(c, k, ts)];
        let mask = filter_mask(li, bl, g(-4), g(-3), g(-2), g(-1), g(0), g(1), g(2), g(3), bd);
        let flat = flat_mask4(1, g(-4), g(-3), g(-2), g(-1), g(0), g(1), g(2), g(3), bd);
        let ix = [idx(c, -4, ts), idx(c, -3, ts), idx(c, -2, ts), idx(c, -1, ts), idx(c, 0, ts), idx(c, 1, ts), idx(c, 2, ts), idx(c, 3, ts)];
        filter8(buf, ix, mask, th, flat, bd);
        c += step;
    }
}

#[allow(clippy::too_many_arguments)]
fn lpf_14(buf: &mut [u16], mut c: isize, ts: isize, step: isize, bl: u8, li: u8, th: u8, bd: i32) {
    for _ in 0..4 {
        let g = |k| buf[idx(c, k, ts)];
        let mask = filter_mask(li, bl, g(-4), g(-3), g(-2), g(-1), g(0), g(1), g(2), g(3), bd);
        let flat = flat_mask4(1, g(-4), g(-3), g(-2), g(-1), g(0), g(1), g(2), g(3), bd);
        let flat2 = flat_mask4(1, g(-7), g(-6), g(-5), g(-1), g(0), g(4), g(5), g(6), bd);
        let mut ix = [0usize; 14];
        for (n, slot) in ix.iter_mut().enumerate() {
            *slot = idx(c, n as isize - 7, ts);
        }
        filter14(buf, ix, mask, th, flat, flat2, bd);
        c += step;
    }
}

/// Highbd horizontal deblock (taps stride by pitch).
#[allow(clippy::too_many_arguments)]
pub fn horizontal(width: u32, buf: &mut [u16], center: usize, p: usize, bl: u8, li: u8, th: u8, bd: i32) {
    let (c, ts, step) = (center as isize, p as isize, 1isize);
    match width {
        4 => lpf_4(buf, c, ts, step, bl, li, th, bd),
        6 => lpf_6(buf, c, ts, step, bl, li, th, bd),
        8 => lpf_8(buf, c, ts, step, bl, li, th, bd),
        14 => lpf_14(buf, c, ts, step, bl, li, th, bd),
        _ => panic!("bad width"),
    }
}

/// Highbd vertical deblock (taps stride by 1).
#[allow(clippy::too_many_arguments)]
pub fn vertical(width: u32, buf: &mut [u16], center: usize, p: usize, bl: u8, li: u8, th: u8, bd: i32) {
    let (c, ts, step) = (center as isize, 1isize, p as isize);
    match width {
        4 => lpf_4(buf, c, ts, step, bl, li, th, bd),
        6 => lpf_6(buf, c, ts, step, bl, li, th, bd),
        8 => lpf_8(buf, c, ts, step, bl, li, th, bd),
        14 => lpf_14(buf, c, ts, step, bl, li, th, bd),
        _ => panic!("bad width"),
    }
}
