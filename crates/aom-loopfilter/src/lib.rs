//! aom-loopfilter — bit-exact AV1 deblocking loop filter (port of libaom
//! v3.14.1 `aom_dsp/loopfilter.c`), lowbd. Both tracks.
//!
//! Each entry point filters 4 positions along an edge. Taps are at
//! `center + k*ts` where `ts` (tap stride) is the pitch for horizontal filters
//! and 1 for vertical filters; the per-position step is the opposite.
//! Validated byte-for-byte against C for random pixel windows + thresholds.


#![forbid(unsafe_code)]
pub mod frame;
pub mod highbd;

#[inline]
fn scc(t: i32) -> i8 {
    t.clamp(-128, 127) as i8
}

#[inline]
fn rpo2(v: i32, n: i32) -> u8 {
    ((v + ((1 << n) >> 1)) >> n) as u8
}

#[inline]
fn iabs(a: u8, b: u8) -> i32 {
    (a as i32 - b as i32).abs()
}

#[inline]
fn hev_mask(thresh: u8, p1: u8, p0: u8, q0: u8, q1: u8) -> i8 {
    let mut hev = 0i8;
    hev |= ((iabs(p1, p0) > thresh as i32) as i8) * -1;
    hev |= ((iabs(q1, q0) > thresh as i32) as i8) * -1;
    hev
}

#[inline]
fn filter_mask2(limit: u8, blimit: u8, p1: u8, p0: u8, q0: u8, q1: u8) -> i8 {
    let l = limit as i32;
    let mut mask = 0i8;
    mask |= ((iabs(p1, p0) > l) as i8) * -1;
    mask |= ((iabs(q1, q0) > l) as i8) * -1;
    mask |= ((iabs(p0, q0) * 2 + iabs(p1, q1) / 2 > blimit as i32) as i8) * -1;
    !mask
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn filter_mask(limit: u8, blimit: u8, p3: u8, p2: u8, p1: u8, p0: u8, q0: u8, q1: u8, q2: u8, q3: u8) -> i8 {
    let l = limit as i32;
    let mut mask = 0i8;
    mask |= ((iabs(p3, p2) > l) as i8) * -1;
    mask |= ((iabs(p2, p1) > l) as i8) * -1;
    mask |= ((iabs(p1, p0) > l) as i8) * -1;
    mask |= ((iabs(q1, q0) > l) as i8) * -1;
    mask |= ((iabs(q2, q1) > l) as i8) * -1;
    mask |= ((iabs(q3, q2) > l) as i8) * -1;
    mask |= ((iabs(p0, q0) * 2 + iabs(p1, q1) / 2 > blimit as i32) as i8) * -1;
    !mask
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn filter_mask3_chroma(limit: u8, blimit: u8, p2: u8, p1: u8, p0: u8, q0: u8, q1: u8, q2: u8) -> i8 {
    let l = limit as i32;
    let mut mask = 0i8;
    mask |= ((iabs(p2, p1) > l) as i8) * -1;
    mask |= ((iabs(p1, p0) > l) as i8) * -1;
    mask |= ((iabs(q1, q0) > l) as i8) * -1;
    mask |= ((iabs(q2, q1) > l) as i8) * -1;
    mask |= ((iabs(p0, q0) * 2 + iabs(p1, q1) / 2 > blimit as i32) as i8) * -1;
    !mask
}

#[inline]
fn flat_mask3_chroma(thresh: u8, p2: u8, p1: u8, p0: u8, q0: u8, q1: u8, q2: u8) -> i8 {
    let t = thresh as i32;
    let mut mask = 0i8;
    mask |= ((iabs(p1, p0) > t) as i8) * -1;
    mask |= ((iabs(q1, q0) > t) as i8) * -1;
    mask |= ((iabs(p2, p0) > t) as i8) * -1;
    mask |= ((iabs(q2, q0) > t) as i8) * -1;
    !mask
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn flat_mask4(thresh: u8, p3: u8, p2: u8, p1: u8, p0: u8, q0: u8, q1: u8, q2: u8, q3: u8) -> i8 {
    let t = thresh as i32;
    let mut mask = 0i8;
    mask |= ((iabs(p1, p0) > t) as i8) * -1;
    mask |= ((iabs(q1, q0) > t) as i8) * -1;
    mask |= ((iabs(p2, p0) > t) as i8) * -1;
    mask |= ((iabs(q2, q0) > t) as i8) * -1;
    mask |= ((iabs(p3, p0) > t) as i8) * -1;
    mask |= ((iabs(q3, q0) > t) as i8) * -1;
    !mask
}

/// `filter4`, operating on buffer indices for op1/op0/oq0/oq1.
fn filter4(buf: &mut [u8], i_op1: usize, i_op0: usize, i_oq0: usize, i_oq1: usize, mask: i8, thresh: u8) {
    let (op1, op0, oq0, oq1) = (buf[i_op1], buf[i_op0], buf[i_oq0], buf[i_oq1]);
    let ps1 = (op1 ^ 0x80) as i8;
    let ps0 = (op0 ^ 0x80) as i8;
    let qs0 = (oq0 ^ 0x80) as i8;
    let qs1 = (oq1 ^ 0x80) as i8;
    let hev = hev_mask(thresh, op1, op0, oq0, oq1);

    let mut filter = scc(ps1 as i32 - qs1 as i32) & hev;
    filter = scc(filter as i32 + 3 * (qs0 as i32 - ps0 as i32)) & mask;

    let filter1 = scc(filter as i32 + 4) >> 3;
    let filter2 = scc(filter as i32 + 3) >> 3;

    buf[i_oq0] = (scc(qs0 as i32 - filter1 as i32) as u8) ^ 0x80;
    buf[i_op0] = (scc(ps0 as i32 + filter2 as i32) as u8) ^ 0x80;

    let f = (((filter1 as i32 + 1) >> 1) as i8) & !hev;
    buf[i_oq1] = (scc(qs1 as i32 - f as i32) as u8) ^ 0x80;
    buf[i_op1] = (scc(ps1 as i32 + f as i32) as u8) ^ 0x80;
}

fn filter6(buf: &mut [u8], idx: [usize; 6], mask: i8, thresh: u8, flat: i8) {
    if flat != 0 && mask != 0 {
        let [i2, i1, i0, j0, j1, j2] = idx;
        let (p2, p1, p0) = (buf[i2] as i32, buf[i1] as i32, buf[i0] as i32);
        let (q0, q1, q2) = (buf[j0] as i32, buf[j1] as i32, buf[j2] as i32);
        buf[i1] = rpo2(p2 * 3 + p1 * 2 + p0 * 2 + q0, 3);
        buf[i0] = rpo2(p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1, 3);
        buf[j0] = rpo2(p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2, 3);
        buf[j1] = rpo2(p0 + q0 * 2 + q1 * 2 + q2 * 3, 3);
    } else {
        filter4(buf, idx[1], idx[2], idx[3], idx[4], mask, thresh);
    }
}

fn filter8(buf: &mut [u8], idx: [usize; 8], mask: i8, thresh: u8, flat: i8) {
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
        filter4(buf, idx[2], idx[3], idx[4], idx[5], mask, thresh);
    }
}

fn filter14(buf: &mut [u8], idx: [usize; 14], mask: i8, thresh: u8, flat: i8, flat2: i8) {
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
        filter8(buf, idx8, mask, thresh, flat);
    }
}

// ---- entry points -------------------------------------------------------
// `ts` = tap stride (pitch for horizontal, 1 for vertical); `step` = the
// per-position advance (1 for horizontal, pitch for vertical).

#[inline]
fn idx(center: isize, k: isize, ts: isize) -> usize {
    (center + k * ts) as usize
}

fn lpf_4(buf: &mut [u8], mut center: isize, ts: isize, step: isize, blimit: u8, limit: u8, thresh: u8) {
    for _ in 0..4 {
        let p1 = buf[idx(center, -2, ts)];
        let p0 = buf[idx(center, -1, ts)];
        let q0 = buf[idx(center, 0, ts)];
        let q1 = buf[idx(center, 1, ts)];
        let mask = filter_mask2(limit, blimit, p1, p0, q0, q1);
        filter4(buf, idx(center, -2, ts), idx(center, -1, ts), idx(center, 0, ts), idx(center, 1, ts), mask, thresh);
        center += step;
    }
}

fn lpf_6(buf: &mut [u8], mut center: isize, ts: isize, step: isize, blimit: u8, limit: u8, thresh: u8) {
    for _ in 0..4 {
        let g = |k| buf[idx(center, k, ts)];
        let (p2, p1, p0) = (g(-3), g(-2), g(-1));
        let (q0, q1, q2) = (g(0), g(1), g(2));
        let mask = filter_mask3_chroma(limit, blimit, p2, p1, p0, q0, q1, q2);
        let flat = flat_mask3_chroma(1, p2, p1, p0, q0, q1, q2);
        let ix = [idx(center, -3, ts), idx(center, -2, ts), idx(center, -1, ts), idx(center, 0, ts), idx(center, 1, ts), idx(center, 2, ts)];
        filter6(buf, ix, mask, thresh, flat);
        center += step;
    }
}

fn lpf_8(buf: &mut [u8], mut center: isize, ts: isize, step: isize, blimit: u8, limit: u8, thresh: u8) {
    for _ in 0..4 {
        let g = |k| buf[idx(center, k, ts)];
        let (p3, p2, p1, p0) = (g(-4), g(-3), g(-2), g(-1));
        let (q0, q1, q2, q3) = (g(0), g(1), g(2), g(3));
        let mask = filter_mask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        let flat = flat_mask4(1, p3, p2, p1, p0, q0, q1, q2, q3);
        let ix = [idx(center, -4, ts), idx(center, -3, ts), idx(center, -2, ts), idx(center, -1, ts), idx(center, 0, ts), idx(center, 1, ts), idx(center, 2, ts), idx(center, 3, ts)];
        filter8(buf, ix, mask, thresh, flat);
        center += step;
    }
}

fn lpf_14(buf: &mut [u8], mut center: isize, ts: isize, step: isize, blimit: u8, limit: u8, thresh: u8) {
    for _ in 0..4 {
        let g = |k| buf[idx(center, k, ts)];
        let (p6, p5, p4, p3, p2, p1, p0) = (g(-7), g(-6), g(-5), g(-4), g(-3), g(-2), g(-1));
        let (q0, q1, q2, q3, q4, q5, q6) = (g(0), g(1), g(2), g(3), g(4), g(5), g(6));
        let mask = filter_mask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        let flat = flat_mask4(1, p3, p2, p1, p0, q0, q1, q2, q3);
        let flat2 = flat_mask4(1, p6, p5, p4, p0, q0, q4, q5, q6);
        let mut ix = [0usize; 14];
        for (n, slot) in ix.iter_mut().enumerate() {
            *slot = idx(center, n as isize - 7, ts);
        }
        filter14(buf, ix, mask, thresh, flat, flat2);
        center += step;
    }
}

/// `p` is the pitch. `center` is the index of `s[0]` in `buf`.
/// Horizontal filters: taps stride by pitch, positions advance by 1.
pub fn horizontal(width: u32, buf: &mut [u8], center: usize, p: usize, blimit: u8, limit: u8, thresh: u8) {
    let (c, ts, step) = (center as isize, p as isize, 1isize);
    match width {
        4 => lpf_4(buf, c, ts, step, blimit, limit, thresh),
        6 => lpf_6(buf, c, ts, step, blimit, limit, thresh),
        8 => lpf_8(buf, c, ts, step, blimit, limit, thresh),
        14 => lpf_14(buf, c, ts, step, blimit, limit, thresh),
        _ => panic!("bad width"),
    }
}

/// Vertical filters: taps stride by 1, positions advance by pitch.
pub fn vertical(width: u32, buf: &mut [u8], center: usize, p: usize, blimit: u8, limit: u8, thresh: u8) {
    let (c, ts, step) = (center as isize, 1isize, p as isize);
    match width {
        4 => lpf_4(buf, c, ts, step, blimit, limit, thresh),
        6 => lpf_6(buf, c, ts, step, blimit, limit, thresh),
        8 => lpf_8(buf, c, ts, step, blimit, limit, thresh),
        14 => lpf_14(buf, c, ts, step, blimit, limit, thresh),
        _ => panic!("bad width"),
    }
}
