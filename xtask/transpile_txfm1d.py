#!/usr/bin/env python3
"""Transpile libaom ping-pong-buffer forward 1D transforms (av1_fwd_txfm1d.c)
into bit-exact Rust. Handles ONLY the regular statement forms used by
fdct8/16/32/64 and fadst8/16. fadst4 + identities are hand-ported.

`--lanes` emits 8-column lane-batched (AVX2 `i32x8`) twins (`<name>_v3`)
instead, mapping each scalar op onto the lane helpers in
`aom-transform/src/simd/mod.rs`: wrapping add/sub stay `+`/`-` (lane adds
wrap), `-a + b` becomes `b - a` (exact in two's complement), a bare `-a` copy
becomes `negv(t, a)`, `half_btf` becomes the exact-i64 `hb` recipe and
`clamp_value` becomes `clampv`. Statement order is preserved verbatim so the
lane kernel IS the scalar kernel, per lane.

Every emitted function is validated byte-for-byte by the differential harness;
this transpiler is a convenience, not a trusted oracle."""
import re, sys

ARGS = sys.argv[1:]
INV = '--inv' in ARGS
LANES = '--lanes' in ARGS
FUNCS = [a for a in ARGS if not a.startswith('--')]  # list of extracted .c files

def translate_operand(tok, m):
    """tok like 'bf0[3]' / '-input[7]' / 'cospi[32]' / '-cospi[16]' -> Rust access, negated flag."""
    neg = tok.startswith('-')
    if neg:
        tok = tok[1:]
    mm = re.match(r'^(\w+)\[(\d+)\]$', tok)
    if not mm:
        raise ValueError(f'operand: {tok!r}')
    arr, idx = mm.group(1), mm.group(2)
    arr = m.get(arr, arr)  # bf0/bf1 -> current buffer; input/cospi pass through
    return f'{arr}[{idx}]', neg

def access(tok, m):
    a, neg = translate_operand(tok, m)
    return f'{a}.wrapping_neg()' if neg else a

def access_lanes(tok, m):
    a, neg = translate_operand(tok, m)
    return f'negv(t, {a})' if neg else a

def translate_rhs(rhs, m, stage):
    rhs = rhs.strip()
    # clamp_value(EXPR, stage_range[stage])  (inverse transforms)
    cm = re.match(r'^clamp_value\((.*), stage_range\[stage\]\)$', rhs)
    if cm:
        inner = translate_rhs(cm.group(1), m, stage)
        if LANES:
            return f'clampv(t, {inner}, stage_range[{stage}])'
        return f'clamp_value({inner}, stage_range[{stage}])'
    # half_btf(w0, in0, w1, in1, cos_bit)
    hb = re.match(r'^half_btf\((.*)\)$', rhs)
    if hb:
        args = [a.strip() for a in hb.group(1).split(',')]
        assert len(args) == 5, args
        w0, in0, w1, in1, _bit = args
        def half_arg(t):  # cospi[k] or -cospi[k] or bf0[k]
            a, neg = translate_operand(t, m)
            return f'-{a}' if neg else a
        if LANES:
            # weights are scalar i32 (splat inside hb); operands are lane values
            return (f'hb(t, {half_arg(w0)}, {access_lanes(in0, m)}, '
                    f'{half_arg(w1)}, {access_lanes(in1, m)}, cos_bit)')
        return (f'half_btf({half_arg(w0)}, {access(in0, m)}, '
                f'{half_arg(w1)}, {access(in1, m)}, cos_bit)')
    # binary: A op B  (op in + -), split on top-level ' + ' / ' - '
    bm = re.match(r'^(-?\w+\[\d+\])\s+([+-])\s+(\w+\[\d+\])$', rhs)
    if bm:
        left, op, right = bm.group(1), bm.group(2), bm.group(3)
        if LANES:
            la, lneg = translate_operand(left, m)
            ra, rneg = translate_operand(right, m)
            assert not rneg
            if lneg:
                # -a + b == b - a (two's complement, wrapping — exact);
                # -a - b would need negv(a) - b, never emitted by libaom.
                assert op == '+', f'unsupported lanes form: {rhs!r}'
                return f'{ra} - {la}'
            return f'{la} {op} {ra}'
        l = access(left, m)
        r = access(right, m)
        fn = 'wrapping_add' if op == '+' else 'wrapping_sub'
        return f'{l}.{fn}({r})'
    # unary copy: X[n] or -X[n]
    um = re.match(r'^-?\w+\[\d+\]$', rhs)
    if um:
        if LANES:
            return access_lanes(rhs, m)
        return access(rhs, m)
    raise ValueError(f'RHS: {rhs!r}')

def transpile(path):
    src = open(path).read()
    src = re.sub(r'//[^\n]*', '', src)  # strip line comments before newline-join
    name = re.search(r'void (av1_\w+)\(', src).group(1)
    size = int(re.search(r'const int32_t size = (\d+);', src).group(1))
    m = {'output': 'out', 'step': 'step', 'input': 'input'}  # array-name -> rust var
    ptr = {}  # bf0/bf1 -> 'out'/'step'/'input'
    body = []
    stage = [0]  # mutable stage counter (incremented on each `stage++`)
    # join statements onto single lines
    text = re.sub(r'\n\s+', ' ', src)
    for stmt in text.split(';'):
        stmt = stmt.strip().strip('}').strip('{').strip()
        if not stmt or stmt.startswith('//'):
            continue
        if stmt == 'stage++':
            stage[0] += 1
            continue
        if re.match(r'(void |const int32_t |int32_t |int8_t |int \b|int stage|stage\b|'
                    r'assert\(|av1_range_check_buf\(|\(void\)|cospi = cospi_arr)', stmt):
            # pointer role assignment?
            pa = re.match(r'^(bf[01]) = (output|step|input)$', stmt)
            if pa:
                ptr[pa.group(1)] = m[pa.group(2)]
            continue
        pa = re.match(r'^(bf[01]) = (output|step|input)$', stmt)
        if pa:
            ptr[pa.group(1)] = m[pa.group(2)]
            continue
        asn = re.match(r'^(bf[01])\[(\d+)\] = (.*)$', stmt, re.S)
        if asn:
            dst = ptr[asn.group(1)]
            idx = asn.group(2)
            mm = dict(m); mm['bf0'] = ptr.get('bf0'); mm['bf1'] = ptr.get('bf1')
            rhs = translate_rhs(asn.group(3), mm, stage[0])
            body.append(f'    {dst}[{idx}] = {rhs};')
            continue
        raise ValueError(f'unhandled stmt in {name}: {stmt!r}')
    if LANES:
        lines = [f'/// 8-column lane-batched twin of [`crate::{name}`] (transpiled; per-lane',
                 '/// bit-identical to the scalar kernel on the full `i32` domain — pinned by',
                 '/// the `simd::tests` differential at every token permutation).',
                 '#[rite]',
                 '#[allow(unused_variables)]',
                 f'pub(crate) fn {name}_v3(t: X64V3Token, input: &[i32x8], out: &mut [i32x8], cos_bit: i32, stage_range: &[i8]) {{',
                 f'    let cospi = cospi_arr(cos_bit);',
                 f'    let mut step = [i32x8::zero(t); {size}];',
                 *body,
                 '}']
    else:
        lines = [f'/// Bit-exact port of libaom `{name}` (transpiled, harness-verified).',
                 '#[allow(unused_variables)]',
                 f'pub fn {name}(input: &[i32], out: &mut [i32], cos_bit: i32, stage_range: &[i8]) {{',
                 f'    let cospi = cospi_arr(cos_bit);',
                 f'    let mut step = [0i32; {size}];',
                 *body,
                 '}']
    return '\n'.join(lines)

kind = 'inverse' if INV else 'forward'
if LANES:
    bodies = [transpile(f) for f in FUNCS]
    all_text = '\n'.join(bodies)
    helpers = [h for h in ('clampv', 'hb', 'negv') if f'{h}(' in all_text]
    print('//! GENERATED by xtask/transpile_txfm1d.py --lanes — do not edit by hand.')
    print(f'//! 8-column lane-batched (AVX2) twins of the scalar 1D {kind} transforms:')
    print('//! per-lane bit-identical to the scalar transcription on the FULL i32 domain')
    print('//! (the `super` helpers reproduce wrapping-i32 / exact-i64 scalar semantics).')
    print('#![allow(clippy::needless_range_loop)]')
    print('use archmage::prelude::*;')
    print('use archmage::X64V3Token;')
    print('use magetypes::simd::i32x8;')
    print('use crate::cospi::cospi_arr;')
    print(f'use super::{{{", ".join(helpers)}}};\n')
    for b in bodies:
        print(b)
        print()
else:
    print('//! GENERATED by xtask/transpile_txfm1d.py — do not edit by hand.')
    print(f'//! Bit-exact ports of libaom v3.14.1 ping-pong 1D {kind} transforms.')
    print('#![allow(clippy::needless_range_loop)]')
    print('use crate::cospi::cospi_arr;')
    if INV:
        print('use crate::fdct::{half_btf, clamp_value};\n')
    else:
        print('use crate::fdct::half_btf;\n')
    for f in FUNCS:
        print(transpile(f))
        print()
