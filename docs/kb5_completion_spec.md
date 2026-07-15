# KB-5 completion spec — lossless (cq0) forward WHT + harness two-pass

Ready-to-apply spec for closing **KB-5** (lossless/cq0 encode divergence, surfaced by
the real-image e2e gate) and its two sub-tasks: **#33** (forward 4×4 Walsh–Hadamard
transform, currently missing on the encode side) and **#32** (encoder e2e harness
two-pass header parse for coded-lossless). All file:line refs and C math below were
traced in source on 2026-07-15; verify they haven't drifted before applying.

Scope note: this is staged design only. Applying it touches `aom-transform` and
`aom-encode` source and one **encoder-owned** RD file (`tx_search.rs`) — see
§2.3 "Ownership / coordination". Do not apply while KB-6 holds those files; coordinate
with the encoder track first.

---

## 1. Forward WHT (#33)

### 1.1 Oracle source — ONE shared function, no highbd variant

`av1_fwht4x4_c` — `reference/libaom/av1/encoder/hybrid_fwd_txfm.c:24`.

**There is no separate `av1_highbd_fwht4x4`.** The file comment (line 20-22) states the
function is *"Shared for both high and low bit depth."* The highbd dispatch
`highbd_fwd_txfm_4x4` (`hybrid_fwd_txfm.c:78-90`) calls `av1_fwht4x4` when
`txfm_param->lossless`, and `av1_lowbd_fwd_txfm_c` (`:241`) just forwards to
`av1_highbd_fwd_txfm` (`:246`) — so both bit-depth paths reach the *same*
`av1_fwht4x4_c`. Porting one bd-independent function is sufficient (mirrors the inverse,
whose `av1_highbd_iwht4x4_add` is likewise bd-independent apart from the final
`highbd_clip_pixel_add`).

### 1.2 Exact math (transcribe faithfully)

Signature: `av1_fwht4x4_c(const int16_t *input, tran_low_t *output, int stride)`.
Intermediates are `tran_high_t` = **i64** (`aom_dsp/aom_dsp_common.h:67`); output is
`tran_low_t` = **i32** (`:68`). `UNIT_QUANT_FACTOR = 1 << UNIT_QUANT_SHIFT = 4`
(`aom_dsp/txfm_common.h:21-22`; `UNIT_QUANT_SHIFT = 2` is already defined in
`inv_txfm2d.rs`).

```
Pass 0 (input columns -> output rows), NO shift, NO factor:
  for i in 0..4:                       // i indexes input columns
    a1 = input[i + 0*stride]
    b1 = input[i + 1*stride]
    c1 = input[i + 2*stride]
    d1 = input[i + 3*stride]
    a1 += b1
    d1  = d1 - c1
    e1  = (a1 - d1) >> 1
    b1  = e1 - b1
    c1  = e1 - c1
    a1 -= c1
    d1 += b1
    output[4*i + 0] = a1            // NOTE output order is (a1, c1, d1, b1)
    output[4*i + 1] = c1
    output[4*i + 2] = d1
    output[4*i + 3] = b1

Pass 1 (output columns -> output columns), APPLY *UNIT_QUANT_FACTOR (== <<2):
  for i in 0..4:                       // i indexes raster columns of the pass-0 result
    a1 = output[i + 4*0]
    b1 = output[i + 4*1]
    c1 = output[i + 4*2]
    d1 = output[i + 4*3]
    a1 += b1
    d1 -= c1
    e1  = (a1 - d1) >> 1
    b1  = e1 - b1
    c1  = e1 - c1
    a1 -= c1
    d1 += b1
    output[i + 4*0] = a1 * UNIT_QUANT_FACTOR    // same (a1, c1, d1, b1) permutation
    output[i + 4*1] = c1 * UNIT_QUANT_FACTOR
    output[i + 4*2] = d1 * UNIT_QUANT_FACTOR
    output[i + 4*3] = b1 * UNIT_QUANT_FACTOR
```

Two subtleties to preserve exactly: (a) the output index permutation is
`(a1, c1, d1, b1)` → positions `(0,1,2,3)`, **not** identity; (b) the `*UNIT_QUANT_FACTOR`
lands only on pass 1 (this is the factor the inverse cancels with its
`input >> UNIT_QUANT_SHIFT`, so `level*4 >> 2 == level` at the lossless unit quantizer —
see the `UNIT_QUANT_SHIFT` doc-comment already in `inv_txfm2d.rs`).

This is the transpose-dual of the inverse `av1_highbd_iwht4x4_16_add`
(`inv_txfm2d.rs` ~272): the inverse runs col-pass then row-pass and shifts the *input*
down; the forward runs col-pass then col-pass (raster) and scales the *output* up.

### 1.3 Where to add it

`crates/aom-transform/src/inv_txfm2d.rs`, adjacent to the inverse WHT at line 256
(`pub fn av1_highbd_iwht4x4_add`). Suggested public signature, mirroring the inverse's
slice/stride shape:

```rust
/// `av1_fwht4x4_c` (av1/encoder/hybrid_fwd_txfm.c:24): the 4x4 reversible
/// Walsh-Hadamard FORWARD transform for `xd->lossless` blocks (forced TX_4X4,
/// tx_type always DCT_DCT). Shared for high and low bit depth. `input` is the
/// 4x4 residual, row-major with row stride `stride`; `output` is the 16-entry
/// raster coefficient block. Intermediates are i64; output is i32.
pub fn av1_fwht4x4(input: &[i16], output: &mut [i32], stride: usize) { ... }
```

(`inv_txfm2d.rs` is the pragmatic home since `UNIT_QUANT_SHIFT` and the inverse WHT
already live there; a rename of the module to `wht.rs` or a `pub use` re-export is
optional polish, not required. Keep it `pub` from `aom_transform` so `aom-encode` and the
differential test can reach it. Adding a public fn is an additive API change — no break.)

---

## 2. Routing (the lossless gate)

### 2.1 The single coding funnel

All coding-path forward transforms funnel through **`xform_quant`**
(`crates/aom-encode/src/lib.rs:155`, the `av1_fwd_txfm2d(...)` call at **lib.rs:172**):

```rust
// av1_xform: forward 2-D transform into a full-size buffer ...
let mut coeff = vec![0i32; full];
av1_fwd_txfm2d(residual, &mut coeff, TX_W[tx_size], tx_type, tx_size);   // <-- branch here
```

Every coding caller reaches this line via `xform_quant` / `xform_quant_optimize`
(callers: `tx_search.rs:669,692`; `encode_intra.rs:414,422,679,687`). So a single branch
at lib.rs:172 covers **both luma and chroma** coding — no separate chroma coding site.

### 2.2 The gate

`xform_quant` currently has no lossless input (`QuantParams` — lib.rs:89 — carries
`bd` but not lossless). Lossless forces `tx_size == TX_4X4` (== 0) and
`tx_type == DCT_DCT` (== 0), so the gate is just a lossless bit plus a debug assert:

```rust
if qp.lossless {                                   // new field, see below
    debug_assert_eq!(tx_size, 0 /* TX_4X4 */);
    debug_assert_eq!(tx_type, 0 /* DCT_DCT */);
    aom_transform::inv_txfm2d::av1_fwht4x4(residual, &mut coeff, TX_W[tx_size]);
} else {
    av1_fwd_txfm2d(residual, &mut coeff, TX_W[tx_size], tx_type, tx_size);
}
```

Thread the flag by adding `pub lossless: bool` to `QuantParams` (lib.rs:89). Every
`QuantParams { .. }` construction then sets `lossless:` from its env's coded-lossless
bit. Alternative (fewer struct-literal edits, more call-site edits): add a `lossless:
bool` parameter to `xform_quant` + `xform_quant_optimize`. Prefer the `QuantParams`
field — it is the params bundle already threaded everywhere and keeps the two public fn
signatures stable.

`n_coeffs`, `log_scale`, `scan`, and the quantizer call are unchanged: at TX_4X4 they are
already the 4×4 values, and the lossless dequant (dc/ac step 4 at qindex 0) is what the
`*UNIT_QUANT_FACTOR`/`>>UNIT_QUANT_SHIFT` pair is calibrated against.

### 2.3 Ownership / coordination — and what does NOT change

- **`lib.rs` (xform_quant)** and **`encode_intra.rs`** are non-forbidden (this track /
  general). Editable.
- **`tx_search.rs:669,692`** constructs `QuantParams` and is **encoder-owned (forbidden)**.
  Adding the `lossless` field forces an edit there (set `lossless: env-coded-lossless`).
  **Coordinate with the encoder track before applying** — do not touch `tx_search.rs`
  unilaterally. (Open question to resolve first: does the mode/tx RD search even execute
  for coded-lossless, where tx_size/tx_type are fixed? If the RD search is bypassed for
  lossless, the `tx_search.rs` construction can default `lossless: false` and only the
  final coding path needs the true bit — which would keep the real fix entirely in
  non-forbidden files. Verify by tracing `av1_xform_quant`'s lossless callers vs the
  lossless RD path before committing to the field-everywhere approach.)
- **DO NOT branch the SATD fast model.** `intra_uv_rd.rs:800`
  (`av1_fwd_txfm2d(..., tx_type=0, tx_size)` inside the SATD model) mirrors C
  `av1_quick_txfm`, which sets **`txfm_param.lossless = 0` unconditionally**
  (`hybrid_fwd_txfm.c:364`). The fast model always uses the DCT even for lossless; adding
  a WHT branch there would diverge from C. Leave it as-is. (The coordinator's
  "intra_uv_rd.rs:800" pointer is the SATD model, not a coding site — flagged here so it
  isn't "fixed" by mistake.)

---

## 3. Harness two-pass for coded-lossless (#32)

### 3.1 The problem

`read_uncompressed_header` gates its loop-filter / CDEF / restoration / tx-mode tail reads
on the `cfg.coded_lossless` / `cfg.all_lossless` **inputs** (writer-mirror design). The
e2e harness `run_case` parses once with `cfg` defaulted to `coded_lossless = false`
(`encoder_gate_bd10_diff.rs:259`, `let p = read_uncompressed_header(&mut rb, &cfg);` — same
pattern in the encoder's `encoder_gate_e2e_byte_match.rs`). For a genuinely coded-lossless
(`cq0` / `--lossless=1`) stream this **misreads the tail**, so the bootstrapped header is
wrong and the whole frame diverges.

### 3.2 The fix — mirror the decoder's probe→reparse

The decoder already solves this at `crates/aom-decode/src/frame.rs:457-487` (probe parse at
`:468`, coded-lossless recompute at `:472`, re-parse at `:481`). Replicate that
shape in `run_case`, immediately after the current single parse:

```rust
// Pass 1 (probe): parse with coded_lossless = false (default cfg). Quant +
// segmentation precede every lossless-gated tail read, so they are exact here.
let mut rb = ReadBitBuffer::new(frame_payload);
let probe = read_uncompressed_header(&mut rb, &cfg);

// Recompute coded_lossless from the probe (frame.rs:355 frame_coded_lossless):
//   all 5 plane delta_q == 0  AND  (no segmentation ? base_qindex==0
//                                   : every MAX_SEGMENTS qindex == 0)
// The harness encodes simple KEY frames with segmentation off, so inline is enough:
let key_shown = !probe.prefix.show_existing_frame
    && probe.prefix.frame_type == 0
    && probe.prefix.show_frame;
let q = &probe.quant;
let coded_lossless = key_shown
    && q.base_qindex == 0
    && q.y_dc_delta_q == 0 && q.u_dc_delta_q == 0 && q.u_ac_delta_q == 0
    && q.v_dc_delta_q == 0 && q.v_ac_delta_q == 0;

// Pass 2 (only when coded-lossless): re-parse the SAME payload with the tail
// gated correctly. No superres in the harness, so all_lossless == coded_lossless.
let p = if coded_lossless {
    let mut cfg2 = cfg.clone();
    cfg2.coded_lossless = true;
    cfg2.all_lossless = true;
    let mut rb2 = ReadBitBuffer::new(frame_payload);
    read_uncompressed_header(&mut rb2, &cfg2)
} else {
    probe
};
```

Then the existing port pipeline consumes `p` unchanged. `p.coded_lossless` /
`p.all_lossless` will now be set, which already drives `env.lossless`, the
`enable_optimize_b` selection, and `av1_get_tx_size_uv` (`encoder_gate_bd10_diff.rs:338,
353` and `intra_uv_rd.rs:96,607`).

### 3.3 `frame_coded_lossless` reuse (optional)

`frame_coded_lossless` (`frame.rs:355`) is currently **private**. `aom-encode` already
depends on `aom-decode` (`crates/aom-encode/Cargo.toml:30`), so exposing it
`pub` (or a thin `pub fn` wrapper) lets the harness call it directly instead of inlining
the recompute. Inlining (§3.2) is fine for the segmentation-off KEY harness; expose the
helper only if a later test needs the segmentation branch.

---

## 4. Test plan

Two differentials, both bd-parameterized (bd8 + bd12), in a **new owned** test file
`crates/aom-encode/tests/fwht4x4_diff.rs` (does not touch encoder-owned files):

### 4.1 Forward WHT vs C oracle (exactness)

- **Add a C shim** (none exists today — verified: no `fwht`/`av1_fwht4x4` in
  `aom-sys-ref`). In `crates/aom-sys-ref/shim/` + `src/lib.rs`, add
  `ref_fwht4x4(input: &[i16], stride) -> Vec<i32>` calling `av1_fwht4x4_c`
  (declared in `av1/encoder/hybrid_fwd_txfm.h`). Bd-independent — one shim covers all
  bit depths.
- Random 4×4 residuals (±255 for bd8 realism, ±4095 for bd12) across several strides;
  assert `av1_fwht4x4(residual, out, stride) == ref_fwht4x4(residual, stride)` element-wise.

### 4.2 Forward→inverse round-trip (reversibility)

- Pick a mid-gray prediction `pred` (e.g. `1 << (bd-1)`) and residual `r` in
  `[-(1<<(bd-1)), (1<<(bd-1))-1]` so `pred + r` stays in `[0, (1<<bd))` (no clip loss).
- `coeff = av1_fwht4x4(r)`; then `av1_highbd_iwht4x4_add(coeff, recon /* =pred */, stride,
  eob, bd)` (inverse entry at `inv_txfm2d.rs:256`); assert `recon[i] == (pred + r[i])`.
- The `*UNIT_QUANT_FACTOR` (forward) and `>> UNIT_QUANT_SHIFT` (inverse) cancel, and the
  butterflies are reversible, so the pair is identity on in-range pixels. Drive both `eob`
  branches of the inverse (a DC-only block for `eob<=1`, a full block for `eob>1`).

### 4.3 e2e gate (after §2 + §3 land)

Once the WHT + routing + harness two-pass are in, a `cq0` KEY-frame cell (mono first, then
4:2:0) in the e2e harness should byte-match real aomenc. That closes KB-5's user-visible
symptom (the repro is already committed per KB-5). Keep it representable-content first;
the KB-4 RD-decision divergence is orthogonal and stays encoder-owned.

---

## 5. Apply order (smallest demoable chunks)

1. **§1** port `av1_fwht4x4` into `aom-transform` + **§4.1/§4.2** differentials — fully
   isolated, no encoder-file contact, lands independently and provably correct.
2. **§3** harness two-pass — test-only, isolated.
3. **§2** routing — the only step that may touch `tx_search.rs`; do §2.3's "does RD search
   run for lossless?" trace first, and coordinate before any forbidden-file edit.
4. **§4.3** e2e cq0 byte-match cell — proves the whole chain.

Steps 1–2 are entirely in non-forbidden files and can land during KB-6. Step 3 is the only
one gated on encoder-track coordination.
