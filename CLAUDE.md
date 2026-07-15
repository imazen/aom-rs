# aom-rs — project instructions & durable bug log

Pure-Rust, **bit-exact** reimplementation of libaom ≥ v3.14.1 as a drop-in replacement.
Validated behind differential harnesses against the REAL exported C functions (priority of
evidence: real exported C fn > synthetic-facade-over-real-fn > verbatim transcription —
transcribed oracles can carry shared bugs).

**Module-progress source of truth:** `STATUS.md` (updated per landing by the track agents).
**This file** holds project-level coordination rules + the durable **Known Bugs** log.

## Gates (definition of done)

- **Gate 1 — Decoder:** bit-identical to C across the AV1 conformance corpus (intra scope
  wired in CI: `xtask/conformance.py --fetch --scope intra`; gate = byte-identity + golden MD5).
- **Gate 2 — Encoder:** bitstream bit-identical for every `--cpu-used 0..9`.
- **Gate 3 — Performance:** ≤ 1.20× C.
- **Gate 4 — Coverage checklist** (+ a zenavif integration gate).

Primary configuration: ALLINTRA (usage=2), speed-0 KEY frame. **Single-frame (KEY-frame)
work must reach byte-exactness across BOTH tracks before inter-frame ("the rest") starts.**

## Known Bugs

Record real bugs here immediately with file:line refs (survives context loss). Do NOT close
an entry by relaxing/excluding a test — only by a landed fix verified on `origin/main`.

### KB-1 — Decoder: recon divergence at base_qindex ≥ 249 (quantizer-62/-63) — REAL CORRUPTION, CI-quarantined
- **Symptom:** decoded RECON diverges from the C oracle at `base_qindex >= 249` — the
  `quantizer-62` / `quantizer-63` conformance vectors. Reproduces at **bd8 AND bd10, luma AND
  chroma**. Divergence is an edge-local ±1 prediction cascade.
- **Localization (Gate-1 agent):** all recon kernels (inv-transform, dequant, intra-pred,
  deblock) + all 16 tx scan orders proven **byte-exact**; the first divergence is in the
  **entropy coefficient-decode path** (`aom-txb` / coeff-context), not reconstruction.
- **CI status (TEMPORARY quarantine):** `.github/workflows/ci.yml:63-64` `rm`s the q62/q63
  vectors after fetch so Gate-1 goes green on the rest. This is a **must-fix corruption bug**
  under the zero-tolerance rule (wrong pixels are a shipping bug, never a known limitation),
  NOT an accepted limitation. The `rm` MUST be reverted in the same PR that lands the fix, and
  the specific q62/q63 vector(s) added as an explicit strong byte-identity case.
- **Tracking:** task **#21** (HIGH). Fix unblock: authorized throwaway reference-*decoder*
  instrumentation to dump the C coefficient + coeff-context/cdf state at the first diverging
  (position, plane, qindex), then revert + rebuild clean (never commit the instrument).
- **Range matters:** q62/q63 is the aggressive end of the quantizer range — exactly the
  web-compression regime this port targets.

### KB-2 — Encoder: `diag+vbars16 256x256 cq62` strong cell does not e2e byte-match
- **Symptom:** in `encoder_gate_e2e_rich_content_strong_lf`, one strong cell (`diag+vbars16`
  256×256 at cq62, real header `[1,17]`) does not match aomenc end-to-end — a residual
  **non-LF coeff/partition near-tie** (the port picks a marginally different coeff/partition
  decision than C). Analogous in kind to the already-fixed partition-RDO 26-bit palette-flag
  bug and the INTERNAL_COST_UPD_SB coeff-trellis bug. See `STATUS.md:2275`, commit `4940315`.
- **Status:** encoder track; not yet root-caused. A single-frame byte-exactness hole → must be
  closed for Gate 2, not excluded. Verify whether any gate currently *excludes* this cell (a
  relaxation to be reverted on fix) vs merely documenting it.

## Encoder single-frame primary envelope (VERIFIED against reference/libaom)

Primary config = ALLINTRA (usage=2), speed-0 KEY frame. libaom's own allintra tuning
(`av1/av1_cx_iface.c:3065`) sets these **defaults** — so matching them, NOT the base defaults,
is what "single-frame exact" means:

- **CDEF: OFF** by default in allintra ("CDEF has been found to blur images, so it's disabled
  in all-intra mode"). Only `--enable-cdef` turns it on.
- **Loop-restoration: OFF** by default in allintra.
- **QM: ON** by default in allintra (`enable_qm=1`, qm_min=4, qm_max=10, alternative QM formula).
- screen_detection_mode = ANTIALIASING_AWARE.

**What the encoder track has byte-matched (`encoder_gate_e2e_*`):** own-search partition / mode /
tx / coefficients + LF-level derivation, in a **CDEF-off + restoration-off** reference encode
(`shim encode_av1_kf`, cdef/restoration/qm passed as explicit params). This envelope MATCHES the
allintra defaults for CDEF+restoration. The frame HEADER is still bootstrapped from the real
parse (qindex, tile info, cdf-update, ...) — only LF-level is port-derived.

**Remaining for single-frame-PRIMARY exactness (blocks "all single frame exactly"):**
- **#8 qindex-from-cq mapping** — port must derive base_qindex from cq_level itself (currently
  read off the real parsed header). Small deterministic function.
- **#23 QM-on encode** — allintra auto-enables QM; confirm the port applies forward quantization
  matrices to byte-match a QM-on allintra encode (decoder QM decode already ported). If the e2e
  gates run QM-off, this is a real open primary hole; if QM-on, already covered — VERIFY.
- **#10 cpu-used 0..9 speed-feature sweep** (Gate 2) — the large remaining item.
- **#21 (decoder q62/63)** — the decoder-side must-fix corruption bug.

**NOT blocking single-frame-primary (deferred to non-default knobs / "the rest"):**
- **#7 CDEF-strength RD search** — off by default in allintra; only for explicit `--enable-cdef`.
  Building blocks exist as shims (`cdef_find_dir`, `cdef_filter_8/16`, `shim_encode_cdef`).
- **Loop-restoration (Wiener/SGR) search** — off by default in allintra; not tracked as a task
  (would be a non-primary item if a `--enable-restoration` config is ever targeted).

## Coordination (parallel tracks)

- Max clean parallelism = **2** (one decoder agent + one encoder agent); cargo's shared
  target-dir lock serializes builds, which keeps the box safe.
- Strict crate ownership; commit with **explicit per-file staging** (`git add <paths>`, never
  `-A`/`-u`/`.`); shared `STATUS.md` via `git add -p`. Push `git push origin HEAD:main`; verify
  `git merge-base --is-ancestor HEAD origin/main`.
- Coordinator independently verifies every landing (on origin, boundary-clean, no `#[ignore]`
  / weakened asserts, gate is a real byte-identity assertion, CI green). Never trust a claim.
