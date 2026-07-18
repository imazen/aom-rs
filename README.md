# aom-rs

Pure-Rust reimplementation of [libaom](https://aomedia.googlesource.com/aom)
(the Alliance for Open Media AV1 reference codec), built module-by-module
behind differential harnesses: every ported module is validated against a
pinned C libaom v3.14.1 oracle, and landed encoder paths are held to
byte-exact bitstream gates.

`#![forbid(unsafe_code)]` | SIMD via [archmage](https://github.com/imazen/archmage)

**Status: early development — not usable yet.** [STATUS.md](STATUS.md) tracks
what has landed; [PARITY.md](PARITY.md) tracks the differential gates.
`just test` runs the pure-Rust suite; see the [justfile](justfile) for the
bench and profiling entry points.

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software — and the 40+
library ecosystem it depends on — full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

Upstream C code from [libaom](https://aomedia.googlesource.com/aom) is
BSD-2-Clause with the Alliance for Open Media Patent License 1.0 — see
[LICENSE](LICENSE) and [PATENTS](PATENTS); those terms continue to cover the
upstream work this port derives from. libaom is battle-tested, carefully
engineered code — this port stands entirely on that foundation.

### Path to MIT

If someone covers Imazen's 2026 AI + server costs, we'll release this port
under MIT — or under the original upstream license (BSD-2-Clause + AOM
Patent License 1.0). Contact support@imazen.io.
