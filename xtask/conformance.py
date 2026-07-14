#!/usr/bin/env python3
"""Define, fetch, and categorize the official AV1 decode-conformance corpus
(Gate 1's authoritative test set) from libaom's own test-data manifest.

libaom ships the canonical vector list as name+sha1 pairs in
`reference/libaom/test/test-data.sha1`, and hosts the bytes at
`https://storage.googleapis.com/aom-test-data/<name>`. Each `av1-1-*.ivf`
vector has a companion `<name>.md5` holding one MD5 per decoded frame over the
raw i420 output -- that per-frame MD5 list is the golden Gate-1 answer libaom's
own decode tests assert against. Our decoder reproduces those frames and MD5s.

This tool does NOT decode with our Rust crates (that comparison test lives in
`crates/aom-decode/tests` and is owned by the decoder track). It only:
  * parses the sha1 manifest and categorizes the AV1 vectors by bit-depth +
    feature family + a coarse decode-scope hint (intra-now vs inter-later),
  * writes `conformance/vectors.json` -- the committed, authoritative Gate-1
    corpus definition,
  * `--fetch` downloads + sha1-verifies vectors (.ivf + .md5) into
    `conformance/data/` (gitignored -- bytes never enter git),
  * `--probe` runs the C `aomdec` to record decoded-frame counts so scope is
    measured, not guessed.

Usage:
  python3 xtask/conformance.py                 # (re)build conformance/vectors.json
  python3 xtask/conformance.py --fetch --family 02-allintra
  python3 xtask/conformance.py --fetch --scope intra --limit 8
  python3 xtask/conformance.py --probe          # probe already-fetched vectors
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "reference", "libaom")
SHA1_MANIFEST = os.path.join(REF, "test", "test-data.sha1")
DATA_URL = "https://storage.googleapis.com/aom-test-data"
DATA_DIR = os.path.join(ROOT, "conformance", "data")
OUT_JSON = os.path.join(ROOT, "conformance", "vectors.json")
AOMDEC = os.path.join(REF, "build", "aomdec")

# Feature-family -> coarse decode scope. "intra" families decode with only
# intra/KEY tooling (in scope for the current decoder); "inter" needs motion
# compensation / multi-ref (not yet ported); "special" needs an extra tool
# (superres, film grain, monochrome, mid-stream CDF carry).
FAMILY_SCOPE = {
    "00-quantizer": "intra",     # per-qindex KEY sweep
    "01-size": "intra",          # frame-dimension conformance (intra frames)
    "02-allintra": "intra",      # AOM_USAGE_ALL_INTRA -- our primary target
    "16-intra": "intra",         # intra-only, incl. intrabc extreme-dv
    "03-sizeup": "special",      # superres upscale
    "03-sizedown": "special",    # superres downscale
    "04-cdfupdate": "special",   # cross-frame CDF carry
    "05-mv": "inter",
    "06-mfmv": "inter",
    "22-svc": "inter",
    "23-film": "special",        # film grain synthesis
    "24-monochrome": "special",  # monochrome (no chroma planes)
}


def parse_sha1():
    """name -> sha1 for every entry in libaom's test-data manifest."""
    out = {}
    with open(SHA1_MANIFEST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sha, name = line.split(None, 1)
            out[name.lstrip("*")] = sha
    return out


def categorize(name):
    """(bitdepth, family, scope_hint) parsed from an av1-1 vector name."""
    m = re.match(r"av1-1-b(8|10)-(\d\d-[a-zA-Z]+)", name)
    if not m:
        return None
    bitdepth = int(m.group(1))
    family = m.group(2).lower()
    # Normalize the "03-size{up,down}" split which encodes direction after 03-.
    scope = FAMILY_SCOPE.get(family, "unknown")
    return bitdepth, family, scope


def build_manifest():
    sha1 = parse_sha1()
    vectors = {}
    families = {}
    for name, sha in sha1.items():
        if not (name.startswith("av1-1-") and name.endswith(".ivf")):
            continue
        cat = categorize(name)
        if cat is None:
            continue
        bitdepth, family, scope = cat
        md5_name = name + ".md5"
        vectors[name] = {
            "sha1": sha,
            "md5_file": md5_name,
            "md5_file_sha1": sha1.get(md5_name),
            "bitdepth": bitdepth,
            "family": family,
            "scope_hint": scope,
            "url": f"{DATA_URL}/{name}",
        }
        families.setdefault(family, {"count": 0, "scope": scope})
        families[family]["count"] += 1

    by_scope = {}
    for v in vectors.values():
        by_scope[v["scope_hint"]] = by_scope.get(v["scope_hint"], 0) + 1

    out = {
        "note": "AUTHORITATIVE Gate-1 AV1 decode-conformance corpus, derived from "
                "libaom's own test/test-data.sha1. Bytes live in conformance/data/ "
                "(gitignored); fetch with xtask/conformance.py --fetch. Each vector's "
                ".md5 companion is the per-frame golden our decoder must reproduce. "
                "scope_hint is a family heuristic; the 'probe' field (frames/decoded) "
                "is measured by C aomdec via --probe.",
        "reference": "libaom v3.14.1",
        "data_url": DATA_URL,
        "summary": {
            "total_av1_vectors": len(vectors),
            "by_scope": by_scope,
            "families": families,
        },
        "vectors": dict(sorted(vectors.items())),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    # Preserve any prior probe results so --probe data survives a rebuild.
    if os.path.exists(OUT_JSON):
        try:
            prior = json.load(open(OUT_JSON)).get("vectors", {})
            for n, v in out["vectors"].items():
                if n in prior and "probe" in prior[n]:
                    v["probe"] = prior[n]["probe"]
        except (ValueError, OSError):
            pass
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    return out


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(name, expect_sha1):
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, name)
    if os.path.exists(dest) and expect_sha1 and sha1_file(dest) == expect_sha1:
        return "cached"
    url = f"{DATA_URL}/{name}"
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    if expect_sha1:
        got = sha1_file(tmp)
        if got != expect_sha1:
            os.remove(tmp)
            raise RuntimeError(f"sha1 mismatch for {name}: got {got} want {expect_sha1}")
    os.replace(tmp, dest)
    return "fetched"


def cmd_fetch(manifest, family=None, scope=None, limit=None):
    picks = []
    for name, v in manifest["vectors"].items():
        if family and v["family"] != family:
            continue
        if scope and v["scope_hint"] != scope:
            continue
        picks.append((name, v))
    if limit:
        picks = picks[:limit]
    if not picks:
        print("no vectors match filter", file=sys.stderr)
        return
    print(f"fetching {len(picks)} vector(s) (+ .md5) into {DATA_DIR}")
    for name, v in picks:
        st = fetch_one(name, v["sha1"])
        st2 = fetch_one(v["md5_file"], v.get("md5_file_sha1"))
        print(f"  {st:8} {name}   ({st2} .md5)")


def cmd_probe(manifest):
    """Record decoded-frame count per fetched vector via C aomdec."""
    if not os.path.isfile(AOMDEC):
        print(f"aomdec not built at {AOMDEC}", file=sys.stderr)
        return
    n = 0
    for name, v in manifest["vectors"].items():
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            continue
        r = subprocess.run([AOMDEC, "--summary", "-o", os.devnull, path],
                           capture_output=True, text=True)
        m = re.search(r"(\d+)\s+decoded frames/(\d+)\s+showed frames", r.stderr)
        frames = int(m.group(1)) if m else None
        v["probe"] = {"decoded_frames": frames, "ok": r.returncode == 0}
        n += 1
        print(f"  probed {name}: frames={frames} ok={r.returncode == 0}")
    with open(OUT_JSON, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"probed {n} fetched vector(s)")


def main():
    manifest = build_manifest()
    s = manifest["summary"]
    if "--fetch" in sys.argv:
        fam = None
        sc = None
        lim = None
        if "--family" in sys.argv:
            fam = sys.argv[sys.argv.index("--family") + 1]
        if "--scope" in sys.argv:
            sc = sys.argv[sys.argv.index("--scope") + 1]
        if "--limit" in sys.argv:
            lim = int(sys.argv[sys.argv.index("--limit") + 1])
        cmd_fetch(manifest, family=fam, scope=sc, limit=lim)
        return
    if "--probe" in sys.argv:
        cmd_probe(manifest)
        return

    print(f"AV1 decode-conformance corpus (Gate 1): {s['total_av1_vectors']} vectors")
    print("  by decode-scope hint:")
    for k, c in sorted(s["by_scope"].items()):
        print(f"    {k:9} {c}")
    print("  by family:")
    for fam, info in sorted(s["families"].items()):
        print(f"    {fam:16} {info['count']:4}  ({info['scope']})")
    print(f"\nwrote {os.path.relpath(OUT_JSON, ROOT)}")
    print("fetch in-scope intra vectors:  python3 xtask/conformance.py "
          "--fetch --scope intra --limit 8 && python3 xtask/conformance.py --probe")


if __name__ == "__main__":
    main()
