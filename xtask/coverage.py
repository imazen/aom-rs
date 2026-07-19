#!/usr/bin/env python3
"""Auto-derive the coverage-gate feature checklist from libaom's live CLI +
control-enum surface, cross-reference it with the hand-authored feature->test
mapping, and print a red/green summary.

The coverage gate is "done" only when every enumerated feature maps to a passing
test. This tool makes that mechanically measurable (and, right now, honestly
shows the large gap). It does NOT invent green: a feature is green only if
coverage/feature_map.json maps it to a test id.

Usage: python3 xtask/coverage.py [--ref <libaom_dir>]
Outputs coverage/features.json and a summary to stdout.
"""
import json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The pinned libaom C oracle lives in the `upstream/` git submodule (repo-reorg
# #3 Phase 1); its aomenc/aomdec + headers are under `upstream/build`. Override
# with `--ref <dir>` (e.g. the fallback `reference/libaom`).
REF = os.path.join(ROOT, "upstream")
if "--ref" in sys.argv:
    REF = sys.argv[sys.argv.index("--ref") + 1]

def cli_options(tool):
    try:
        out = subprocess.run([os.path.join(REF, "build", tool), "--help"],
                             capture_output=True, text=True).stderr or ""
        out2 = subprocess.run([os.path.join(REF, "build", tool), "--help"],
                              capture_output=True, text=True).stdout or ""
        text = out + out2
    except FileNotFoundError:
        return []
    opts = sorted(set(re.findall(r"^\s+(--[a-zA-Z0-9-]+)", text, re.M)))
    return opts

def control_enums():
    hdr = os.path.join(REF, "aom", "aomcx.h")
    if not os.path.exists(hdr):
        return []
    text = open(hdr).read()
    return sorted(set(re.findall(r"\b(AV1E_SET_[A-Z0-9_]+|AOME_SET_[A-Z0-9_]+)\b", text)))

def build_surface():
    feats = {}
    for o in cli_options("aomenc"):
        feats[f"cli.enc{o}"] = {"kind": "cli-enc", "name": o}
    for o in cli_options("aomdec"):
        feats[f"cli.dec{o}"] = {"kind": "cli-dec", "name": o}
    for e in control_enums():
        feats[f"ctrl.{e}"] = {"kind": "control", "name": e}
    return feats

def main():
    surface = build_surface()
    map_path = os.path.join(ROOT, "coverage", "feature_map.json")
    mapping = json.load(open(map_path)) if os.path.exists(map_path) else {}

    green = 0
    for key, item in surface.items():
        m = mapping.get(key)
        if m:
            item["status"] = "green"
            item["test"] = m
            green += 1
        else:
            item["status"] = "red"

    total = len(surface)
    out = {
        "note": "AUTO-DERIVED by xtask/coverage.py from live aomenc/aomdec --help "
                "+ aomcx.h control enums. Green requires a mapping in feature_map.json "
                "to a passing test. Low-level kernel coverage is tracked separately "
                "in checklist.json.",
        "reference": "libaom v3.14.1",
        "summary": {"total": total, "green": green, "red": total - green,
                    "percent": round(100.0 * green / total, 2) if total else 0.0},
        "features": surface,
    }
    with open(os.path.join(ROOT, "coverage", "features.json"), "w") as f:
        json.dump(out, f, indent=2)

    s = out["summary"]
    print(f"Coverage gate (feature surface): {s['green']}/{s['total']} green "
          f"({s['percent']}%), {s['red']} red")
    by_kind = {}
    for it in surface.values():
        k = it["kind"]
        by_kind.setdefault(k, [0, 0])
        by_kind[k][0] += 1
        if it["status"] == "green":
            by_kind[k][1] += 1
    for k, (t, g) in sorted(by_kind.items()):
        print(f"  {k:10} {g}/{t}")
    print("\nGate is GREEN only at 100%. Kernel-level differential coverage "
          "(transform/quant/entropy/intra/loopfilter/dist) is in checklist.json.")

if __name__ == "__main__":
    main()
