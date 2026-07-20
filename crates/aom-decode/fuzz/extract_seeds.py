#!/usr/bin/env python3
"""Extract tiny AV1 OBU temporal-unit seeds from the conformance IVF vectors.

Seeds are RAW OBU bytes (the IVF container is stripped), matching what
`decode_frames` / `decode_frame_obus` parse. Kept small (<8 KB each) and
few (hand-curated) per the fuzz-corpus discipline: seeds go in git, the
working corpus does not.
"""
import os
import struct
import sys

SRC = "/root/aom-rs/conformance/data"
BASE = os.path.dirname(os.path.abspath(__file__))
SEED_FRAMES = os.path.join(BASE, "seeds", "decode_frames")
SEED_OBUS = os.path.join(BASE, "seeds", "decode_obus")
os.makedirs(SEED_FRAMES, exist_ok=True)
os.makedirs(SEED_OBUS, exist_ok=True)


def temporal_units(data: bytes):
    assert data[:4] == b"DKIF", "not an IVF file"
    hdr_len = struct.unpack_from("<H", data, 6)[0]
    off = hdr_len
    tus = []
    while off + 12 <= len(data):
        sz = struct.unpack_from("<I", data, off)[0]
        off += 12
        if off + sz > len(data):
            break
        tus.append(data[off : off + sz])
        off += sz
    return tus


# (vector name, how many leading TUs to concatenate for the decode_frames seed)
FRAMES_VECTORS = [
    ("av1-1-b8-01-size-16x16", 2),
    ("av1-1-b8-01-size-18x16", 2),
    ("av1-1-b8-01-size-32x16", 2),
    ("av1-1-b8-01-size-34x16", 3),
    ("av1-1-b8-01-size-18x34", 3),
    ("av1-1-b8-01-size-66x66", 2),
]

# KEY-frame-only single-TU seeds for decode_obus.
OBUS_VECTORS = [
    "av1-1-b8-01-size-16x16",
    "av1-1-b8-01-size-18x16",
    "av1-1-b8-01-size-32x16",
    "av1-1-b8-01-size-34x16",
    "av1-1-b8-01-size-66x66",
    "av1-1-b8-00-quantizer-63",  # KEY frame 0: CDEF/LR/SB, high-q => small
    "av1-1-b10-00-quantizer-63",  # 10-bit KEY frame 0
]

MAX_SEED = 8192
total = 0

for name, nframes in FRAMES_VECTORS:
    p = os.path.join(SRC, name + ".ivf")
    if not os.path.exists(p):
        print(f"skip (absent): {name}", file=sys.stderr)
        continue
    tus = temporal_units(open(p, "rb").read())
    stream = b"".join(tus[:nframes])
    if len(stream) > MAX_SEED:
        continue
    out = os.path.join(SEED_FRAMES, f"{name}-f{nframes}.obu")
    open(out, "wb").write(stream)
    total += len(stream)
    print(f"frames seed {name}: {len(stream)} bytes ({nframes} TUs)")

for name in OBUS_VECTORS:
    p = os.path.join(SRC, name + ".ivf")
    if not os.path.exists(p):
        print(f"skip (absent): {name}", file=sys.stderr)
        continue
    tus = temporal_units(open(p, "rb").read())
    tu0 = tus[0]
    if len(tu0) > MAX_SEED:
        print(f"skip (KEY TU too big): {name} = {len(tu0)}", file=sys.stderr)
        continue
    out = os.path.join(SEED_OBUS, f"{name}-key.obu")
    open(out, "wb").write(tu0)
    total += len(tu0)
    print(f"obus seed {name}: {len(tu0)} bytes")

print(f"total seed bytes: {total}")
