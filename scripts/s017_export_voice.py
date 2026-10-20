"""Write the S017 fp32 voice table from the upstream af_heart voice.

    python scripts/s017_export_voice.py [--voice models/pytorch/voices/af_heart.pt]
        -> artifacts/s017/af_heart_f32.bin   (510 x 256 little-endian float32 = 522,240 bytes)
    python scripts/s017_bench_inputs.py
        -> artifacts/s017/bench-s017.json    (frozen token ids and sample counts; needs the same voice file)

Upstream: hexgrad/Kokoro-82M revision f3ff3571791e39611d31c381e3a41a3af07b4987,
voices/af_heart.pt, SHA-256 0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff.
The expected output hashes are in MANIFEST.json; this script checks them.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VOICE_SHA256 = "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", type=Path, default=ROOT / "models/pytorch/voices/af_heart.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/s017/af_heart_f32.bin")
    args = parser.parse_args()

    got = hashlib.sha256(args.voice.read_bytes()).hexdigest()
    if got != VOICE_SHA256:
        raise SystemExit(f"{args.voice}: sha256 {got} is not the pinned upstream af_heart.pt")
    voice = torch.load(args.voice, map_location="cpu")  # [510, 1, 256] float32
    assert tuple(voice.shape) == (510, 1, 256) and voice.dtype == torch.float32
    data = np.ascontiguousarray(voice.squeeze(1).numpy().astype("<f4")).tobytes()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(args.out, len(data), digest)
    manifest = ROOT / "MANIFEST.json"
    if manifest.exists():
        want = json.loads(manifest.read_text())["voice_table"]["sha256"]
        print("matches MANIFEST.json" if digest == want else f"DIFFERS from MANIFEST.json ({want})")


if __name__ == "__main__":
    main()
