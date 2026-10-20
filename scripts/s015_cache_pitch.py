"""Cache fp32 teacher pitch targets (F0, energy) for every usable S015 train utterance.

    train utterances with >= 40 tokens (the trainer's pool), in batches of 8:
      teacher-cache inputs and S014-student inputs (s015_train.Batcher.inputs)
      -> fp32 Kokoro F0Ntrain on the full utterance -> f0, n [2F]
    -> artifacts/s015/cache/pitch/{teacher,s014}/{f0,n,offsets}.npy + meta.json
       offsets has one entry per train utterance (+1); unused utterances are empty

The S014 student is padding-invariant (batch vs alone: same frames, max diff about 1e-6),
so a cached target matches the live one; the trainer still checks the frame count.
Audio targets are not cached: crops and sine noise are drawn fresh every step.

Usage: s015_cache_pitch.py --device cuda [--variant teacher|s014]  (one process per variant can run in parallel)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
from s014_student import Student  # noqa: E402
from s014_train import Split  # noqa: E402
from s015_train import S014_STUDENT, Batcher, pitch_teacher  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/s015/cache/pitch"
MIN_TOKENS = 40


class PitchCache:
    """Read side: memory-mapped f0 / n per (variant, utterance)."""

    @staticmethod
    def available() -> bool:
        """Both variants must be complete; otherwise the trainer uses the live teacher."""
        return all(
            (CACHE / variant / f"{name}.npy").is_file()
            for variant in ("teacher", "s014")
            for name in ("offsets", "f0", "n")
        )

    def __init__(self, device):
        self.device = device
        self.parts = {}
        for variant in ("teacher", "s014"):
            root = CACHE / variant
            load = lambda n: np.load(root / f"{n}.npy", mmap_mode="r")
            self.parts[variant] = (load("offsets"), load("f0"), load("n"))

    def get(self, index: int, use_student: bool, frames: int):
        offsets, f0, n = self.parts["s014" if use_student else "teacher"]
        a, b = int(offsets[index]), int(offsets[index + 1])
        if b - a != 2 * frames:
            return None
        to = lambda x: torch.from_numpy(np.array(x[a:b], np.float32))[None].to(self.device)
        return to(f0), to(n)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", choices=("teacher", "s014"), help="only this variant (default both)")
    parser.add_argument("--limit", type=int, default=0, help="first N usable utterances only (probe)")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # exact fp32 teacher

    model = teacher.load_teacher(args.device)
    voice = teacher.load_voice()
    saved = torch.load(S014_STUDENT, map_location="cpu")
    s014 = Student(**saved["config"])
    s014.load_state_dict(saved["model"])
    s014 = s014.to(args.device).eval()
    split = Split("train")
    batcher = Batcher(split, voice, args.device, s014)
    usable = [i for i in range(len(split)) if split.lengths[i] >= MIN_TOKENS]
    if args.limit:
        usable = usable[: args.limit]

    start = time.time()
    for variant, use_student in (("teacher", False), ("s014", True)):
        if args.variant and variant != args.variant:
            continue
        lengths = np.zeros(len(split), np.int64)
        f0_parts, n_parts = {}, {}
        for k in range(0, len(usable), 8):
            indices = usable[k : k + 8]
            for i, item in zip(indices, batcher.inputs(indices, use_student)):
                f0, n = pitch_teacher(model, item)
                f0_parts[i] = f0[0].float().cpu().numpy()
                n_parts[i] = n[0].float().cpu().numpy()
                lengths[i] = f0_parts[i].shape[0]
            if k % 8000 == 0:
                print(json.dumps({"variant": variant, "done": k, "of": len(usable), "sec": round(time.time() - start)}),
                      flush=True)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        order = sorted(f0_parts)
        out = CACHE / variant
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "offsets.npy", offsets)
        np.save(out / "f0.npy", np.concatenate([f0_parts[i] for i in order]).astype(np.float32))
        np.save(out / "n.npy", np.concatenate([n_parts[i] for i in order]).astype(np.float32))
        meta = {"utterances": len(order), "values": int(offsets[-1]), "min_tokens": MIN_TOKENS,
                "device": args.device, "torch": torch.__version__, "sec": round(time.time() - start)}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps({"variant": variant, **meta}), flush=True)


if __name__ == "__main__":
    main()
