"""Run the fp32 Kokoro teacher once over a corpus split and store its targets.

    rows (train.jsonl + templates.jsonl, or holdout.jsonl)
    -> length-sorted batches of ~6k tokens -> batched teacher frontend (MPS)
    -> per token:
         ids            uint8
         duration       fp32   sigmoid(duration_logits).sum(-1), before rounding
         hidden  [512]  int8 + fp16 scale   duration_hidden without the style copy
         text    [512]  int8 + fp16 scale   text_features
    -> artifacts/s014/cache/<split>/*.npy + index.json (token offsets)

Per-token int8 scaling keeps about 40 dB, far finer than the student's error.
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

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/s014/corpus"
SPLITS = {"train": ("train.jsonl", "templates.jsonl"), "holdout": ("holdout.jsonl",)}


def quantize(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """x [N, C] -> int8 [N, C], fp16 scale [N]."""
    scale = x.abs().amax(-1).clamp(min=1e-8) / 127
    q = torch.round(x / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.cpu().numpy(), scale.half().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=tuple(SPLITS))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-tokens", type=int, default=6000)
    args = parser.parse_args()

    rows = [json.loads(line) for name in SPLITS[args.split] for line in open(CORPUS / name)]
    lengths = np.array([len(r["ids"]) + 2 for r in rows])  # with both pads
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    total = int(offsets[-1])
    out = ROOT / "artifacts/s014/cache" / args.split
    out.mkdir(parents=True, exist_ok=True)
    mm = lambda name, dtype, shape: np.lib.format.open_memmap(out / name, "w+", dtype, shape)
    ids_all = mm("ids.npy", np.uint8, (total,))
    dur_all = mm("duration.npy", np.float32, (total,))
    hid_all = mm("hidden_q.npy", np.int8, (total, 512))
    hid_scale = mm("hidden_scale.npy", np.float16, (total,))
    txt_all = mm("text_q.npy", np.int8, (total, 512))
    txt_scale = mm("text_scale.npy", np.float16, (total,))

    model = teacher.load_teacher(args.device)
    voice = teacher.load_voice()
    order = np.argsort(lengths)
    batches, current = [], []
    for i in order:
        if current and (len(current) + 1) * lengths[i] > args.batch_tokens:
            batches.append(current)
            current = []
        current.append(i)
    batches.append(current)

    start = time.time()
    for b, batch in enumerate(batches):
        L = int(lengths[batch].max())
        ids = np.zeros((len(batch), L), dtype=np.int64)
        for j, i in enumerate(batch):
            ids[j, : lengths[i]] = [0, *rows[i]["ids"], 0]
        style = teacher.style_rows(voice, [len(rows[i]["ids"]) for i in batch])
        hidden, logits, text = teacher.frontend(
            model,
            torch.from_numpy(ids).to(args.device),
            torch.from_numpy(lengths[batch]).to(args.device),
            torch.from_numpy(style).float().to(args.device),
        )
        duration = torch.sigmoid(logits).sum(-1)
        for j, i in enumerate(batch):
            n, o = int(lengths[i]), int(offsets[i])
            ids_all[o : o + n] = ids[j, :n]
            dur_all[o : o + n] = duration[j, :n].cpu().numpy()
            hid_all[o : o + n], hid_scale[o : o + n] = quantize(hidden[j, :n, :512])
            txt_all[o : o + n], txt_scale[o : o + n] = quantize(text[j, :, :n].T)
        if b % 50 == 0:
            done = int(lengths[np.concatenate(batches[: b + 1])].sum())
            print(f"batch {b}/{len(batches)} tokens {done}/{total} {time.time() - start:.0f}s", flush=True)

    for array in (ids_all, dur_all, hid_all, hid_scale, txt_all, txt_scale):
        array.flush()
    (out / "index.json").write_text(json.dumps({
        "offsets": offsets.tolist(),
        "books": [r["book"] for r in rows],
        "tokens": total,
        "seconds": round(time.time() - start, 1),
    }))
    print(f"done {args.split}: {len(rows)} utterances, {total} tokens, {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
