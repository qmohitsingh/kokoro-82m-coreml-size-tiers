"""Run the fp32 Kokoro teacher ONCE over the recipe corpus and store its targets.

Same logic and file layout as s014_cache_teacher.py, with one change: every
utterance is cached once per voice, because the duration path depends on the
style row. Rows are (utterance, voice) pairs. The teacher is the PyTorch
checkpoint s014_teacher.py loads (models/pytorch/kokoro-v1_0.pth + config.json).

    python scripts/s018_cache_teacher.py train   --work runs-s018 --voice-dir voices
    python scripts/s018_cache_teacher.py heldout --work runs-s018 --voice-dir voices   # validation only, never trained on

Style row = voice[token_count - 1] from 510x256 fp32 voice tables, one file per
voice named <voice-dir>/<voice>_f32.bin. Input: <work>/<corpus>/train.jsonl or
<work>/heldout/tokens.json; output: <work>/cache/<name>/.
Per token: ids uint8, duration fp32 (sigmoid sum, before rounding),
hidden int8+fp16 scale (512), text int8+fp16 scale (512).
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
from s014_cache_teacher import quantize  # noqa: E402

DEFAULT_VOICES = ("af_heart",)


def load_voices(voice_dir: Path, voices=DEFAULT_VOICES) -> dict[str, np.ndarray]:
    return {v: np.fromfile(Path(voice_dir) / f"{v}_f32.bin", dtype="<f4").reshape(510, 256) for v in voices}


def read_rows(work: Path, split: str, corpus: str = "corpus") -> list[dict]:
    if split == "train":
        return [json.loads(l) for l in open(work / corpus / "train.jsonl")]
    items = json.load(open(work / "heldout/tokens.json"))
    return [dict(text=i["text"], ids=i["ids"][1:-1]) for i in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("train", "heldout"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--voice-dir", type=Path, required=True, help="folder with <voice>_f32.bin tables")
    parser.add_argument("--voices", nargs="+", default=list(DEFAULT_VOICES))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-tokens", type=int, default=6000)
    parser.add_argument("--corpus", default="corpus", help="<work>/<corpus>/train.jsonl")
    parser.add_argument("--cache-name", help="output dir under <work>/cache (default: the split)")
    parser.add_argument("--limit", type=int, help="smoke test: first N utterances")
    args = parser.parse_args()
    VOICES = tuple(args.voices)

    utts = read_rows(args.work, args.split, args.corpus)[: args.limit]
    rows = [(u, v) for v in VOICES for u in utts]  # row = (utterance, voice)
    lengths = np.array([len(u["ids"]) + 2 for u, _ in rows])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    total = int(offsets[-1])
    out = args.work / "cache" / ((args.cache_name or args.split) + ("_smoke" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    mm = lambda name, dtype, shape: np.lib.format.open_memmap(out / name, "w+", dtype, shape)
    ids_all = mm("ids.npy", np.uint8, (total,))
    dur_all = mm("duration.npy", np.float32, (total,))
    hid_all = mm("hidden_q.npy", np.int8, (total, 512))
    hid_scale = mm("hidden_scale.npy", np.float16, (total,))
    txt_all = mm("text_q.npy", np.int8, (total, 512))
    txt_scale = mm("text_scale.npy", np.float16, (total,))

    model = teacher.load_teacher(args.device)
    voices = load_voices(args.voice_dir, VOICES)
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
            ids[j, : lengths[i]] = [0, *rows[i][0]["ids"], 0]
        style = np.stack([voices[rows[i][1]][len(rows[i][0]["ids"]) - 1] for i in batch])
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
            print(f"batch {b}/{len(batches)} {time.time() - start:.0f}s", flush=True)

    for array in (ids_all, dur_all, hid_all, hid_scale, txt_all, txt_scale):
        array.flush()
    (out / "index.json").write_text(json.dumps({
        "offsets": offsets.tolist(),
        "voices": [v for _, v in rows],
        "texts": [u["text"] for u, _ in rows],
        "tokens": total,
        "seconds": round(time.time() - start, 1),
    }))
    print(f"done {args.split}: {len(rows)} rows ({len(utts)} utterances x {len(VOICES)} voices), {total} tokens, {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
