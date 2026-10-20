"""Make the S014 blind listening clips for the Mac arms.

Two steps, two venvs. The hidden sentence text never leaves step 1's memory:
no text, phonemes, or text-derived file names are written or printed.

  1. .venv-convert/bin/python -I scripts/s014_listen_prep.py ids
       hidden_test/sentences.txt (sha256 checked) -> espeak en-us phonemes
       -> token ids with pad 0 at both ends (as s014_cache_teacher.py)
       -> artifacts/s014/listen/ids.json   [{id, category, ids}]
  2. .venv-convert/bin/python -B scripts/s014_listen_prep.py audio
       ids.json -> style row, reference arm (fp32 Kokoro) and student arm
       (w256-b step10000) through the same fp32 downstream, seed 1000 + index
       -> artifacts/s014/listen/{ref,student}/<id>.wav (24 kHz 16-bit mono,
          no loudness normalization), sentences.json, key.json, mac_stats.json
  3. ... -B scripts/s014_listen_prep.py check   (after the phone clips are pulled)
       -> artifacts/s014/listen/s014_check.json

The hidden set is used only to make listening clips. Never for tuning.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
LISTEN = ROOT / "artifacts/s014/listen"
HIDDEN = ROOT / "artifacts/s014/hidden_test/sentences.txt"
HIDDEN_SHA = "ec1902203acbb1013f38daa9d525586feba19d2469df85d8bb0546efcfbe8167"
CHECKPOINT = ROOT / "artifacts/s014/runs/w256-b/step10000.pt"
MAX_TOKENS = 508  # 510 with both pads, same bound as the corpus (3..508)


def make_ids() -> None:
    data = HIDDEN.read_bytes()
    if hashlib.sha256(data).hexdigest() != HIDDEN_SHA:
        raise SystemExit("hidden_test/sentences.txt sha256 mismatch")
    rows = []
    for line in data.decode("utf-8").split("\n"):
        if line and not line.startswith("#"):
            category, text = line.split("\t", 1)
            rows.append((category, text))
    from s014_corpus import phonemize_batch

    phon = phonemize_batch([t for _, t in rows])
    out, dropped = [], 0
    for index, ((category, _), (_, ids)) in enumerate(zip(rows, phon)):
        if len(ids) > MAX_TOKENS or len(ids) < 1:
            dropped += 1
            continue
        out.append({"id": f"h{index:03d}", "index": index, "category": category, "ids": [0, *ids, 0]})
    LISTEN.mkdir(parents=True, exist_ok=True)
    (LISTEN / "ids.json").write_text(json.dumps(out))
    print(f"sentences {len(rows)}, kept {len(out)}, dropped {dropped}")


def level_db(x) -> float:
    import numpy as np

    return 20 * math.log10(max(float(np.sqrt(np.mean(x.astype("float64") ** 2))), 1e-9))


def make_audio(device: str) -> None:
    import numpy as np
    import soundfile as sf
    import torch

    import s014_teacher as teacher
    from s014_eval import run
    from s014_student import Student

    items = json.loads((LISTEN / "ids.json").read_text())
    saved = torch.load(CHECKPOINT, map_location="cpu")
    student = Student(**saved["config"])
    student.load_state_dict(saved["model"])
    student = student.to(device).eval()
    model = teacher.load_teacher(device)
    voice = teacher.load_voice()
    for arm in ("ref", "student"):
        (LISTEN / arm).mkdir(parents=True, exist_ok=True)

    sentences, stats = [], []
    for item in items:
        ids = item["ids"]
        style_row = teacher.style_rows(voice, [len(ids) - 2])[0]
        seed = 1000 + item["index"]
        with torch.no_grad():
            ids_t = torch.tensor([ids], device=device)
            style = torch.from_numpy(style_row[None].astype(np.float32)).to(device)
            lengths = torch.tensor([len(ids)], device=device)
            h_t, logits_t, text_t = teacher.frontend(model, ids_t, lengths, style)
            h_s, logits_s, text_s = student(ids_t, style)
            f_t = teacher.frames_from_logits(logits_t)[0]
            f_s = teacher.frames_from_logits(logits_s)[0]
            _, _, audio_t = run(model, h_t, text_t, f_t, style, seed)
            _, _, audio_s = run(model, h_s, text_s, f_s, style, seed)
        for arm, audio, frames in (("ref", audio_t, int(f_t.sum())), ("student", audio_s, int(f_s.sum()))):
            assert len(audio) == 600 * frames, (item["id"], arm, len(audio), frames)
            assert np.isfinite(audio).all() and np.abs(audio).max() < 1.0, (item["id"], arm)
            sf.write(LISTEN / arm / f"{item['id']}.wav", audio, 24000, subtype="PCM_16")
        sentences.append({
            "id": item["id"], "input_ids": ids, "style": [float(v) for v in style_row],
            "category": item["category"],
        })
        stats.append({
            "id": item["id"], "tokens": len(ids), "frames_ref": int(f_t.sum()), "frames_student": int(f_s.sum()),
            "peak_ref": float(np.abs(audio_t).max()), "peak_student": float(np.abs(audio_s).max()),
            "rms_db_ref": level_db(audio_t), "rms_db_student": level_db(audio_s),
        })
        print(item["id"], stats[-1]["frames_ref"], stats[-1]["frames_student"], flush=True)

    (LISTEN / "sentences.json").write_text(json.dumps(sentences))
    (LISTEN / "mac_stats.json").write_text(json.dumps(stats, indent=1))
    # Analysis key only. Nothing here reaches the app UI. Sentence text is not stored.
    (LISTEN / "key.json").write_text(json.dumps({
        "arms": {
            "student": "ref (original fp32 Kokoro) vs w256-b step10000 student, fp32 downstream, Mac MPS, seed 1000+index",
            "s014": "ref vs the S014 Core ML chain on iPhone 12 (student w256 int/fp16 + int8 pitch + int4 decoder/generator)",
        },
        "ref": "original fp32 Kokoro af_heart, en-us, seed 1000+index",
        "sentences": [{"id": s["id"], "category": s["category"], "tokens": len(s["input_ids"])} for s in sentences],
    }, indent=1))
    print(f"wrote {len(sentences)} clips per arm")


def check_s014(arm: str = "s014") -> None:
    import numpy as np
    import soundfile as sf

    phone = json.loads((LISTEN / arm / "frames.json").read_text())
    mac = {s["id"]: s for s in json.loads((LISTEN / "mac_stats.json").read_text())}
    rows, fails = [], []
    for item_id, info in sorted(phone.items()):
        path = LISTEN / arm / f"{item_id}.wav"
        if not path.exists():
            fails.append((item_id, "missing wav"))
            continue
        audio, rate = sf.read(path, dtype="float64")
        ref, _ = sf.read(LISTEN / "ref" / f"{item_id}.wav", dtype="float64")
        frames = info["frames"]
        row = {
            "id": item_id, "rate": rate, "samples": len(audio), "frames_phone": frames,
            "frames_mac_student": mac[item_id]["frames_student"], "frames_ref": mac[item_id]["frames_ref"],
            "length_ok": len(audio) == 600 * frames, "nan": bool(not np.isfinite(audio).all()),
            "peak": float(np.abs(audio).max()), "seconds": len(audio) / rate,
            "rms_db": level_db(audio), "rms_db_ref": level_db(ref),
        }
        row["rms_diff_db"] = row["rms_db"] - row["rms_db_ref"]
        rows.append(row)
        if not row["length_ok"] or row["nan"] or row["peak"] >= 1.0 or rate != 24000:
            fails.append((item_id, "length/nan/peak/rate"))
    diffs = [abs(r["rms_diff_db"]) for r in rows]
    summary = {
        "count": len(rows), "expected": len(phone), "failures": fails,
        "mean_abs_rms_diff_db": float(np.mean(diffs)), "max_abs_rms_diff_db": float(np.max(diffs)),
        "mean_signed_rms_diff_db": float(np.mean([r["rms_diff_db"] for r in rows])),
        "under_0.5s": [r["id"] for r in rows if r["seconds"] < 0.5],
        "silent": [r["id"] for r in rows if r["rms_db"] < -60],
        "max_peak": max(r["peak"] for r in rows),
        "frames_equal_mac_student": sum(r["frames_phone"] == r["frames_mac_student"] for r in rows),
        "frames_phone_vs_mac_student_max_abs_diff": max(abs(r["frames_phone"] - r["frames_mac_student"]) for r in rows),
    }
    (LISTEN / f"{arm}_check.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "ids":
        make_ids()
    elif mode == "audio":
        make_audio(sys.argv[2] if len(sys.argv) > 2 else "mps")
    elif mode == "check":
        check_s014(sys.argv[2] if len(sys.argv) > 2 else "s014")  # s014-m8: mixed8 generator
    else:
        raise SystemExit("usage: s014_listen_prep.py ids|audio [device]|check [arm]")
