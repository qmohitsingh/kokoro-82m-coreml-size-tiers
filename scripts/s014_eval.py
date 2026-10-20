"""Objective diagnostics for an S014 student checkpoint (PyTorch, fp32 downstream).

    for each utterance (5 frozen chunks, plus N validation utterances):
      teacher:  fp32 front end -> durations -> F0Ntrain -> decoder   (seed k)
      student:  student front end -> its durations -> same frozen downstream (seed k)
      noise:    teacher again with seed k+1, the floor for audio distances
      aligned:  student features on the teacher's durations, for frame-level
                F0 / energy error and waveform SNR (equal lengths)
    -> per utterance and mean: exact-duration rate, total frames, sample count
       check (600 x frames), F0 / energy error, DTW log-mel, waveform SNR,
       peak, RMS, clipping
    -> artifacts/s014/eval/<run>/{frozen,val}.json (+ frozen wavs, not listened to)

The sealed hidden test is never read here. No intelligibility/WER: no offline
ASR model is installed; WER waits for the candidate stage.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
from logmel_dtw import dtw_logmel_db  # noqa: E402
from s014_student import Student  # noqa: E402
from s014_train import Split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run(model, hidden, text, frames, style, seed):
    n = frames.shape[0]
    aln = teacher.alignment(frames, n, int(frames.sum()))
    torch.manual_seed(seed)
    f0, energy, audio = teacher.synthesize(model, hidden, text, aln, style)
    return f0[0], energy[0], audio.float().cpu().numpy()


def level_db(x: np.ndarray) -> float:
    return 20 * math.log10(max(float(np.sqrt(np.mean(x**2))), 1e-9))


@torch.no_grad()
def compare(model, student, ids: list[int], style_row: np.ndarray, device: str, seed: int) -> dict:
    ids_t = torch.tensor([ids], device=device)
    style = torch.from_numpy(style_row[None].astype(np.float32)).to(device)
    lengths = torch.tensor([len(ids)], device=device)
    h_t, logits_t, text_t = teacher.frontend(model, ids_t, lengths, style)
    h_s, logits_s, text_s = student(ids_t, style)
    f_t = teacher.frames_from_logits(logits_t)[0]
    f_s = teacher.frames_from_logits(logits_s)[0]

    f0_t, n_t, audio_t = run(model, h_t, text_t, f_t, style, seed)
    _, _, audio_noise = run(model, h_t, text_t, f_t, style, seed + 1)
    _, _, audio_s = run(model, h_s, text_s, f_s, style, seed)
    f0_a, n_a, audio_a = run(model, h_s, text_s, f_t, style, seed)
    _, _, audio_d = run(model, h_t, text_t, f_s, style, seed)
    band = lambda a, b: 64 + abs(len(a) - len(b)) // 256 + 1

    voiced = f0_t > 10
    snr = 10 * math.log10(float(np.sum(audio_t**2)) / max(float(np.sum((audio_t - audio_a) ** 2)), 1e-12))
    return dict(
        tokens=len(ids),
        duration_exact=float((f_s == f_t).float().mean()),
        frames_teacher=int(f_t.sum()),
        frames_student=int(f_s.sum()),
        samples_student=len(audio_s),
        samples_ok=len(audio_s) == 600 * int(f_s.sum()),
        f0_mae_hz=float((f0_a - f0_t).abs()[voiced].mean()),
        f0_voicing_agree=float(((f0_a > 10) == voiced).float().mean()),
        energy_mae=float((n_a - n_t).abs().mean()),
        # The band must cover the length difference, or no path exists.
        dtw_logmel_db=dtw_logmel_db(audio_t, audio_s, band=band(audio_t, audio_s))[0],
        # Split: student features on teacher durations; teacher features on student durations.
        dtw_features_only_db=dtw_logmel_db(audio_t, audio_a)[0],
        dtw_durations_only_db=dtw_logmel_db(audio_t, audio_d, band=band(audio_t, audio_d))[0],
        dtw_noise_floor_db=dtw_logmel_db(audio_t, audio_noise)[0],
        waveform_snr_db=snr,
        peak_teacher=float(np.abs(audio_t).max()),
        peak_student=float(np.abs(audio_s).max()),
        rms_diff_db=level_db(audio_s) - level_db(audio_t),
        clipped_samples=int((np.abs(audio_s) >= 0.999).sum()),
    ), audio_t, audio_s


def summary(rows: list[dict]) -> dict:
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["samples_ok_all"] = all(r["samples_ok"] for r in rows)
    out["total_rel_median"] = float(np.median([abs(r["frames_student"] - r["frames_teacher"]) / r["frames_teacher"] for r in rows]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--val", type=int, default=100, help="validation utterances (0 to skip)")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu")
    student = Student(**saved["config"])
    student.load_state_dict(saved["model"])
    student = student.to(args.device).eval()
    model = teacher.load_teacher(args.device)
    voice = teacher.load_voice()
    name = f"{args.checkpoint.parent.name}-{args.checkpoint.stem}"
    out = ROOT / "artifacts/s014/eval" / name
    (out / "frozen").mkdir(parents=True, exist_ok=True)

    manifest = json.loads((ROOT / "benchmark/inputs/manifest.json").read_text())
    rows = []
    for item in manifest["items"]:
        for chunk in item["chunks"]:
            frozen = np.load(ROOT / "benchmark/inputs" / chunk["file"])
            ids = frozen["input_ids"][0].tolist()
            row, audio_t, audio_s = compare(model, student, ids, frozen["style"][0], args.device, 1234)
            row["chunk"] = chunk["file"]
            rows.append(row)
            stem = Path(chunk["file"]).stem
            sf.write(out / "frozen" / f"{stem}-teacher.wav", audio_t, 24000)
            sf.write(out / "frozen" / f"{stem}-student.wav", audio_s, 24000)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    (out / "frozen.json").write_text(json.dumps({"rows": rows, "mean": summary(rows)}, indent=2))
    print("frozen mean", json.dumps({k: round(v, 4) for k, v in summary(rows).items()}), flush=True)

    if args.val:
        split = Split("holdout")
        pick = np.random.default_rng(0).choice(len(split), args.val, replace=False)
        rows = []
        for k, i in enumerate(pick):
            o, n = int(split.offsets[i]), int(split.lengths[i])
            ids = split.ids[o : o + n].astype(int).tolist()
            style_row = teacher.style_rows(voice, [n - 2])[0]
            rows.append(compare(model, student, ids, style_row, args.device, 1234 + k)[0])
        (out / "val.json").write_text(json.dumps({"rows": rows, "mean": summary(rows)}, indent=2))
        print("val mean", json.dumps({k: round(v, 4) for k, v in summary(rows).items()}), flush=True)


if __name__ == "__main__":
    main()
