"""Objective diagnostics for an S015 downstream checkpoint (PyTorch, fp32).

    for each utterance (5 frozen chunks, plus N holdout utterances):
      teacher:  fp32 Kokoro end to end (front end, durations, F0Ntrain, decoder)   seed k
      noise:    teacher again with seed k + 1, the floor for audio distances
      s014:     S014 student front end -> its durations -> fp32 Kokoro downstream    seed k
      s015:     S014 student front end -> its durations -> S015 pitch + vocoder      seed k
      s015_tf:  teacher front end + durations -> S015 pitch + vocoder (equal length,
                isolates the downstream: F0 / energy error, waveform SNR)
    -> finite, non-silent, samples = 600 x own frames, RMS vs teacher, clipping,
       DTW log-mel vs teacher, band energy 6-8 / 8-10 / 10-12 kHz vs teacher
    -> artifacts/s015/eval/<run>-<ckpt>/{frozen,val}.json (+ frozen wavs)

Metrics are diagnostics; the blind listening test decides. The sealed hidden
test is never read here.
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
from s015_downstream import PitchStudent, VocoderStudent  # noqa: E402
from s015_train import S014_STUDENT, sine_spectrum  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BANDS = ((6000, 8000), (8000, 10000), (10000, 12000))


def teacher_audio(model, hidden, text, frames, style, seed):
    aln = teacher.alignment(frames, frames.shape[0], int(frames.sum()))
    torch.manual_seed(seed)
    f0, n, audio = teacher.synthesize(model, hidden, text, aln, style)
    return f0[0], n[0], audio.float().cpu().numpy()


def s015_audio(model, pitch, vocoder, hidden, text, frames, style, seed):
    aln = teacher.alignment(frames, frames.shape[0], int(frames.sum()))
    f0, n = pitch(hidden.transpose(-1, -2) @ aln, style[:, 128:])
    torch.manual_seed(seed)
    har = sine_spectrum(model, f0)
    mag, phase = vocoder(text @ aln, f0, n, style[:, :128], har)
    audio = model.decoder.generator.stft.inverse(mag, phase).squeeze()
    return f0[0], n[0], audio.float().cpu().numpy()


def level_db(x: np.ndarray) -> float:
    return 20 * math.log10(max(float(np.sqrt(np.mean(x**2))), 1e-9))


def band_db(x: np.ndarray) -> list[float]:
    """Energy in each band relative to total energy, dB."""
    power = np.abs(np.fft.rfft(x)) ** 2
    hz = np.fft.rfftfreq(len(x), 1 / 24000)
    total = max(float(power.sum()), 1e-20)
    return [10 * math.log10(max(float(power[(hz >= lo) & (hz < hi)].sum()), 1e-20) / total) for lo, hi in BANDS]


def audio_stats(name: str, audio: np.ndarray, frames: int, ref: np.ndarray) -> dict:
    band = 64 + abs(len(ref) - len(audio)) // 256 + 1
    finite = bool(np.isfinite(audio).all())
    out = {
        f"{name}_finite": finite,
        f"{name}_samples_ok": len(audio) == 600 * frames,
        f"{name}_rms_db": level_db(audio),
        f"{name}_rms_diff_db": level_db(audio) - level_db(ref),
        f"{name}_peak": float(np.abs(audio).max()),
        f"{name}_clipped": int((np.abs(audio) >= 0.999).sum()),
        f"{name}_dtw_logmel_db": dtw_logmel_db(ref, audio, band=band)[0] if finite else float("nan"),
    }
    out[f"{name}_silent"] = out[f"{name}_rms_db"] < -50
    for (lo, hi), a, r in zip(BANDS, band_db(audio), band_db(ref)):
        out[f"{name}_band{lo // 1000}_{hi // 1000}k_diff_db"] = a - r
    return out


@torch.no_grad()
def compare(model, student, pitch, vocoder, ids, style_row, device, seed):
    ids_t = torch.tensor([ids], device=device)
    style = torch.from_numpy(style_row[None].astype(np.float32)).to(device)
    h_t, logits_t, text_t = teacher.frontend(model, ids_t, torch.tensor([len(ids)], device=device), style)
    h_s, logits_s, text_s = student(ids_t, style)
    f_t = teacher.frames_from_logits(logits_t)[0]
    f_s = teacher.frames_from_logits(logits_s)[0]

    f0_t, n_t, a_teacher = teacher_audio(model, h_t, text_t, f_t, style, seed)
    _, _, a_noise = teacher_audio(model, h_t, text_t, f_t, style, seed + 1)
    _, _, a_s014 = teacher_audio(model, h_s, text_s, f_s, style, seed)
    _, _, a_s015 = s015_audio(model, pitch, vocoder, h_s, text_s, f_s, style, seed)
    f0_x, n_x, a_tf = s015_audio(model, pitch, vocoder, h_t, text_t, f_t, style, seed)

    voiced = f0_t > 10
    snr = lambda a: 10 * math.log10(float(np.sum(a_teacher**2)) / max(float(np.sum((a_teacher - a) ** 2)), 1e-12))
    row = dict(
        tokens=len(ids),
        frames_teacher=int(f_t.sum()),
        frames_student=int(f_s.sum()),
        noise_dtw_logmel_db=dtw_logmel_db(a_teacher, a_noise)[0],
        noise_snr_db=snr(a_noise),
        s015_tf_f0_mae_hz=float((f0_x - f0_t).abs()[voiced].mean()),
        s015_tf_voicing_agree=float(((f0_x > 10) == voiced).float().mean()),
        s015_tf_energy_mae=float((n_x - n_t).abs().mean()),
        s015_tf_snr_db=snr(a_tf),
    )
    for name, audio, frames in [("s014", a_s014, int(f_s.sum())), ("s015", a_s015, int(f_s.sum())),
                                ("s015_tf", a_tf, int(f_t.sum()))]:
        row.update(audio_stats(name, audio, frames, a_teacher))
    return row, dict(teacher=a_teacher, s014=a_s014, s015=a_s015, s015_tf=a_tf)


def summary(rows: list[dict]) -> dict:
    out = {}
    for k, v in rows[0].items():
        if isinstance(v, bool):
            out[f"{k}_all"] = all(r[k] for r in rows)
        elif isinstance(v, (int, float)):
            out[k] = float(np.nanmean([r[k] for r in rows]))
    return out


def load_downstream(path: Path, device: str):
    state = torch.load(path, map_location="cpu")
    pitch, vocoder = PitchStudent(**state.get("pitch_config") or {}), VocoderStudent(**state.get("vocoder_config", {}))
    pitch.load_state_dict(state["pitch"])
    vocoder.load_state_dict(state["vocoder"])
    return pitch.to(device).eval(), vocoder.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="artifacts/s015/runs/<name>/<step>.pt")
    parser.add_argument("--val", type=int, default=100, help="holdout utterances (0 to skip)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    saved = torch.load(S014_STUDENT, map_location="cpu")
    student = Student(**saved["config"])
    student.load_state_dict(saved["model"])
    student = student.to(args.device).eval()
    pitch, vocoder = load_downstream(args.checkpoint, args.device)
    model = teacher.load_teacher(args.device)
    voice = teacher.load_voice()
    out = ROOT / "artifacts/s015/eval" / f"{args.checkpoint.parent.name}-{args.checkpoint.stem}"
    (out / "frozen").mkdir(parents=True, exist_ok=True)
    fmt = lambda row: json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    manifest = json.loads((ROOT / "benchmark/inputs/manifest.json").read_text())
    rows = []
    for item in manifest["items"]:
        for chunk in item["chunks"]:
            frozen = np.load(ROOT / "benchmark/inputs" / chunk["file"])
            row, audio = compare(model, student, pitch, vocoder, frozen["input_ids"][0].tolist(),
                                 frozen["style"][0], args.device, 1234)
            row["chunk"] = chunk["file"]
            rows.append(row)
            for arm, wave in audio.items():
                sf.write(out / "frozen" / f"{Path(chunk['file']).stem}-{arm}.wav", wave, 24000)
            print(fmt(row), flush=True)
    (out / "frozen.json").write_text(json.dumps({"rows": rows, "mean": summary(rows)}, indent=2))
    print("frozen mean", fmt(summary(rows)), flush=True)

    if args.val:
        split = Split("holdout")
        pick = np.random.default_rng(0).choice(len(split), args.val, replace=False)
        rows = []
        for k, i in enumerate(pick):
            o, n = int(split.offsets[i]), int(split.lengths[i])
            ids = split.ids[o : o + n].astype(int).tolist()
            style_row = teacher.style_rows(voice, [n - 2])[0]
            rows.append(compare(model, student, pitch, vocoder, ids, style_row, args.device, 1234 + k)[0])
        (out / "val.json").write_text(json.dumps({"rows": rows, "mean": summary(rows)}, indent=2))
        print("val mean", fmt(summary(rows)), flush=True)


if __name__ == "__main__":
    main()
