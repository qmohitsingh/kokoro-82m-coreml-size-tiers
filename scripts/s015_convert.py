"""Convert a trained S015 downstream checkpoint to int8 Core ML and check it.

    checkpoint -> PitchStudent / VocoderStudent -> trace -> Core ML (flexible F)
    -> int8 per-channel weights (as the size probe) -> coremlcompiler
    -> frozen chunks, S014 student front end, PyTorch fp32 vs Core ML int8:
         pitch / energy MAE, audio SNR and DTW log-mel (same sine spectrum)
    -> Core ML end-to-end WAVs for the frozen chunks (diagnostic, not listened to)
    -> warm Mac latency (median of 5) per model and compute unit
    -> compiled bytes + S014 student + required non-model bytes = projected app
    -> artifacts/s015/coreml/<run>-<ckpt>/report.json

Usage: s015_convert.py artifacts/s015/runs/<name>/<step>.pt
"""

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
from logmel_dtw import dtw_logmel_db  # noqa: E402
from s014_student import Student  # noqa: E402
from s015_eval import load_downstream  # noqa: E402
from s015_size_probe import compile_bytes, quantize, to_coreml  # noqa: E402
from s015_train import S014_STUDENT, sine_spectrum  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UNITS = {"cpu": ct.ComputeUnit.CPU_ONLY, "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU}


def warm_ms(model, feed, runs=5):
    model.predict(feed)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(feed)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def snr_db(ref, x):
    return 10 * math.log10(float(np.sum(ref**2)) / max(float(np.sum((ref - x) ** 2)), 1e-12))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    out = ROOT / "artifacts/s015/coreml" / f"{args.checkpoint.parent.name}-{args.checkpoint.stem}"
    (out / "frozen").mkdir(parents=True, exist_ok=True)
    pitch, vocoder = load_downstream(args.checkpoint, "cpu")

    report = {"checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)), "models": {}}
    packages = {}
    for stage, module in [("pitch", pitch), ("vocoder", vocoder)]:
        package = out / f"kokoro_s015_{stage}_int8.mlpackage"
        shutil.rmtree(package, ignore_errors=True)
        quantize(to_coreml(stage, module)).save(str(package))
        packages[stage] = package
        report["models"][stage] = {"compiled_bytes": compile_bytes(package, out / "compiled" / stage)}
    budget = json.loads((ROOT / "artifacts/s015/budget.json").read_text())
    downstream = sum(m["compiled_bytes"] for m in report["models"].values())
    report["downstream_compiled_bytes"] = downstream
    report["projected_app_bytes"] = downstream + budget["student_compiled"] + budget["required_non_model_bytes"]
    report["fits_20mb"] = report["projected_app_bytes"] <= 20_000_000
    print(json.dumps(report), flush=True)

    saved = torch.load(S014_STUDENT, map_location="cpu")
    student = Student(**saved["config"])
    student.load_state_dict(saved["model"])
    student.eval()
    model = teacher.load_teacher("cpu")
    stft = model.decoder.generator.stft
    ml = {(stage, unit): ct.models.MLModel(str(packages[stage]), compute_units=cu)
          for stage in packages for unit, cu in UNITS.items()}

    manifest = json.loads((ROOT / "benchmark/inputs/manifest.json").read_text())
    rows = []
    for item in manifest["items"]:
        for chunk in item["chunks"]:
            frozen = np.load(ROOT / "benchmark/inputs" / chunk["file"])
            ids = torch.tensor(frozen["input_ids"][:1].astype(np.int64))
            style = torch.from_numpy(frozen["style"][:1].astype(np.float32))
            hidden, logits, text = student(ids, style)
            frames = teacher.frames_from_logits(logits)[0]
            aln = teacher.alignment(frames, frames.shape[0], int(frames.sum()))
            aligned, text_a = hidden.transpose(-1, -2) @ aln, text @ aln

            f0, n = pitch(aligned, style[:, 128:])
            pfeed = {"aligned": aligned.numpy(), "style": style[:, 128:].numpy()}
            p_ml = ml["pitch", "cpu"].predict(pfeed)
            f0_ml, n_ml = torch.from_numpy(p_ml["pitch"]), torch.from_numpy(p_ml["energy"])

            torch.manual_seed(1234)
            har = sine_spectrum(model, f0)
            audio = stft.inverse(*vocoder(text_a, f0, n, style[:, :128], har)).squeeze().numpy()
            vfeed = {"text": text_a.numpy(), "pitch": f0.numpy(), "energy": n.numpy(),
                     "style": style[:, :128].numpy(), "spectrum": har.numpy()}
            v_ml = ml["vocoder", "cpu_gpu"].predict(vfeed)
            audio_ml = stft.inverse(torch.from_numpy(v_ml["magnitude"]), torch.from_numpy(v_ml["phase"])).squeeze().numpy()

            # end to end through Core ML: int8 pitch -> its own sine spectrum -> int8 vocoder
            torch.manual_seed(1234)
            har_ml = sine_spectrum(model, f0_ml)
            e2e = ml["vocoder", "cpu_gpu"].predict({**vfeed, "pitch": f0_ml.numpy(), "energy": n_ml.numpy(),
                                                    "spectrum": har_ml.numpy()})
            audio_e2e = stft.inverse(torch.from_numpy(e2e["magnitude"]), torch.from_numpy(e2e["phase"])).squeeze().numpy()
            stem = Path(chunk["file"]).stem
            sf.write(out / "frozen" / f"{stem}-s015-coreml-int8.wav", audio_e2e, 24000)

            voiced = f0 > 10
            row = dict(
                chunk=chunk["file"], frames=int(frames.sum()),
                pitch_int8_f0_mae_hz=float((f0_ml - f0).abs()[voiced].mean()),
                pitch_int8_energy_mae=float((n_ml - n).abs().mean()),
                vocoder_int8_snr_db=snr_db(audio, audio_ml),
                vocoder_int8_dtw_logmel_db=dtw_logmel_db(audio, audio_ml)[0],
                e2e_finite=bool(np.isfinite(audio_e2e).all()),
                e2e_samples_ok=len(audio_e2e) == 600 * int(frames.sum()),
                e2e_peak=float(np.abs(audio_e2e).max()),
            )
            if stem.startswith("30s"):
                row["pitch_warm_ms"] = {u: warm_ms(ml["pitch", u], pfeed) for u in UNITS}
                row["vocoder_warm_ms"] = {u: warm_ms(ml["vocoder", u], vfeed) for u in UNITS}
            rows.append(row)
            print(json.dumps(row), flush=True)
    report["frozen"] = rows
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
