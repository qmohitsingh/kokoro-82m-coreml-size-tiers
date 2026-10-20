"""Make S015 ear-test clips on the Mac (Core ML int8 downstream), next to S014's references.

    artifacts/s014/listen/ids.json (token ids only; the hidden text is never read)
      -> S014 student front end (PyTorch fp32) -> its durations
      -> S015 Core ML int8 pitch -> sine spectrum (seed 1000 + index) -> S015 Core ML int8 vocoder
      -> iSTFT -> artifacts/s015/listen/<tag>/<id>.wav  (24 kHz 16-bit mono, no normalization)
    reference arm: artifacts/s014/listen/ref/<id>.wav (original fp32 Kokoro, seed 1000 + index)
    -> artifacts/s015/listen/<tag>/stats.json: finite, samples = 600 x frames, RMS and DTW
       log-mel vs reference, peak, clipping

Entries 0-31 were heard in the S014 sessions; the next blind session should use 32+.
Usage: s015_listen_prep.py <tag>   (tag = artifacts/s015/coreml/<tag>)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
from logmel_dtw import dtw_logmel_db  # noqa: E402
from s014_student import Student  # noqa: E402
from s015_train import S014_STUDENT, sine_spectrum  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S014_LISTEN = ROOT / "artifacts/s014/listen"


def level_db(x) -> float:
    return 20 * math.log10(max(float(np.sqrt(np.mean(x**2))), 1e-9))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    args = parser.parse_args()
    coreml = ROOT / "artifacts/s015/coreml" / args.tag
    out = ROOT / "artifacts/s015/listen" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    saved = torch.load(S014_STUDENT, map_location="cpu")
    student = Student(**saved["config"])
    student.load_state_dict(saved["model"])
    student.eval()
    model = teacher.load_teacher("cpu")
    voice = teacher.load_voice()
    pitch = ct.models.MLModel(str(coreml / "kokoro_s015_pitch_int8.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    vocoder = ct.models.MLModel(str(coreml / "kokoro_s015_vocoder_int8.mlpackage"),
                                compute_units=ct.ComputeUnit.CPU_AND_GPU)

    rows = []
    for item in json.loads((S014_LISTEN / "ids.json").read_text()):
        ids = item["ids"]
        style = torch.from_numpy(teacher.style_rows(voice, [len(ids) - 2])[0][None].astype(np.float32))
        hidden, logits, text = student(torch.tensor([ids]), style)
        frames = teacher.frames_from_logits(logits)[0]
        aln = teacher.alignment(frames, frames.shape[0], int(frames.sum()))
        p = pitch.predict({"aligned": (hidden.transpose(-1, -2) @ aln).numpy(), "style": style[:, 128:].numpy()})
        torch.manual_seed(1000 + item["index"])
        har = sine_spectrum(model, torch.from_numpy(p["pitch"]))
        v = vocoder.predict({"text": (text @ aln).numpy(), "pitch": p["pitch"], "energy": p["energy"],
                             "style": style[:, :128].numpy(), "spectrum": har.numpy()})
        audio = model.decoder.generator.stft.inverse(torch.from_numpy(v["magnitude"]),
                                                     torch.from_numpy(v["phase"])).squeeze().numpy()
        sf.write(out / f"{item['id']}.wav", audio, 24000, subtype="PCM_16")
        # measure the 16-bit file, as the reference is 16-bit: float silence sits under the
        # -80 dB log-mel floor while quantization noise does not, which inflates DTW
        audio, _ = sf.read(out / f"{item['id']}.wav", dtype="float32")
        ref, _ = sf.read(S014_LISTEN / "ref" / f"{item['id']}.wav", dtype="float32")
        band = 64 + abs(len(ref) - len(audio)) // 256 + 1
        rows.append(dict(id=item["id"], frames=int(frames.sum()), finite=bool(np.isfinite(audio).all()),
                         samples_ok=len(audio) == 600 * int(frames.sum()),
                         rms_diff_db=level_db(audio) - level_db(ref), peak=float(np.abs(audio).max()),
                         clipped=int((np.abs(audio) >= 0.999).sum()),
                         dtw_logmel_db=dtw_logmel_db(ref, audio, band=band)[0]))
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("rms_diff_db", "peak", "clipped", "dtw_logmel_db")}
    summary.update(n=len(rows), finite_all=all(r["finite"] for r in rows),
                   samples_ok_all=all(r["samples_ok"] for r in rows))
    (out / "stats.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
