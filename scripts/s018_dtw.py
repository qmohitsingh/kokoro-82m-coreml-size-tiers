"""DTW log-mel of the full chain with a candidate student, on 5 frozen benchmark items plus 10 of your held-out sentences. Record it, do not gate on it.

Same code as s017_eval_full_chain.py main() (imported helpers, same reference, same
metric (the 5 frozen items are benchmark/inputs/manifest.json), int8 pitch + int8 vocoder, CPU_ONLY); the student package, the fp32 reference
student, the pitch and vocoder packages and the output folder are arguments, and
nothing under benchmark/ is written. In S018 this number got worse (1.79 -> 2.58 dB)
while the WER improved: it is a regression report, not a pass/fail gate.

    python scripts/s018_dtw.py --work runs-s018 --name run1 --package runs-s018/runs/run1/kokoro_student_w256.mlpackage \
        --student-ckpt student-w256-b-step10000.pt --pitch 25mb/kokoro_pitch_int8_flex.mlpackage \
        --vocoder 25mb/kokoro_s015_vocoder_int8.mlpackage [--voice-table 25mb/af_heart_f32.bin]
    # baseline: run again with --package <the original student package> --name baseline

Writes <work>/dtw/<name>/dtw.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s017_eval_full_chain as E  # noqa: E402


def items(voice, heldout: Path, n: int = 10):
    """(case, ids with pads, style row): the frozen benchmark chunks, then n held-out sentences (fixed seed)."""
    out = []
    manifest = json.loads((E.ROOT / "benchmark/inputs/manifest.json").read_text())
    for item in manifest["items"]:
        for chunk in item["chunks"]:
            frozen = np.load(E.ROOT / "benchmark/inputs" / chunk["file"])
            out.append((Path(chunk["file"]).stem, frozen["input_ids"][0].tolist(), frozen["style"][0]))
    held = json.load(open(heldout))
    for k, i in enumerate(np.random.default_rng(7).choice(len(held), min(n, len(held)), replace=False)):
        out.append((f"val{k}", held[i]["ids"], voice[held[i]["style"]]))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--package", type=Path, required=True, help="student .mlpackage to evaluate")
    ap.add_argument("--student-ckpt", type=Path, required=True, help="fp32 student .pt used for the reference (the published step10000.pt)")
    ap.add_argument("--pitch", type=Path, required=True)
    ap.add_argument("--vocoder", type=Path, required=True)
    ap.add_argument("--voice-table", type=Path, help="510x256 fp32 voice table (default: the teacher's af_heart)")
    args = ap.parse_args()
    out = args.work / "dtw" / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    F32 = E.F32
    model = E.teacher.load_teacher("cpu")
    E.use_float_residual_scale()
    voice = np.fromfile(args.voice_table, dtype="<f4").reshape(510, 256) if args.voice_table else E.teacher.load_voice()
    saved = torch.load(args.student_ckpt, map_location="cpu")
    student = E.Student(**saved["config"])
    student.load_state_dict(saved["model"])
    student.eval()
    ml = {"pitch": E.load_ml(args.pitch), "vocoder": E.load_ml(args.vocoder), "student": E.load_ml(args.package)}
    rows = []
    for case, ids, style_row in items(voice, args.work / "heldout/tokens.json"):
        style = torch.from_numpy(style_row[None].astype(F32))
        hidden, logits_t, text = student(torch.tensor([ids]), style)
        frames_t = E.teacher.frames_from_logits(logits_t)[0]
        aln = E.teacher.alignment(frames_t, frames_t.shape[0], int(frames_t.sum()))
        timbre, prosody = style[:, :128], style[:, 128:]
        f0_ref, energy_ref = model.predictor.F0Ntrain(hidden.transpose(-1, -2) @ aln, prosody)
        _, _, ref = E.capture_reference(model, text @ aln, f0_ref, energy_ref, timbre)
        ref = ref.squeeze().numpy()

        dh, dl, tf = E.student_step(ml["student"], ids, style_row)
        frames, _ = E.swift_durations(dl)
        index = np.repeat(np.arange(len(ids)), frames)
        aligned = np.ascontiguousarray(dh[0][index].T[None])
        text_al = np.ascontiguousarray(tf[0][:, index][None])
        p = ml["pitch"].predict({"aligned": aligned, "style": style_row[None, 128:].astype(F32)})
        pitch, energy = np.asarray(p["pitch"], F32), np.asarray(p["energy"], F32)
        v = ml["vocoder"].predict({"text": text_al, "pitch": pitch, "energy": energy,
                                   "style": style_row[None, :128].astype(F32), "spectrum": E.swift_spectrum(pitch)[None]})
        audio = E.swift_istft(np.asarray(v["magnitude"], F32)[0], np.asarray(v["phase"], F32)[0])
        bands = [a - r for a, r in zip(E.band_db(audio), E.band_db(ref))]
        rows.append(dict(case=case, frames=int(frames.sum()), dtw=float(E.dtw_logmel_db(ref, audio)[0]),
                         rms=float(E.level_db(audio) - E.level_db(ref)), b6_8=float(bands[0]), b8_10=float(bands[1]),
                         b10_12=float(bands[2]), finite=bool(np.isfinite(audio).all())))
        print(json.dumps({k: round(x, 3) if isinstance(x, float) else x for k, x in rows[-1].items()}), flush=True)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("dtw", "rms", "b6_8", "b8_10", "b10_12")}
    (out / "dtw.json").write_text(json.dumps(dict(package=args.package.name, mean=mean, rows=rows), indent=2))
    print("mean", json.dumps({k: round(x, 3) for k, x in mean.items()}))


if __name__ == "__main__":
    main()
