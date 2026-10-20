"""S015 vs accepted S014 mixed8, same inputs and same sine noise, both against fp32 Kokoro.

Items are the S014 vocoder-isolation set (the 1.77 dB baseline): 5 frozen chunks + 10
holdout utterances, np.random.default_rng(7).choice(len(holdout), 10). The hidden
listening set is never read. Everything runs on CPU (MPS draws a different sine source).

    per item: S014 student front end (fp32) -> its durations -> fp32 F0Ntrain
              -> fp32 decoder + generator, sine seed NOISE_SEED          = reference
    arms ("voc" = given fp32 F0/N and the reference's captured spectrum;
          "full" = the arm's own F0/N, sine spectrum from it with seed NOISE_SEED):
      fp32-seed     fp32 downstream, seed NOISE_SEED + 1                  (noise floor)
      s014m8-voc    int4 decoder + mixed8 generator (Core ML)            (reproduces 1.77)
      s014m8-full   int8 pitch + int4 decoder + mixed8 generator (Core ML)
      s015-voc      S015 vocoder, PyTorch fp32 (--checkpoint)
      s015-full     S015 pitch + vocoder, PyTorch fp32 (--checkpoint)
      s015ml-voc    S015 vocoder int8 (Core ML, --coreml dir)
      s015ml-full   S015 pitch + vocoder int8 (Core ML, --coreml dir)
      s017ml-full   S014's Kokoro int8 pitch + S015 vocoder int8 (Core ML, --coreml dir), the S017 app;
                    also SNR against the same chain in fp32 (Kokoro F0Ntrain + --checkpoint vocoder)
    -> artifacts/s015/compare/<tag>/<arm>.json; --summary prints the table

One arm per process (several Core ML models in one process can segfault):
    s015_compare_s014.py <tag> --arm s014m8-voc
    s015_compare_s014.py <tag> --arm s015-full --checkpoint artifacts/s015/runs/v2-a100/best.pt
    s015_compare_s014.py <tag> --summary
"""

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
from align_and_convert_pitch import use_float_residual_scale  # noqa: E402
from convert_generator import NOISE_SEED, capture_reference  # noqa: E402
from logmel_dtw import dtw_logmel_db  # noqa: E402
from s014_student import Student  # noqa: E402
from s014_train import Split  # noqa: E402
from s015_eval import band_db, level_db, load_downstream  # noqa: E402
from s015_train import S014_STUDENT, sine_spectrum  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/s015/compare"
S014 = ROOT / "artifacts/s014/downstream"
S014_PACKAGES = {"pitch": S014 / "kokoro_pitch_int8_flex.mlpackage",
                 "decoder": S014 / "kokoro_decoder_int4_flex.mlpackage",
                 "generator": S014 / "kokoro_generator_mixed8_flex.mlpackage"}
ARMS = ("fp32-seed", "s014m8-voc", "s014m8-full", "s015-voc", "s015-full", "s015ml-voc", "s015ml-full", "s017ml-full")


def items(voice):
    out = []
    manifest = json.loads((ROOT / "benchmark/inputs/manifest.json").read_text())
    for item in manifest["items"]:
        for chunk in item["chunks"]:
            frozen = np.load(ROOT / "benchmark/inputs" / chunk["file"])
            out.append((Path(chunk["file"]).stem, frozen["input_ids"][0].tolist(), frozen["style"][0]))
    split = Split("holdout")
    for k, i in enumerate(np.random.default_rng(7).choice(len(split), 10, replace=False)):
        o, n = int(split.offsets[i]), int(split.lengths[i])
        out.append((f"val{k}", split.ids[o:o + n].astype(int).tolist(), teacher.style_rows(voice, [n - 2])[0]))
    return out


def mlmodel(path):
    return ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--checkpoint", type=Path, help="S015 .pt (s015-* arms)")
    parser.add_argument("--coreml", type=Path, help="dir with kokoro_s015_{pitch,vocoder}_int8.mlpackage (s015ml-*)")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    out = OUT / args.tag
    if args.summary:
        return summary(out)
    assert args.arm, "--arm or --summary"
    torch.set_grad_enabled(False)

    model = teacher.load_teacher("cpu")
    use_float_residual_scale()  # as in s014_vocoder_isolate; fp32 output unchanged
    voice = teacher.load_voice()
    saved = torch.load(S014_STUDENT, map_location="cpu")
    student = Student(**saved["config"])
    student.load_state_dict(saved["model"])
    student.eval()
    stft = model.decoder.generator.stft
    n = lambda t: t.numpy().astype(np.float32)

    arm = args.arm
    if arm.startswith("s014m8"):
        ml = {k: mlmodel(S014_PACKAGES[k]) for k in (("decoder", "generator") if arm.endswith("voc")
                                                    else ("pitch", "decoder", "generator"))}
    elif arm.startswith("s015ml"):
        ml = {k: mlmodel(args.coreml / f"kokoro_s015_{k}_int8.mlpackage")
              for k in (("vocoder",) if arm.endswith("voc") else ("pitch", "vocoder"))}
    elif arm == "s017ml-full":
        ml = {"pitch": mlmodel(S014_PACKAGES["pitch"]),
              "vocoder": mlmodel(args.coreml / "kokoro_s015_vocoder_int8.mlpackage")}
        _, vocoder = load_downstream(args.checkpoint, "cpu")
    elif arm.startswith("s015"):
        pitch, vocoder = load_downstream(args.checkpoint, "cpu")

    rows = []
    for case, ids, style_row in items(voice):
        style = torch.from_numpy(style_row[None].astype(np.float32))
        hidden, logits, text = student(torch.tensor([ids]), style)
        frames = teacher.frames_from_logits(logits)[0]
        aln = teacher.alignment(frames, frames.shape[0], int(frames.sum()))
        aligned, text_al, timbre, prosody = hidden.transpose(-1, -2) @ aln, text @ aln, style[:, :128], style[:, 128:]
        f0, energy = model.predictor.F0Ntrain(aligned, prosody)
        _, spectrum, ref = capture_reference(model, text_al, f0, energy, timbre)
        ref = ref.squeeze().numpy()
        torch.manual_seed(NOISE_SEED)
        assert torch.equal(sine_spectrum(model, f0), spectrum), "sine seed does not line up with the reference"

        def own_spectrum(f0_own):
            torch.manual_seed(NOISE_SEED)
            return sine_spectrum(model, f0_own)

        f0_arm, e_arm = f0, energy
        if arm == "fp32-seed":
            torch.manual_seed(NOISE_SEED + 1)
            audio = model.decoder(text_al, f0, energy, timbre).squeeze().numpy()
        elif arm.startswith("s014m8"):
            e_arm, spec = energy, spectrum
            if arm.endswith("full"):
                p = ml["pitch"].predict({"aligned": n(aligned), "style": n(prosody)})
                f0_arm, e_arm = torch.from_numpy(np.asarray(p["pitch"])), torch.from_numpy(np.asarray(p["energy"]))
                spec = own_spectrum(f0_arm)
            feats = ml["decoder"].predict({"text": n(text_al), "pitch": n(f0_arm), "energy": n(e_arm),
                                           "style": n(timbre)})["features"]
            g = ml["generator"].predict({"features": np.asarray(feats, np.float32), "style": n(timbre),
                                         "spectrum": n(spec)})
            audio = stft.inverse(*(torch.from_numpy(np.asarray(g[k])) for k in ("magnitude", "phase"))).squeeze().numpy()
        elif arm.startswith("s015ml") or arm == "s017ml-full":
            e_arm, spec = energy, spectrum
            if arm.endswith("full"):
                p = ml["pitch"].predict({"aligned": n(aligned), "style": n(prosody)})
                f0_arm, e_arm = torch.from_numpy(np.asarray(p["pitch"])), torch.from_numpy(np.asarray(p["energy"]))
                spec = own_spectrum(f0_arm)
            v = ml["vocoder"].predict({"text": n(text_al), "pitch": n(f0_arm), "energy": n(e_arm),
                                       "style": n(timbre), "spectrum": n(spec)})
            audio = stft.inverse(*(torch.from_numpy(np.asarray(v[k])) for k in ("magnitude", "phase"))).squeeze().numpy()
        else:
            e_arm, spec = energy, spectrum
            if arm.endswith("full"):
                f0_arm, e_arm = pitch(aligned, prosody)
                spec = own_spectrum(f0_arm)
            audio = stft.inverse(*vocoder(text_al, f0_arm, e_arm, timbre, spec)).squeeze().numpy()

        voiced = f0 > 10
        bands = [a - r for a, r in zip(band_db(audio), band_db(ref))]
        row = dict(case=case, frames=int(frames.sum()), dtw=dtw_logmel_db(ref, audio)[0],
                   rms=level_db(audio) - level_db(ref), b6_8=bands[0], b8_10=bands[1], b10_12=bands[2],
                   f0_mae_hz=float((f0_arm - f0).abs()[voiced].mean()), peak=float(np.abs(audio).max()),
                   finite=bool(np.isfinite(audio).all()), samples_ok=audio.shape[0] == 600 * int(frames.sum()))
        if arm == "s017ml-full":
            fp32 = stft.inverse(*vocoder(text_al, f0, energy, timbre, spectrum)).squeeze().numpy()
            row.update(energy_mae=float((e_arm - energy).abs().mean()),
                       snr_vs_fp32_db=float(10 * np.log10((fp32 ** 2).sum() / ((audio - fp32) ** 2).sum())),
                       dtw_vs_fp32=dtw_logmel_db(fp32, audio)[0])
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    keys = ("dtw", "rms", "b6_8", "b8_10", "b10_12", "f0_mae_hz", "peak")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    mean["all_finite"] = all(r["finite"] and r["samples_ok"] for r in rows)
    source = {"checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "coreml": str(args.coreml) if args.coreml else None}
    (out / f"{arm}.json").write_text(json.dumps({"arm": arm, **source, "rows": rows, "mean": mean}, indent=2))
    print("mean", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in mean.items()}), flush=True)


def summary(out: Path) -> None:
    print(f"| Arm vs fp32 Kokoro (15 items, CPU) | DTW dB | RMS Δ dB | 6–8k | 8–10k | 10–12k | F0 MAE Hz | finite |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for arm in ARMS:
        path = out / f"{arm}.json"
        if path.exists():
            m = json.loads(path.read_text())["mean"]
            print(f"| {arm} | {m['dtw']:.3f} | {m['rms']:+.2f} | {m['b6_8']:+.2f} | {m['b8_10']:+.2f} | "
                  f"{m['b10_12']:+.2f} | {m['f0_mae_hz']:.2f} | {m['all_finite']} |")


if __name__ == "__main__":
    main()
