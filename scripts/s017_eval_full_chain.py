"""S017 full shipped chain on the Mac, 15 gate items, same metric as the gate.

Unlike the s017ml-full arm of s015_compare_s014.py (PyTorch fp32 student + torch durations
and sine source), every stage here is what the phone runs, ported from CoreMLChain.swift:

    ids + af_heart row -> kokoro_student_w256 (Core ML, CPU_ONLY; fp16 outputs)
      -> Swift duration rule (Float sigmoid sum over 50 bins, round half away from zero, min 1)
      -> repeat tokens to frames -> kokoro_pitch_int8_flex (Core ML, CPU_ONLY)
      -> Swift SourceDSP (SplitMix64 noise, 9 harmonics, merge, tanh, STFT 20/5)
      -> kokoro_s015_vocoder_int8 (Core ML, CPU_ONLY like the phone) -> Swift inverseSTFT.

Reference and metric are the gate's: fp32 S014-student -> fp32 Kokoro, torch sine seed NOISE_SEED,
dtw_logmel_db / band_db / level_db. Nothing is tuned; the mean is reported as it comes.
Phone ran the student and pitch on .all (Neural Engine); here CPU_ONLY, so fp16 outputs can differ.

    python scripts/s017_eval_full_chain.py            # eval -> artifacts/s017/eval-full-chain/, benchmark/results/size-s017-eval.json
    python scripts/s017_eval_full_chain.py --frames   # frame counts for listen sentences vs listen-frames.json (phone)
"""

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
import s015_compare_s014 as gate  # noqa: E402
from align_and_convert_pitch import use_float_residual_scale  # noqa: E402
from convert_generator import NOISE_SEED, capture_reference  # noqa: E402
from logmel_dtw import dtw_logmel_db  # noqa: E402
from s014_student import Student  # noqa: E402
from s015_eval import band_db, level_db  # noqa: E402
from s015_train import S014_STUDENT, sine_spectrum  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/s017/eval-full-chain"
PKG = {"student": ROOT / "artifacts/s014/student_coreml/kokoro_student_w256.mlpackage",
       "pitch": ROOT / "artifacts/s014/downstream/kokoro_pitch_int8_flex.mlpackage",
       "vocoder": ROOT / "artifacts/s015/coreml/v4-prune-step10000/kokoro_s015_vocoder_int8.mlpackage"}
COMPILED = {"student": ROOT / "artifacts/s014/student_coreml/compiled/kokoro_student_w256.mlmodelc",
            "pitch": ROOT / "artifacts/s014/downstream/compiled/pitch/kokoro_pitch_int8_flex.mlmodelc",
            "vocoder": ROOT / "artifacts/s015/coreml/v4-prune-step10000/compiled/vocoder/kokoro_s015_vocoder_int8.mlmodelc"}
F32 = np.float32


def swift_durations(logits):
    """CoreMLChain.durations: Float sigmoid, sequential Float sum, rounded() (half away), max 1."""
    x = np.asarray(logits, F32)[0]  # [T, 50]
    s = (F32(1) / (F32(1) + np.exp(-x))).astype(F32)
    total = np.cumsum(s, axis=-1, dtype=F32)[:, -1]
    ties = int((np.abs(total - np.floor(total) - 0.5) < 1e-6).sum())
    return np.maximum(1, np.floor(total + F32(0.5)).astype(np.int64)), ties


class SplitMix:
    """SplitMix64 + Box-Muller exactly as the Swift struct; returns n normals in call order."""

    def __init__(self, seed=0):
        self.state = seed

    def normals(self, n):
        pairs = (n + 1) // 2
        m = (1 << 64) - 1
        z = (self.state + (np.arange(1, 2 * pairs + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15))) & np.uint64(m)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        u = ((z >> np.uint64(11)).astype(np.float64) / float(1 << 53)).astype(F32)
        u1, u2 = np.maximum(u[0::2], np.finfo(F32).smallest_subnormal), u[1::2]
        r = np.sqrt(-2 * np.log(u1)).astype(F32)
        a = (F32(2) * F32(np.pi) * u2).astype(F32)
        out = np.empty(2 * pairs, F32)
        out[0::2], out[1::2] = r * np.cos(a), r * np.sin(a)
        return out[:n]


MERGE_W = np.array([-0.0815418661, -0.185196668, -0.182633981, -0.178372055, -0.0987389460, 0.0826403946,
                    0.0874399915, -0.390685469, -0.547744334], F32)
MERGE_B = F32(-0.0294502601)


def swift_spectrum(pitch):
    """SourceDSP.spectrum -> [22, 120*F + 1] (11 magnitude rows, 11 phase rows)."""
    pitch = np.asarray(pitch, F32).reshape(-1)
    nf, H, up = pitch.shape[0], 9, 300
    cycles = np.fmod(pitch[:, None] * (np.arange(H, dtype=F32) + 1) / F32(24000), F32(1)).astype(F32)
    cycles = np.where(cycles < 0, cycles + 1, cycles).astype(F32)
    coarse = (np.cumsum(cycles, axis=0, dtype=F32) * F32(2) * F32(np.pi) * F32(up)).astype(F32)  # [nf, H]
    n = nf * up
    s = np.arange(n, dtype=F32)
    coord = ((s + F32(0.5)) / F32(up) - F32(0.5)).astype(F32)
    low = np.clip(np.floor(coord).astype(np.int64), 0, nf - 1)
    high = np.clip(low + 1, 0, nf - 1)
    frac = np.clip(coord - low.astype(F32), 0, 1).astype(F32)[:, None]
    angle = (coarse[low] + (coarse[high] - coarse[low]) * frac).astype(F32)
    voiced = (pitch[np.minimum(nf - 1, np.arange(n) // up)] > 10)[:, None]
    scale = np.where(voiced, F32(0.003), F32(0.1 / 3)).astype(F32)
    noise = SplitMix(0).normals(n * H).reshape(n, H)
    noisy = (np.where(voiced, np.sin(angle) * F32(0.1), F32(0)) + scale * noise).astype(F32)
    wave_ = np.tanh(MERGE_B + (noisy * MERGE_W).sum(axis=1, dtype=F32)).astype(F32)
    sp = torch.stft(torch.from_numpy(wave_), 20, 5, 20, torch.hann_window(20), center=True, pad_mode="reflect",
                    return_complex=True)  # periodic Hann, one-sided: [11, n/5+1]
    return np.concatenate([sp.abs().numpy(), sp.angle().numpy()], axis=0).astype(F32)


def swift_istft(mag, phase):
    """SourceDSP.inverseSTFT: mag/phase [11, frames] -> overlap-add with squared Hann envelope."""
    nfft, hop = 20, 5
    frames = mag.shape[1]
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)).astype(F32)
    seg = np.fft.irfft(mag * np.exp(1j * phase), n=nfft, axis=0).astype(F32) * win[:, None]  # [20, frames]
    total = (frames - 1) * hop + nfft
    ola, env = np.zeros(total, F32), np.zeros(total, F32)
    for k in range(nfft // hop):  # frames f, f+4, ... never overlap within one hop phase
        idx = (np.arange(frames)[None, :] * hop + np.arange(k * hop, (k + 1) * hop)[:, None])
        np.add.at(ola, idx, seg[k * hop:(k + 1) * hop])
        np.add.at(env, idx, np.broadcast_to((win ** 2)[k * hop:(k + 1) * hop, None], idx.shape))
    ola, env = ola[nfft // 2:total - nfft // 2], env[nfft // 2:total - nfft // 2]
    return np.where(env > 1e-11, ola / np.where(env > 1e-11, env, 1), 0).astype(F32)


def load_ml(path):
    return ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)


def student_step(student_ml, ids, style_row):
    r = student_ml.predict({"input_ids": np.asarray([ids], np.int32), "style": style_row[None].astype(F32)})
    return (np.asarray(r[k], F32) for k in ("duration_hidden", "duration_logits", "text_features"))


def sha_dir(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def write_wav(path: Path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000)
        w.writeframes(pcm.tobytes())


def frames_proof():
    student_ml = load_ml(PKG["student"])
    rows = json.loads((ROOT / "artifacts/s014/listen/sentences.json").read_text())
    phone = json.loads((ROOT / "artifacts/s017/listen-frames.json").read_text())
    diff = {}
    for r in rows:
        if r["id"] not in phone:
            continue
        _, logits, _ = student_step(student_ml, r["input_ids"], np.asarray(r["style"], F32))
        f = int(swift_durations(logits)[0].sum())
        if f != phone[r["id"]]["frames"]:
            diff[r["id"]] = (f, phone[r["id"]]["frames"])
    result = {"compared": sum(r["id"] in phone for r in rows), "mismatches": diff}
    print(json.dumps(result))
    out = ROOT / "benchmark/results/size-s017-eval.json"
    if out.exists():
        summary = json.loads(out.read_text())
        summary["frames_vs_phone_listen_clips"] = result
        out.write_text(json.dumps(summary, indent=2))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", action="store_true")
    if ap.parse_args().frames:
        return frames_proof()
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)

    model = teacher.load_teacher("cpu")
    use_float_residual_scale()
    voice = teacher.load_voice()
    saved = torch.load(S014_STUDENT, map_location="cpu")
    student = Student(**saved["config"])
    student.load_state_dict(saved["model"])
    student.eval()
    stft = model.decoder.generator.stft
    ml = {k: load_ml(PKG[k]) for k in PKG}

    phone = {r["id"]: r for r in json.loads((ROOT / "artifacts/s017/coreml-speed-s017.json").read_text())}
    rows = []
    for case, ids, style_row in gate.items(voice):
        style = torch.from_numpy(style_row[None].astype(F32))
        # Reference exactly as the gate (fp32 PyTorch student + Kokoro fp32 chain).
        hidden, logits_t, text = student(torch.tensor([ids]), style)
        frames_t = teacher.frames_from_logits(logits_t)[0]
        aln = teacher.alignment(frames_t, frames_t.shape[0], int(frames_t.sum()))
        aligned_t, text_al_t = hidden.transpose(-1, -2) @ aln, text @ aln
        timbre, prosody = style[:, :128], style[:, 128:]
        f0_ref, energy_ref = model.predictor.F0Ntrain(aligned_t, prosody)
        _, spectrum, ref = capture_reference(model, text_al_t, f0_ref, energy_ref, timbre)
        ref = ref.squeeze().numpy()

        # Shipped chain.
        dh, dl, tf = student_step(ml["student"], ids, style_row)  # [1,T,640], [1,T,50], [1,512,T]
        frames, ties = swift_durations(dl)
        T, Fr = len(ids), int(frames.sum())
        index = np.repeat(np.arange(T), frames)
        aligned = np.ascontiguousarray(dh[0][index].T[None])  # [1,640,F]
        text_al = np.ascontiguousarray(tf[0][:, index][None])  # [1,512,F]
        p = ml["pitch"].predict({"aligned": aligned, "style": style_row[None, 128:].astype(F32)})
        pitch, energy = np.asarray(p["pitch"], F32), np.asarray(p["energy"], F32)
        spec = swift_spectrum(pitch)[None]
        v = ml["vocoder"].predict({"text": text_al, "pitch": pitch, "energy": energy,
                                   "style": style_row[None, :128].astype(F32), "spectrum": spec})
        audio = swift_istft(np.asarray(v["magnitude"], F32)[0], np.asarray(v["phase"], F32)[0])
        wav = OUT / f"{case}.wav"
        write_wav(wav, audio)

        bands = [a - r for a, r in zip(band_db(audio), band_db(ref))]
        voiced = f0_ref > 10
        rows.append(dict(
            case=case, frames=Fr, frames_pytorch_student=int(frames_t.sum()), duration_half_ties=ties,
            samples=int(audio.shape[0]), samples_ok=audio.shape[0] == 600 * Fr, reference_samples=int(ref.shape[0]),
            dtw=float(dtw_logmel_db(ref, audio)[0]), rms=float(level_db(audio) - level_db(ref)),
            b6_8=float(bands[0]), b8_10=float(bands[1]), b10_12=float(bands[2]),
            f0_mae_hz_vs_fp32=(float(np.abs(pitch.reshape(-1) - f0_ref.numpy().reshape(-1))[voiced.numpy().reshape(-1)].mean())
                               if pitch.size == f0_ref.numel() else None),
            peak=float(np.abs(audio).max()), finite=bool(np.isfinite(audio).all()),
            wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest()))
        print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in rows[-1].items()
                          if k != "wav_sha256"}), flush=True)

    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("dtw", "rms", "b6_8", "b8_10", "b10_12", "peak")}
    mean["all_finite"] = all(r["finite"] and r["samples_ok"] for r in rows)
    models = {k: dict(package_sha256=sha_dir(PKG[k]), compiled_logical_bytes=dir_bytes(COMPILED[k]),
                      compute_units_in_eval="CPU_ONLY") for k in PKG}
    total = sum(m["compiled_logical_bytes"] for m in models.values())
    full = {"chain": "student_w256 CoreML -> Swift durations -> pitch int8 flex CoreML -> Swift SourceDSP "
                     "-> vocoder int8 CoreML (CPU_ONLY) -> Swift iSTFT",
            "gate_dB": 1.86, "mean": mean, "models": models, "compiled_total_bytes": total, "rows": rows}
    (OUT / "eval.json").write_text(json.dumps(full, indent=2))
    # Frame proof against the phone's speed-run audio lengths (3s, 7s, 15s single chunk; 30s = two chunks).
    by = {r["case"]: r["frames"] for r in rows}
    cmp = {"3s": by["3s"], "7s": by["7s"], "15s": by["15s"], "30s": by["30s-0"] + by["30s-1"]}
    proof = {k: dict(mac=cmp[k], phone=round(phone[k]["audioSeconds"] * 40)) for k in cmp}
    summary = {"experiment": "S017 full shipped chain, 15 gate items, Mac Core ML CPU_ONLY",
               "mean_dtw_db": mean["dtw"], "gate_db": 1.86, "gate_pass": mean["dtw"] <= 1.86,
               "mean": mean, "per_item": [{k: r[k] for k in ("case", "frames", "samples", "dtw", "wav_sha256")}
                                           for r in rows],
               "models": models, "compiled_total_bytes": total, "frames_vs_phone_speed_run": proof}
    Path(ROOT / "benchmark/results/size-s017-eval.json").write_text(json.dumps(summary, indent=2))
    print("mean", json.dumps({k: round(x, 3) if isinstance(x, float) else x for k, x in mean.items()}))
    print("frames vs phone", json.dumps(proof), "total", total)


if __name__ == "__main__":
    main()
