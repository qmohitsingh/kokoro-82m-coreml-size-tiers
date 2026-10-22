"""Score a retrained student: WER of its speech against the teacher on a held-out set.

    python scripts/s018_score.py --work runs-s018 --name run1 --voice-dir voices \
        --teacher-onnx kokoro-v1.0.onnx --pitch 25mb/kokoro_pitch_int8_flex.mlpackage \
        --vocoder 25mb/kokoro_s015_vocoder_int8.mlpackage \
        --whisper-cli whisper-cli --whisper-model ggml-medium.en.bin [--baseline <original student package>]

Arms: ref (teacher ONNX, CPU), base (optional: the student you started from), cand
(<work>/runs/<name>/kokoro_student_w256.mlpackage unless --package). Pitch and vocoder are
the published Core ML packages, all CPU_ONLY, speed 1.0; the Swift stages (duration rounding,
source DSP, inverse STFT) are reproduced in numpy below. Transcription: whisper.cpp
(whisper-cli, ggml-medium.en) after ffmpeg resampling to 16 kHz. Sets: heldout
(<work>/heldout/tokens.json) plus any --set NAME=tokens.json (same format).
Gate on this WER, not on listening or log-mel DTW: both missed the S017 mispronunciations.
Writes <work>/runs/<name>/score/{wav,txt}/... and score.json.
"""

import argparse
import json
import re
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import coremltools as ct  # noqa: E402
import onnxruntime as ort  # noqa: E402

F32 = np.float32
CU = ct.ComputeUnit


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
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(20) / 20)).astype(np.float64)
    x = np.pad(wave_.astype(np.float64), (10, 10), mode="reflect")
    nfr = 1 + (len(x) - 20) // 5
    idx = np.arange(nfr)[:, None] * 5 + np.arange(20)[None, :]
    sp = np.fft.rfft(x[idx] * win, axis=1).T  # [11, nfr]
    return np.concatenate([np.abs(sp), np.angle(sp)], axis=0).astype(F32)


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


def sets(work: Path, extra: list[str], limit: int | None):
    paths = {"heldout": work / "heldout/tokens.json", **{e.split("=", 1)[0]: Path(e.split("=", 1)[1]) for e in extra}}
    out = {}
    for name, path in paths.items():
        items = json.load(open(path))[:limit]
        out[name] = ([dict(ids=i["ids"], style=i["style"]) for i in items], [i["text"] for i in items])
    return out


def wav_out(path: Path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm.tobytes())


def chain(stu, pit, voc, ids, row):
    r = stu.predict({"input_ids": np.asarray([ids], np.int32), "style": row[None].astype(F32)})
    dh, dl, tf = (np.asarray(r[k], F32) for k in ("duration_hidden", "duration_logits", "text_features"))
    frames, _ = swift_durations(dl)
    index = np.repeat(np.arange(len(ids)), frames)
    aligned = np.ascontiguousarray(dh[0][index].T[None]); text_al = np.ascontiguousarray(tf[0][:, index][None])
    p = pit.predict({"aligned": aligned, "style": row[None, 128:].astype(F32)})
    pitch, energy = np.asarray(p["pitch"], F32), np.asarray(p["energy"], F32)
    v = voc.predict({"text": text_al, "pitch": pitch, "energy": energy, "style": row[None, :128].astype(F32), "spectrum": swift_spectrum(pitch)[None]})
    return swift_istft(np.asarray(v["magnitude"], F32)[0], np.asarray(v["phase"], F32)[0])


def transcribe(wav: Path, txt: Path, whisper: Path, wmodel: Path):
    if txt.exists():
        return
    w16 = wav.with_suffix(".16k.wav")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-ar", "16000", "-ac", "1", str(w16)], check=True)
    out = subprocess.run([str(whisper), "-m", str(wmodel), "-f", str(w16), "-nt", "-np", "-t", "2"], capture_output=True, text=True).stdout
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text(" ".join(out.split()))
    w16.unlink()


def norm(s):
    s = s.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9' ]", " ", s).split()


def align(ref, hyp):
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, errs = n, m, []
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]: errs.append((ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1: errs.append((ref[i - 1], "")); i -= 1
        else: errs.append(("", hyp[j - 1])); j -= 1
    return d[n][m], errs[::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--name", required=True, help="run name under <work>/runs")
    ap.add_argument("--package", type=Path, help="candidate student .mlpackage (default <work>/runs/<name>/kokoro_student_w256.mlpackage)")
    ap.add_argument("--baseline", type=Path, help="optional .mlpackage of the student you started from; adds the base arm")
    ap.add_argument("--teacher-onnx", type=Path, required=True, help="reference Kokoro ONNX (input_ids/tokens, style, speed)")
    ap.add_argument("--pitch", type=Path, required=True, help="kokoro_pitch_int8_flex.mlpackage")
    ap.add_argument("--vocoder", type=Path, required=True, help="kokoro_s015_vocoder_int8.mlpackage")
    ap.add_argument("--voice-dir", type=Path, required=True, help="folder with <voice>_f32.bin tables")
    ap.add_argument("--whisper-cli", type=Path, required=True)
    ap.add_argument("--whisper-model", type=Path, required=True, help="ggml-medium.en.bin")
    ap.add_argument("--voices", nargs="+", default=["af_heart"])
    ap.add_argument("--set", action="append", default=[], metavar="NAME=tokens.json", help="extra sentence set, same format as heldout/tokens.json")
    ap.add_argument("--limit", type=int, help="first N sentences of each set")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    run = args.work / "runs" / args.name
    cand_pkg = args.package or run / "kokoro_student_w256.mlpackage"
    out = run / "score"
    voice_rows = {v: np.fromfile(args.voice_dir / f"{v}_f32.bin", dtype="<f4").reshape(510, 256) for v in args.voices}
    S = sets(args.work, args.set, args.limit)
    load = lambda p, cu=CU.CPU_ONLY: ct.models.MLModel(str(p), compute_units=cu)
    pit, voc, cand = load(args.pitch), load(args.vocoder), load(cand_pkg)
    base = load(args.baseline) if args.baseline else None
    arms = ("ref",) + (("base",) if base else ()) + ("cand",)

    # interface check: names, dtypes, shapes of inputs and outputs
    def iface(m):
        sp = m.get_spec().description
        f = lambda x: [(a.name, list(a.type.multiArrayType.shape), a.type.multiArrayType.dataType) for a in x]
        return f(sp.input), f(sp.output)
    iface_same = iface(base) == iface(cand) if base else None
    print("interface identical to baseline:", iface_same, iface(cand), flush=True)

    sess = ort.InferenceSession(str(args.teacher_onnx), providers=["CPUExecutionProvider"])
    tok_name = sess.get_inputs()[0].name  # "input_ids" or "tokens", depending on the export
    t0 = time.time()
    jobs = []
    for sname, (items, texts) in S.items():
        for arm in arms:
            for v in args.voices:
                for i, it in enumerate(items, 1):
                    wav = out / "wav" / sname / arm / v / f"{i:02d}.wav"
                    txt = out / "txt" / sname / arm / v / f"{i:02d}.txt"
                    if txt.exists():
                        continue
                    row = voice_rows[v][it["style"]]
                    if arm == "ref":
                        audio = sess.run(None, {tok_name: np.asarray([it["ids"]], np.int64), "style": row[None], "speed": np.array([1.0], F32)})[0].reshape(-1)  # [1, N] or [N], depending on the export
                    else:
                        audio = chain(base if arm == "base" else cand, pit, voc, it["ids"], row)
                    wav_out(wav, audio)
                    jobs.append((wav, txt))
    print(f"rendered {len(jobs)} clips in {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(lambda j: transcribe(*j, args.whisper_cli, args.whisper_model), jobs))
    print(f"transcribed in {time.time() - t1:.0f}s", flush=True)

    result = {"interface_identical": iface_same, "render_s": round(t1 - t0), "whisper_s": round(time.time() - t1), "sets": {}}
    for sname, (items, texts) in S.items():
        res = result["sets"][sname] = {}
        for v in args.voices:
            per = {}
            for arm in arms:
                errs, words, bad = 0, 0, []
                for i, s in enumerate(texts, 1):
                    ref = norm(s); hyp = norm((out / "txt" / sname / arm / v / f"{i:02d}.txt").read_text())
                    e, el = align(ref, hyp); errs += e; words += len(ref)
                    bad += [(i, r, h) for r, h in el]
                per[arm] = dict(errs=errs, words=words, wer=round(100 * errs / words, 2), bad=bad)
            key = lambda arm: {(i, r) for i, r, h in per[arm]["bad"] if r}
            if base:
                per["fixed"] = sorted(f"#{i} {r}" for i, r in key("base") - key("cand"))
                per["newly_broken"] = sorted(f"#{i} {r}" for i, r in key("cand") - key("base") if (i, r) not in key("ref"))
            per["ref_ok_cand_wrong"] = sorted(f"#{i} {r}" for i, r in key("cand") - key("ref"))
            # repeat-word rule: a word the reference says correctly must not fail in more than one sentence
            cnt = {}
            for i, r, h in per["cand"]["bad"]:
                if r and (i, r) not in key("ref"):
                    cnt.setdefault(r, set()).add(i)
            per["repeat_word_violations"] = {w: sorted(x) for w, x in cnt.items() if len(x) > 1}
            res[v] = per
    (run / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for sname, res in result["sets"].items():
        for v, per in res.items():
            print(f"{sname:8} {v:11} ref {per['ref']['errs']}/{per['ref']['words']} {per['ref']['wer']}%  base {per.get('base', {}).get('wer')}%  cand {per['cand']['wer']}%  repeat_violations {list(per['repeat_word_violations'])}")


if __name__ == "__main__":
    main()
