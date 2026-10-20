"""S017 four-length blind test: Mac-side reference wavs.

Reference = original Kokoro af_heart, the benchmark/inputs/*.wav files (kokoro-onnx 0.6.1), the same
files the S004/S007/S011 four-length tests played. 30s is 30s-0 then 30s-1 joined with no gap, as the
phone joins its two S017 chunks. Output: artifacts/s017/four-length/ref/{3s,7s,15s,30s}.wav
Push: see docs/EXPERIMENT_LOG.md (S017 full-chain section).
"""

import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "benchmark/inputs"
OUT = ROOT / "artifacts/s017/four-length/ref"
OUT.mkdir(parents=True, exist_ok=True)
for name, parts in {"3s": ["3s"], "7s": ["7s"], "15s": ["15s"], "30s": ["30s-0", "30s-1"]}.items():
    frames, params = b"", None
    for part in parts:
        with wave.open(str(SRC / f"{part}.wav")) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
            params = params or w.getparams()
            frames += w.readframes(w.getnframes())
    with wave.open(str(OUT / f"{name}.wav"), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)
    print(name, len(frames) // 2, "samples")
