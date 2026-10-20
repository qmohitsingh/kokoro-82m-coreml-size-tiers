"""S015: exact production-app byte budget, from the built S014 Release app.

    S014 app files -> required runtime | bench/listening only | replaced models
    + af_heart voice table (production needs it; S014 reads styles from bench.json)
    -> maximum compiled-model bytes under 20,000,000, and what is left for the downstream

The Swift binary is counted at its S014 size, which still contains the bench and
listening UI, so the budget is conservative until a production binary is built.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logical_size import logical_bytes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "benchmark/ios/build/s014/Build/Products/Release-iphoneos/KokoroSize.app"
LIMIT = 20_000_000
STUDENT = "kokoro_student_w256.mlmodelc"
DOWNSTREAM = ("kokoro_pitch_int8_flex.mlmodelc", "kokoro_decoder_int4_flex.mlmodelc",
              "kokoro_generator_mixed8_flex.mlmodelc")
BENCH_ONLY = ("3s.wav", "7s.wav", "15s.wav", "30s-0.wav", "30s-1.wav", "bench.json")
RUNTIME = ("KokoroSize", "Info.plist", "PkgInfo")


def main() -> None:
    sizes = {p.name: logical_bytes(p) for p in APP.iterdir()}
    total = logical_bytes(APP)
    assert sum(sizes.values()) == total, "every app byte is assigned to one entry"
    unknown = set(sizes) - set(BENCH_ONLY) - set(RUNTIME) - set(DOWNSTREAM) - {STUDENT}
    assert not unknown, unknown

    voices = np.load(ROOT / "models/voices-v1.0.bin")
    heart = voices["af_heart"]  # [510, 1, 256] fp32
    voice_fp32 = int(heart.size * 4)

    runtime = {k: sizes[k] for k in RUNTIME}
    bench = {k: sizes[k] for k in BENCH_ONLY}
    required_non_model = sum(runtime.values()) + voice_fp32
    model_budget = LIMIT - required_non_model
    downstream_budget = model_budget - sizes[STUDENT]
    report = {
        "s014_app_bytes": total,
        "runtime_files": runtime,
        "voice_table_af_heart_fp32_bytes": voice_fp32,
        "voice_table_shape": list(heart.shape),
        "bench_and_listening_only": bench,
        "bench_and_listening_total": sum(bench.values()),
        "student_compiled": sizes[STUDENT],
        "s014_downstream_compiled": {k: sizes[k] for k in DOWNSTREAM},
        "s014_downstream_total": sum(sizes[k] for k in DOWNSTREAM),
        "required_non_model_bytes": required_non_model,
        "max_compiled_model_bytes": model_budget,
        "max_downstream_compiled_bytes_with_s014_student": downstream_budget,
        "s014_downstream_over_budget_by": sum(sizes[k] for k in DOWNSTREAM) - downstream_budget,
        "note": "binary counted at S014 size (includes diagnostic UI); upper bound until a production binary is built",
    }
    out = ROOT / "artifacts/s015/budget.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
