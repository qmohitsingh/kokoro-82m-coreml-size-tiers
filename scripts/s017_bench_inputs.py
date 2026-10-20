"""S017 production bench.json: the frozen S014 speed cases without the per-chunk style rows.

Each stored style row is exactly af_heart[token count without pads - 1], the row the
app reads from its bundled voice table, so the app looks it up instead of carrying a
second copy. Token ids and sample counts are unchanged.

    python scripts/s017_bench_inputs.py -> artifacts/s017/bench-s017.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

bench = json.loads((ROOT / "benchmark/inputs/bench.json").read_text())
voice = teacher.load_voice()
for case in bench["cases"]:
    for chunk in case["chunks"]:
        ids = chunk["input_ids"]
        assert ids[0] == 0 and ids[-1] == 0
        row = teacher.style_rows(voice, [len(ids) - 2])[0]
        assert np.array_equal(np.asarray(chunk["style"], np.float32), row), case["id"]
slim = {"sample_rate": bench["sample_rate"], "voice": bench["voice"],
        "cases": [{"id": c["id"], "chunks": [{"input_ids": ch["input_ids"], "samples": ch["samples"]}
                                              for ch in c["chunks"]]} for c in bench["cases"]]}
out = ROOT / "artifacts/s017/bench-s017.json"
out.write_text(json.dumps(slim, separators=(",", ":")))
print(out, out.stat().st_size)
