"""Check the flexible-length Core ML student against the PyTorch fp32 student.

Frozen chunks + 100 holdout utterances, computeUnits CPU_ONLY and ALL:
frame agreement, SNR / max_abs of duration_hidden and text_features, output
shapes (incl. T=1, 2, 512), and Mac latency hint (ALL, median of 5 warm runs).
Writes artifacts/s014/student_coreml/report.json.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
from s014_student import Student  # noqa: E402
from s014_train import Split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/s014/student_coreml"
NAMES = ("duration_hidden", "duration_logits", "text_features")


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    ref, test = ref.astype(np.float64), test.astype(np.float64)
    return 10 * np.log10(max(np.sum(ref**2), 1e-30) / max(np.sum((ref - test) ** 2), 1e-30))


def frames(logits: np.ndarray) -> np.ndarray:
    return teacher.frames_from_logits(torch.from_numpy(logits))[0].numpy()


def pt_run(student, ids, style):
    with torch.no_grad():
        return [o.numpy() for o in student(torch.tensor([ids]), torch.from_numpy(style[None].astype(np.float32)))]


def cm_run(ml, ids, style):
    out = ml.predict({"input_ids": np.array([ids], np.int32), "style": style[None].astype(np.float32)})
    return [np.asarray(out[n]) for n in NAMES]


def compare(student, ml, items):
    exact_tok = total_tok = 0
    diffs, mismatched, snr_h, snr_t, abs_h, abs_t = [], 0, [], [], [], []
    shapes_ok = True
    for ids, style in items:
        p, c = pt_run(student, ids, style), cm_run(ml, ids, style)
        shapes_ok &= all(a.shape == b.shape for a, b in zip(p, c))
        fp, fc = frames(p[1]), frames(c[1])
        exact_tok += int((fp == fc).sum())
        total_tok += len(fp)
        diffs.append(abs(int(fp.sum()) - int(fc.sum())))
        mismatched += int((fp != fc).any())
        snr_h.append(snr_db(p[0], c[0])); snr_t.append(snr_db(p[2], c[2]))
        abs_h.append(float(np.abs(p[0] - c[0]).max())); abs_t.append(float(np.abs(p[2] - c[2]).max()))
    mm = lambda v: dict(min=float(np.min(v)), mean=float(np.mean(v)), max=float(np.max(v)))
    return dict(
        utterances=len(items), tokens=total_tok,
        frames_exact=exact_tok / total_tok,
        total_frames_abs_diff=dict(mean=float(np.mean(diffs)), max=int(np.max(diffs))),
        utterances_with_frame_mismatch=mismatched,
        duration_hidden=dict(snr_db=mm(snr_h), max_abs=mm(abs_h)),
        text_features=dict(snr_db=mm(snr_t), max_abs=mm(abs_t)),
        shapes_match=bool(shapes_ok),
    )


def main() -> None:
    saved = torch.load(ROOT / "artifacts/s014/runs/w256-b/step10000.pt", map_location="cpu")
    student = Student(**saved["config"])
    student.load_state_dict(saved["model"])
    student = student.eval()

    manifest = json.loads((ROOT / "benchmark/inputs/manifest.json").read_text())
    frozen = []
    for item in manifest["items"]:
        for chunk in item["chunks"]:
            z = np.load(ROOT / "benchmark/inputs" / chunk["file"])
            frozen.append((z["input_ids"][0].astype(int).tolist(), z["style"][0]))
    split = Split("holdout")
    voice = teacher.load_voice()
    pick = np.random.default_rng(0).choice(len(split), 100, replace=False)
    holdout = []
    for i in pick:
        o, n = int(split.offsets[i]), int(split.lengths[i])
        holdout.append((split.ids[o : o + n].astype(int).tolist(), teacher.style_rows(voice, [n - 2])[0]))
    lens = [len(x[0]) for x in frozen + holdout]
    print("T range", min(lens), max(lens), flush=True)

    package = str(OUT / "kokoro_student_w256.mlpackage")
    report = dict(frozen_lengths=[len(x[0]) for x in frozen], holdout_length_range=[min(lens[5:]), max(lens[5:])])
    rng = np.random.default_rng(1)
    style0 = frozen[0][1]
    for units in ("CPU_ONLY", "ALL"):
        ml = ct.models.MLModel(package, compute_units=getattr(ct.ComputeUnit, units))
        res = dict(frozen=compare(student, ml, frozen), holdout=compare(student, ml, holdout))
        shapes = {}
        for T in (1, 2, 3, 8, 64, 255, 256, 511, 512):
            ids = rng.integers(1, 178, T).astype(int).tolist()
            ids[0] = ids[-1] = 0
            p, c = pt_run(student, ids, style0), cm_run(ml, ids, style0)
            shapes[str(T)] = dict(
                pytorch=[list(a.shape) for a in p], coreml=[list(a.shape) for a in c],
                match=all(a.shape == b.shape for a, b in zip(p, c)),
                finite=bool(all(np.isfinite(a).all() for a in c)),
            )
        res["shape_checks"] = shapes
        res["shape_checks_pass"] = all(v["match"] and v["finite"] for v in shapes.values())
        if units == "ALL":
            lat = {}
            for T in (32, 128, 256, 512):
                ids = rng.integers(1, 178, T).astype(int).tolist()
                ids[0] = ids[-1] = 0
                for _ in range(2):
                    cm_run(ml, ids, style0)
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter(); cm_run(ml, ids, style0); ts.append((time.perf_counter() - t0) * 1000)
                lat[str(T)] = dict(median_ms=statistics.median(ts), runs_ms=ts)
            res["latency_all_ms"] = lat
        report[units] = res
        print(units, json.dumps({k: v for k, v in res.items() if k not in ("shape_checks", "latency_all_ms")}), flush=True)
    report["size"] = json.loads((OUT / "size.json").read_text())
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["ALL"].get("latency_all_ms"), indent=1))
    print("shape pass", report["CPU_ONLY"]["shape_checks_pass"], report["ALL"]["shape_checks_pass"])


if __name__ == "__main__":
    main()
