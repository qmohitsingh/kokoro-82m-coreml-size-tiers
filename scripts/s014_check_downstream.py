"""S014: check the flexible downstream models against the S011 fixed functions.

One stage per process (loading many Core ML models in one process segfaults):
    s014_check_downstream.py {pitch,decoder,generator}
Writes artifacts/s014/downstream/check_<stage>.json. Torch runs on CPU.
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

from align_and_convert_pitch import reference_outputs, snr_db, use_float_residual_scale  # noqa: E402
from convert_generator import capture_reference  # noqa: E402
from s014_convert_downstream import load_model, OUT, ROOT, NAMES  # noqa: E402
from convert_decoder_body import DecoderBody  # noqa: E402

CASES = {"3s": 146, "7s": 275, "15s": 567, "30s-0": 614, "30s-1": 633}
FIXED = {
    "pitch": (ROOT / "artifacts/int8/kokoro_pitch_int8.mlpackage"),
    "decoder": (ROOT / "artifacts/sweeps/kokoro_decoder-all-int4.mlpackage"),
    "generator": (ROOT / "artifacts/sweeps/kokoro_generator-all-int4.mlpackage"),
}
OUTPUTS = {"pitch": ("pitch", "energy"), "decoder": ("features",), "generator": ("magnitude", "phase")}


def case_inputs(model, case):
    frozen = np.load(ROOT / f"benchmark/inputs/{case}.npz")
    ids = torch.from_numpy(frozen["input_ids"].astype(np.int32))
    style = torch.from_numpy(frozen["style"].astype(np.float32))
    frames, _, _, aligned, text, pitch, energy, prosody = reference_outputs(model, ids, style)
    timbre = style[:, :128]
    features, spectrum, _ = capture_reference(model, text, pitch, energy, timbre)
    n = lambda t: t.numpy().astype(np.float32)
    return int(frames.sum()), {
        "pitch": {"aligned": n(aligned), "style": n(prosody)},
        "decoder": {"text": n(text), "pitch": n(pitch), "energy": n(energy), "style": n(timbre)},
        "generator": {"features": n(features), "style": n(timbre), "spectrum": n(spectrum)},
    }


def tile(array, length, axis=-1):
    reps = -(-length // array.shape[axis])
    return np.take(np.tile(array, [reps if i == (axis % array.ndim) else 1 for i in range(array.ndim)]),
                   range(length), axis=axis)


def odd_inputs(stage, base, f):
    """Plausible inputs at frame count f, tiled from real 30s-1 data."""
    if stage == "pitch":
        return {"aligned": tile(base["aligned"], f), "style": base["style"]}
    if stage == "decoder":
        return {"text": tile(base["text"], f), "pitch": tile(base["pitch"], 2 * f),
                "energy": tile(base["energy"], 2 * f), "style": base["style"]}
    return {"features": tile(base["features"], 2 * f), "style": base["style"],
            "spectrum": tile(base["spectrum"], 120 * f + 1)}


def expected_shapes(stage, ref_out, ref_f, f):
    """Scale the last axis of a reference output from ref_f frames to f frames."""
    shapes = {}
    for name, array in ref_out.items():
        last = array.shape[-1]
        # last axis is a*ref_f + b with b in {0,1}; recover a, b
        b = last % ref_f
        a = (last - b) // ref_f
        shapes[name] = tuple(array.shape[:-1]) + (a * f + b,)
    return shapes


def predict(mlmodel, inputs):
    return {k: np.asarray(v) for k, v in mlmodel.predict(inputs).items()}


def main():
    stage = sys.argv[1]
    torch.set_grad_enabled(False)
    model = load_model()
    if stage != "generator":
        use_float_residual_scale()
    flex_path = OUT / f"{NAMES[stage]}.mlpackage"
    flex_cpu = ct.models.MLModel(str(flex_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    result = {"stage": stage, "chunks": {}, "odd": {}, "latency": {}}
    data = {}
    for case, f in CASES.items():
        frames, inputs = case_inputs(model, case)
        assert frames == f, (case, frames, f)
        data[case] = inputs[stage]
        fixed = ct.models.MLModel(str(FIXED[stage]), function_name=f"f{f}",
                                  compute_units=ct.ComputeUnit.CPU_ONLY)
        a = predict(flex_cpu, inputs[stage])
        b = predict(fixed, inputs[stage])
        row = {}
        for name in OUTPUTS[stage]:
            row[name] = {"shape_flex": list(a[name].shape), "shape_fixed": list(b[name].shape),
                         "max_abs": float(np.max(np.abs(a[name] - b[name]))),
                         "snr_db": snr_db(b[name], a[name])}
            print(case, stage, name, row[name])
        result["chunks"][case] = row
        ref_out = b
        result.setdefault("_ref", {})[case] = {k: list(v.shape) for k, v in b.items()}
        del fixed
    # odd lengths on the flexible model only
    base = data["30s-1"]
    ref_shapes = {k: np.zeros(v) for k, v in result["_ref"]["30s-1"].items()}
    for f in (401, 1500):
        out = predict(flex_cpu, odd_inputs(stage, base, f))
        want = expected_shapes(stage, ref_shapes, 633, f)
        row = {}
        for name, array in out.items():
            row[name] = {"shape": list(array.shape), "expected": list(want[name]),
                         "shape_ok": tuple(array.shape) == want[name],
                         "finite": bool(np.isfinite(array).all()),
                         "abs_max": float(np.max(np.abs(array)))}
        row["pass"] = all(v["shape_ok"] and v["finite"] for v in row.values())
        print("odd", f, stage, row)
        result["odd"][str(f)] = row
    result.pop("_ref")
    # latency, computeUnits ALL, F=633
    inputs633 = data["30s-1"]
    for label, loader in (
        ("flex", lambda: ct.models.MLModel(str(flex_path), compute_units=ct.ComputeUnit.ALL)),
        ("fixed_f633", lambda: ct.models.MLModel(str(FIXED[stage]), function_name="f633",
                                                 compute_units=ct.ComputeUnit.ALL)),
    ):
        m = loader()
        t0 = time.perf_counter()
        m.predict(inputs633)  # first call, includes compile/warm-up
        first = time.perf_counter() - t0
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            m.predict(inputs633)
            times.append(time.perf_counter() - t0)
        result["latency"][label] = {"first_s": first, "median_s": statistics.median(times),
                                    "runs_s": times}
        print("latency", stage, label, result["latency"][label])
        del m
    (OUT / f"check_{stage}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
