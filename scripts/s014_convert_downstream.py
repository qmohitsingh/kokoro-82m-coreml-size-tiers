"""S014: convert pitch/energy, decoder body, and generator with a flexible frame length.

Same wrapper modules, same weights, same compression recipes as S011; the only
change is that the frame dimension is a ct.RangeDim instead of a fixed size.

Frame dimension relations (F = speech frames):
    pitch input  aligned  [1, 640, F]          F     in [1, 2048]
    pitch/energy          [1, 2F]              2F    in [2, 4096]   (own RangeDim)
    decoder text          [1, 512, F]          F     in [1, 2048]
    generator features    [1, 512, 2F]         2F    in [2, 4096]   (own RangeDim)
    generator spectrum    [1, 22, 120F + 1]    120F+1 in [121, 245761] (own RangeDim)
The relations 2F and 120F+1 are not enforced by Core ML; the caller must pass
consistent lengths (the chain does).

Usage (one stage per process):
    s014_convert_downstream.py {pitch,decoder,generator} [--fp32-only]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from align_and_convert_pitch import (  # noqa: E402
    PitchEnergy,
    reference_outputs,
    snr_db,
    use_float_residual_scale,
)
from convert_decoder_body import DecoderBody  # noqa: E402
from convert_generator import WaveformGenerator, capture_reference, NOISE_SEED  # noqa: E402
from kokoro import KModel  # noqa: E402
from logical_size import logical_bytes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/s014/downstream"
MAX_F = 2048
S011_COMPILED = {"pitch": 8_892_606, "decoder": 17_369_455, "generator": 11_392_268}
NAMES = {"pitch": "kokoro_pitch_int8_flex", "decoder": "kokoro_decoder_int4_flex",
         "generator": "kokoro_generator_int4_flex"}
FP32_NAMES = {"pitch": "kokoro_pitch_energy_flex_fp32", "decoder": "kokoro_decoder_body_flex_fp32",
              "generator": "kokoro_generator_flex_fp32"}
DTYPE = {"pitch": "int8", "decoder": "int4", "generator": "int4"}


def rd(lo, hi):
    return ct.RangeDim(lower_bound=lo, upper_bound=hi, default=None)


def load_model():
    return KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(ROOT / "models/pytorch/config.json"),
        model=str(ROOT / "models/pytorch/kokoro-v1_0.pth"),
    ).eval()


def convert(module, example, inputs, outputs):
    traced = torch.jit.trace(module, example, strict=True)
    return ct.convert(
        traced, convert_to="mlprogram", inputs=inputs, outputs=outputs,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )


def tensor(name, dims):
    return ct.TensorType(name=name, shape=ct.Shape(shape=dims), dtype=np.float32)


def build_fp32(stage: str, package: Path):
    torch.set_grad_enabled(False)
    frozen = np.load(ROOT / "benchmark/inputs/3s.npz")  # F = 146, as the original traces
    input_ids = torch.from_numpy(frozen["input_ids"].astype(np.int32))
    style = torch.from_numpy(frozen["style"].astype(np.float32))
    model = load_model()
    frames, _, _, aligned, text, pitch, energy, prosody = reference_outputs(model, input_ids, style)
    timbre = style[:, :128]
    f = int(frames.sum())
    if stage == "pitch":
        use_float_residual_scale()
        mlmodel = convert(
            PitchEnergy(model).eval(), (aligned, prosody),
            [tensor("aligned", (1, 640, rd(1, MAX_F))), tensor("style", (1, 128))],
            [ct.TensorType(name="pitch"), ct.TensorType(name="energy")],
        )
    elif stage == "decoder":
        use_float_residual_scale()
        mlmodel = convert(
            DecoderBody(model).eval(), (text, pitch, energy, timbre),
            [tensor("text", (1, 512, rd(1, MAX_F))),
             tensor("pitch", (1, rd(2, 2 * MAX_F))),
             tensor("energy", (1, rd(2, 2 * MAX_F))),
             tensor("style", (1, 128))],
            [ct.TensorType(name="features")],
        )
    else:
        features, spectrum, _ = capture_reference(model, text, pitch, energy, timbre)
        mlmodel = convert(
            WaveformGenerator(model).eval(), (features, timbre, spectrum),
            [tensor("features", (1, 512, rd(2, 2 * MAX_F))),
             tensor("style", (1, 128)),
             tensor("spectrum", (1, 22, rd(121, 120 * MAX_F + 1)))],
            [ct.TensorType(name="magnitude"), ct.TensorType(name="phase")],
        )
    shutil.rmtree(package, ignore_errors=True)
    mlmodel.save(str(package))
    print(f"fp32 saved {package} (traced at F={f})")


def compress(stage: str, source: Path, destination: Path):
    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype=DTYPE[stage],
            granularity="per_channel", weight_threshold=2048,
        )
    )
    model = ct.models.MLModel(str(source), skip_model_load=True)
    compressed = ct.optimize.coreml.linear_quantize_weights(model, config=config)
    shutil.rmtree(destination, ignore_errors=True)
    compressed.save(str(destination))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(NAMES))
    parser.add_argument("--fp32-only", action="store_true")
    args = parser.parse_args()
    stage = args.stage
    OUT.mkdir(parents=True, exist_ok=True)
    fp32 = OUT / "fp32" / f"{FP32_NAMES[stage]}.mlpackage"
    fp32.parent.mkdir(parents=True, exist_ok=True)
    build_fp32(stage, fp32)
    if args.fp32_only:
        return
    final = OUT / f"{NAMES[stage]}.mlpackage"
    compress(stage, fp32, final)
    compiled_root = OUT / "compiled" / stage
    shutil.rmtree(compiled_root, ignore_errors=True)
    compiled_root.mkdir(parents=True)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(final), str(compiled_root)], check=True)
    compiled = next(compiled_root.glob("*.mlmodelc"))
    size = {
        "stage": stage, "dtype": DTYPE[stage],
        "fp32_package_bytes": logical_bytes(fp32),
        "package_bytes": logical_bytes(final),
        "compiled_bytes": logical_bytes(compiled),
        "s011_compiled_bytes": S011_COMPILED[stage],
    }
    size["delta_vs_s011"] = size["compiled_bytes"] - size["s011_compiled_bytes"]
    (OUT / f"size_{stage}.json").write_text(json.dumps(size, indent=2))
    print(json.dumps(size, indent=2))


if __name__ == "__main__":
    main()
