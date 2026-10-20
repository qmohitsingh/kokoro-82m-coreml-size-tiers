"""S015 size gate before training: convert untrained pitch and vocoder students.

    untrained PitchStudent / VocoderStudent -> trace -> Core ML (flexible F, 1-2048)
    -> int8 per-channel weights (same recipe as S014's pitch) -> coremlcompiler
    -> compiled bytes + S014 student + required non-model bytes = projected app

Weight bytes do not depend on weight values, so untrained weights give the
real size. Usage: s015_size_probe.py {pitch,vocoder} [--last C]  (vocoder final-stage channels)
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
from logical_size import logical_bytes  # noqa: E402
from s015_downstream import PitchStudent, VocoderStudent, parameter_count  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/s015/probe"
MAX_F = 2048


def rd(lo, hi):
    return ct.RangeDim(lower_bound=lo, upper_bound=hi, default=None)


def tensor(name, dims):
    return ct.TensorType(name=name, shape=ct.Shape(shape=dims), dtype=np.float32)


def to_coreml(stage: str, module: torch.nn.Module):
    torch.set_grad_enabled(False)
    module = module.eval()
    f = 146
    if stage == "pitch":
        example = (torch.randn(1, 640, f), torch.randn(1, 128))
        inputs = [tensor("aligned", (1, 640, rd(1, MAX_F))), tensor("style", (1, 128))]
        outputs = [ct.TensorType(name="pitch"), ct.TensorType(name="energy")]
    else:
        example = (torch.randn(1, 512, f), torch.randn(1, 2 * f), torch.randn(1, 2 * f),
                   torch.randn(1, 128), torch.randn(1, 22, 120 * f + 1))
        inputs = [tensor("text", (1, 512, rd(1, MAX_F))), tensor("pitch", (1, rd(2, 2 * MAX_F))),
                  tensor("energy", (1, rd(2, 2 * MAX_F))), tensor("style", (1, 128)),
                  tensor("spectrum", (1, 22, rd(121, 120 * MAX_F + 1)))]
        outputs = [ct.TensorType(name="magnitude"), ct.TensorType(name="phase")]
    traced = torch.jit.trace(module, example, strict=True)
    return ct.convert(traced, convert_to="mlprogram", inputs=inputs, outputs=outputs,
                      minimum_deployment_target=ct.target.iOS18,
                      compute_precision=ct.precision.FLOAT32,
                      compute_units=ct.ComputeUnit.CPU_ONLY)


def quantize(model):
    config = ct.optimize.coreml.OptimizationConfig(global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=2048))
    return ct.optimize.coreml.linear_quantize_weights(model, config=config)


def compile_bytes(package: Path, root: Path) -> int:
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(root)], check=True,
                   stdout=subprocess.DEVNULL)
    return logical_bytes(next(root.glob("*.mlmodelc")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pitch", "vocoder"))
    parser.add_argument("--last", type=int, help="vocoder final-stage channels (default 64)")
    parser.add_argument("--pitch-config", type=json.loads, default={}, help='PitchStudent kwargs, e.g. {"width": 256, "lstm": 128, "inp": false}')
    args = parser.parse_args()
    torch.manual_seed(0)
    module = PitchStudent(**args.pitch_config) if args.stage == "pitch" else VocoderStudent(last=args.last)
    name = f"{args.stage}_int8" + (f"_last{args.last}" if args.last else "") + "".join(f"_{k}{v}" for k, v in args.pitch_config.items())
    mlmodel = quantize(to_coreml(args.stage, module))
    OUT.mkdir(parents=True, exist_ok=True)
    package = OUT / f"s015_{name}.mlpackage"
    shutil.rmtree(package, ignore_errors=True)
    mlmodel.save(str(package))
    size = compile_bytes(package, OUT / "compiled" / name)
    row = {"stage": args.stage, "dtype": "int8", "parameters": parameter_count(module),
           "config": module.config, "package_bytes": logical_bytes(package), "compiled_bytes": size}
    (OUT / f"size_{name}.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row))


if __name__ == "__main__":
    main()
