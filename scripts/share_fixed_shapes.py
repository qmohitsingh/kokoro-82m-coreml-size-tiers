"""Join one stage's fixed-shape packages and measure whether weights stay shared.

coremltools save_multifunction deduplicates identical constants. That is a claim
about the package. This script also compiles with coremlcompiler and records the
compiled size, which is what the Release app actually ships.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
from coremltools.proto import FeatureTypes_pb2 as ft


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark/ios/Resources"
OUT = ROOT / "artifacts/shared"

# Five shapes cover 3s, 7s, 15s, and both halves of 30s.
STAGES = {
    "text": [
        "kokoro_text_t56_fp32",
        "kokoro_text_t110_fp32",
        "kokoro_text_t237_fp32",
        "kokoro_text_t260_fp32",
        "kokoro_text_t280_fp32",
    ],
    "prosody": [
        "kokoro_prosody_text_t56_fp32",
        "kokoro_prosody_text_t110_fp32",
        "kokoro_prosody_text_t237_fp32",
        "kokoro_prosody_text_t260_fp32",
        "kokoro_prosody_text_t280_fp32",
    ],
    "pitch": [
        "kokoro_pitch_energy_f146_fp32",
        "kokoro_pitch_energy_f275_fp32",
        "kokoro_pitch_energy_f567_fp32",
        "kokoro_pitch_energy_f614_fp32",
        "kokoro_pitch_energy_f633_fp32",
    ],
    "decoder": [
        "kokoro_decoder_body_f146_fp32",
        "kokoro_decoder_body_f275_fp32",
        "kokoro_decoder_body_f567_fp32",
        "kokoro_decoder_body_f614_fp32",
        "kokoro_decoder_body_f633_fp32",
    ],
    "generator": [
        "kokoro_generator_f146_fp32",
        "kokoro_generator_f275_fp32",
        "kokoro_generator_f567_fp32",
        "kokoro_generator_f614_fp32",
        "kokoro_generator_f633_fp32",
    ],
}


def logical_bytes(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            total += (Path(dirpath) / name).stat().st_size
    return total


def weight_bytes(package: Path) -> int:
    matches = list(package.rglob("weight.bin"))
    return sum(path.stat().st_size for path in matches)


def function_name(package_stem: str) -> str:
    # kokoro_text_t56_fp32 -> t56; kokoro_pitch_energy_f146_fp32 -> f146
    parts = package_stem.split("_")
    for part in parts:
        if part.startswith("t") and part[1:].isdigit():
            return part
        if part.startswith("f") and part[1:].isdigit():
            return part
    raise ValueError(f"no shape tag in {package_stem}")


def sample_inputs(spec) -> dict:
    rng = np.random.default_rng(0)
    values = {}
    for feature in spec.description.input:
        array_type = feature.type.multiArrayType
        shape = tuple(int(dim) for dim in array_type.shape)
        if array_type.dataType == ft.ArrayFeatureType.INT32:
            values[feature.name] = rng.integers(0, 178, size=shape, dtype=np.int32)
        elif array_type.dataType == ft.ArrayFeatureType.FLOAT32:
            values[feature.name] = rng.standard_normal(shape).astype(np.float32)
        else:
            raise RuntimeError(f"unsupported input {feature.name} type {array_type.dataType}")
    return values


def check_parity(names: list[str], shared_path: Path) -> None:
    for name in names:
        function = function_name(name)
        separate_path = str((SOURCE / f"{name}.mlpackage").resolve())
        separate = ct.models.MLModel(separate_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        shared = ct.models.MLModel(
            str(shared_path),
            function_name=function,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        inputs = sample_inputs(separate.get_spec())
        left = separate.predict(inputs)
        right = shared.predict(inputs)
        worst = 0.0
        for key, value in left.items():
            error = float(np.max(np.abs(np.asarray(value) - np.asarray(right[key]))))
            worst = max(worst, error)
            print(f"parity {function} {key} max_abs={error:.9g}")
        if worst != 0.0:
            raise RuntimeError(f"{function} shared output differs, max_abs={worst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(STAGES))
    args = parser.parse_args()
    names = STAGES[args.stage]
    OUT.mkdir(parents=True, exist_ok=True)
    compiled_dir = OUT / "compiled" / args.stage
    if compiled_dir.exists():
        shutil.rmtree(compiled_dir)
    compiled_dir.mkdir(parents=True)

    print(f"coremltools {ct.__version__}")
    separate = 0
    separate_weights = 0
    for name in names:
        package = (SOURCE / f"{name}.mlpackage").resolve()
        size = logical_bytes(package)
        weights = weight_bytes(package)
        separate += size
        separate_weights += weights
        print(f"separate {name} package={size} weight.bin={weights}")

    descriptor = MultiFunctionDescriptor()
    for name in names:
        package = str((SOURCE / f"{name}.mlpackage").resolve())
        target = function_name(name)
        descriptor.add_function(package, "main", target)
        print(f"add {target} from {name}")
    descriptor.default_function_name = function_name(names[0])

    destination = OUT / f"kokoro_{args.stage}_fp32.mlpackage"
    if destination.exists():
        shutil.rmtree(destination)
    print(f"save_multifunction -> {destination}")
    save_multifunction(descriptor, str(destination))
    shared = logical_bytes(destination)
    shared_weights = weight_bytes(destination)
    print(f"shared package={shared} weight.bin={shared_weights}")
    print(f"package ratio shared/separate={shared / separate:.4f}")
    print(f"weight ratio shared/separate={shared_weights / separate_weights:.4f}")

    print("compile shared package")
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(destination), str(compiled_dir)],
        check=True,
    )
    compiled = next(compiled_dir.glob("*.mlmodelc"))
    compiled_size = logical_bytes(compiled)
    compiled_weights = weight_bytes(compiled)
    print(f"compiled {compiled.name} package={compiled_size} weight.bin={compiled_weights}")
    print(f"compiled/separate={compiled_size / separate:.4f}")
    print(f"compiled_weight/separate_weight={compiled_weights / separate_weights:.4f}")
    check_parity(names, destination)


if __name__ == "__main__":
    main()
