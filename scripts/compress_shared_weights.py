"""Apply 8-bit weight-only quantization to a shared multifunction stage.

Activations and model I/O remain fp32. The stored weight constants become int8
plus per-channel scales, then Core ML dequantizes them for the existing graph.
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
SOURCE = ROOT / "artifacts" / "fp32-source"  # override with --fp32-source
SHARED = ROOT / "artifacts/shared"
OUTPUT = ROOT / "artifacts/int8"
SHAPES = {
    "text": ("t56", "t110", "t237", "t260", "t280"),
    "prosody": ("t56", "t110", "t237", "t260", "t280"),
    "pitch": ("f146", "f275", "f567", "f614", "f633"),
    "decoder": ("f146", "f275", "f567", "f614", "f633"),
    "generator": ("f146", "f275", "f567", "f614", "f633"),
}
PREFIX = {
    "text": "kokoro_text",
    "prosody": "kokoro_prosody_text",
    "pitch": "kokoro_pitch_energy",
    "decoder": "kokoro_decoder_body",
    "generator": "kokoro_generator",
}
PACKAGES = {
    stage: tuple(f"{PREFIX[stage]}_{shape}_fp32" for shape in shapes)
    for stage, shapes in SHAPES.items()
}


def logical_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(
        (Path(directory) / name).stat().st_size
        for directory, _, names in os.walk(path)
        for name in names
    )


def sample_inputs(spec) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    result = {}
    for feature in spec.description.input:
        array_type = feature.type.multiArrayType
        shape = tuple(int(value) for value in array_type.shape)
        if array_type.dataType == ft.ArrayFeatureType.INT32:
            result[feature.name] = rng.integers(0, 178, size=shape, dtype=np.int32)
        else:
            result[feature.name] = rng.standard_normal(shape).astype(np.float32)
    return result


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    signal = np.sum(reference * reference)
    noise = np.sum((reference - candidate) ** 2)
    return float("inf") if noise == 0 else float(10 * np.log10(signal / noise))


def shape_tag(package_name: str) -> str:
    return next(
        part for part in package_name.split("_")
        if len(part) > 1 and part[0] in "ft" and part[1:].isdigit()
    )


def check_parity(compressed: Path, packages: tuple[str, ...]) -> None:
    for package_name in packages:
        function = shape_tag(package_name)
        source = SOURCE / f"{package_name}.mlpackage"
        original = ct.models.MLModel(
            str(source),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        candidate = ct.models.MLModel(
            str(compressed),
            function_name=function,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        inputs = sample_inputs(original.get_spec())
        reference_outputs = original.predict(inputs)
        compressed_outputs = candidate.predict(inputs)
        for name, reference in reference_outputs.items():
            output = np.asarray(compressed_outputs[name])
            reference = np.asarray(reference)
            maximum = float(np.max(np.abs(reference - output)))
            snr = snr_db(reference, output)
            print(f"parity {function} {name} max_abs={maximum:.9g} snr_db={snr:.3f}")


def main() -> None:
    global SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(PACKAGES))
    parser.add_argument("--fp32-source", type=Path, default=SOURCE,
                        help="folder holding the fixed-shape fp32 .mlpackage files from the earlier conversion project")
    args = parser.parse_args()
    SOURCE = args.fp32_source

    source = SHARED / f"kokoro_{args.stage}_fp32.mlpackage"
    destination = OUTPUT / f"kokoro_{args.stage}_int8.mlpackage"
    pieces = OUTPUT / "pieces" / args.stage
    compiled_root = OUTPUT / "compiled" / args.stage
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(destination, ignore_errors=True)
    shutil.rmtree(pieces, ignore_errors=True)
    shutil.rmtree(compiled_root, ignore_errors=True)
    pieces.mkdir(parents=True)
    compiled_root.mkdir(parents=True)

    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
            weight_threshold=2048,
        )
    )
    descriptor = MultiFunctionDescriptor()
    for package_name in PACKAGES[args.stage]:
        original = SOURCE / f"{package_name}.mlpackage"
        piece = pieces / f"{package_name.replace('_fp32', '_int8')}.mlpackage"
        print(f"compress {original.name} with int8 per-channel weights")
        model = ct.models.MLModel(str(original), skip_model_load=True)
        compressed = ct.optimize.coreml.linear_quantize_weights(model, config=config)
        compressed.save(str(piece))
        descriptor.add_function(str(piece), "main", shape_tag(package_name))
    descriptor.default_function_name = shape_tag(PACKAGES[args.stage][0])
    print(f"merge compressed shapes -> {destination.name}")
    save_multifunction(descriptor, str(destination))

    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(destination), str(compiled_root)],
        check=True,
    )
    compiled = next(compiled_root.glob("*.mlmodelc"))
    source_bytes = logical_bytes(source)
    package_bytes = logical_bytes(destination)
    compiled_bytes = logical_bytes(compiled)
    print(f"source_package_bytes={source_bytes}")
    print(f"int8_package_bytes={package_bytes}")
    print(f"int8_compiled_bytes={compiled_bytes}")
    print(f"compiled/source={compiled_bytes / source_bytes:.4f}")
    check_parity(destination, PACKAGES[args.stage])


if __name__ == "__main__":
    main()
