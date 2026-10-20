"""Compress weights that are not on the duration or pitch-output path.

Prosody keeps the duration LSTMs and duration projection in fp32. Only the
text-feature branch (embedding, convolutions, and its LSTM) is int8.

Pitch keeps the final F0 convolution and the final energy convolution in fp32.
`internal` also compresses the front LSTM and the earlier convolutions.
`energy` compresses only the loudness convolutions before the final one.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import coremltools as ct
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "fp32-source"  # override with --fp32-source
OUTPUT = ROOT / "artifacts/selective"
MINIMUM_WEIGHT = 2048
INT8 = ct.optimize.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int8",
    granularity="per_channel",
    weight_threshold=MINIMUM_WEIGHT,
)


def logical_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(
        (Path(directory) / name).stat().st_size
        for directory, _, names in os.walk(path)
        for name in names
    )


def block_of(path: Path):
    model = ct.models.MLModel(str(path), skip_model_load=True)
    function = model.get_spec().mlProgram.functions["main"]
    return next(iter(function.block_specializations.values()))


def const_sizes(block) -> dict[str, int]:
    sizes = {}
    for op in block.operations:
        if op.type != "const" or not op.outputs:
            continue
        output = op.outputs[0]
        size = 1
        for dimension in output.type.tensorType.dimensions:
            size *= dimension.constant.size if dimension.HasField("constant") else 1
        sizes[output.name] = size
    return sizes


def large_inputs(op, sizes: dict[str, int]) -> list[str]:
    names = []
    for binding in op.inputs.values():
        for argument in binding.arguments:
            if sizes.get(argument.name, 0) >= MINIMUM_WEIGHT:
                names.append(argument.name)
    return names


def prosody_weights(block) -> list[str]:
    """Text-feature weights. Duration LSTMs and the duration projection stay out."""
    sizes = const_sizes(block)
    selected = []
    for op in block.operations:
        if op.type == "gather" or op.type == "conv":
            selected.extend(large_inputs(op, sizes))
        if op.type == "lstm" and any(item.name.startswith("x_batch_first") for item in op.outputs):
            selected.extend(large_inputs(op, sizes))
    return selected


def pitch_weights(block, mode: str) -> list[str]:
    """Internal pitch weights, leaving the two output convolutions in fp32."""
    sizes = const_sizes(block)
    selected = []
    f0_output = []
    energy_started = False
    last_conv = []
    for op in block.operations:
        if op.type == "conv":
            weights = large_inputs(op, sizes)
            if not weights:
                continue
            if mode == "internal" or energy_started:
                selected.extend(weights)
            if not energy_started:
                f0_output = weights
            last_conv = weights
        if op.type == "linear" and any(
            argument.name and "predictor_N" in argument.name
            for binding in op.inputs.values()
            for argument in binding.arguments
        ):
            energy_started = True
        if mode == "internal" and op.type == "lstm" and not energy_started:
            selected.extend(large_inputs(op, sizes))
    skip = set(f0_output) | set(last_conv)
    return [name for name in selected if name not in skip]


def compress_stage(stage: str, packages: tuple[str, ...], selector) -> None:
    destination = OUTPUT / f"kokoro_{stage}.mlpackage"
    pieces = OUTPUT / "pieces" / stage
    compiled_root = OUTPUT / "compiled" / stage
    shutil.rmtree(destination, ignore_errors=True)
    shutil.rmtree(pieces, ignore_errors=True)
    shutil.rmtree(compiled_root, ignore_errors=True)
    pieces.mkdir(parents=True)
    compiled_root.mkdir(parents=True)
    descriptor = MultiFunctionDescriptor()
    for package_name in packages:
        original = SOURCE / f"{package_name}.mlpackage"
        chosen = selector(block_of(original))
        print(f"{stage} {package_name} weights={len(chosen)} {','.join(chosen)}")
        config = ct.optimize.coreml.OptimizationConfig(
            op_name_configs={name: INT8 for name in chosen}
        )
        model = ct.models.MLModel(str(original), skip_model_load=True)
        compressed = ct.optimize.coreml.linear_quantize_weights(model, config=config)
        piece = pieces / f"{package_name}.mlpackage"
        compressed.save(str(piece))
        tag = next(
            part for part in package_name.split("_")
            if len(part) > 1 and part[0] in "ft" and part[1:].isdigit()
        )
        descriptor.add_function(str(piece), "main", tag)
    descriptor.default_function_name = next(
        part for part in packages[0].split("_")
        if len(part) > 1 and part[0] in "ft" and part[1:].isdigit()
    )
    print(f"merge -> {destination.name}")
    save_multifunction(descriptor, str(destination))
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(destination), str(compiled_root)],
        check=True,
    )
    compiled = next(compiled_root.glob("*.mlmodelc"))
    print(f"{stage}_compiled_bytes={logical_bytes(compiled)}")


def main() -> None:
    global SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prosody", "pitch-internal", "pitch-energy"))
    parser.add_argument("--fp32-source", type=Path, default=SOURCE,
                        help="folder holding the fixed-shape fp32 .mlpackage files from the earlier conversion project")
    args = parser.parse_args()
    SOURCE = args.fp32_source
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "prosody":
        packages = tuple(
            f"kokoro_prosody_text_{tag}_fp32"
            for tag in ("t56", "t110", "t237", "t260", "t280")
        )
        compress_stage("prosody", packages, prosody_weights)
        return
    mode = "internal" if args.stage == "pitch-internal" else "energy"
    packages = tuple(
        f"kokoro_pitch_energy_{tag}_fp32"
        for tag in ("f146", "f275", "f567", "f614", "f633")
    )
    compress_stage(args.stage, packages, lambda block: pitch_weights(block, mode))


if __name__ == "__main__":
    main()
