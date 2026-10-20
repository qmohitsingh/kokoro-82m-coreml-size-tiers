"""Build one selectively compressed multifunction model for sensitivity sweeps."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import coremltools as ct
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "fp32-source"  # override with --fp32-source
OUTPUT = ROOT / "artifacts/sweeps"
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
GROUPS = {
    "pitch-f0-lstm": ("concat_4", "concat_5", "concat_6", "concat_7"),
    "pitch-f0-block0": ("weight_3", "weight_7"),
    "pitch-f0-block1": ("weight_11", "weight_15", "weight_17"),
    "pitch-f0-block2": ("weight_21", "weight_25"),
    "pitch-f0-style": (
        "predictor_F0_0_norm1_fc_weight",
        "predictor_F0_0_norm2_fc_weight",
        "predictor_F0_1_norm1_fc_weight",
        "predictor_F0_1_norm2_fc_weight",
        "predictor_F0_2_norm1_fc_weight",
        "predictor_F0_2_norm2_fc_weight",
    ),
    "text-embedding": (
        "bert_embeddings_word_embeddings_weight",
        "token_type_embeddings_1",
        "position_embeddings_1",
        "bert_encoder_embedding_hidden_mapping_in_weight",
    ),
    "text-attention": (
        "bert_encoder_albert_layer_groups_0_albert_layers_0_attention_query_weight",
        "bert_encoder_albert_layer_groups_0_albert_layers_0_attention_key_weight",
        "bert_encoder_albert_layer_groups_0_albert_layers_0_attention_value_weight",
        "bert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense_weight",
    ),
    "text-ffn": (
        "bert_encoder_albert_layer_groups_0_albert_layers_0_ffn_weight",
        "bert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output_weight",
    ),
    "text-output": ("bert_encoder_weight",),
    "prosody-text": (
        "text_encoder_embedding_weight",
        "weight_3",
        "weight_7",
        "weight_11",
        "concat_44",
        "concat_45",
        "concat_46",
        "concat_47",
    ),
    "decoder-encode": (
        "decoder_encode_norm1_fc_weight",
        "decoder_encode_norm2_fc_weight",
        "weight_7",
        "weight_11",
        "weight_13",
        "weight_15",
    ),
    "decoder-block0": (
        "decoder_decode_0_norm1_fc_weight",
        "decoder_decode_0_norm2_fc_weight",
        "weight_19",
        "weight_23",
        "weight_25",
    ),
    "decoder-block1": (
        "decoder_decode_1_norm1_fc_weight",
        "decoder_decode_1_norm2_fc_weight",
        "weight_29",
        "weight_33",
        "weight_35",
    ),
    "decoder-block2": (
        "decoder_decode_2_norm1_fc_weight",
        "decoder_decode_2_norm2_fc_weight",
        "weight_39",
        "weight_43",
        "weight_45",
    ),
    "decoder-block3": (
        "decoder_decode_3_norm1_fc_weight",
        "decoder_decode_3_norm2_fc_weight",
        "var_502",
        "weight_49",
        "weight_53",
        "weight_55",
    ),
    "generator-upsample": ("var_365", "var_1538"),
    "generator-noise": (
        "generator_noise_convs_0_weight",
        "weight_5", "weight_9", "weight_13", "weight_17", "weight_21", "weight_25",
        "generator_noise_convs_1_weight",
        "weight_103", "weight_107", "weight_111", "weight_115", "weight_119", "weight_123",
    ),
    "generator-res0-2": (
        "weight_29", "weight_33", "weight_37", "weight_41", "weight_45", "weight_49",
        "weight_53", "weight_57", "weight_61", "weight_65", "weight_69", "weight_73",
        "weight_77", "weight_81", "weight_85", "weight_89", "weight_93", "weight_97",
    ),
    "generator-res3-5": (
        "weight_127", "weight_131", "weight_135", "weight_139", "weight_143", "weight_147",
        "weight_151", "weight_155", "weight_159", "weight_163", "weight_167", "weight_171",
        "weight_175", "weight_179", "weight_183", "weight_187", "weight_191", "weight_195",
    ),
    "generator-output": ("weight_197",),
    # Last-stage residual blocks, kernel 3, 7, 11 (128 channels).
    "generator-res3": tuple(f"weight_{i}" for i in range(127, 148, 4)),
    "generator-res4": tuple(f"weight_{i}" for i in range(151, 172, 4)),
    "generator-res5": tuple(f"weight_{i}" for i in range(175, 196, 4)),
    # res5 split into its three (convs1, convs2) pairs, one per dilation.
    "generator-res5-p0": ("weight_175", "weight_179"),
    "generator-res5-p1": ("weight_183", "weight_187"),
    "generator-res5-p2": ("weight_191", "weight_195"),
}


def logical_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(
        (Path(directory) / name).stat().st_size
        for directory, _, names in os.walk(path)
        for name in names
    )


def compressor(kind: str):
    if kind in ("int8", "int4"):
        return ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=kind,
            granularity="per_channel",
            weight_threshold=2048,
        )
    bits = int(kind.removeprefix("pal"))
    return ct.optimize.coreml.OpPalettizerConfig(
        mode="kmeans",
        nbits=bits,
        granularity="per_tensor",
        weight_threshold=2048,
    )


def main() -> None:
    global SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(SHAPES))
    parser.add_argument(
        "kind", choices=("int8", "int4", "mixed4", "mixed8", "pal4", "pal6", "pal8")
    )
    parser.add_argument("--group", choices=tuple(GROUPS), nargs="+", default=[])
    parser.add_argument("--name")
    parser.add_argument("--fp32-source", type=Path, default=SOURCE,
                        help="folder holding the fixed-shape fp32 .mlpackage files from the earlier conversion project")
    args = parser.parse_args()
    SOURCE = args.fp32_source
    for group in args.group:
        if not group.startswith(args.stage + "-"):
            raise SystemExit(f"{group} is not a {args.stage} group")
    label = "+".join(group.removeprefix(args.stage + "-") for group in args.group)
    name = args.name or (f"{args.stage}-{label}" if label else f"{args.stage}-all") + "-" + args.kind
    destination = OUTPUT / f"kokoro_{name}.mlpackage"
    pieces = OUTPUT / "pieces" / name
    compiled_root = OUTPUT / "compiled" / name
    shutil.rmtree(destination, ignore_errors=True)
    shutil.rmtree(pieces, ignore_errors=True)
    shutil.rmtree(compiled_root, ignore_errors=True)
    pieces.mkdir(parents=True)
    compiled_root.mkdir(parents=True)
    selected = tuple(op for group in args.group for op in GROUPS[group])
    if args.kind == "mixed4":
        if not selected:
            raise SystemExit("mixed4 requires --group")
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=compressor("int8"),
            op_name_configs={op_name: compressor("int4") for op_name in selected},
        )
    elif args.kind == "mixed8":
        # The reverse of mixed4: int4 everywhere, the selected group stays int8.
        if not selected:
            raise SystemExit("mixed8 requires --group")
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=compressor("int4"),
            op_name_configs={op_name: compressor("int8") for op_name in selected},
        )
    else:
        config_value = compressor(args.kind)
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=None if selected else config_value,
            op_name_configs={op_name: config_value for op_name in selected},
        )
    descriptor = MultiFunctionDescriptor()
    for tag in SHAPES[args.stage]:
        package_name = f"{PREFIX[args.stage]}_{tag}_fp32"
        original = SOURCE / f"{package_name}.mlpackage"
        piece = pieces / f"{package_name}_{args.kind}.mlpackage"
        print(f"compress {package_name} kind={args.kind} group={label or 'all'}")
        model = ct.models.MLModel(str(original), skip_model_load=True)
        if args.kind.startswith("pal"):
            compressed = ct.optimize.coreml.palettize_weights(model, config=config)
        else:
            compressed = ct.optimize.coreml.linear_quantize_weights(model, config=config)
        compressed.save(str(piece))
        descriptor.add_function(str(piece), "main", tag)
    descriptor.default_function_name = SHAPES[args.stage][0]
    save_multifunction(descriptor, str(destination))
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(destination), str(compiled_root)],
        check=True,
    )
    compiled = next(compiled_root.glob("*.mlmodelc"))
    print(f"variant={name}")
    print(f"compiled_bytes={logical_bytes(compiled)}")
    print(f"package={destination}")


if __name__ == "__main__":
    main()
