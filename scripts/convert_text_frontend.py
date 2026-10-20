"""Convert Kokoro's deterministic ALBERT text front end for one fixed input length.

Plain-language flow:

    token ids -> ALBERT understands their context -> linear layer makes 512 features

This is only the first conversion slice. It does not make audio.
"""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from kokoro import KModel


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "benchmark/inputs/3s.npz"
CONFIG = ROOT / "models/pytorch/config.json"
CHECKPOINT = ROOT / "models/pytorch/kokoro-v1_0.pth"
OUTPUT_DIR = ROOT / "artifacts"


class TextFrontEnd(torch.nn.Module):
    """Turn speech-sound token numbers into contextual 512-value features."""

    def __init__(self, model: KModel, sequence_length: int):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        # Hugging Face slices these from a 512-position buffer using a tensor
        # converted to a Python int. Core ML cannot lower that helper. The
        # graph is fixed at 56 tokens, so provide the exact equivalent tensors.
        self.register_buffer(
            "position_ids",
            torch.arange(sequence_length, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros((1, sequence_length), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "attention_bias",
            torch.zeros((1, 1, 1, sequence_length), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.bert.embeddings(
            input_ids,
            position_ids=self.position_ids,
            token_type_ids=self.token_type_ids,
        )
        # All 56 positions are real. The equivalent expanded attention mask is
        # zero. Supplying it directly avoids Hugging Face's finfo.min cast,
        # which overflows when coremltools changes the graph to fp16.
        contextual = self.bert.encoder(
            embeddings,
            attention_mask=self.attention_bias,
            return_dict=False,
        )[0]
        return self.bert_encoder(contextual).transpose(1, 2)


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    signal = np.sum(reference.astype(np.float64) ** 2)
    noise = np.sum((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2)
    return float("inf") if noise == 0 else float(10 * np.log10(signal / noise))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = np.load(INPUT)
    input_ids = frozen["input_ids"].astype(np.int32)

    print("Load pinned PyTorch Kokoro checkpoint")
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(CONFIG),
        model=str(CHECKPOINT),
    ).eval()
    wrapper = TextFrontEnd(model, sequence_length=input_ids.shape[1]).eval()

    example = torch.from_numpy(input_ids)
    with torch.no_grad():
        original = model.bert(
            example,
            attention_mask=torch.ones_like(example, dtype=torch.int32),
        )
        original = model.bert_encoder(original).transpose(1, 2).cpu().numpy()
        pytorch_output = wrapper(example).cpu().numpy()
    rewrite_max_abs = float(np.max(np.abs(original - pytorch_output)))
    if rewrite_max_abs > 1e-6:
        raise RuntimeError(f"wrapper changed PyTorch output: {rewrite_max_abs}")
    np.save(OUTPUT_DIR / "text_t56_pytorch.npy", pytorch_output)
    print("PyTorch output", pytorch_output.shape, pytorch_output.dtype)
    print(f"wrapper_max_abs={rewrite_max_abs:.9g}")

    print("Trace the fixed 56-token graph")
    traced = torch.jit.trace(wrapper, example, strict=True)

    precision = (
        ct.precision.FLOAT32 if args.precision == "fp32" else ct.precision.FLOAT16
    )
    print(f"Convert to {args.precision} Core ML")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=input_ids.shape,
                dtype=np.int32,
            )
        ],
        outputs=[ct.TensorType(name="encoded")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    package = OUTPUT_DIR / f"kokoro_text_t56_{args.precision}.mlpackage"
    mlmodel.save(package)

    print("Run Core ML on the Mac")
    coreml_output = mlmodel.predict({"input_ids": input_ids})["encoded"]
    np.save(OUTPUT_DIR / f"text_t56_coreml_{args.precision}.npy", coreml_output)
    max_abs = float(np.max(np.abs(pytorch_output - coreml_output)))
    snr = snr_db(pytorch_output, coreml_output)
    print(f"Core ML output {coreml_output.shape} {coreml_output.dtype}")
    print(f"max_abs={max_abs:.9g}")
    print(f"snr_db={snr:.3f}")
    print(f"saved={package}")


if __name__ == "__main__":
    main()
