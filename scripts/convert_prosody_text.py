"""Convert Kokoro's fixed-length recurrent text section.

Plain-language flow:

    ALBERT features + voice -> duration features -> frames per token
    token ids -> text features used later by the decoder

The wrappers skip pack/unpack helpers because all 56 positions are real.
The LSTM math and weights are unchanged.
"""

from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from kokoro import KModel
from kokoro.modules import AdaLayerNorm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts"


class ProsodyText(torch.nn.Module):
    def __init__(self, model: KModel, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.duration_blocks = model.predictor.text_encoder.lstms
        self.duration_dropout = model.predictor.text_encoder.dropout
        self.duration_lstm = model.predictor.lstm
        self.duration_proj = model.predictor.duration_proj
        self.text_encoder = model.text_encoder

    def duration_encoder(
        self, encoded: torch.Tensor, style: torch.Tensor
    ) -> torch.Tensor:
        # encoded [B, 512, T], style [B, 128]
        x = encoded.permute(2, 0, 1)
        repeated_style = style.unsqueeze(0).repeat(self.sequence_length, 1, 1)
        x = torch.cat([x, repeated_style], dim=-1)
        x = x.transpose(0, 1).transpose(-1, -2)
        for block in self.duration_blocks:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, repeated_style.permute(1, 2, 0)], dim=1)
            else:
                x, _ = block(x.transpose(-1, -2))
                x = F.dropout(x, p=self.duration_dropout, training=False)
                x = x.transpose(-1, -2)
        return x.transpose(-1, -2)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.text_encoder.embedding(input_ids).transpose(1, 2)
        for block in self.text_encoder.cnn:
            x = block(x)
        x, _ = self.text_encoder.lstm(x.transpose(1, 2))
        return x.transpose(1, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoded: torch.Tensor,
        style: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prosody_style = style[:, 128:]
        duration_hidden = self.duration_encoder(encoded, prosody_style)
        recurrent, _ = self.duration_lstm(duration_hidden)
        duration_logits = self.duration_proj(recurrent)
        text_features = self.encode_text(input_ids)
        return duration_hidden, duration_logits, text_features


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    signal = np.sum(reference.astype(np.float64) ** 2)
    noise = np.sum((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2)
    return float("inf") if noise == 0 else float(10 * np.log10(signal / noise))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = np.load(ROOT / "benchmark/inputs/3s.npz")
    input_ids = frozen["input_ids"].astype(np.int32)
    style = frozen["style"].astype(np.float32)
    encoded = np.load(ROOT / "artifacts/text_t56_pytorch.npy").astype(np.float32)

    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(ROOT / "models/pytorch/config.json"),
        model=str(ROOT / "models/pytorch/kokoro-v1_0.pth"),
    ).eval()
    wrapper = ProsodyText(model, sequence_length=input_ids.shape[1]).eval()
    ids_tensor = torch.from_numpy(input_ids)
    encoded_tensor = torch.from_numpy(encoded)
    style_tensor = torch.from_numpy(style)

    # Prove that skipping pack/unpack did not change PyTorch.
    length = input_ids.shape[1]
    lengths = torch.tensor([length], dtype=torch.long)
    mask = torch.zeros((1, length), dtype=torch.bool)
    with torch.no_grad():
        original_duration = model.predictor.text_encoder(
            encoded_tensor, style_tensor[:, 128:], lengths, mask
        )
        original_recurrent, _ = model.predictor.lstm(original_duration)
        original_logits = model.predictor.duration_proj(original_recurrent)
        original_text = model.text_encoder(ids_tensor, lengths, mask)
        rewritten = wrapper(ids_tensor, encoded_tensor, style_tensor)

    originals = (original_duration, original_logits, original_text)
    names = ("duration_hidden", "duration_logits", "text_features")
    for name, reference, candidate in zip(names, originals, rewritten):
        error = float(torch.max(torch.abs(reference - candidate)))
        wrapper_snr = snr_db(reference.numpy(), candidate.numpy())
        print(f"wrapper_{name}_max_abs={error:.9g}")
        print(f"wrapper_{name}_snr_db={wrapper_snr:.3f}")
        if wrapper_snr < 100:
            raise RuntimeError(f"wrapper changed {name}: {wrapper_snr} dB")

    traced = torch.jit.trace(
        wrapper,
        (ids_tensor, encoded_tensor, style_tensor),
        strict=True,
    )
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
            ct.TensorType(name="encoded", shape=encoded.shape, dtype=np.float32),
            ct.TensorType(name="style", shape=style.shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="duration_hidden"),
            ct.TensorType(name="duration_logits"),
            ct.TensorType(name="text_features"),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    package = OUTPUT_DIR / "kokoro_prosody_text_t56_fp32.mlpackage"
    mlmodel.save(package)

    result = mlmodel.predict(
        {"input_ids": input_ids, "encoded": encoded, "style": style}
    )
    for name, reference in zip(names, originals):
        expected = reference.numpy()
        candidate = result[name]
        print(
            f"{name}: shape={candidate.shape}, "
            f"max_abs={np.max(np.abs(expected - candidate)):.9g}, "
            f"snr_db={snr_db(expected, candidate):.3f}"
        )

    reference_frames = (
        torch.round(torch.sigmoid(original_logits).sum(axis=-1))
        .clamp(min=1)
        .long()
        .numpy()
    )
    coreml_frames = np.maximum(
        np.round(
            (1 / (1 + np.exp(-result["duration_logits"]))).sum(axis=-1)
        ),
        1,
    ).astype(np.int64)
    print(f"changed_duration_tokens={np.count_nonzero(reference_frames != coreml_frames)}")
    print(f"saved={package}")


if __name__ == "__main__":
    main()
