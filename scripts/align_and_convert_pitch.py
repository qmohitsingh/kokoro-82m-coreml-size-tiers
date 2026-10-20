"""Stretch token features into speech frames, then convert pitch and energy.

Plain-language flow:

    frames per token -> copy each token's features that many times
    stretched features + voice -> pitch curve and loudness curve

The copy step cannot live inside Core ML. Its output length depends on the
predicted frame counts. Pitch and energy are a normal fixed-length network
once that length is known.
"""

from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from kokoro import KModel
from kokoro.istftnet import AdainResBlk1d


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts"


def use_float_residual_scale() -> None:
    """Keep the block math, but multiply by the float 1/sqrt(2).

    The original calls rsqrt on an integer tensor. Eager PyTorch turns that
    into a float. Core ML keeps the integer and then rejects the square root.
    """

    def forward(self, x, s):
        out = self._residual(x, s)
        return (out + self._shortcut(x)) * (2.0 ** -0.5)

    AdainResBlk1d.forward = forward


class PitchEnergy(torch.nn.Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.predictor = model.predictor

    def forward(self, aligned: torch.Tensor, style: torch.Tensor):
        # aligned: one 640-value column per speech frame.
        # style: the 128 voice numbers that control prosody.
        return self.predictor.F0Ntrain(aligned, style)


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    signal = np.sum(reference.astype(np.float64) ** 2)
    noise = np.sum((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2)
    return float("inf") if noise == 0 else float(10 * np.log10(signal / noise))


def repeat_features(features: torch.Tensor, frames_per_token: torch.Tensor) -> torch.Tensor:
    """Copy each token column once per predicted speech frame.

    features is [1, channels, tokens]. The result is [1, channels, frames].
    """
    columns = [
        features[:, :, index : index + 1].repeat(1, 1, int(count))
        for index, count in enumerate(frames_per_token.tolist())
    ]
    return torch.cat(columns, dim=-1)


def reference_outputs(model: KModel, input_ids: torch.Tensor, style: torch.Tensor):
    """Run Kokoro's own math from token ids through pitch and energy."""
    length = input_ids.shape[-1]
    input_lengths = torch.tensor([length], dtype=torch.long)
    text_mask = torch.arange(length).unsqueeze(0) + 1 > input_lengths.unsqueeze(1)
    with torch.no_grad():
        contextual = model.bert(input_ids, attention_mask=(~text_mask).int())
        encoded = model.bert_encoder(contextual).transpose(-1, -2)
        prosody = style[:, 128:]
        duration_features = model.predictor.text_encoder(
            encoded, prosody, input_lengths, text_mask
        )
        recurrent, _ = model.predictor.lstm(duration_features)
        duration = torch.sigmoid(model.predictor.duration_proj(recurrent)).sum(-1)
        frames = torch.round(duration).clamp(min=1).long().squeeze(0)
        token_index = torch.repeat_interleave(torch.arange(length), frames)
        alignment = torch.zeros((length, token_index.shape[0]))
        alignment[token_index, torch.arange(token_index.shape[0])] = 1
        alignment = alignment.unsqueeze(0)
        aligned_prosody = duration_features.transpose(-1, -2) @ alignment
        text_features = model.text_encoder(input_ids, input_lengths, text_mask)
        aligned_text = text_features @ alignment
        pitch, energy = model.predictor.F0Ntrain(aligned_prosody, prosody)
    return frames, duration_features, text_features, aligned_prosody, aligned_text, pitch, energy, prosody


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = np.load(ROOT / "benchmark/inputs/3s.npz")
    input_ids = torch.from_numpy(frozen["input_ids"].astype(np.int32))
    style = torch.from_numpy(frozen["style"].astype(np.float32))
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(ROOT / "models/pytorch/config.json"),
        model=str(ROOT / "models/pytorch/kokoro-v1_0.pth"),
    ).eval()

    (
        frames,
        duration_features,
        text_features,
        aligned,
        aligned_text,
        pitch,
        energy,
        prosody,
    ) = reference_outputs(model, input_ids, style)
    copied_prosody = repeat_features(duration_features.transpose(1, 2), frames)
    copied_text = repeat_features(text_features, frames)
    print(f"frames={int(frames.sum())} tokens={int(frames.numel())}")
    print(f"alignment_prosody_max_abs={torch.max(torch.abs(aligned - copied_prosody)):.9g}")
    print(f"alignment_text_max_abs={torch.max(torch.abs(aligned_text - copied_text)):.9g}")
    if not torch.equal(aligned, copied_prosody) or not torch.equal(aligned_text, copied_text):
        raise RuntimeError("repeat alignment does not match Kokoro's alignment matrix")

    use_float_residual_scale()
    with torch.no_grad():
        patched_pitch, patched_energy = model.predictor.F0Ntrain(copied_prosody, prosody)
    print(f"scale_patch_pitch_max_abs={torch.max(torch.abs(pitch - patched_pitch)):.9g}")
    print(f"scale_patch_energy_max_abs={torch.max(torch.abs(energy - patched_energy)):.9g}")
    if not torch.allclose(pitch, patched_pitch, atol=0, rtol=0):
        raise RuntimeError("residual scale patch changed pitch")
    if not torch.allclose(energy, patched_energy, atol=0, rtol=0):
        raise RuntimeError("residual scale patch changed energy")

    np.save(OUTPUT_DIR / "3s_frames.npy", frames.numpy())
    np.save(OUTPUT_DIR / "3s_aligned_prosody.npy", copied_prosody.numpy())
    np.save(OUTPUT_DIR / "3s_pitch_pytorch.npy", pitch.numpy())
    np.save(OUTPUT_DIR / "3s_energy_pytorch.npy", energy.numpy())

    wrapper = PitchEnergy(model).eval()
    traced = torch.jit.trace(wrapper, (copied_prosody, prosody), strict=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="aligned", shape=tuple(copied_prosody.shape), dtype=np.float32),
            ct.TensorType(name="style", shape=tuple(prosody.shape), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="pitch"), ct.TensorType(name="energy")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    package = OUTPUT_DIR / f"kokoro_pitch_energy_f{int(frames.sum())}_fp32.mlpackage"
    mlmodel.save(package)
    result = mlmodel.predict(
        {"aligned": copied_prosody.numpy(), "style": prosody.numpy()}
    )
    for name, reference in (("pitch", pitch.numpy()), ("energy", energy.numpy())):
        candidate = result[name]
        print(
            f"{name}: shape={candidate.shape}, "
            f"max_abs={np.max(np.abs(reference - candidate)):.9g}, "
            f"snr_db={snr_db(reference, candidate):.3f}"
        )
    print(f"saved={package}")


if __name__ == "__main__":
    main()
