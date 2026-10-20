"""Convert the decoder body for the frozen 3-second clip.

Plain-language flow:

    stretched text + pitch + loudness + voice
        -> the feature map the waveform generator expects

This slice does not make audio. The sine-wave source and the generator
that follows it are later pieces.
"""

from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from kokoro import KModel

from align_and_convert_pitch import reference_outputs, snr_db, use_float_residual_scale


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts"


class DecoderBody(torch.nn.Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        text: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        style: torch.Tensor,
    ):
        # text: 512 values per speech frame.
        # pitch and energy: curves at twice that frame rate.
        # style: the 128 voice numbers that control timbre.
        pitch_features = self.decoder.F0_conv(pitch.unsqueeze(1))
        energy_features = self.decoder.N_conv(energy.unsqueeze(1))
        hidden = torch.cat([text, pitch_features, energy_features], dim=1)
        hidden = self.decoder.encode(hidden, style)
        text_side = self.decoder.asr_res(text)
        for block in self.decoder.decode:
            # Every block in this model still sees the original text, pitch, and energy.
            hidden = torch.cat([hidden, text_side, pitch_features, energy_features], dim=1)
            hidden = block(hidden, style)
        return hidden


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

    frames, _, _, _, text, pitch, energy, _ = reference_outputs(model, input_ids, style)
    timbre = style[:, :128]
    captured = {}

    def remember_generator_input(_module, args):
        captured["features"] = args[0].detach().clone()

    handle = model.decoder.generator.register_forward_pre_hook(remember_generator_input)
    with torch.no_grad():
        model.decoder(text, pitch, energy, timbre)
    handle.remove()
    reference = captured["features"]

    body = DecoderBody(model).eval()
    with torch.no_grad():
        eager = body(text, pitch, energy, timbre)
    print(f"frames={int(frames.sum())} body_shape={tuple(reference.shape)}")
    print(f"wrapper_max_abs={torch.max(torch.abs(reference - eager)):.9g}")
    if not torch.equal(reference, eager):
        raise RuntimeError("decoder body wrapper does not match Kokoro")

    use_float_residual_scale()
    with torch.no_grad():
        patched = body(text, pitch, energy, timbre)
    print(f"scale_patch_max_abs={torch.max(torch.abs(reference - patched)):.9g}")
    if not torch.equal(reference, patched):
        raise RuntimeError("residual scale patch changed the decoder body")

    np.save(OUTPUT_DIR / "3s_decoder_body_pytorch.npy", reference.numpy())
    traced = torch.jit.trace(body, (text, pitch, energy, timbre), strict=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="text", shape=tuple(text.shape), dtype=np.float32),
            ct.TensorType(name="pitch", shape=tuple(pitch.shape), dtype=np.float32),
            ct.TensorType(name="energy", shape=tuple(energy.shape), dtype=np.float32),
            ct.TensorType(name="style", shape=tuple(timbre.shape), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="features")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    package = OUTPUT_DIR / f"kokoro_decoder_body_f{int(frames.sum())}_fp32.mlpackage"
    mlmodel.save(package)
    result = mlmodel.predict(
        {
            "text": text.numpy(),
            "pitch": pitch.numpy(),
            "energy": energy.numpy(),
            "style": timbre.numpy(),
        }
    )
    candidate = result["features"]
    print(
        f"features: shape={candidate.shape}, "
        f"max_abs={np.max(np.abs(reference.numpy() - candidate)):.9g}, "
        f"snr_db={snr_db(reference.numpy(), candidate):.3f}"
    )
    print(f"saved={package}")


if __name__ == "__main__":
    main()
