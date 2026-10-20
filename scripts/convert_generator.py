"""Convert the waveform generator for one frozen noise draw.

Plain-language flow:

    pitch curve -> sine source -> spectrum   (this stays in PyTorch)
    decoder features + spectrum + voice -> waveform

The sine source uses a running phase sum and fresh random noise. Those stay
out of Core ML. The inverse spectrogram uses complex numbers, which this
converter cannot load, so it stays in PyTorch too. This script freezes one
noise draw and converts the conv stack through the magnitude and phase.
"""

from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from kokoro import KModel

from align_and_convert_pitch import reference_outputs, snr_db


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts"
NOISE_SEED = 0


class WaveformGenerator(torch.nn.Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.generator = model.decoder.generator

    def forward(self, features: torch.Tensor, style: torch.Tensor, spectrum: torch.Tensor):
        # features: decoder body output, 512 values at the doubled frame rate.
        # spectrum: magnitude then phase of the sine source.
        # style: the 128 voice numbers that control timbre.
        generator = self.generator
        hidden = features
        for stage in range(generator.num_upsamples):
            hidden = F.leaky_relu(hidden, negative_slope=0.1)
            source = generator.noise_convs[stage](spectrum)
            source = generator.noise_res[stage](source, style)
            hidden = generator.ups[stage](hidden)
            if stage == generator.num_upsamples - 1:
                hidden = generator.reflection_pad(hidden)
            hidden = hidden + source
            combined = None
            for kernel in range(generator.num_kernels):
                block = generator.resblocks[stage * generator.num_kernels + kernel](hidden, style)
                combined = block if combined is None else combined + block
            hidden = combined / generator.num_kernels
        hidden = F.leaky_relu(hidden)
        hidden = generator.conv_post(hidden)
        bins = generator.post_n_fft // 2 + 1
        magnitude = torch.exp(hidden[:, :bins, :])
        phase = torch.sin(hidden[:, bins:, :])
        return magnitude, phase


def capture_reference(model: KModel, text, pitch, energy, timbre):
    """Run Kokoro's decoder once and keep the sine spectrum and waveform."""
    captured = {}
    generator = model.decoder.generator
    original = generator.stft.transform

    def remember(source_wave):
        magnitude, phase = original(source_wave)
        captured["spectrum"] = torch.cat([magnitude, phase], dim=1).detach().clone()
        return magnitude, phase

    def remember_features(_module, args):
        captured["features"] = args[0].detach().clone()

    generator.stft.transform = remember
    feature_hook = generator.register_forward_pre_hook(remember_features)
    torch.manual_seed(NOISE_SEED)
    with torch.no_grad():
        audio = model.decoder(text, pitch, energy, timbre)
    feature_hook.remove()
    generator.stft.transform = original
    return captured["features"], captured["spectrum"], audio


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
    features, spectrum, audio = capture_reference(model, text, pitch, energy, timbre)
    wrapper = WaveformGenerator(model).eval()
    with torch.no_grad():
        magnitude, phase = wrapper(features, timbre, spectrum)
        eager = model.decoder.generator.stft.inverse(magnitude, phase)
    print(f"frames={int(frames.sum())} audio_shape={tuple(audio.shape)} spectrum_shape={tuple(spectrum.shape)}")
    print(f"wrapper_max_abs={torch.max(torch.abs(audio - eager)):.9g}")
    if not torch.equal(audio, eager):
        raise RuntimeError("generator wrapper does not match Kokoro")

    np.save(OUTPUT_DIR / "3s_generator_features.npy", features.numpy())
    np.save(OUTPUT_DIR / "3s_source_spectrum.npy", spectrum.numpy())
    np.save(OUTPUT_DIR / "3s_audio_pytorch.npy", audio.numpy())
    traced = torch.jit.trace(wrapper, (features, timbre, spectrum), strict=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="features", shape=tuple(features.shape), dtype=np.float32),
            ct.TensorType(name="style", shape=tuple(timbre.shape), dtype=np.float32),
            ct.TensorType(name="spectrum", shape=tuple(spectrum.shape), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="magnitude"), ct.TensorType(name="phase")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    package = OUTPUT_DIR / f"kokoro_generator_f{int(frames.sum())}_fp32.mlpackage"
    mlmodel.save(package)
    result = mlmodel.predict(
        {
            "features": features.numpy(),
            "style": timbre.numpy(),
            "spectrum": spectrum.numpy(),
        }
    )
    with torch.no_grad():
        reconstructed = model.decoder.generator.stft.inverse(
            torch.from_numpy(result["magnitude"]),
            torch.from_numpy(result["phase"]),
        )
    candidate = reconstructed.numpy()
    reference = audio.numpy()
    print(
        f"magnitude: shape={result['magnitude'].shape}, "
        f"max_abs={np.max(np.abs(magnitude.numpy() - result['magnitude'])):.9g}, "
        f"snr_db={snr_db(magnitude.numpy(), result['magnitude']):.3f}"
    )
    print(
        f"phase: shape={result['phase'].shape}, "
        f"max_abs={np.max(np.abs(phase.numpy() - result['phase'])):.9g}, "
        f"snr_db={snr_db(phase.numpy(), result['phase']):.3f}"
    )
    print(
        f"audio: shape={candidate.shape}, "
        f"max_abs={np.max(np.abs(reference - candidate)):.9g}, "
        f"snr_db={snr_db(reference, candidate):.3f}"
    )
    print(f"saved={package}")


if __name__ == "__main__":
    main()
