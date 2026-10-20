"""Check the sine source, then run every converted slice in order.

Plain-language flow:

    pitch curve + two noise arrays -> sine spectrum
    token ids -> text -> timing -> pitch -> decoder -> spectrum -> waveform

The spectrum step stays in PyTorch. It is the same math Kokoro uses, with the
random numbers passed in instead of drawn inside the model. The inverse
spectrogram stays in PyTorch too.
"""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from kokoro import KModel

from align_and_convert_pitch import reference_outputs, repeat_features, snr_db


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts"
REFERENCE_PACKAGES = ROOT / "artifacts" / "fp32-source"  # override with --fp32-source


def source_wave(
    generator: torch.nn.Module,
    pitch: torch.Tensor,
    phase_noise: torch.Tensor,
    wave_noise: torch.Tensor,
    phase_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the sine spectrum the generator expects.

    pitch: [1, frames], the predicted tone curve.
    phase_noise: [1, 9], one starting offset per harmonic. The first is cleared.
    wave_noise: [1, samples, 9], unit noise added onto the sine waves.
    """
    sine = generator.m_source.l_sin_gen
    sampled = generator.f0_upsamp(pitch[:, None]).transpose(1, 2)
    harmonics = torch.arange(1, sine.harmonic_num + 2, dtype=sampled.dtype)
    partials = sampled * harmonics.view(1, 1, -1)
    cycles = (partials / sine.sampling_rate) % 1
    cycles = cycles.clone()
    starting = phase_noise.clone()
    starting[:, 0] = 0
    cycles[:, 0, :] = cycles[:, 0, :] + starting
    cycles = F.interpolate(
        cycles.transpose(1, 2),
        scale_factor=1 / sine.upsample_scale,
        mode="linear",
    ).transpose(1, 2)
    if phase_dtype != cycles.dtype:
        cycles = cycles.to(phase_dtype)
    phase = torch.cumsum(cycles, dim=1) * (2 * torch.pi)
    phase = F.interpolate(
        phase.transpose(1, 2) * sine.upsample_scale,
        scale_factor=sine.upsample_scale,
        mode="linear",
    ).transpose(1, 2)
    waves = torch.sin(phase.to(torch.float32)) * sine.sine_amp
    voiced = (sampled > sine.voiced_threshold).to(waves.dtype)
    noise_amp = voiced * sine.noise_std + (1 - voiced) * sine.sine_amp / 3
    waves = waves * voiced + noise_amp * wave_noise
    merged = generator.m_source.l_tanh(generator.m_source.l_linear(waves))
    return merged.transpose(1, 2).squeeze(1)


def source_spectrum(
    generator: torch.nn.Module,
    pitch: torch.Tensor,
    phase_noise: torch.Tensor,
    wave_noise: torch.Tensor,
    phase_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    waveform = source_wave(generator, pitch, phase_noise, wave_noise, phase_dtype)
    magnitude, angle = generator.stft.transform(waveform)
    return torch.cat([magnitude, angle], dim=1)


def capture_noise(model: KModel, text, pitch, energy, timbre):
    """Run Kokoro's decoder once and keep the random draws it used."""
    drawn = []
    original_rand = torch.rand
    original_randn_like = torch.randn_like
    spectrum = {}

    def remember_rand(*args, **kwargs):
        value = original_rand(*args, **kwargs)
        drawn.append(value.detach().clone())
        return value

    def remember_randn_like(tensor, *args, **kwargs):
        value = original_randn_like(tensor, *args, **kwargs)
        drawn.append(value.detach().clone())
        return value

    generator = model.decoder.generator
    original_transform = generator.stft.transform

    def remember_spectrum(wave):
        magnitude, angle = original_transform(wave)
        spectrum["value"] = torch.cat([magnitude, angle], dim=1).detach().clone()
        return magnitude, angle

    torch.rand = remember_rand
    torch.randn_like = remember_randn_like
    generator.stft.transform = remember_spectrum
    try:
        with torch.no_grad():
            audio = model.decoder(text, pitch, energy, timbre)
    finally:
        torch.rand = original_rand
        torch.randn_like = original_randn_like
        generator.stft.transform = original_transform
    return drawn, spectrum["value"], audio


def predict(path: Path, feeds: dict, function_name: str | None = None) -> dict:
    model = ct.models.MLModel(
        str(path),
        function_name=function_name,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    result = model.predict(feeds)
    return {key: np.asarray(value) for key, value in result.items()}


def report(name: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    print(
        f"{name}: shape={candidate.shape}, "
        f"max_abs={np.max(np.abs(reference - candidate)):.9g}, "
        f"snr_db={snr_db(reference, candidate):.3f}"
    )


def main() -> None:
    global REFERENCE_PACKAGES
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--int8-stages",
        default="",
        help="comma-separated later stages: pitch,decoder,generator",
    )
    parser.add_argument("--fp32-source", type=Path, default=REFERENCE_PACKAGES,
                        help="folder holding the fixed-shape fp32 .mlpackage files from the earlier conversion project")
    args = parser.parse_args()
    REFERENCE_PACKAGES = args.fp32_source
    int8_stages = {stage for stage in args.int8_stages.split(",") if stage}
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
    torch.manual_seed(0)
    drawn, reference_spectrum, reference_audio = capture_noise(
        model, text, pitch, energy, timbre
    )
    print("noise_shapes=" + ",".join(str(tuple(item.shape)) for item in drawn))
    phase_noise = next(item for item in drawn if item.shape[-1] == 9 and item.ndim == 2)
    wave_noise = next(item for item in drawn if item.ndim == 3 and item.shape[-1] == 9)

    with torch.no_grad():
        rebuilt = source_spectrum(model.decoder.generator, pitch, phase_noise, wave_noise)
        reference_wave = source_wave(model.decoder.generator, pitch, phase_noise, wave_noise)
        double_wave = source_wave(
            model.decoder.generator,
            pitch,
            phase_noise,
            wave_noise,
            phase_dtype=torch.float64,
        )
    print(f"source_max_abs={torch.max(torch.abs(reference_spectrum - rebuilt)):.9g}")
    if not torch.equal(reference_spectrum, rebuilt):
        raise RuntimeError("sine source does not match Kokoro")
    print(
        f"double_phase_wave_snr_db={snr_db(reference_wave.numpy(), double_wave.numpy()):.3f} "
        f"max_abs={torch.max(torch.abs(reference_wave - double_wave)):.9g}"
    )

    encoded = predict(
        REFERENCE_PACKAGES / "kokoro_text_t56_fp32.mlpackage",
        {"input_ids": input_ids.numpy()},
    )["encoded"]
    prosody = predict(
        REFERENCE_PACKAGES / "kokoro_prosody_text_t56_fp32.mlpackage",
        {
            "input_ids": input_ids.numpy(),
            "encoded": encoded.astype(np.float32),
            "style": style.numpy(),
        },
    )
    logits = torch.from_numpy(prosody["duration_logits"].astype(np.float32))
    chain_frames = torch.round(torch.sigmoid(logits).sum(-1)).clamp(min=1).long().squeeze(0)
    changed = int(torch.count_nonzero(chain_frames != frames))
    print(f"changed_duration_tokens={changed} frames={int(chain_frames.sum())}")
    if changed or int(chain_frames.sum()) != int(frames.sum()):
        raise RuntimeError("Core ML durations changed the frame count")

    hidden = torch.from_numpy(prosody["duration_hidden"].astype(np.float32))
    text_features = torch.from_numpy(prosody["text_features"].astype(np.float32))
    aligned_prosody = repeat_features(hidden.transpose(1, 2), chain_frames)
    aligned_text = repeat_features(text_features, chain_frames)
    pitch_package = (
        OUTPUT_DIR / "int8/kokoro_pitch_int8.mlpackage"
        if "pitch" in int8_stages
        else REFERENCE_PACKAGES / "kokoro_pitch_energy_f146_fp32.mlpackage"
    )
    decoder_package = (
        OUTPUT_DIR / "int8/kokoro_decoder_int8.mlpackage"
        if "decoder" in int8_stages
        else REFERENCE_PACKAGES / "kokoro_decoder_body_f146_fp32.mlpackage"
    )
    generator_package = (
        OUTPUT_DIR / "int8/kokoro_generator_int8.mlpackage"
        if "generator" in int8_stages
        else REFERENCE_PACKAGES / "kokoro_generator_f146_fp32.mlpackage"
    )
    curves = predict(
        pitch_package,
        {
            "aligned": aligned_prosody.numpy(),
            "style": style[:, 128:].numpy(),
        },
        "f146" if "pitch" in int8_stages else None,
    )
    chain_pitch = torch.from_numpy(curves["pitch"].astype(np.float32))
    chain_energy = torch.from_numpy(curves["energy"].astype(np.float32))
    body = predict(
        decoder_package,
        {
            "text": aligned_text.numpy(),
            "pitch": chain_pitch.numpy(),
            "energy": chain_energy.numpy(),
            "style": timbre.numpy(),
        },
        "f146" if "decoder" in int8_stages else None,
    )["features"]
    with torch.no_grad():
        chain_wave = source_wave(
            model.decoder.generator, chain_pitch, phase_noise, wave_noise
        )
        chain_spectrum = source_spectrum(
            model.decoder.generator, chain_pitch, phase_noise, wave_noise
        )
    report("source_wave", reference_wave.numpy(), chain_wave.numpy())
    generated = predict(
        generator_package,
        {
            "features": np.asarray(body, dtype=np.float32),
            "style": timbre.numpy(),
            "spectrum": chain_spectrum.numpy(),
        },
        "f146" if "generator" in int8_stages else None,
    )
    with torch.no_grad():
        chain_audio = model.decoder.generator.stft.inverse(
            torch.from_numpy(generated["magnitude"].astype(np.float32)),
            torch.from_numpy(generated["phase"].astype(np.float32)),
        )
    report("pitch", pitch.numpy(), chain_pitch.numpy())
    report("energy", energy.numpy(), chain_energy.numpy())
    report("spectrum", reference_spectrum.numpy(), chain_spectrum.numpy())
    report("audio", reference_audio.numpy(), chain_audio.numpy())
    suffix = "int8_" + "_".join(sorted(int8_stages)) if int8_stages else "fp32"
    np.save(OUTPUT_DIR / f"3s_chain_audio_{suffix}.npy", chain_audio.numpy())
    reference_wav = OUTPUT_DIR / "3s_pytorch_seed0.wav"
    chain_wav = OUTPUT_DIR / f"3s_chain_seed0_{suffix}.wav"
    sf.write(reference_wav, reference_audio.detach().float().squeeze().numpy(), 24000)
    sf.write(chain_wav, chain_audio.detach().float().squeeze().numpy(), 24000)
    print(f"saved={reference_wav}")
    print(f"saved={chain_wav}")


if __name__ == "__main__":
    main()
