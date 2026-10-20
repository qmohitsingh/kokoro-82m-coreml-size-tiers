"""Convert one frozen clip and compare the joined waveform with PyTorch.

Plain-language flow:

    token ids -> text -> timing -> pitch -> decoder -> sine spectrum -> waveform

Each clip has its own fixed sizes. The 3-second packages stay as they are.
"""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from kokoro import KModel

from align_and_convert_pitch import (
    reference_outputs,
    repeat_features,
    snr_db,
    use_float_residual_scale,
    PitchEnergy,
)
from check_source_and_chain import capture_noise, predict, report, source_spectrum, source_wave
from convert_decoder_body import DecoderBody
from convert_generator import WaveformGenerator
from convert_prosody_text import ProsodyText
from convert_text_frontend import TextFrontEnd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts"
CASES = ("7s", "15s", "30s-0", "30s-1")


def load_model() -> KModel:
    return KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(ROOT / "models/pytorch/config.json"),
        model=str(ROOT / "models/pytorch/kokoro-v1_0.pth"),
    ).eval()


def convert(module, example, inputs, outputs, package: Path):
    traced = torch.jit.trace(module, example, strict=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    mlmodel.save(package)
    return mlmodel


def convert_text(model, input_ids: torch.Tensor) -> Path:
    tokens = input_ids.shape[1]
    wrapper = TextFrontEnd(model, sequence_length=tokens).eval()
    with torch.no_grad():
        original = model.bert(input_ids, attention_mask=torch.ones_like(input_ids))
        original = model.bert_encoder(original).transpose(1, 2)
        rewritten = wrapper(input_ids)
    error = torch.max(torch.abs(original - rewritten))
    print(f"text_wrapper_max_abs={error:.9g}")
    if float(error) > 1e-6:
        raise RuntimeError("text wrapper changed PyTorch")
    package = OUTPUT_DIR / f"kokoro_text_t{tokens}_fp32.mlpackage"
    mlmodel = convert(
        wrapper,
        input_ids,
        [ct.TensorType(name="input_ids", shape=tuple(input_ids.shape), dtype=np.int32)],
        [ct.TensorType(name="encoded")],
        package,
    )
    encoded = mlmodel.predict({"input_ids": input_ids.numpy()})["encoded"]
    report("text", original.numpy(), encoded)
    del mlmodel
    return package


def convert_prosody(model, input_ids, style, encoded) -> tuple[Path, torch.Tensor]:
    tokens = input_ids.shape[1]
    wrapper = ProsodyText(model, sequence_length=tokens).eval()
    lengths = torch.tensor([tokens], dtype=torch.long)
    mask = torch.zeros((1, tokens), dtype=torch.bool)
    with torch.no_grad():
        duration = model.predictor.text_encoder(encoded, style[:, 128:], lengths, mask)
        recurrent, _ = model.predictor.lstm(duration)
        logits = model.predictor.duration_proj(recurrent)
        text = model.text_encoder(input_ids, lengths, mask)
        rewritten = wrapper(input_ids, encoded, style)
    for name, reference, candidate in zip(
        ("duration_hidden", "duration_logits", "text_features"),
        (duration, logits, text),
        rewritten,
    ):
        score = snr_db(reference.numpy(), candidate.numpy())
        print(f"prosody_wrapper_{name}_snr_db={score:.3f}")
        if score < 100:
            raise RuntimeError(f"prosody wrapper changed {name}")
    package = OUTPUT_DIR / f"kokoro_prosody_text_t{tokens}_fp32.mlpackage"
    mlmodel = convert(
        wrapper,
        (input_ids, encoded, style),
        [
            ct.TensorType(name="input_ids", shape=tuple(input_ids.shape), dtype=np.int32),
            ct.TensorType(name="encoded", shape=tuple(encoded.shape), dtype=np.float32),
            ct.TensorType(name="style", shape=tuple(style.shape), dtype=np.float32),
        ],
        [
            ct.TensorType(name="duration_hidden"),
            ct.TensorType(name="duration_logits"),
            ct.TensorType(name="text_features"),
        ],
        package,
    )
    result = mlmodel.predict(
        {"input_ids": input_ids.numpy(), "encoded": encoded.numpy(), "style": style.numpy()}
    )
    for name, reference in zip(
        ("duration_hidden", "duration_logits", "text_features"),
        (duration, logits, text),
    ):
        report(name, reference.numpy(), result[name])
    reference_frames = torch.round(torch.sigmoid(logits).sum(-1)).clamp(min=1).long()
    candidate_frames = torch.round(
        torch.sigmoid(torch.from_numpy(result["duration_logits"])).sum(-1)
    ).clamp(min=1).long()
    changed = int(torch.count_nonzero(reference_frames != candidate_frames))
    print(f"prosody_changed_duration_tokens={changed}")
    if changed:
        raise RuntimeError("Core ML duration rounding changed a token")
    del mlmodel
    return package, reference_frames.squeeze(0)


def convert_pitch(model, aligned, prosody, pitch, energy) -> Path:
    use_float_residual_scale()
    with torch.no_grad():
        patched_pitch, patched_energy = model.predictor.F0Ntrain(aligned, prosody)
    if not torch.equal(pitch, patched_pitch) or not torch.equal(energy, patched_energy):
        raise RuntimeError("residual scale patch changed pitch or energy")
    frames = aligned.shape[-1]
    package = OUTPUT_DIR / f"kokoro_pitch_energy_f{frames}_fp32.mlpackage"
    wrapper = PitchEnergy(model).eval()
    mlmodel = convert(
        wrapper,
        (aligned, prosody),
        [
            ct.TensorType(name="aligned", shape=tuple(aligned.shape), dtype=np.float32),
            ct.TensorType(name="style", shape=tuple(prosody.shape), dtype=np.float32),
        ],
        [ct.TensorType(name="pitch"), ct.TensorType(name="energy")],
        package,
    )
    result = mlmodel.predict({"aligned": aligned.numpy(), "style": prosody.numpy()})
    report("pitch_slice", pitch.numpy(), result["pitch"])
    report("energy_slice", energy.numpy(), result["energy"])
    del mlmodel
    return package


def convert_decoder(model, text, pitch, energy, timbre, frames: int) -> Path:
    captured = {}

    def remember(_module, args):
        captured["features"] = args[0].detach().clone()

    handle = model.decoder.generator.register_forward_pre_hook(remember)
    with torch.no_grad():
        model.decoder(text, pitch, energy, timbre)
    handle.remove()
    reference = captured["features"]
    body = DecoderBody(model).eval()
    with torch.no_grad():
        eager = body(text, pitch, energy, timbre)
    if not torch.equal(reference, eager):
        raise RuntimeError("decoder body wrapper does not match Kokoro")
    package = OUTPUT_DIR / f"kokoro_decoder_body_f{frames}_fp32.mlpackage"
    mlmodel = convert(
        body,
        (text, pitch, energy, timbre),
        [
            ct.TensorType(name="text", shape=tuple(text.shape), dtype=np.float32),
            ct.TensorType(name="pitch", shape=tuple(pitch.shape), dtype=np.float32),
            ct.TensorType(name="energy", shape=tuple(energy.shape), dtype=np.float32),
            ct.TensorType(name="style", shape=tuple(timbre.shape), dtype=np.float32),
        ],
        [ct.TensorType(name="features")],
        package,
    )
    result = mlmodel.predict(
        {
            "text": text.numpy(),
            "pitch": pitch.numpy(),
            "energy": energy.numpy(),
            "style": timbre.numpy(),
        }
    )
    report("decoder_body", reference.numpy(), result["features"])
    del mlmodel
    return package


def convert_waveform(model, features, timbre, spectrum, audio, frames: int) -> Path:
    wrapper = WaveformGenerator(model).eval()
    with torch.no_grad():
        magnitude, phase = wrapper(features, timbre, spectrum)
        eager = model.decoder.generator.stft.inverse(magnitude, phase)
    if not torch.equal(audio, eager):
        raise RuntimeError("generator wrapper does not match Kokoro")
    package = OUTPUT_DIR / f"kokoro_generator_f{frames}_fp32.mlpackage"
    mlmodel = convert(
        wrapper,
        (features, timbre, spectrum),
        [
            ct.TensorType(name="features", shape=tuple(features.shape), dtype=np.float32),
            ct.TensorType(name="style", shape=tuple(timbre.shape), dtype=np.float32),
            ct.TensorType(name="spectrum", shape=tuple(spectrum.shape), dtype=np.float32),
        ],
        [ct.TensorType(name="magnitude"), ct.TensorType(name="phase")],
        package,
    )
    result = mlmodel.predict(
        {
            "features": features.numpy(),
            "style": timbre.numpy(),
            "spectrum": spectrum.numpy(),
        }
    )
    with torch.no_grad():
        reconstructed = model.decoder.generator.stft.inverse(
            torch.from_numpy(np.asarray(result["magnitude"], dtype=np.float32)),
            torch.from_numpy(np.asarray(result["phase"], dtype=np.float32)),
        )
    report("generator_audio", audio.numpy(), reconstructed.numpy())
    del mlmodel
    return package


def joined_waveform(model, case: str, input_ids, style, text, pitch, energy, frames) -> None:
    timbre = style[:, :128]
    tokens = input_ids.shape[1]
    torch.manual_seed(0)
    drawn, reference_spectrum, reference_audio = capture_noise(
        model, text, pitch, energy, timbre
    )
    phase_noise = next(item for item in drawn if item.ndim == 2)
    wave_noise = next(item for item in drawn if item.ndim == 3 and item.shape[-1] == 9)
    with torch.no_grad():
        reference_wave = source_wave(model.decoder.generator, pitch, phase_noise, wave_noise)
        rebuilt = source_spectrum(model.decoder.generator, pitch, phase_noise, wave_noise)
    if not torch.equal(reference_spectrum, rebuilt):
        raise RuntimeError("sine source does not match Kokoro")

    encoded = predict(
        OUTPUT_DIR / f"kokoro_text_t{tokens}_fp32.mlpackage",
        {"input_ids": input_ids.numpy()},
    )["encoded"]
    prosody = predict(
        OUTPUT_DIR / f"kokoro_prosody_text_t{tokens}_fp32.mlpackage",
        {
            "input_ids": input_ids.numpy(),
            "encoded": encoded.astype(np.float32),
            "style": style.numpy(),
        },
    )
    logits = torch.from_numpy(prosody["duration_logits"].astype(np.float32))
    chain_frames = torch.round(torch.sigmoid(logits).sum(-1)).clamp(min=1).long().squeeze(0)
    changed = int(torch.count_nonzero(chain_frames != frames))
    print(f"chain_changed_duration_tokens={changed} frames={int(chain_frames.sum())}")
    if changed:
        raise RuntimeError("joined chain changed the frame count")

    hidden = torch.from_numpy(prosody["duration_hidden"].astype(np.float32))
    text_features = torch.from_numpy(prosody["text_features"].astype(np.float32))
    aligned_prosody = repeat_features(hidden.transpose(1, 2), chain_frames)
    aligned_text = repeat_features(text_features, chain_frames)
    curves = predict(
        OUTPUT_DIR / f"kokoro_pitch_energy_f{int(frames.sum())}_fp32.mlpackage",
        {"aligned": aligned_prosody.numpy(), "style": style[:, 128:].numpy()},
    )
    chain_pitch = torch.from_numpy(curves["pitch"].astype(np.float32))
    chain_energy = torch.from_numpy(curves["energy"].astype(np.float32))
    body = predict(
        OUTPUT_DIR / f"kokoro_decoder_body_f{int(frames.sum())}_fp32.mlpackage",
        {
            "text": aligned_text.numpy(),
            "pitch": chain_pitch.numpy(),
            "energy": chain_energy.numpy(),
            "style": timbre.numpy(),
        },
    )["features"]
    with torch.no_grad():
        chain_wave = source_wave(model.decoder.generator, chain_pitch, phase_noise, wave_noise)
        chain_spectrum = source_spectrum(
            model.decoder.generator, chain_pitch, phase_noise, wave_noise
        )
    generated = predict(
        OUTPUT_DIR / f"kokoro_generator_f{int(frames.sum())}_fp32.mlpackage",
        {
            "features": np.asarray(body, dtype=np.float32),
            "style": timbre.numpy(),
            "spectrum": chain_spectrum.numpy(),
        },
    )
    with torch.no_grad():
        chain_audio = model.decoder.generator.stft.inverse(
            torch.from_numpy(np.asarray(generated["magnitude"], dtype=np.float32)),
            torch.from_numpy(np.asarray(generated["phase"], dtype=np.float32)),
        )
    report("chain_pitch", pitch.numpy(), chain_pitch.numpy())
    report("chain_source_wave", reference_wave.numpy(), chain_wave.numpy())
    report("chain_audio", reference_audio.numpy(), chain_audio.numpy())
    pytorch_wav = OUTPUT_DIR / f"{case}_pytorch_seed0.wav"
    chain_wav = OUTPUT_DIR / f"{case}_chain_seed0.wav"
    sf.write(pytorch_wav, reference_audio.detach().float().squeeze().numpy(), 24000)
    sf.write(chain_wav, chain_audio.detach().float().squeeze().numpy(), 24000)
    print(f"saved={pytorch_wav}")
    print(f"saved={chain_wav}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=CASES)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = np.load(ROOT / "benchmark/inputs" / f"{args.case}.npz")
    input_ids = torch.from_numpy(frozen["input_ids"].astype(np.int32))
    style = torch.from_numpy(frozen["style"].astype(np.float32))
    print(f"case={args.case} tokens={input_ids.shape[1]}")
    model = load_model()
    frames, _, _, aligned, text, pitch, energy, prosody = reference_outputs(
        model, input_ids, style
    )
    print(f"frames={int(frames.sum())} pitch={tuple(pitch.shape)}")
    encoded = convert_text(model, input_ids)
    del encoded
    # convert_text already checked parity. Reload PyTorch encoded for prosody,
    # matching the 3-second conversion, which was traced on PyTorch features.
    with torch.no_grad():
        length = input_ids.shape[1]
        mask = torch.arange(length).unsqueeze(0) + 1 > length
        contextual = model.bert(input_ids, attention_mask=(~mask).int())
        pytorch_encoded = model.bert_encoder(contextual).transpose(-1, -2)
    convert_prosody(model, input_ids, style, pytorch_encoded)
    convert_pitch(model, aligned, prosody, pitch, energy)
    timbre = style[:, :128]
    convert_decoder(model, text, pitch, energy, timbre, int(frames.sum()))
    torch.manual_seed(0)
    drawn, spectrum, audio = capture_noise(model, text, pitch, energy, timbre)
    features = {}

    def remember(_module, args):
        features["value"] = args[0].detach().clone()

    # The capture above already consumed seed 0. Trace the generator on that
    # same spectrum, then the joined check draws seed 0 again.
    handle = model.decoder.generator.register_forward_pre_hook(remember)
    torch.manual_seed(0)
    with torch.no_grad():
        replay = model.decoder(text, pitch, energy, timbre)
    handle.remove()
    if not torch.equal(audio, replay):
        raise RuntimeError("seed 0 did not replay the same waveform")
    convert_waveform(
        model, features["value"], timbre, spectrum, audio, int(frames.sum())
    )
    joined_waveform(model, args.case, input_ids, style, text, pitch, energy, frames)


if __name__ == "__main__":
    main()
