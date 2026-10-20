"""Batched fp32 Kokoro teacher for S014 distillation.

    padded token ids + lengths + style
      -> ALBERT -> bert_encoder -> DurationEncoder -> LSTM -> duration_proj
      -> TextEncoder
    same math as KModel.forward_with_tokens; the one unpacked LSTM there is
    packed here so padding cannot leak into the backward direction.

Downstream helpers align features with durations and run the frozen
pitch, decoder, and generator, for pitch/energy and log-mel losses.
"""

import json
from pathlib import Path

import numpy as np
import torch
from kokoro import KModel
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load_teacher(device: str) -> KModel:
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(ROOT / "models/pytorch/config.json"),
        model=str(ROOT / "models/pytorch/kokoro-v1_0.pth"),
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def load_voice() -> np.ndarray:
    """af_heart pack [510, 256]; row = token count without pads, minus one."""
    path = next((ROOT / "models").rglob("af_heart.pt"), None)
    if path is not None:
        return torch.load(path, map_location="cpu").squeeze(1).numpy()
    voices = np.load(ROOT / "models/voices-v1.0.bin")
    return voices["af_heart"].squeeze(1)


def style_rows(voice: np.ndarray, token_counts: list[int]) -> np.ndarray:
    return np.stack([voice[min(n, len(voice)) - 1] for n in token_counts])


@torch.no_grad()
def frontend(model: KModel, ids: torch.Tensor, lengths: torch.Tensor, style: torch.Tensor):
    """ids [B, T] with pads 0 at both ends, zeros after length. Returns
    duration_hidden [B, T, 640], duration_logits [B, T, 50], text_features [B, 512, T]."""
    T = ids.shape[1]
    mask = torch.arange(T, device=ids.device)[None] >= lengths[:, None]
    bert = model.bert(ids, attention_mask=(~mask).int())
    d_en = model.bert_encoder(bert).transpose(-1, -2)
    s = style[:, 128:]
    d = model.predictor.text_encoder(d_en, s, lengths, mask)
    packed = nn.utils.rnn.pack_padded_sequence(d, lengths.cpu(), batch_first=True, enforce_sorted=False)
    x, _ = model.predictor.lstm(packed)
    x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=T)
    logits = model.predictor.duration_proj(x)
    text = model.text_encoder(ids, lengths, mask)
    return d, logits, text


def frames_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.round(torch.sigmoid(logits).sum(-1)).clamp(min=1).long()


def alignment(frames: torch.Tensor, length: int, total: int) -> torch.Tensor:
    """[T, F] one-hot token-to-frame matrix for one utterance."""
    index = torch.repeat_interleave(torch.arange(length, device=frames.device), frames[:length])
    aln = torch.zeros((frames.shape[0], total), device=frames.device)
    aln[index, torch.arange(index.shape[0], device=frames.device)] = 1
    return aln


def synthesize(model: KModel, hidden, text, aln, style):
    """One utterance: [1,T,640], [1,512,T], [T,F], [1,256] -> F0, N, audio."""
    en = hidden.transpose(-1, -2) @ aln
    f0, n = model.predictor.F0Ntrain(en, style[:, 128:])
    audio = model.decoder(text @ aln, f0, n, style[:, :128]).squeeze()
    return f0, n, audio


def reference_voice_check() -> None:
    """The frozen chunks' style must equal voice[token_count - 1]."""
    voice = load_voice()
    manifest = json.loads((ROOT / "benchmark/inputs/manifest.json").read_text())
    for item in manifest["items"]:
        for chunk in item["chunks"]:
            frozen = np.load(ROOT / "benchmark/inputs" / chunk["file"])
            row = style_rows(voice, [chunk["token_count"]])[0]
            print(chunk["file"], "style_max_abs", float(np.abs(frozen["style"][0] - row).max()))
