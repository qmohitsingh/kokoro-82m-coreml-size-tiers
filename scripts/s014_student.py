"""S014 student: one small model replaces Kokoro's text body and duration path.

Plain-language flow:

    token ids + af_heart style -> conv/attention trunk -> three heads
        duration_logits [B, T, 50]   same meaning as Kokoro's duration_proj
        duration_hidden [B, T, 640]  512 learned values + the 128 prosody style
        text_features   [B, 512, T]  what Kokoro's TextEncoder gives the decoder

The outputs keep the S011 prosody model's names and shapes, so pitch, decoder,
and generator stay unchanged. There is no position table: depthwise
convolutions carry position, so the five fixed shapes share every weight.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """Depthwise conv for local phoneme context, then a small feed-forward."""

    def __init__(self, width: int, kernel: int, expand: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width)
        self.norm2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, width * expand)
        self.down = nn.Linear(width * expand, width)

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
        y = self.norm1(x)
        if keep is not None:
            # Padding must look like the zeros past the end at inference.
            y = y * keep
        x = x + self.conv(y.transpose(1, 2)).transpose(1, 2)
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class AttentionBlock(nn.Module):
    """Whole-sentence context, which ALBERT gave the teacher."""

    def __init__(self, width: int, heads: int, expand: int):
        super().__init__()
        self.heads = heads
        self.head = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, width * expand)
        self.down = nn.Linear(width * expand, width)

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
        # No shape reads, so tracing leaves no int casts for Core ML.
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.unflatten(-1, (self.heads, self.head)).transpose(1, 2)
        k = k.unflatten(-1, (self.heads, self.head)).transpose(1, 2)
        v = v.unflatten(-1, (self.heads, self.head)).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) * (self.head**-0.5)
        if keep is not None:
            # keep [B, T, 1] -> mask padded keys only.
            scores = scores.masked_fill(keep.transpose(1, 2).unsqueeze(1) == 0, -1e4)
        y = (scores.softmax(-1) @ v).transpose(1, 2).flatten(2)
        x = x + self.out(y)
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class Student(nn.Module):
    def __init__(
        self,
        width: int = 256,
        layout: str = "ccaccacc",
        kernel: int = 5,
        expand: int = 2,
        heads: int = 4,
        n_token: int = 178,
        style_dim: int = 128,
        max_dur: int = 50,
    ):
        super().__init__()
        self.config = dict(
            width=width, layout=layout, kernel=kernel, expand=expand, heads=heads
        )
        self.embed = nn.Embedding(n_token, width)
        self.style = nn.Linear(style_dim, width)
        self.blocks = nn.ModuleList(
            ConvBlock(width, kernel, expand) if kind == "c"
            else AttentionBlock(width, heads, expand)
            for kind in layout
        )
        self.norm = nn.LayerNorm(width)
        self.duration = nn.Linear(width, max_dur)
        self.hidden = nn.Linear(width, 512)
        self.text = nn.Linear(width, 512)

    def forward(
        self,
        input_ids: torch.Tensor,
        style: torch.Tensor,
        keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # style [B, 256]; Kokoro's prosody half is the last 128 values.
        prosody_style = style[:, 128:]
        x = self.embed(input_ids) + self.style(prosody_style).unsqueeze(1)
        for block in self.blocks:
            x = block(x, keep)
        x = self.norm(x)
        duration_logits = self.duration(x)
        repeated_style = prosody_style.unsqueeze(1) + torch.zeros_like(x[..., :128])
        duration_hidden = torch.cat([self.hidden(x), repeated_style], dim=-1)
        text_features = self.text(x).transpose(1, 2)
        return duration_hidden, duration_logits, text_features


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
