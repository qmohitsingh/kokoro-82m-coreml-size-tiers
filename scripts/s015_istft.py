"""Real-valued inverse STFT for training (n_fft 20, hop 5, periodic Hann).

torch.istft's backward returns NaN gradients on MPS; this is the same
math as overlap-add with conv_transpose1d, so gradients stay finite:
    frame[n] = sum_k c_k (re_k cos(2 pi k n / N) - im_k sin(2 pi k n / N)) / N
    audio    = overlap_add(frame * window) / overlap_add(window^2), center trimmed
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class RealISTFT(nn.Module):
    def __init__(self, n_fft=20, hop=5):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None]
        c = torch.full_like(k, 2.0)
        c[0], c[-1] = 1.0, 1.0
        angle = 2 * math.pi * k * n / n_fft
        self.register_buffer("w_re", (c * torch.cos(angle) * window / n_fft)[:, None].float())
        self.register_buffer("w_im", (-c * torch.sin(angle) * window / n_fft)[:, None].float())
        self.register_buffer("w_env", (window ** 2)[None, None].float())

    def forward(self, magnitude, phase):
        """[B, 11, T] magnitude and phase -> [B, hop * (T - 1)] audio."""
        frames = magnitude.shape[-1]
        audio = (F.conv_transpose1d(magnitude * torch.cos(phase), self.w_re, stride=self.hop)
                 + F.conv_transpose1d(magnitude * torch.sin(phase), self.w_im, stride=self.hop))
        ones = torch.ones(1, 1, frames, device=magnitude.device, dtype=magnitude.dtype)
        env = F.conv_transpose1d(ones, self.w_env, stride=self.hop)
        start, length = self.n_fft // 2, self.hop * (frames - 1)
        # trim before dividing: the envelope is zero at the untrimmed first sample
        return audio[:, 0, start:start + length] / env[:, 0, start:start + length]
