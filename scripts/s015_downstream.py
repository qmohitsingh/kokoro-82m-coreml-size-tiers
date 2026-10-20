"""S015 downstream student: replaces Kokoro's pitch, decoder and generator.

Same interfaces as the S014 flexible models, so the Swift sine source and
inverse STFT stay as they are:

    PitchStudent    aligned [1, 640, F] + style[128:] [1, 128] -> pitch [1, 2F], energy [1, 2F]
    VocoderStudent  text [1, 512, F] + pitch, energy [1, 2F] + style[:128] [1, 128]
                    + spectrum [1, 22, 120F + 1] -> magnitude, phase [1, 11, 120F + 1]

Same building blocks as the teacher (AdaIN residual blocks, Snake resblocks,
sine-source injection, n_fft 20 / hop 5 iSTFT head), narrower:
    pitch     BiLSTM 128x2 instead of 256x2; branch widths 256/128 instead of 512/256
    decoder   width 256 instead of 1024; output 256 channels instead of 512
    generator 256 -> 128 -> 64 channels instead of 512 -> 256 -> 128
No weight norm and no dropout (both are identities at inference).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        gamma, beta = self.fc(s)[:, :, None].chunk(2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk1d(nn.Module):
    """Kokoro's AdainResBlk1d without weight norm and dropout."""

    def __init__(self, dim_in, dim_out, style_dim=128, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, bias=False) if dim_in != dim_out else None
        self.pool = (nn.ConvTranspose1d(dim_in, dim_in, 3, 2, groups=dim_in, padding=1, output_padding=1)
                     if upsample else nn.Identity())

    def forward(self, x, s):
        h = F.leaky_relu(self.norm1(x, s), 0.2)
        h = self.conv1(self.pool(h))
        h = self.conv2(F.leaky_relu(self.norm2(h, s), 0.2))
        sc = F.interpolate(x, scale_factor=2.0, mode="nearest") if self.upsample else x
        if self.conv1x1 is not None:
            sc = self.conv1x1(sc)
        return (h + sc) * (1 / math.sqrt(2))


class SnakeResBlock(nn.Module):
    """Kokoro's AdaINResBlock1: three dilated conv pairs, AdaIN + Snake."""

    def __init__(self, channels, kernel, dilations=(1, 3, 5), style_dim=128):
        super().__init__()
        pad = lambda d: (kernel * d - d) // 2
        self.convs1 = nn.ModuleList([nn.Conv1d(channels, channels, kernel, 1, dilation=d, padding=pad(d)) for d in dilations])
        self.convs2 = nn.ModuleList([nn.Conv1d(channels, channels, kernel, 1, padding=pad(1)) for _ in dilations])
        self.adain1 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in dilations])
        self.adain2 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in dilations])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in dilations])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in dilations])
        for conv in [*self.convs1, *self.convs2]:
            nn.init.normal_(conv.weight, 0.0, 0.01)

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            h = n1(x, s)
            h = h + (1 / a1) * torch.sin(a1 * h) ** 2
            h = c1(h)
            h = n2(h, s)
            h = h + (1 / a2) * torch.sin(a2 * h) ** 2
            x = x + c2(h)
        return x


class PitchStudent(nn.Module):
    def __init__(self, width=256, lstm=128, style_dim=128, inp=True, half=None, n_half=None):
        super().__init__()
        self.config = dict(width=width, lstm=lstm)
        if not inp:  # S016: Kokoro's layout, the LSTM reads the 640-dim aligned input directly
            self.config["inp"] = False
        if half is not None:
            self.config["half"] = half
        if n_half is not None:  # S016: the energy branch narrower than the F0 branch
            self.config["n_half"] = n_half
        self.inp = nn.Conv1d(640, width, 1) if inp else nn.Identity()
        self.shared = nn.LSTM(width if inp else 640, lstm, batch_first=True, bidirectional=True)
        h, half = 2 * lstm, half or width // 2

        n_half = n_half or half

        def branch(half):
            return nn.ModuleList([
                AdainResBlk1d(h, width, style_dim),
                AdainResBlk1d(width, half, style_dim, upsample=True),
                AdainResBlk1d(half, half, style_dim),
            ])

        self.f0, self.n = branch(half), branch(n_half)
        self.f0_proj = nn.Conv1d(half, 1, 1)
        self.n_proj = nn.Conv1d(n_half, 1, 1)

    def forward(self, aligned, style):
        x, _ = self.shared(self.inp(aligned).transpose(1, 2))
        x = x.transpose(1, 2)
        f0, n = x, x
        for block in self.f0:
            f0 = block(f0, style)
        for block in self.n:
            n = block(n, style)
        return self.f0_proj(f0).squeeze(1), self.n_proj(n).squeeze(1)


class VocoderStudent(nn.Module):
    def __init__(self, width=256, gen=256, asr=32, style_dim=128,
                 kernels=(3, 7, 11), up_rates=(10, 6), up_kernels=(20, 12), last=None):
        super().__init__()
        self.config = dict(width=width, gen=gen, asr=asr)
        if last is not None:  # final-stage channels; default gen // 2 ** len(up_rates)
            self.config["last"] = last
        # decoder body at frame rate F, upsampled to 2F at the end
        self.f0_conv = nn.Conv1d(1, 1, 3, 2, 1)
        self.n_conv = nn.Conv1d(1, 1, 3, 2, 1)
        self.asr_res = nn.Conv1d(512, asr, 1)
        self.encode = AdainResBlk1d(512 + 2, width, style_dim)
        extra = 2 + asr
        self.decode = nn.ModuleList([
            AdainResBlk1d(width + extra, width, style_dim),
            AdainResBlk1d(width + extra, width, style_dim),
            AdainResBlk1d(width + extra, gen, style_dim, upsample=True),
        ])
        # generator: 2F -> 20F -> 120F + 1, sine source injected at each stage
        self.ups, self.noise_convs, self.noise_res, self.resblocks = (nn.ModuleList() for _ in range(4))
        ch = gen
        for i, (u, k) in enumerate(zip(up_rates, up_kernels)):
            prev, ch = ch, gen // 2 ** (i + 1)
            if last is not None and i + 1 == len(up_rates):
                ch = last
            self.ups.append(nn.ConvTranspose1d(prev, ch, k, u, padding=(k - u) // 2))
            nn.init.normal_(self.ups[-1].weight, 0.0, 0.01)
            if i + 1 < len(up_rates):
                stride = math.prod(up_rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(22, ch, stride * 2, stride, padding=(stride + 1) // 2))
                self.noise_res.append(SnakeResBlock(ch, 7, style_dim=style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(22, ch, 1))
                self.noise_res.append(SnakeResBlock(ch, 11, style_dim=style_dim))
            for k_res in kernels:
                self.resblocks.append(SnakeResBlock(ch, k_res, style_dim=style_dim))
        self.num_kernels = len(kernels)
        self.conv_post = nn.Conv1d(ch, 22, 7, 1, padding=3)
        nn.init.normal_(self.conv_post.weight, 0.0, 0.01)

    def features(self, text, pitch, energy, style):
        f0 = self.f0_conv(pitch[:, None])
        n = self.n_conv(energy[:, None])
        x = self.encode(torch.cat([text, f0, n], 1), style)
        res = self.asr_res(text)
        for block in self.decode:
            x = block(torch.cat([x, res, f0, n], 1), style)
        return x

    def logits(self, text, pitch, energy, style, spectrum):
        """Pre-activation head [B, 22, 120F + 1]: log-magnitude then phase argument."""
        x = self.features(text, pitch, energy, style)
        for i in range(len(self.ups)):
            x = F.leaky_relu(x, 0.1)
            source = self.noise_res[i](self.noise_convs[i](spectrum), style)
            x = self.ups[i](x)
            if i == len(self.ups) - 1:
                x = F.pad(x, (1, 0), mode="reflect")
            x = x + source
            xs = None
            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x, style)
                xs = y if xs is None else xs + y
            x = xs / self.num_kernels
        return self.conv_post(F.leaky_relu(x, 0.01))

    def forward(self, text, pitch, energy, style, spectrum):
        x = self.logits(text, pitch, energy, style, spectrum)
        return torch.exp(x[:, :11]), torch.sin(x[:, 11:])


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    p, v = PitchStudent(), VocoderStudent()
    print("pitch", parameter_count(p), "vocoder", parameter_count(v), "total", parameter_count(p) + parameter_count(v))
    F_ = 50
    f0, n = p(torch.randn(1, 640, F_), torch.randn(1, 128))
    mag, ph = v(torch.randn(1, 512, F_), f0, n, torch.randn(1, 128), torch.randn(1, 22, 120 * F_ + 1))
    print(f0.shape, n.shape, mag.shape, ph.shape)
