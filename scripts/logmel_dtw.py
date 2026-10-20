"""DTW-aligned log-mel distance between two 24 kHz waveforms.

Waveform SNR punishes small pitch changes that shift harmonic phase without an
equivalent audible change. This compares 80-band log-mel frames after dynamic
time warping, so only spectral differences remain. It is a preliminary metric;
blind listening is the audio gate.
"""

import numpy as np

SAMPLE_RATE = 24_000
N_FFT = 1024
HOP = 256
MELS = 80


def mel_filters() -> np.ndarray:
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    edges = to_hz(np.linspace(to_mel(0), to_mel(SAMPLE_RATE / 2), MELS + 2))
    bins = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    filters = np.zeros((MELS, bins.size))
    for band in range(MELS):
        low, mid, high = edges[band : band + 3]
        rising = (bins - low) / (mid - low)
        falling = (high - bins) / (high - mid)
        filters[band] = np.clip(np.minimum(rising, falling), 0, None)
    return filters


def log_mel_db(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float64).reshape(-1)
    window = np.hanning(N_FFT)
    padded = np.pad(wave, N_FFT // 2, mode="reflect")
    count = 1 + (padded.size - N_FFT) // HOP
    frames = np.stack([padded[i * HOP : i * HOP + N_FFT] * window for i in range(count)])
    power = np.abs(np.fft.rfft(frames, axis=-1)) ** 2
    # Floor at -80 dB so silence does not dominate the distance.
    return 10 * np.log10(np.maximum(power @ mel_filters().T, 1e-8))


def dtw_logmel_db(
    reference: np.ndarray, candidate: np.ndarray, band: int = 64
) -> tuple[float, float]:
    """Mean |dB| per mel bin along the DTW path, and the undistorted (diagonal) value.

    The path stays within `band` frames (about 0.7 s) of the diagonal. Durations
    are identical, so drift beyond that is not expected, and a full matrix for a
    30-second chunk would not fit in memory.
    """
    left, right = log_mel_db(reference), log_mel_db(candidate)
    total = np.full((len(left) + 1, len(right) + 1), np.inf)
    steps = np.zeros_like(total)
    total[0, 0] = 0
    for i in range(1, len(left) + 1):
        low, high = max(1, i - band), min(len(right), i + band)
        cost = np.mean(np.abs(left[i - 1] - right[low - 1 : high]), axis=-1)
        for j in range(low, high + 1):
            options = (total[i - 1, j - 1], total[i - 1, j], total[i, j - 1])
            best = int(np.argmin(options))
            previous = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[best]
            total[i, j] = cost[j - low] + options[best]
            steps[i, j] = steps[previous] + 1
    shared = min(len(left), len(right))
    diagonal = float(np.mean(np.abs(left[:shared] - right[:shared])))
    return float(total[-1, -1] / steps[-1, -1]), diagonal
