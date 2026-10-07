"""FFT tools: look at what frequencies a sound contains, and how they move.

- `spectrum` / `plot_spectrum`: which frequencies are in a sound, and how loud.
- `harmonic_levels`: how loud each harmonic is compared to the fundamental.
  This is what makes a square wave sound different from a sine at the same pitch.
- `pitch_track` / `plot_spectrogram`: how the main frequency changes over time,
  which is how you reverse-engineer a sweep from a recording.
- `resynthesize`: play a pitch track back through a wavetable, to check by ear
  that the track captured the sound.

Plotting functions import matplotlib only when called.
"""

import numpy as np

from .oscillator import SR, Shape, osc


def to_db(mag, floor: float = -120.0) -> np.ndarray:
    return np.maximum(20 * np.log10(np.maximum(mag, 1e-12)), floor)


def spectrum(x, sr: int = SR, db: bool = True):
    """Magnitude spectrum of `x` using a Hann window.

    Returns (freqs, mags). Scaled so a full-scale sine reads 1.0 (0 dB).
    """
    x = np.asarray(x, dtype=float)
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w)) / (w.sum() / 2)
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return freqs, to_db(mag) if db else mag


def harmonic_levels(x, f0: float, sr: int = SR, count: int = 8) -> np.ndarray:
    """Level in dB of harmonics 1..count relative to the fundamental `f0`.

    Reads the spectrum peak within f0/4 of each multiple of f0. A square wave
    gives about [0, -inf, -9.5, -inf, -14, ...]: only odd harmonics, at 1/k.
    """
    freqs, mag = spectrum(x, sr, db=False)
    levels = []
    for k in range(1, count + 1):
        band = np.abs(freqs - k * f0) < f0 / 4
        levels.append(mag[band].max() if band.any() else 0.0)
    levels = np.array(levels)
    return to_db(levels / levels[0])


def pitch_track(x, sr: int = SR, frame: int = 2048, hop: int = 256,
                fmin: float = 60.0, fmax: float = 4000.0, silence_db: float = -40.0):
    """Estimate the main frequency over time.

    Splits `x` into overlapping frames, takes the FFT of each one, and picks
    the strongest peak between fmin and fmax. Parabolic interpolation around
    the peak gives a more precise value than the FFT bin spacing alone.

    Returns (times, freqs). freqs is NaN for frames quieter than `silence_db`
    relative to the loudest sample.

    This works well on chip sounds, which have one loud fundamental. With
    several voices it follows the loudest, and it can lock onto a harmonic
    (2x or 3x the real note) if that harmonic is louder than the fundamental.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    nfft = frame * 4  # zero-padding for a smoother spectrum
    win = np.hanning(frame)
    starts = np.arange(0, len(x) - frame + 1, hop)
    bin_freqs = np.fft.rfftfreq(nfft, 1 / sr)
    bins = np.flatnonzero((bin_freqs >= fmin) & (bin_freqs <= fmax))
    lo, hi = bins[0], bins[-1]
    peak = np.max(np.abs(x)) or 1.0

    freqs = np.full(len(starts), np.nan)
    for i, s in enumerate(starts):
        seg = x[s : s + frame]
        if to_db(np.sqrt(np.mean(seg**2)) / peak) < silence_db:
            continue
        mag = np.abs(np.fft.rfft(seg * win, nfft))
        k = lo + int(np.argmax(mag[lo : hi + 1]))
        offset = 0.0
        if lo < k < hi:
            a, b, c = np.log(mag[k - 1 : k + 2] + 1e-12)
            denom = a - 2 * b + c
            if denom:
                offset = 0.5 * (a - c) / denom
        freqs[i] = (k + offset) * sr / nfft
    return (starts + frame / 2) / sr, freqs


def resynthesize(times, freqs, shape: Shape, length: int, sr: int = SR) -> np.ndarray:
    """Play a pitch track (from `pitch_track`) through `shape`, `length` samples long.

    Frames where the track is NaN come out silent.
    """
    times, freqs = np.asarray(times), np.asarray(freqs)
    voiced = ~np.isnan(freqs)
    if not voiced.any():
        return np.zeros(length)
    t = np.arange(length) / sr
    f = np.interp(t, times[voiced], freqs[voiced])
    gate = np.interp(t, times, voiced.astype(float)) > 0.5
    return osc(shape, f, sr) * gate


def plot_spectrum(x, sr: int = SR, fmax: float = 5000, ax=None, title: str | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.figure(figsize=(9, 3.5)).gca()
    freqs, db = spectrum(x, sr)
    ax.plot(freqs, db, lw=0.8)
    ax.set(xlim=(0, fmax), ylim=(-90, 5), xlabel="Frequency (Hz)", ylabel="Level (dB)",
           title=title or "Spectrum")
    ax.grid(alpha=0.3)
    return ax


def plot_spectrogram(x, sr: int = SR, fmax: float = 4000, ax=None, track=None,
                     title: str | None = None):
    """Spectrogram (time vs frequency, brightness = loudness).

    Pass `track=(times, freqs)` from `pitch_track` to draw it on top.
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.figure(figsize=(9, 4)).gca()
    ax.specgram(x, Fs=sr, NFFT=1024, noverlap=896, cmap="magma")
    if track is not None:
        ax.plot(*track, color="cyan", lw=1, label="pitch track")
        ax.legend(loc="upper right")
    ax.set(ylim=(0, fmax), xlabel="Time (s)", ylabel="Frequency (Hz)",
           title=title or "Spectrogram")
    return ax


def plot_waveform(x, sr: int = SR, seconds: float = 0.01, ax=None, title: str | None = None):
    """The first `seconds` of the raw samples. Short windows show the 4-bit stairs."""
    import matplotlib.pyplot as plt

    ax = ax or plt.figure(figsize=(9, 3)).gca()
    n = min(len(x), int(seconds * sr))
    ax.step(np.arange(n) / sr * 1000, x[:n], where="post", lw=0.9)
    ax.set(xlabel="Time (ms)", ylabel="Amplitude", title=title or "Waveform")
    ax.grid(alpha=0.3)
    return ax
