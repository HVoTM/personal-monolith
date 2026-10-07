"""Curves over time: pitch sweeps to feed into `osc()`, and volume shapes.

Every function returns one value per sample, so curves can be joined end to
end and multiplied with each other or with audio.
"""

import numpy as np

from .oscillator import SR, Shape, phase, triangle


def n_samples(seconds: float, sr: int = SR) -> int:
    return int(round(seconds * sr))


def hold(value: float, seconds: float, sr: int = SR) -> np.ndarray:
    """A constant value."""
    return np.full(n_samples(seconds, sr), float(value))


def linear(start: float, end: float, seconds: float, sr: int = SR) -> np.ndarray:
    """Straight-line move from `start` to `end`."""
    return np.linspace(start, end, n_samples(seconds, sr), endpoint=False)


def exponential(start: float, end: float, seconds: float, sr: int = SR) -> np.ndarray:
    """Move from `start` to `end` by a constant ratio per sample.

    For pitch this sounds like an even slide (the same number of semitones
    per second), which is how the ear hears frequency. Both ends must be > 0.
    """
    n = n_samples(seconds, sr)
    return start * (end / start) ** (np.arange(n) / n)


def wobble(center: float, depth: float, rate: float, seconds: float,
           shape: Shape = triangle, sr: int = SR) -> np.ndarray:
    """Oscillate between center - depth and center + depth, `rate` times a second."""
    n = n_samples(seconds, sr)
    return center + depth * shape(phase(np.full(n, float(rate)), sr))


def join(*parts) -> np.ndarray:
    """Concatenate curves (or audio) end to end."""
    return np.concatenate([np.asarray(p, dtype=float) for p in parts])


def frame_steps(curve, rate: float = 60.0, sr: int = SR) -> np.ndarray:
    """Sample-and-hold `curve`, changing value only `rate` times a second.

    Arcade games updated the sound chip once per video frame (60 Hz), so pitch
    sweeps moved in small stairs rather than smoothly. Running a smooth sweep
    through this gives it the same stepped texture.
    """
    curve = np.asarray(curve, dtype=float)
    hop = sr / rate
    idx = (np.floor(np.arange(len(curve)) / hop) * hop).astype(int)
    return curve[idx]


def adsr(seconds: float, attack: float = 0.005, decay: float = 0.05,
         sustain: float = 0.7, release: float = 0.05, sr: int = SR) -> np.ndarray:
    """Attack-decay-sustain-release volume shape lasting `seconds` in total."""
    n = n_samples(seconds, sr)
    a, d, r = (n_samples(t, sr) for t in (attack, decay, release))
    env = join(
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, sustain, d, endpoint=False),
        np.full(max(n - a - d - r, 0), sustain),
        np.linspace(sustain, 0, r),
    )[:n]
    return np.pad(env, (0, n - len(env)))


def fade(x, fade_in: float = 0.002, fade_out: float = 0.01, sr: int = SR) -> np.ndarray:
    """Short linear fades at both ends, to avoid clicks when a note starts or stops."""
    x = np.array(x, dtype=float)
    a = min(n_samples(fade_in, sr), len(x))
    r = min(n_samples(fade_out, sr), len(x))
    x[:a] *= np.linspace(0, 1, a, endpoint=False)
    if r:
        x[-r:] *= np.linspace(1, 0, r)
    return x


def volume_steps(env, levels: int = 16) -> np.ndarray:
    """Quantize a 0..1 volume curve to the chip's 4-bit volume (16 levels)."""
    return np.round(np.clip(env, 0, 1) * (levels - 1)) / (levels - 1)
