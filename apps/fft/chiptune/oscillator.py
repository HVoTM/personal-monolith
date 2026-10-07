"""Oscillators: turn a frequency curve into audio samples.

A "frequency curve" is an array with one Hz value per output sample. A steady
tone, a sweep and a vibrato are all just different curves, so every sound in
this package goes through the same `osc()` call.
"""

import re
from typing import Callable

import numpy as np

SR = 44_100
"""Default output sample rate in Hz."""

Shape = Callable[[np.ndarray], np.ndarray]
"""Maps phase (0..1, position within one cycle) to a sample value (-1..1)."""

_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_number(name: str) -> int:
    """MIDI note number for a note name: 'A4' -> 69, 'C#5' -> 73, 'Bb3' -> 58.

    C4 is middle C (60).
    """
    m = _NOTE_RE.match(name.strip())
    if not m:
        raise ValueError(f"not a note name: {name!r}")
    letter, accidental, octave = m.groups()
    n = _SEMITONES[letter.upper()] + 12 * (int(octave) + 1)
    return n + {"#": 1, "b": -1, "": 0}[accidental]


def note_freq(note):
    """Frequency in Hz of a note name or MIDI number (A4 = 440 Hz).

    Each semitone multiplies the frequency by 2**(1/12), so 12 semitones up
    (one octave) doubles it.
    """
    if isinstance(note, str):
        note = note_number(note)
    f = 440.0 * 2.0 ** ((np.asarray(note, dtype=float) - 69) / 12)
    return float(f) if f.ndim == 0 else f


def phase(freqs, sr: int = SR) -> np.ndarray:
    """Phase accumulator: position within the waveform cycle for every sample.

    Each sample adds freq/sr to a running total and keeps only the fractional
    part, so the result is always in [0, 1) and starts at 0. Sound chips do
    the same thing in hardware with an integer counter that wraps around.
    """
    step = np.asarray(freqs, dtype=float) / sr
    return (np.cumsum(step) - step) % 1.0


def sine(p):
    return np.sin(2 * np.pi * p)


def square(p):
    return np.where(p < 0.5, 1.0, -1.0)


def pulse(duty: float) -> Shape:
    """Square wave that is high for `duty` of each cycle (0.5 = square)."""
    return lambda p: np.where(p < duty, 1.0, -1.0)


def triangle(p):
    return 1.0 - 4.0 * np.abs(p - 0.5)


def saw(p):
    return 2.0 * p - 1.0


def osc(shape: Shape, freqs, sr: int = SR) -> np.ndarray:
    """Play `shape` following the per-sample frequency curve `freqs`."""
    return shape(phase(freqs, sr))


def tone(shape: Shape, freq: float, seconds: float, sr: int = SR) -> np.ndarray:
    """A steady tone at one frequency."""
    return osc(shape, np.full(int(round(seconds * sr)), float(freq)), sr)
