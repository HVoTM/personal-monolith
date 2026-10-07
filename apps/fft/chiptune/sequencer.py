"""A tiny tracker: turn lists of notes into audio, one voice per list.

The Pac-Man board has three WSG voices, so a song is at most three Voices
mixed together.
"""

from dataclasses import dataclass

import numpy as np

from .envelope import fade, n_samples
from .oscillator import SR, Shape, note_freq, osc

Note = tuple[str | int | None, float]
"""(pitch, length in steps). Pitch is a note name, a MIDI number, or None for a rest."""


@dataclass
class Voice:
    shape: Shape
    notes: list[Note]
    volume: float = 1.0
    gate: float = 0.9
    """Fraction of each note's length that actually sounds; the rest is silence,
    so repeated notes are heard as separate notes."""


def render_voice(voice: Voice, step_seconds: float, sr: int = SR) -> np.ndarray:
    parts = []
    for pitch, steps in voice.notes:
        n = n_samples(steps * step_seconds, sr)
        out = np.zeros(n)
        if pitch is not None:
            on = int(n * voice.gate)
            out[:on] = fade(osc(voice.shape, np.full(on, note_freq(pitch)), sr), sr=sr)
        parts.append(out)
    return np.concatenate(parts) * voice.volume if parts else np.zeros(0)


def mix(*tracks) -> np.ndarray:
    """Add tracks together, padding shorter ones with silence.

    Divides by the number of tracks so the mix stays within -1..1.
    """
    out = np.zeros(max(len(t) for t in tracks))
    for t in tracks:
        out[: len(t)] += t
    return out / len(tracks)


def render_song(voices: list[Voice], bpm: float, steps_per_beat: int = 4,
                sr: int = SR) -> np.ndarray:
    """Render voices whose note lengths are in steps (default: 16th notes)."""
    step = 60 / bpm / steps_per_beat
    return mix(*(render_voice(v, step, sr) for v in voices))
