"""Pac-Man style sound effects, built from pitch curves and wavetables.

These are first attempts by ear, not transcriptions from the game. To get
closer: record the original, run `python -m chiptune analyze original.wav`,
read the pitch track it prints, and adjust the numbers here to match.
"""

import numpy as np

from .envelope import exponential, frame_steps, hold, join, linear, n_samples, wobble
from .oscillator import SR, osc, phase, saw
from .sequencer import Voice, render_song
from .wavetable import TABLES


def munch(cycles: int = 4, sr: int = SR) -> np.ndarray:
    """Waka-waka: alternating falling and rising chirps while eating dots."""
    wa = exponential(640, 190, 0.075, sr)
    ka = exponential(190, 640, 0.075, sr)
    pitch = frame_steps(np.tile(join(wa, ka), cycles), sr=sr)
    return osc(TABLES["triangle"], pitch, sr)


def siren(seconds: float = 3.0, level: int = 1, sr: int = SR) -> np.ndarray:
    """Background siren. Level 1-5: the game speeds it up as the maze empties."""
    rate = 1.8 + 0.35 * (level - 1)
    center = 560 + 50 * (level - 1)
    pitch = frame_steps(wobble(center, 170, rate, seconds, sr=sr), sr=sr)
    return 0.8 * osc(TABLES["sine"], pitch, sr)


def power_pellet(seconds: float = 2.0, sr: int = SR) -> np.ndarray:
    """Frightened-ghost loop: fast repeating upward sweeps."""
    per_second = 7.5
    one = exponential(180, 720, 1 / per_second, sr)
    reps = int(np.ceil(seconds * per_second))
    pitch = frame_steps(np.tile(one, reps)[: n_samples(seconds, sr)], sr=sr)
    return osc(TABLES["hollow"], pitch, sr)


def eat_ghost(sr: int = SR) -> np.ndarray:
    """Eating a frightened ghost: one fast climb."""
    seconds = 0.5
    pitch = frame_steps(exponential(150, 2400, seconds, sr), sr=sr)
    return osc(TABLES["triangle"], pitch, sr) * linear(1.0, 0.4, seconds, sr)


def death(sr: int = SR) -> np.ndarray:
    """Losing a life: a falling warble, then two short blips."""
    seconds = 1.4
    n = n_samples(seconds, sr)
    base = exponential(900, 160, seconds, sr)
    warble = 1 + 0.35 * saw(phase(np.full(n, 9.0), sr))  # re-rises 9 times a second
    fall = osc(TABLES["square"], frame_steps(base * warble, sr=sr), sr) * linear(0.7, 0.4, seconds, sr)
    blip = 0.6 * osc(TABLES["square"], frame_steps(exponential(1000, 120, 0.12, sr), sr=sr), sr)
    gap = hold(0, 0.08, sr)
    return join(fall, gap, blip, gap, blip)


def _bass_phrase(root: str) -> list:
    return [(f"{root}2", 3), (f"{root}3", 1), (f"{root}2", 3), (f"{root}3", 1)]


# Start-of-game jingle, in 16th-note steps. Transcribed by ear, so treat the
# rhythm as approximate.
INTRO_MELODY = [
    ("B4", 1), ("B5", 1), ("F#5", 1), ("D#5", 1), ("B5", 0.5), ("F#5", 1.5), ("D#5", 2),
    ("C5", 1), ("C6", 1), ("G5", 1), ("E5", 1), ("C6", 0.5), ("G5", 1.5), ("E5", 2),
    ("B4", 1), ("B5", 1), ("F#5", 1), ("D#5", 1), ("B5", 0.5), ("F#5", 1.5), ("D#5", 2),
    ("D#5", 0.5), ("E5", 0.5), ("F5", 1),
    ("F5", 0.5), ("F#5", 0.5), ("G5", 1),
    ("G5", 0.5), ("G#5", 0.5), ("A5", 1),
    ("B5", 2),
]
INTRO_BASS = (
    _bass_phrase("B") + _bass_phrase("C") + _bass_phrase("B")
    + [("F#2", 2), ("G#2", 2), ("A#2", 2), ("B2", 2)]
)


def intro(bpm: float = 115, sr: int = SR) -> np.ndarray:
    """Start-of-game jingle on two voices."""
    return render_song(
        [Voice(TABLES["hollow"], INTRO_MELODY), Voice(TABLES["triangle"], INTRO_BASS, volume=0.9)],
        bpm=bpm, sr=sr,
    )


SOUNDS = {
    "intro": intro,
    "munch": munch,
    "siren": siren,
    "power_pellet": power_pellet,
    "eat_ghost": eat_ghost,
    "death": death,
}
