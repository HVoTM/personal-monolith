"""Wavetables in the style of the Namco WSG, the sound chip in Pac-Man.

The chip holds 8 waveforms. Each is 32 samples long and each sample is a
4-bit number (0-15). To play a note it steps through one of those tables at a
speed set by a frequency register. The tables are so short and coarse that
even a "sine" comes out slightly buzzy, and that grit is a big part of the
arcade sound. Compare `TABLES["sine"]` with a clean `oscillator.sine` in a
spectrum plot to see the extra harmonics.

The tables in TABLES are hand-made approximations of that style. They are not
dumps of the original Pac-Man sound ROM.
"""

from dataclasses import dataclass

import numpy as np

from .oscillator import pulse, saw, sine, square, triangle

TABLE_SIZE = 32
LEVELS = 16  # 4-bit samples

# WSG timing as emulators document it: the phase accumulator is 20 bits and
# is updated at 96 kHz, and its top 5 bits select one of the 32 table entries.
# The accumulator wraps once per waveform cycle, so the output frequency is
# register * 96000 / 2**20 (about 0.09 Hz per register step).
WSG_RATE = 96_000
WSG_ACC_BITS = 20


@dataclass(frozen=True, eq=False)
class Wavetable:
    """One cycle of a waveform stored as TABLE_SIZE integers in 0..LEVELS-1.

    Calling it with a phase array returns samples in -1..1, so a Wavetable can
    be passed to `osc()` anywhere a shape function can.
    """

    name: str
    samples: np.ndarray

    def __call__(self, p):
        idx = (np.asarray(p) * len(self.samples)).astype(int) % len(self.samples)
        mid = (LEVELS - 1) / 2
        return (self.samples[idx] - mid) / mid

    @classmethod
    def from_function(cls, name: str, f, size: int = TABLE_SIZE) -> "Wavetable":
        """Sample one cycle of `f` (phase 0..1 -> -1..1) and quantize to 4 bits."""
        x = np.clip(f(np.arange(size) / size), -1, 1)
        return cls(name, np.round((x + 1) / 2 * (LEVELS - 1)).astype(int))

    @classmethod
    def from_harmonics(cls, name: str, amps, size: int = TABLE_SIZE) -> "Wavetable":
        """Build a table by adding sine harmonics: amps[0] is the fundamental,
        amps[1] the 2nd harmonic (one octave up), and so on."""

        def f(p):
            x = sum(a * np.sin(2 * np.pi * (k + 1) * p) for k, a in enumerate(amps))
            return x / np.max(np.abs(x))

        return cls.from_function(name, f, size)

    def ascii(self) -> str:
        """Text plot of the table, one column per sample, for a quick look."""
        rows = []
        for level in range(LEVELS - 1, -1, -1):
            rows.append(f"{level:2d} " + "".join("#" if s == level else " " for s in self.samples))
        return "\n".join(rows)


def _tables(*tables: Wavetable) -> dict[str, Wavetable]:
    return {t.name: t for t in tables}


TABLES = _tables(
    Wavetable.from_function("sine", sine),
    Wavetable.from_function("triangle", triangle),
    Wavetable.from_function("square", square),
    Wavetable.from_function("saw", saw),
    Wavetable.from_function("pulse25", pulse(0.25)),
    Wavetable.from_harmonics("hollow", [1, 0, 0.45]),
    Wavetable.from_harmonics("organ", [1, 0.5, 0, 0.25]),
    Wavetable.from_harmonics("bright", [1, 0.7, 0.5, 0.35, 0.25]),
)


def wsg_hz(register: int) -> float:
    """Frequency the WSG plays for a given frequency register value."""
    return register * WSG_RATE / 2**WSG_ACC_BITS


def wsg_register(hz: float) -> int:
    """Frequency register value the WSG needs to play `hz`."""
    return int(round(hz * 2**WSG_ACC_BITS / WSG_RATE))
