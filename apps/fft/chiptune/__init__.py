"""Pac-Man style arcade sound synthesis, and FFT tools to analyze it."""

from .analysis import (harmonic_levels, pitch_track, plot_spectrogram, plot_spectrum,
                       plot_waveform, resynthesize, spectrum)
from .envelope import (adsr, exponential, fade, frame_steps, hold, join, linear, n_samples,
                       volume_steps, wobble)
from .oscillator import (SR, note_freq, note_number, osc, phase, pulse, saw, sine, square,
                         tone, triangle)
from .sequencer import Voice, mix, render_song
from .sounds import SOUNDS
from .wav import load_wav, play, save_wav
from .wavetable import TABLES, Wavetable, wsg_hz, wsg_register
