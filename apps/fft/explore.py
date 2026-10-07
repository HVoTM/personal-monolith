# %% [markdown]
# # Exploring chip sounds with the FFT
#
# Run this file cell by cell in VS Code: each `# %%` line starts a cell, and
# "Run Cell" opens an interactive window (needs the Jupyter extension and
# `pip install ipykernel`). It also runs as a plain script.

# %%
import matplotlib.pyplot as plt
import numpy as np

from chiptune import (SOUNDS, TABLES, exponential, frame_steps, harmonic_levels, osc, pitch_track,
                      play, plot_spectrogram, plot_spectrum, plot_waveform, sine, square, tone)

# %% [markdown]
# ## 1. Same pitch, different waveforms
# Every wave below plays A4 (440 Hz), but they sound different because each
# has a different set of harmonics (multiples of 440 Hz). The spectrum shows
# them as peaks.

# %%
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for ax, (name, shape) in zip(axes, [("clean sine", sine), ("clean square", square),
                                    ("4-bit sine table", TABLES["sine"])]):
    plot_spectrum(tone(shape, 440, 1.0), ax=ax, fmax=16000, title=name)
plt.tight_layout()

# %% [markdown]
# The clean square only has odd harmonics (3x, 5x, 7x...), each weaker by 1/k.
# The 4-bit sine table is *almost* a pure sine, but the 32-step staircase adds
# harmonics near 31x and 33x the note (about 30 dB down), and the 4-bit
# rounding adds a few weaker ones. That's the chip grit.

# %%
print("square harmonics (dB):", np.round(harmonic_levels(tone(square, 441, 1.0), 441), 1))
plot_waveform(tone(TABLES["sine"], 440, 0.01), title="4-bit sine table: visible stairs")
print(TABLES["sine"].ascii())

# %% [markdown]
# ## 2. Sweeps, and the 60 Hz stairs
# Arcade games changed the pitch once per video frame. Compare a smooth sweep
# with the same sweep passed through `frame_steps`.

# %%
smooth = exponential(200, 1600, 1.0)
fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
plot_spectrogram(osc(TABLES["triangle"], smooth), ax=a, title="smooth sweep")
plot_spectrogram(osc(TABLES["triangle"], frame_steps(smooth)), ax=b, title="60 Hz stepped sweep")
plt.tight_layout()

# %%
play(osc(TABLES["triangle"], smooth))
play(osc(TABLES["triangle"], frame_steps(smooth)))

# %% [markdown]
# ## 3. The built-in sounds
# Spectrogram with the pitch track on top. Edit `chiptune/sounds.py` and rerun.

# %%
name = "intro"
x = SOUNDS[name]()
plot_spectrogram(x, track=pitch_track(x), title=name)
play(x)

# %% [markdown]
# ## 4. Your turn
# Build a sound from pieces: a pitch curve, a wavetable, and a volume curve.

# %%
pitch = frame_steps(np.concatenate([exponential(300, 900, 0.15), exponential(900, 450, 0.25)]))
x = osc(TABLES["organ"], pitch)
plot_spectrogram(x, track=pitch_track(x))
play(x)

# %%
plt.show()
