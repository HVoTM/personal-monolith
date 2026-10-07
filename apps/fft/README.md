# fft

Pac-Man style arcade sounds made from scratch, plus FFT tools to analyze them and to reverse-engineer the originals.

Written in Python with numpy and matplotlib. The package is called `chiptune`.

## How the sounds are made

Pac-Man's sound chip, the Namco WSG, is simple:

- It has 3 voices. Each one loops over a **32-sample wavetable of 4-bit values** (0–15).
- A **phase accumulator** sets the pitch. It is a counter that adds a frequency value on every tick, and its top bits choose which table entry to output.
- The game changes the frequency and volume registers **once per video frame (60 Hz)**. The waka-waka, the siren and the death sound are all pitch curves updated this way.

This package copies that design:

| Module | What it does |
| --- | --- |
| `oscillator.py` | Phase accumulator, basic waveforms, note names to Hz |
| `wavetable.py` | 32 × 4-bit wavetables (8 of them, like the chip) and WSG register math |
| `envelope.py` | Pitch and volume curves: sweeps, wobbles, ADSR, and `frame_steps` for the 60 Hz stairs |
| `sequencer.py` | A small 3-voice tracker for melodies |
| `sounds.py` | The effects: intro, munch, siren, power pellet, eat ghost, death |
| `analysis.py` | Spectrum, harmonic levels, spectrogram, pitch tracking, resynthesis |
| `wav.py` | WAV read/write, and playback (the built-in `winsound` on Windows) |

The wavetables and effects are approximations made by ear. They are not taken from the game's ROMs.

## Setup

Requires Python 3.10+.

```sh
cd apps/fft
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"
pytest
```

## Usage

```sh
python -m chiptune list                   # the built-in sounds
python -m chiptune play munch death       # listen
python -m chiptune render                 # write every sound to out/*.wav
python -m chiptune analyze some.wav       # pitch track + spectrogram + resynthesis
```

`explore.py` walks through the ideas step by step: harmonics, the 4-bit staircase, 60 Hz pitch steps, and building your own sound. Open it in VS Code and run it cell by cell. Each `# %%` line starts a cell, and running cells needs the Jupyter extension and `pip install ipykernel`.

## Matching the original sounds

1. Get a clip of the original. `yt-dlp -x --audio-format wav -o "refs/%(title)s.%(ext)s" <url>` needs [yt-dlp](https://github.com/yt-dlp/yt-dlp) and ffmpeg. `refs/` is git-ignored.
2. Trim it to a single sound and analyze it:
   ```sh
   python -m chiptune analyze refs/clip.wav --start 12.3 --end 13.1
   ```
   This prints the main frequency once per video frame and saves a resynthesis to `out/`. It also plots the original's spectrogram, with the pitch track drawn on top, above the resynthesis.
3. Check that the resynthesis sounds like the original. If it does, the pitch track is correct. Then copy the start and end frequencies and the timings into `chiptune/sounds.py`.

Tips:
- For fast chirps like the munch, use `--frame 1024` or `512` to follow the sweep more closely. For steady notes, use `--frame 4096` for more precise frequencies.
- The tracker follows the loudest peak. If the track jumps to double or triple the note, that harmonic was louder than the fundamental. Lower `--fmax` to keep it below the harmonic.
- Gameplay recordings often have several sounds playing at once, so pick moments where the sound you want plays alone.
