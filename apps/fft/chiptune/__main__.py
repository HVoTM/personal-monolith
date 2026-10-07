"""Command line: python -m chiptune {list,render,play,analyze} ..."""

import argparse
from pathlib import Path

import numpy as np

from .analysis import pitch_track, plot_spectrogram, resynthesize
from .sounds import SOUNDS
from .wav import load_wav, play, save_wav
from .wavetable import TABLES


def _names(names: list[str]) -> list[str]:
    unknown = [n for n in names if n not in SOUNDS]
    if unknown:
        raise SystemExit(f"unknown sound(s): {', '.join(unknown)}. Try `list`.")
    return names or list(SOUNDS)


def _list(args) -> None:
    for name, fn in SOUNDS.items():
        print(f"{name:14} {fn.__doc__.strip().splitlines()[0]}")


def _render(args) -> None:
    for name in _names(args.names):
        x = SOUNDS[name]()
        path = save_wav(Path(args.out) / f"{name}.wav", x)
        print(f"wrote {path}")
        if args.play:
            play(x)


def _play(args) -> None:
    for name in _names(args.names):
        print(f"playing {name}")
        play(SOUNDS[name]())


def _analyze(args) -> None:
    x, sr = load_wav(args.path)
    x = x[int(args.start * sr) : int(args.end * sr) if args.end else None]
    times, freqs = pitch_track(x, sr, frame=args.frame, hop=args.frame // 8,
                               fmin=args.fmin, fmax=args.fmax)

    print(f"{args.path}: {len(x) / sr:.2f} s at {sr} Hz")
    next_t = 0.0
    for t, f in zip(times, freqs):
        if t >= next_t:
            print(f"  {t:6.3f} s  " + ("   (silent)" if np.isnan(f) else f"{f:7.1f} Hz"))
            next_t = t + args.every

    resynth = resynthesize(times, freqs, TABLES[args.table], len(x), sr)
    out = save_wav(args.out or Path("out") / f"{Path(args.path).stem}_resynth.wav", resynth, sr)
    print(f"resynthesis written to {out}")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        _, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
        plot_spectrogram(x, sr, fmax=args.fmax, ax=top, track=(times, freqs), title="Original")
        plot_spectrogram(resynth, sr, fmax=args.fmax, ax=bottom,
                         title=f"Resynthesis ({args.table} wavetable)")
        plt.tight_layout()
        plt.show()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m chiptune",
                                     description="Pac-Man style sound synthesis and analysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the built-in sounds").set_defaults(run=_list)

    p = sub.add_parser("render", help="write sounds to WAV files (all of them by default)")
    p.add_argument("names", nargs="*")
    p.add_argument("-o", "--out", default="out", help="output folder (default: out)")
    p.add_argument("--play", action="store_true", help="also play each sound")
    p.set_defaults(run=_render)

    p = sub.add_parser("play", help="play sounds without writing files")
    p.add_argument("names", nargs="*")
    p.set_defaults(run=_play)

    p = sub.add_parser("analyze", help="pitch-track a WAV file, resynthesize it, and plot both")
    p.add_argument("path")
    p.add_argument("--start", type=float, default=0.0, help="clip start in seconds")
    p.add_argument("--end", type=float, help="clip end in seconds")
    p.add_argument("--fmin", type=float, default=60.0, help="lowest frequency to track (Hz)")
    p.add_argument("--fmax", type=float, default=4000.0, help="highest frequency to track (Hz)")
    p.add_argument("--frame", type=int, default=2048,
                   help="FFT frame size in samples. Smaller follows fast sweeps better, "
                        "larger measures steady notes more precisely (default: 2048)")
    p.add_argument("--every", type=float, default=1 / 60,
                   help="print interval in seconds (default: one video frame)")
    p.add_argument("--table", choices=list(TABLES), default="sine",
                   help="wavetable for the resynthesis")
    p.add_argument("-o", "--out", help="resynthesis WAV path (default: out/<name>_resynth.wav)")
    p.add_argument("--no-plot", action="store_true")
    p.set_defaults(run=_analyze)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
