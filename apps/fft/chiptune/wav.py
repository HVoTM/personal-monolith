"""Read, write and play audio using only the standard library (plus numpy)."""

import io
import sys
import wave
from pathlib import Path

import numpy as np

from .oscillator import SR


def _pcm16(x, volume: float) -> bytes:
    x = np.clip(np.asarray(x, dtype=float) * volume, -1, 1)
    return (x * 32767).astype("<i2").tobytes()


def _write(target, x, sr: int, volume: float) -> None:
    with wave.open(target, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(_pcm16(x, volume))


def save_wav(path, x, sr: int = SR, volume: float = 0.8) -> Path:
    """Write mono 16-bit WAV. Samples are scaled by `volume` and clipped to -1..1."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(str(path), x, sr, volume)
    return path


def wav_bytes(x, sr: int = SR, volume: float = 0.8) -> bytes:
    buf = io.BytesIO()
    _write(buf, x, sr, volume)
    return buf.getvalue()


def load_wav(path):
    """Read an integer PCM WAV file as mono floats in -1..1. Returns (x, sr).

    Stereo is averaged to mono. Float WAVs are not supported by the standard
    library; convert with `ffmpeg -i in.wav -c:a pcm_s16le out.wav`.
    """
    with wave.open(str(path), "rb") as w:
        channels, width, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(float) - 128) / 128
    elif width == 2:
        x = np.frombuffer(raw, "<i2") / 2**15
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 2**23, v - 2**24, v) / 2**23
    elif width == 4:
        x = np.frombuffer(raw, "<i4") / 2**31
    else:
        raise ValueError(f"unsupported sample width: {width} bytes")
    return x.reshape(-1, channels).mean(axis=1), sr


def play(x, sr: int = SR, volume: float = 0.8) -> None:
    """Play audio and wait until it finishes.

    Uses the built-in winsound module on Windows. Elsewhere it needs
    `pip install sounddevice`.
    """
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(wav_bytes(x, sr, volume), winsound.SND_MEMORY)
        return
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError("playback needs `pip install sounddevice` on this platform") from None
    sd.play(np.clip(np.asarray(x) * volume, -1, 1), sr)
    sd.wait()
