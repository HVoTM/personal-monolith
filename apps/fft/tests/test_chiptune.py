import numpy as np
import pytest

from chiptune import (SOUNDS, SR, TABLES, Voice, exponential, frame_steps, harmonic_levels,
                      linear, load_wav, note_freq, note_number, osc, phase, pitch_track,
                      render_song, resynthesize, save_wav, sine, spectrum, tone, wsg_hz,
                      wsg_register)


def cents(a, b):
    return 1200 * np.log2(np.asarray(a) / np.asarray(b))


@pytest.mark.parametrize("name, number", [("A4", 69), ("C4", 60), ("C#5", 73), ("Bb3", 58), ("b2", 47)])
def test_note_number(name, number):
    assert note_number(name) == number


def test_note_freq():
    assert note_freq("A4") == pytest.approx(440)
    assert note_freq("A5") == pytest.approx(880)
    assert note_freq(60) == pytest.approx(261.626, abs=1e-3)


def test_phase_starts_at_zero_and_wraps():
    p = phase(np.full(100, SR / 10))  # 10 samples per cycle
    assert p[0] == 0
    assert p.min() >= 0 and p.max() < 1
    assert min(p[10], 1 - p[10]) < 1e-9  # back at the start of a cycle (0 and 1 are the same point)


def test_sine_peak_is_at_its_frequency():
    freqs, db = spectrum(tone(sine, 1000, 1.0))
    assert freqs[np.argmax(db)] == pytest.approx(1000, abs=1)
    assert db.max() == pytest.approx(0, abs=0.1)


def test_tables_are_32_steps_of_4_bits():
    assert len(TABLES) == 8
    for t in TABLES.values():
        assert len(t.samples) == 32
        assert t.samples.min() >= 0 and t.samples.max() <= 15


def test_square_table_has_only_odd_harmonics():
    x = tone(TABLES["square"], 441, 1.0)
    levels = harmonic_levels(x, 441)
    assert levels[1] < -40  # 2nd harmonic: absent
    assert levels[2] == pytest.approx(20 * np.log10(1 / 3), abs=1.5)  # 3rd: about 1/3


def test_wsg_register_round_trip():
    assert wsg_hz(wsg_register(440)) == pytest.approx(440, abs=0.1)


def test_frame_steps_changes_at_most_60_times_a_second():
    stepped = frame_steps(linear(200, 800, 1.0))
    assert len(np.unique(stepped)) <= 60


def test_pitch_track_follows_a_sweep():
    truth = exponential(300, 1200, 1.0)
    times, freqs = pitch_track(osc(TABLES["square"], truth))
    expected = np.interp(times, np.arange(len(truth)) / SR, truth)
    assert np.median(np.abs(cents(freqs, expected))) < 20


def test_resynthesis_keeps_the_pitch():
    x = osc(TABLES["sine"], exponential(400, 900, 0.8))
    times, freqs = pitch_track(x)
    t2, f2 = pitch_track(resynthesize(times, freqs, TABLES["triangle"], len(x)))
    ok = ~np.isnan(freqs) & ~np.isnan(f2)
    assert np.median(np.abs(cents(f2[ok], freqs[ok]))) < 20


def test_wav_round_trip(tmp_path):
    x = tone(sine, 440, 0.1)
    path = save_wav(tmp_path / "a.wav", x, volume=1.0)
    y, sr = load_wav(path)
    assert sr == SR
    assert np.max(np.abs(x - y)) < 1e-4


def test_song_length():
    song = render_song([Voice(sine, [("C4", 4), (None, 4), ("E4", 8)])], bpm=120)
    assert len(song) == pytest.approx(2 * SR, abs=2)  # 16 sixteenths at 120 bpm = 2 s


@pytest.mark.parametrize("name", list(SOUNDS))
def test_sounds_render(name):
    x = SOUNDS[name]()
    assert len(x) > SR // 10
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x)) <= 1.0
