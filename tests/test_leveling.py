"""Section leveling: the bed lands on its per-section target and the vocal
sits a fixed margin above it, regardless of how quiet the source was."""

import numpy as np
import pyloudnorm
import pytest

from musicmixer.models import Section
from musicmixer.services.leveling import (
    SECTION_TARGET_LUFS,
    ENERGY_OFFSET_DB,
    VOCAL_OVER_BED_DB,
    level_buses,
)

SR = 44100
BPM = 120.0
SPB = 60.0 / BPM


def _beat_frames(n_beats: int) -> np.ndarray:
    # librosa-style frames at 22050 Hz / hop 512
    return np.arange(n_beats + 1) * SPB * 22050 / 512


def _noise(n: int, amplitude: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mono = rng.standard_normal(n).astype(np.float32) * amplitude
    return np.column_stack([mono, mono])


def _section(label, start, end, energy="medium", vocal_gain=0.0):
    return Section(label=label, start_beat=start, end_beat=end,
                   stem_gains={"lead_vocals": vocal_gain},
                   transition_in="cut", transition_beats=0, energy=energy)


def _lufs(audio):
    return pyloudnorm.Meter(SR).integrated_loudness(audio)


@pytest.fixture
def buses():
    """Three 8s sections: quiet bed, loud bed, mid bed. Vocal planned only in
    the middle section and much louder than the bed, with a short ramp tail
    bleeding into the outro (as the renderer's transitions produce)."""
    beats = 48
    n = int(beats * SPB * SR)
    third = n // 3
    bed = np.zeros((n, 2), dtype=np.float32)
    bed[:third] = _noise(third, 0.01, 1)
    bed[third:2 * third] = _noise(third, 0.08, 2)
    bed[2 * third:] = _noise(n - 2 * third, 0.05, 3)
    vocal = np.zeros((n, 2), dtype=np.float32)
    vocal[third:2 * third + SR // 2] = _noise(third + SR // 2, 0.15, 4)
    sections = [
        _section("intro", 0, 16, "low"),
        _section("verse", 16, 32, "high", vocal_gain=0.9),
        _section("outro", 32, 48, "low"),
    ]
    return vocal, bed, sections, _beat_frames(beats), third


def test_each_section_mix_lands_on_energy_target(buses):
    vocal, bed, sections, frames, third = buses
    leveled_vocal, leveled_bed, levels = level_buses(vocal, bed, sections, frames, SR, BPM)
    mix = leveled_vocal + leveled_bed

    for i, energy in enumerate(("low", "high", "low")):
        got = _lufs(mix[i * third:(i + 1) * third])
        assert got == pytest.approx(SECTION_TARGET_LUFS + ENERGY_OFFSET_DB[energy], abs=1.0), (
            f"section {i} ({energy}) landed at {got:.1f} LUFS"
        )
    assert levels[0].bed_gain_db > 6.0   # the quiet intro was actually raised


def test_vocal_sits_margin_above_bed_and_bed_makes_room(buses):
    vocal, bed, sections, frames, third = buses
    leveled_vocal, leveled_bed, levels = level_buses(vocal, bed, sections, frames, SR, BPM)

    bed_under_vocal = _lufs(leveled_bed[third:2 * third])
    vocal_lufs = _lufs(leveled_vocal[third:2 * third])
    assert vocal_lufs - bed_under_vocal == pytest.approx(VOCAL_OVER_BED_DB, abs=1.0)
    assert levels[1].vocal_gain_db < 0  # it started far too hot

    # bed sits below the section target under the vocal, at target without it
    assert bed_under_vocal < SECTION_TARGET_LUFS - 2.0
    bed_alone = _lufs(leveled_bed[2 * third:])
    assert bed_alone == pytest.approx(SECTION_TARGET_LUFS + ENERGY_OFFSET_DB["low"], abs=1.0)


def test_unplanned_vocal_sections_inherit_neighbour_gain(buses):
    """A ramp tail in a section with no planned vocal must not be measured
    (it would read as active and get boosted to the clamp)."""
    vocal, bed, sections, frames, third = buses
    _, _, levels = level_buses(vocal, bed, sections, frames, SR, BPM)

    assert np.isnan(levels[0].vocal_lufs) and np.isnan(levels[2].vocal_lufs)
    assert levels[0].vocal_gain_db == levels[1].vocal_gain_db
    assert levels[2].vocal_gain_db == levels[1].vocal_gain_db


def test_empty_sections_is_passthrough():
    bed = _noise(SR, 0.1, 5)
    vocal = _noise(SR, 0.1, 6)
    v, b, levels = level_buses(vocal, bed, [], _beat_frames(4), SR, BPM)
    assert v is vocal and b is bed and levels == []
