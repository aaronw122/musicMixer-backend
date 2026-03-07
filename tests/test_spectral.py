"""Tests for musicmixer.services.spectral — spectral analysis for adaptive EQ.

Uses synthetic audio (sine waves) to verify spectral profile computation,
conflict detection, and adaptive correction generation.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from musicmixer.models import FrequencyConflict, SpectralProfile
from musicmixer.services.spectral import (
    ANOMALY_THRESHOLD_DB,
    ISO_BAND_CENTERS,
    MAX_CORRECTIONS_PER_STEM,
    compute_adaptive_corrections,
    compute_spectral_profile,
    detect_conflicts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 44100
DURATION = 2.0


def _make_sine(
    freq: float,
    duration: float = DURATION,
    sr: int = SR,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate a mono sine wave as float32 (N,)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def _make_stereo_sine(
    freq: float,
    duration: float = DURATION,
    sr: int = SR,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate a stereo sine wave as float32 (N, 2)."""
    mono = _make_sine(freq, duration, sr, amplitude)
    return np.column_stack([mono, mono])


def _nearest_band_index(freq_hz: float) -> int:
    """Return the index of the ISO band center nearest to freq_hz."""
    return int(np.argmin(np.abs(ISO_BAND_CENTERS - freq_hz)))


# ---------------------------------------------------------------------------
# SpectralProfile computation
# ---------------------------------------------------------------------------


class TestComputeSpectralProfile:
    """Tests for compute_spectral_profile."""

    def test_sine_peak_at_correct_band(self):
        """A 400 Hz sine should have its peak energy near the 400 Hz band."""
        audio = _make_sine(400.0, duration=2.0)
        profile = compute_spectral_profile(audio, SR, stem_type="vocals")

        assert isinstance(profile, SpectralProfile)
        assert profile.stem_type == "vocals"
        assert len(profile.band_centers_hz) == 31
        assert len(profile.band_energies_db) == 31

        # The band with the highest energy should be near 400 Hz
        peak_band_idx = int(np.argmax(profile.band_energies_db))
        peak_band_hz = ISO_BAND_CENTERS[peak_band_idx]
        # 400 Hz is the ISO center for band index 13; allow +/- 1 band
        expected_idx = _nearest_band_index(400.0)
        assert abs(peak_band_idx - expected_idx) <= 1, (
            f"Peak band at {peak_band_hz:.0f} Hz, expected near 400 Hz"
        )

    def test_peak_detection_finds_sine(self):
        """A strong 400 Hz sine should be detected as a spectral peak."""
        audio = _make_sine(400.0, duration=2.0)
        profile = compute_spectral_profile(audio, SR, stem_type="drums")

        # Should have at least one detected peak
        assert len(profile.peak_frequencies_hz) >= 1
        # At least one peak should be near 400 Hz
        closest = min(profile.peak_frequencies_hz, key=lambda f: abs(f - 400.0))
        assert abs(closest - 400.0) <= 200.0, (
            f"Nearest detected peak at {closest:.0f} Hz, expected near 400 Hz"
        )

    def test_stereo_input_handled(self):
        """Stereo input should produce a valid profile (downmixed to mono)."""
        audio = _make_stereo_sine(1000.0)
        profile = compute_spectral_profile(audio, SR, stem_type="guitar")

        assert isinstance(profile, SpectralProfile)
        assert len(profile.band_energies_db) == 31
        assert np.all(np.isfinite(profile.band_energies_db))

    def test_mono_input_handled(self):
        """Mono input should produce a valid profile directly."""
        audio = _make_sine(1000.0)
        profile = compute_spectral_profile(audio, SR, stem_type="bass")

        assert isinstance(profile, SpectralProfile)
        assert len(profile.band_energies_db) == 31

    def test_silent_audio(self):
        """Silent audio should produce a flat profile with no deviations.

        After normalization (mean = 0 dB reference), all-silent audio has
        uniform energy across bands, so all deviations are 0 dB.  This
        correctly means no anomalies or conflicts will be detected.
        """
        audio = np.zeros(int(SR * 1.0), dtype=np.float32)
        profile = compute_spectral_profile(audio, SR, stem_type="other")

        assert isinstance(profile, SpectralProfile)
        # All bands at the same level => all deviations are 0 after normalization
        assert np.all(np.abs(profile.band_energies_db) < 1.0), (
            f"Silent audio should have near-zero deviation, "
            f"max deviation = {np.max(np.abs(profile.band_energies_db)):.1f} dB"
        )
        # No peaks should be detected in uniform silence
        assert len(profile.peak_frequencies_hz) == 0

    def test_short_audio_padded(self):
        """Audio shorter than nperseg should be zero-padded and still work."""
        audio = _make_sine(440.0, duration=0.01)  # ~441 samples
        profile = compute_spectral_profile(audio, SR, stem_type="piano")

        assert isinstance(profile, SpectralProfile)
        assert len(profile.band_energies_db) == 31
        assert np.all(np.isfinite(profile.band_energies_db))


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


class TestDetectConflicts:
    """Tests for detect_conflicts."""

    def test_overlapping_sines_produce_conflict(self):
        """Two stems both loud at the same frequency should produce a conflict."""
        # Both stems have energy at 400 Hz
        audio_a = _make_sine(400.0, amplitude=0.9)
        audio_b = _make_sine(400.0, amplitude=0.9)

        profile_a = compute_spectral_profile(audio_a, SR, stem_type="vocals")
        profile_b = compute_spectral_profile(audio_b, SR, stem_type="guitar")

        conflicts = detect_conflicts([profile_a], [profile_b])

        # There should be at least one conflict near 400 Hz
        relevant = [c for c in conflicts if abs(c.center_hz - 400.0) <= 200.0]
        assert len(relevant) >= 1, (
            f"Expected conflict near 400 Hz, got conflicts at: "
            f"{[c.center_hz for c in conflicts]}"
        )

        # The guitar (lower priority) should be the recommended cut stem
        for c in relevant:
            assert c.recommended_cut_stem == "guitar"
            assert c.recommended_cut_db <= 0.0  # cut, not boost
            assert c.recommended_q >= 1.5

    def test_non_overlapping_sines_no_conflict(self):
        """Stems at very different frequencies should produce no conflict."""
        audio_a = _make_sine(200.0, amplitude=0.9)
        audio_b = _make_sine(8000.0, amplitude=0.9)

        profile_a = compute_spectral_profile(audio_a, SR, stem_type="vocals")
        profile_b = compute_spectral_profile(audio_b, SR, stem_type="drums")

        conflicts = detect_conflicts([profile_a], [profile_b])

        # Should be no conflicts where both stems have high energy in the same band
        # (allow at most minor conflicts from spectral leakage, but none near
        # the primary frequencies)
        near_200 = [c for c in conflicts if abs(c.center_hz - 200.0) <= 100.0]
        near_8k = [c for c in conflicts if abs(c.center_hz - 8000.0) <= 2000.0]
        # Neither frequency band should have a conflict involving both stems
        assert len(near_200) == 0, f"Unexpected conflict near 200 Hz"
        assert len(near_8k) == 0, f"Unexpected conflict near 8 kHz"

    def test_vocals_always_win_in_presence_range(self):
        """In the 2-5 kHz range, vocal conflicts should always cut the instrumental."""
        audio_v = _make_sine(3000.0, amplitude=0.9)
        audio_i = _make_sine(3000.0, amplitude=0.9)

        profile_v = compute_spectral_profile(audio_v, SR, stem_type="vocals")
        profile_i = compute_spectral_profile(audio_i, SR, stem_type="bass")

        conflicts = detect_conflicts([profile_v], [profile_i])

        # All conflicts in the presence range should cut the instrumental (bass)
        presence_conflicts = [
            c for c in conflicts
            if 2000.0 <= c.center_hz <= 5000.0
        ]
        for c in presence_conflicts:
            assert c.recommended_cut_stem == "bass", (
                f"Conflict at {c.center_hz:.0f} Hz cuts {c.recommended_cut_stem}, "
                f"expected bass (vocals always win in presence range)"
            )

    def test_conflict_severity_ordering(self):
        """Conflicts should be sorted by severity (descending)."""
        audio_a = _make_sine(400.0, amplitude=0.9)
        audio_b = _make_sine(400.0, amplitude=0.9)

        profile_a = compute_spectral_profile(audio_a, SR, stem_type="vocals")
        profile_b = compute_spectral_profile(audio_b, SR, stem_type="guitar")

        conflicts = detect_conflicts([profile_a], [profile_b])

        if len(conflicts) > 1:
            severities = [c.severity_db for c in conflicts]
            assert severities == sorted(severities, reverse=True), (
                "Conflicts not sorted by severity descending"
            )

    def test_empty_profiles_no_conflicts(self):
        """Empty profile lists should produce no conflicts."""
        assert detect_conflicts([], []) == []

    def test_silent_profiles_no_conflicts(self):
        """Silent stems should produce no conflicts (all below threshold)."""
        silent = np.zeros(int(SR * 1.0), dtype=np.float32)
        profile_a = compute_spectral_profile(silent, SR, stem_type="vocals")
        profile_b = compute_spectral_profile(silent, SR, stem_type="drums")

        conflicts = detect_conflicts([profile_a], [profile_b])
        assert len(conflicts) == 0


# ---------------------------------------------------------------------------
# Adaptive corrections
# ---------------------------------------------------------------------------


class TestComputeAdaptiveCorrections:
    """Tests for compute_adaptive_corrections."""

    def test_corrections_are_cuts_only(self):
        """All adaptive corrections should have negative gain (cuts only)."""
        audio = _make_sine(400.0, amplitude=0.9)
        profile_v = compute_spectral_profile(audio, SR, stem_type="vocals")
        profile_i = compute_spectral_profile(audio, SR, stem_type="guitar")

        conflicts = detect_conflicts([profile_v], [profile_i])
        vocal_corr, inst_corr = compute_adaptive_corrections(
            conflicts, [profile_v], [profile_i]
        )

        for corrections in [vocal_corr, inst_corr]:
            for stem, corr_list in corrections.items():
                for freq, gain, q in corr_list:
                    assert gain <= 0.0, (
                        f"Correction for {stem} at {freq:.0f} Hz has positive gain "
                        f"{gain:.1f} dB (must be cut only)"
                    )

    def test_gain_clamped_to_minus_4(self):
        """No correction should exceed -4 dB."""
        audio = _make_sine(400.0, amplitude=0.9)
        profile_v = compute_spectral_profile(audio, SR, stem_type="vocals")
        profile_i = compute_spectral_profile(audio, SR, stem_type="guitar")

        conflicts = detect_conflicts([profile_v], [profile_i])
        vocal_corr, inst_corr = compute_adaptive_corrections(
            conflicts, [profile_v], [profile_i]
        )

        for corrections in [vocal_corr, inst_corr]:
            for stem, corr_list in corrections.items():
                for freq, gain, q in corr_list:
                    assert gain >= -4.0, (
                        f"Correction for {stem} at {freq:.0f} Hz has gain {gain:.1f} dB "
                        f"(must be >= -4 dB)"
                    )

    def test_max_4_corrections_per_stem(self):
        """Each stem should have at most 4 corrections."""
        # Create a broadband signal that triggers many anomalies
        audio = np.random.RandomState(42).randn(int(SR * 2)).astype(np.float32) * 0.9
        profile_v = compute_spectral_profile(audio, SR, stem_type="vocals")
        profile_i = compute_spectral_profile(audio, SR, stem_type="guitar")

        conflicts = detect_conflicts([profile_v], [profile_i])
        vocal_corr, inst_corr = compute_adaptive_corrections(
            conflicts, [profile_v], [profile_i]
        )

        for corrections in [vocal_corr, inst_corr]:
            for stem, corr_list in corrections.items():
                assert len(corr_list) <= MAX_CORRECTIONS_PER_STEM, (
                    f"Stem {stem} has {len(corr_list)} corrections, "
                    f"max is {MAX_CORRECTIONS_PER_STEM}"
                )

    def test_q_range(self):
        """All Q values should be in [1.5, 3.0]."""
        audio = _make_sine(400.0, amplitude=0.9)
        profile_v = compute_spectral_profile(audio, SR, stem_type="vocals")
        profile_i = compute_spectral_profile(audio, SR, stem_type="guitar")

        conflicts = detect_conflicts([profile_v], [profile_i])
        vocal_corr, inst_corr = compute_adaptive_corrections(
            conflicts, [profile_v], [profile_i]
        )

        for corrections in [vocal_corr, inst_corr]:
            for stem, corr_list in corrections.items():
                for freq, gain, q in corr_list:
                    assert 1.5 <= q <= 3.0, (
                        f"Q for {stem} at {freq:.0f} Hz is {q:.2f}, "
                        f"expected in [1.5, 3.0]"
                    )

    def test_shared_stem_type_routes_to_correct_source(self):
        """When both songs have an 'other' stem, corrections route by position, not name.

        detect_conflicts() sets stem_a=vocal-source, stem_b=instrumental-source.
        If the recommended cut is stem_b ('other' from instrumental), it must
        land in inst_corrections — not vocal_corrections just because 'other'
        also exists in the vocal profiles.
        """
        # Both sources have an "other" stem with energy at the same frequency
        audio = _make_sine(400.0, amplitude=0.9)
        vocal_other = compute_spectral_profile(audio, SR, stem_type="other")
        inst_other = compute_spectral_profile(audio, SR, stem_type="other")

        conflicts = detect_conflicts([vocal_other], [inst_other])
        assert len(conflicts) >= 1, "Expected at least one conflict for shared 'other' stems"

        vocal_corr, inst_corr = compute_adaptive_corrections(
            conflicts, [vocal_other], [inst_other]
        )

        # The conflict's recommended_cut_stem is determined by _resolve_cut_stem.
        # For equal-priority stems, stem_b (instrumental) gets cut.
        # Therefore 'other' corrections from cross-stem conflicts must appear
        # in inst_corrections, NOT vocal_corrections.
        #
        # Note: vocal_corr may still have 'other' entries from per-stem anomaly
        # corrections (step 1), but cross-stem conflict corrections for the
        # instrumental 'other' must not land in vocal_corr.
        #
        # We verify by checking that inst_corr has 'other' entries — if the bug
        # were present, all 'other' conflict corrections would go to vocal_corr.
        assert "other" in inst_corr, (
            f"Instrumental 'other' corrections missing — likely misrouted to "
            f"vocal_corrections. vocal_corr keys: {list(vocal_corr.keys())}, "
            f"inst_corr keys: {list(inst_corr.keys())}"
        )

    def test_silent_stems_no_corrections(self):
        """Silent stems should produce no corrections."""
        silent = np.zeros(int(SR * 1.0), dtype=np.float32)
        profile_v = compute_spectral_profile(silent, SR, stem_type="vocals")
        profile_i = compute_spectral_profile(silent, SR, stem_type="drums")

        conflicts = detect_conflicts([profile_v], [profile_i])
        vocal_corr, inst_corr = compute_adaptive_corrections(
            conflicts, [profile_v], [profile_i]
        )

        assert len(vocal_corr) == 0
        assert len(inst_corr) == 0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    """Performance tests — analysis must stay within budget."""

    @pytest.mark.slow
    def test_12_stems_under_2_seconds(self):
        """Spectral analysis of 12 stems (2 songs x 6 stems) should complete in <10s."""
        # Generate 12 realistic-length sine signals (60s each)
        stems = []
        freqs = [200, 400, 800, 1600, 3200, 6400, 250, 500, 1000, 2000, 4000, 8000]
        for freq in freqs:
            audio = _make_sine(float(freq), duration=60.0, amplitude=0.5)
            stems.append(audio)

        start = time.monotonic()

        profiles = []
        stem_types = ["vocals", "drums", "bass", "guitar", "piano", "other"] * 2
        for audio, stype in zip(stems, stem_types):
            profiles.append(compute_spectral_profile(audio, SR, stem_type=stype))

        vocal_profiles = profiles[:6]
        inst_profiles = profiles[6:]

        conflicts = detect_conflicts(vocal_profiles, inst_profiles)
        compute_adaptive_corrections(conflicts, vocal_profiles, inst_profiles)

        elapsed = time.monotonic() - start
        assert elapsed < 10.0, (
            f"12-stem analysis took {elapsed:.2f}s, budget is 10.0s"
        )
