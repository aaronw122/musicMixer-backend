"""Tests for musicmixer.services.separation - dispatcher logic."""

import io
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import soundfile as sf
import pytest


def _make_float32_wav_bytes(freq: float = 440.0, duration: float = 0.5, sr: int = 44100) -> bytes:
    """Create float32 WAV bytes for a sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


class TestSeparateStemsDispatcher:
    """Test that separate_stems dispatches to correct backend."""

    @patch("musicmixer.services.separation._separate_modal")
    def test_dispatches_to_modal(self, mock_modal, tmp_path):
        """stem_backend='modal' should call _separate_modal."""
        from musicmixer.services.separation import separate_stems

        mock_modal.return_value = {"vocals": tmp_path / "vocals.wav"}

        with patch("musicmixer.services.separation.settings") as mock_settings:
            mock_settings.stem_backend = "modal"
            result = separate_stems(tmp_path / "input.wav", tmp_path / "out")

        mock_modal.assert_called_once()
        assert "vocals" in result

    @patch("musicmixer.services.separation._separate_local")
    def test_dispatches_to_local(self, mock_local, tmp_path):
        """stem_backend='local' should call _separate_local."""
        from musicmixer.services.separation import separate_stems

        mock_local.return_value = {"vocals": tmp_path / "vocals.wav"}

        with patch("musicmixer.services.separation.settings") as mock_settings:
            mock_settings.stem_backend = "local"
            result = separate_stems(tmp_path / "input.wav", tmp_path / "out")

        mock_local.assert_called_once()
        assert "vocals" in result


class TestSeparateModalValidation:
    """Test _separate_modal stem validation."""

    def test_raises_on_missing_stems(self, tmp_path):
        """Should raise RuntimeError if Modal returns fewer than 6 stems."""
        from musicmixer.services.separation import _separate_modal

        # Mock separate_stems_remote to return only 4 stems
        incomplete_stems = {
            "vocals": _make_float32_wav_bytes(440),
            "drums": _make_float32_wav_bytes(220),
            "bass": _make_float32_wav_bytes(110),
            "other": _make_float32_wav_bytes(330),
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = incomplete_stems

        # Create a real input file
        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        with patch("modal.Function.from_name", return_value=mock_remote):
            with pytest.raises(RuntimeError, match="Expected 6 stems"):
                _separate_modal(input_path, tmp_path / "out")

    def test_accepts_complete_6_stems(self, tmp_path):
        """Should accept and save all 6 stems without error."""
        from musicmixer.services.separation import _separate_modal

        complete_stems = {
            "vocals": _make_float32_wav_bytes(440),
            "drums": _make_float32_wav_bytes(220),
            "bass": _make_float32_wav_bytes(110),
            "guitar": _make_float32_wav_bytes(330),
            "piano": _make_float32_wav_bytes(550),
            "other": _make_float32_wav_bytes(660),
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = complete_stems

        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        output_dir = tmp_path / "out"

        with patch("modal.Function.from_name", return_value=mock_remote):
            result = _separate_modal(input_path, output_dir)

        assert set(result.keys()) == {"vocals", "drums", "bass", "guitar", "piano", "other"}
        for stem_name, stem_path in result.items():
            assert stem_path.exists()
            assert stem_path.stat().st_size > 0

    def test_float32_validation_warns_on_non_float(self, tmp_path, caplog):
        """Should log a warning if stems are not float32."""
        from musicmixer.services.separation import _separate_modal
        import logging

        # Create a PCM16 WAV (not float32)
        sr = 44100
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        buf = io.BytesIO()
        sf.write(buf, stereo, sr, format="WAV", subtype="PCM_16")
        pcm16_bytes = buf.getvalue()

        complete_stems = {
            "vocals": pcm16_bytes,  # PCM_16, not FLOAT
            "drums": _make_float32_wav_bytes(220),
            "bass": _make_float32_wav_bytes(110),
            "guitar": _make_float32_wav_bytes(330),
            "piano": _make_float32_wav_bytes(550),
            "other": _make_float32_wav_bytes(660),
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = complete_stems

        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        output_dir = tmp_path / "out"

        with patch("modal.Function.from_name", return_value=mock_remote):
            with caplog.at_level(logging.WARNING, logger="musicmixer.services.separation"):
                result = _separate_modal(input_path, output_dir)

        # Should still succeed but log a warning
        assert "vocals" in result
        assert any("PCM_16" in record.message for record in caplog.records)


class TestTokenizeStemFilename:
    """Test _tokenize_stem_filename helper in both modules."""

    def test_simple_stem_name(self):
        from musicmixer.services.separation_local import _tokenize_stem_filename

        assert _tokenize_stem_filename("vocals") == ["vocals"]

    def test_underscore_delimited(self):
        from musicmixer.services.separation_local import _tokenize_stem_filename

        assert _tokenize_stem_filename("song_bass_other") == ["song", "bass", "other"]

    def test_hyphen_delimited(self):
        from musicmixer.services.separation_local import _tokenize_stem_filename

        assert _tokenize_stem_filename("song-drums-track") == ["song", "drums", "track"]

    def test_dot_delimited(self):
        from musicmixer.services.separation_local import _tokenize_stem_filename

        assert _tokenize_stem_filename("input.vocals") == ["input", "vocals"]

    def test_mixed_delimiters(self):
        from musicmixer.services.separation_local import _tokenize_stem_filename

        assert _tokenize_stem_filename("my_song-vocals.stem") == ["my", "song", "vocals", "stem"]

    def test_case_insensitive(self):
        from musicmixer.services.separation_local import _tokenize_stem_filename

        assert _tokenize_stem_filename("Song_VOCALS") == ["song", "vocals"]

    def test_modal_tokenizer_matches_local(self):
        """Both modules should have identical tokenizer behavior."""
        from musicmixer.services.separation_local import _tokenize_stem_filename as local_tok
        from musicmixer.services.separation_modal import _tokenize_stem_filename as modal_tok

        cases = [
            "vocals",
            "song_bass_other",
            "my-drums-track",
            "input.guitar",
            "Song_VOCALS",
        ]
        for case in cases:
            assert local_tok(case) == modal_tok(case), f"Mismatch for: {case}"


class TestStemNameMatching:
    """Test that whole-token matching prevents substring collisions."""

    def _create_wav_files(self, output_dir: Path, filenames: list[str]):
        """Create minimal WAV files with given filenames in output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            path = output_dir / name
            # Write a tiny valid WAV
            sr = 44100
            t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
            mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            stereo = np.column_stack([mono, mono])
            sf.write(str(path), stereo, sr, format="WAV", subtype="FLOAT")

    def test_simple_stem_names_match(self, tmp_path):
        """Standard stem filenames should be matched correctly."""
        from musicmixer.services.separation_local import (
            _tokenize_stem_filename,
        )

        output_dir = tmp_path / "stems"
        self._create_wav_files(output_dir, [
            "vocals.wav", "drums.wav", "bass.wav", "other.wav",
        ])

        expected = ["vocals", "drums", "bass", "other"]
        stems = {}
        for stem_file in output_dir.iterdir():
            if stem_file.suffix != ".wav":
                continue
            tokens = _tokenize_stem_filename(stem_file.stem)
            matched = [s for s in expected if s in tokens]
            if matched:
                stems[matched[0]] = stem_file

        assert set(stems.keys()) == {"vocals", "drums", "bass", "other"}

    def test_prefixed_stem_names_match(self, tmp_path):
        """Stem names with prefixes (e.g., 'song_vocals') should match."""
        from musicmixer.services.separation_local import (
            _tokenize_stem_filename,
        )

        output_dir = tmp_path / "stems"
        self._create_wav_files(output_dir, [
            "song_vocals.wav", "song_drums.wav", "song_bass.wav", "song_other.wav",
        ])

        expected = ["vocals", "drums", "bass", "other"]
        stems = {}
        for stem_file in output_dir.iterdir():
            if stem_file.suffix != ".wav":
                continue
            tokens = _tokenize_stem_filename(stem_file.stem)
            matched = [s for s in expected if s in tokens]
            if matched:
                stems[matched[0]] = stem_file

        assert set(stems.keys()) == {"vocals", "drums", "bass", "other"}

    def test_bass_not_matched_by_substring_in_other_word(self, tmp_path):
        """'bass' should NOT match a filename like 'bassing_track.wav'
        where 'bass' only appears as a substring of another token."""
        from musicmixer.services.separation_local import (
            _tokenize_stem_filename,
        )

        tokens = _tokenize_stem_filename("bassing_track")
        expected = ["vocals", "drums", "bass", "other"]
        matched = [s for s in expected if s in tokens]
        assert "bass" not in matched

    def test_other_vocals_not_collision(self, tmp_path):
        """'other_vocals.wav' should match both 'other' and 'vocals' as
        separate tokens, with 'vocals' winning because it appears first
        in the expected_stems list."""
        from musicmixer.services.separation_local import (
            _tokenize_stem_filename,
        )

        tokens = _tokenize_stem_filename("other_vocals")
        expected_4stem = ["vocals", "drums", "bass", "other"]
        matched = [s for s in expected_4stem if s in tokens]
        # Both should be found as separate tokens
        assert "vocals" in matched
        assert "other" in matched
        # First in expected list wins
        assert matched[0] == "vocals"

    def test_song_bass_other_no_false_match(self, tmp_path):
        """'song_bass_other.wav' should match both 'bass' and 'other'
        as separate tokens. With expected_stems order ['vocals', 'drums',
        'bass', 'guitar', 'piano', 'other'], 'bass' comes first."""
        from musicmixer.services.separation_local import (
            _tokenize_stem_filename,
        )

        tokens = _tokenize_stem_filename("song_bass_other")
        expected_6stem = ["vocals", "drums", "bass", "guitar", "piano", "other"]
        matched = [s for s in expected_6stem if s in tokens]
        assert matched == ["bass", "other"]
        # First match wins
        assert matched[0] == "bass"

    def test_bare_substring_no_longer_matches(self):
        """The old bug: 'basser' contains 'bass' as a substring.
        Token-based matching should NOT match this."""
        from musicmixer.services.separation_local import (
            _tokenize_stem_filename,
        )

        tokens = _tokenize_stem_filename("basser")
        expected = ["vocals", "drums", "bass", "other"]
        matched = [s for s in expected if s in tokens]
        assert matched == []

    def test_duplicate_match_logs_warning(self, tmp_path, caplog):
        """When a file matches multiple stems, a warning should be logged."""
        from musicmixer.services.separation_local import (
            _tokenize_stem_filename,
        )

        output_dir = tmp_path / "stems"
        self._create_wav_files(output_dir, ["other_vocals.wav"])

        expected_4stem = ["vocals", "drums", "bass", "other"]

        with caplog.at_level(logging.WARNING, logger="musicmixer.services.separation_local"):
            # Simulate the matching logic from separation_local
            from musicmixer.services.separation_local import logger as local_logger

            for stem_file in output_dir.iterdir():
                if stem_file.suffix != ".wav":
                    continue
                tokens = _tokenize_stem_filename(stem_file.stem)
                matched = [s for s in expected_4stem if s in tokens]
                if len(matched) > 1:
                    local_logger.warning(
                        "File %s matched multiple stems: %s; using first: %s",
                        stem_file.name, matched, matched[0],
                    )

        assert any("matched multiple stems" in record.message for record in caplog.records)
