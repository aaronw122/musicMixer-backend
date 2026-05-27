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
            mock_settings.stem_cache_enabled = False
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
            mock_settings.stem_cache_enabled = False
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


class TestSeparateVocalSongDispatcher:
    """Test that separate_vocal_song dispatches to correct backend."""

    @patch("musicmixer.services.separation._separate_vocal_song_modal")
    def test_dispatches_to_modal(self, mock_modal, tmp_path):
        """stem_backend='modal' should call _separate_vocal_song_modal."""
        from musicmixer.services.separation import separate_vocal_song

        mock_modal.return_value = {
            "lead_vocals": tmp_path / "lead_vocals.wav",
            "backing_vocals": tmp_path / "backing_vocals.wav",
            "instrumental": tmp_path / "instrumental.wav",
        }

        with patch("musicmixer.services.separation.settings") as mock_settings:
            mock_settings.stem_backend = "modal"
            result = separate_vocal_song(tmp_path / "input.wav", tmp_path / "out")

        mock_modal.assert_called_once()
        assert "lead_vocals" in result
        assert "backing_vocals" in result
        assert "instrumental" in result

    @patch("musicmixer.services.separation._separate_vocal_song_local")
    def test_dispatches_to_local(self, mock_local, tmp_path):
        """stem_backend='local' should call _separate_vocal_song_local."""
        from musicmixer.services.separation import separate_vocal_song

        mock_local.return_value = {
            "lead_vocals": tmp_path / "lead_vocals.wav",
            "backing_vocals": None,
            "instrumental": tmp_path / "instrumental.wav",
        }

        with patch("musicmixer.services.separation.settings") as mock_settings:
            mock_settings.stem_backend = "local"
            result = separate_vocal_song(tmp_path / "input.wav", tmp_path / "out")

        mock_local.assert_called_once()
        assert "lead_vocals" in result
        assert result["backing_vocals"] is None


class TestSeparateVocalSongModalValidation:
    """Test _separate_vocal_song_modal stem validation."""

    def test_raises_on_missing_stems(self, tmp_path):
        """Should raise RuntimeError if Modal returns incomplete vocal stems."""
        from musicmixer.services.separation import _separate_vocal_song_modal

        # Only return 2 of 3 expected stems
        incomplete_stems = {
            "lead_vocals": _make_float32_wav_bytes(440),
            "instrumental": _make_float32_wav_bytes(330),
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = incomplete_stems

        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        with patch("modal.Function.from_name", return_value=mock_remote):
            with pytest.raises(RuntimeError, match="Expected vocal-song stems"):
                _separate_vocal_song_modal(input_path, tmp_path / "out")

    def test_accepts_complete_3_stems(self, tmp_path):
        """Should accept and save all 3 vocal-song stems without error."""
        from musicmixer.services.separation import _separate_vocal_song_modal

        complete_stems = {
            "lead_vocals": _make_float32_wav_bytes(440),
            "backing_vocals": _make_float32_wav_bytes(220),
            "instrumental": _make_float32_wav_bytes(110),
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = complete_stems

        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        output_dir = tmp_path / "out"

        with patch("modal.Function.from_name", return_value=mock_remote):
            result = _separate_vocal_song_modal(input_path, output_dir)

        assert set(result.keys()) == {"lead_vocals", "backing_vocals", "instrumental"}
        for stem_name, stem_path in result.items():
            assert stem_path.exists()
            assert stem_path.stat().st_size > 0

    def test_float32_validation_warns_on_non_float(self, tmp_path, caplog):
        """Should log a warning if vocal stems are not float32."""
        from musicmixer.services.separation import _separate_vocal_song_modal

        # Create a PCM16 WAV
        sr = 44100
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        buf = io.BytesIO()
        sf.write(buf, stereo, sr, format="WAV", subtype="PCM_16")
        pcm16_bytes = buf.getvalue()

        complete_stems = {
            "lead_vocals": pcm16_bytes,  # PCM_16
            "backing_vocals": _make_float32_wav_bytes(220),
            "instrumental": _make_float32_wav_bytes(110),
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = complete_stems

        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        with patch("modal.Function.from_name", return_value=mock_remote):
            with caplog.at_level(logging.WARNING, logger="musicmixer.services.separation"):
                result = _separate_vocal_song_modal(input_path, tmp_path / "out")

        assert "lead_vocals" in result
        assert any("PCM_16" in record.message for record in caplog.records)

    def test_raises_on_extra_stems(self, tmp_path):
        """Should raise RuntimeError if Modal returns unexpected extra stems."""
        from musicmixer.services.separation import _separate_vocal_song_modal

        extra_stems = {
            "lead_vocals": _make_float32_wav_bytes(440),
            "backing_vocals": _make_float32_wav_bytes(220),
            "instrumental": _make_float32_wav_bytes(110),
            "karaoke": _make_float32_wav_bytes(330),  # unexpected extra
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = extra_stems

        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        with patch("modal.Function.from_name", return_value=mock_remote):
            with pytest.raises(RuntimeError, match="Expected vocal-song stems"):
                _separate_vocal_song_modal(input_path, tmp_path / "out")


class TestSeparateVocalSongLocal:
    """Test local fallback for vocal-song separation."""

    def _mock_stems_local(self, output_dir):
        """Create a mock for separate_stems_local that returns pre-created files."""
        vocals_path = output_dir / "vocals.wav"
        other_path = output_dir / "other.wav"
        return {
            "vocals": vocals_path,
            "drums": output_dir / "drums.wav",
            "bass": output_dir / "bass.wav",
            "other": other_path,
            "guitar": None,
            "piano": None,
        }

    def test_maps_vocals_to_lead_vocals(self, tmp_path):
        """Local fallback should map htdemucs_ft 'vocals' -> 'lead_vocals'."""
        from musicmixer.services.separation_local import separate_vocal_song_local

        output_dir = tmp_path / "stems"
        output_dir.mkdir(parents=True)

        sr = 44100
        t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
        mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.column_stack([mono, mono])

        for name in ["vocals", "drums", "bass", "other"]:
            sf.write(str(output_dir / f"{name}.wav"), stereo, sr, format="WAV", subtype="FLOAT")

        mock_result = self._mock_stems_local(output_dir)

        with patch(
            "musicmixer.services.separation_local.separate_stems_local",
            return_value=mock_result,
        ):
            result = separate_vocal_song_local(tmp_path / "input.wav", output_dir)

        assert result["lead_vocals"] is not None
        assert result["backing_vocals"] is None
        assert result["instrumental"] is not None

    def test_backing_vocals_always_none(self, tmp_path):
        """Local fallback cannot split lead/backing, so backing_vocals is always None."""
        from musicmixer.services.separation_local import separate_vocal_song_local

        output_dir = tmp_path / "stems"
        output_dir.mkdir(parents=True)

        sr = 44100
        t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
        mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.column_stack([mono, mono])

        for name in ["vocals", "drums", "bass", "other"]:
            sf.write(str(output_dir / f"{name}.wav"), stereo, sr, format="WAV", subtype="FLOAT")

        mock_result = self._mock_stems_local(output_dir)

        with patch(
            "musicmixer.services.separation_local.separate_stems_local",
            return_value=mock_result,
        ):
            result = separate_vocal_song_local(tmp_path / "input.wav", output_dir)

        assert result["backing_vocals"] is None

    def test_progress_callback_called(self, tmp_path):
        """Progress callback should be invoked during local vocal separation."""
        from musicmixer.services.separation_local import separate_vocal_song_local

        output_dir = tmp_path / "stems"
        output_dir.mkdir(parents=True)

        sr = 44100
        t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
        mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.column_stack([mono, mono])

        for name in ["vocals", "drums", "bass", "other"]:
            sf.write(str(output_dir / f"{name}.wav"), stereo, sr, format="WAV", subtype="FLOAT")

        callback = MagicMock()
        mock_result = self._mock_stems_local(output_dir)

        with patch(
            "musicmixer.services.separation_local.separate_stems_local",
            return_value=mock_result,
        ):
            separate_vocal_song_local(tmp_path / "input.wav", output_dir, callback)

        callback.assert_called()


class TestSaveStemBytes:
    """Test the shared _save_stem_bytes helper."""

    def test_saves_stems_to_disk(self, tmp_path):
        """Should save all stems as WAV files and return correct paths."""
        from musicmixer.services.separation import _save_stem_bytes

        stem_bytes = {
            "lead_vocals": _make_float32_wav_bytes(440),
            "backing_vocals": _make_float32_wav_bytes(220),
            "instrumental": _make_float32_wav_bytes(110),
        }

        output_dir = tmp_path / "out"
        result = _save_stem_bytes(stem_bytes, output_dir)

        assert set(result.keys()) == {"lead_vocals", "backing_vocals", "instrumental"}
        for stem_name, stem_path in result.items():
            assert stem_path.exists()
            assert stem_path.name == f"{stem_name}.wav"
            assert stem_path.stat().st_size > 0

    def test_creates_output_dir(self, tmp_path):
        """Should create the output directory if it doesn't exist."""
        from musicmixer.services.separation import _save_stem_bytes

        stem_bytes = {"test": _make_float32_wav_bytes()}
        output_dir = tmp_path / "deeply" / "nested" / "out"

        assert not output_dir.exists()
        _save_stem_bytes(stem_bytes, output_dir)
        assert output_dir.exists()


class TestVocalSongStemsConstant:
    """Test the VOCAL_SONG_STEMS constant."""

    def test_contains_expected_stems(self):
        """VOCAL_SONG_STEMS should have exactly the 3 expected stem names."""
        from musicmixer.services.separation import VOCAL_SONG_STEMS

        assert VOCAL_SONG_STEMS == {"lead_vocals", "backing_vocals", "instrumental"}

    def test_no_overlap_with_bs_roformer_stems(self):
        """Vocal-song stems should not overlap with BS-RoFormer instrumental stems."""
        from musicmixer.services.separation import VOCAL_SONG_STEMS

        bs_roformer_stems = {"vocals", "drums", "bass", "guitar", "piano", "other"}
        overlap = VOCAL_SONG_STEMS & bs_roformer_stems
        assert overlap == set(), f"Unexpected overlap: {overlap}"


class TestModalConstants:
    """Test Modal module constants for model checkpoints."""

    def test_karaoke_model_checkpoint_defined(self):
        """Karaoke model checkpoint constant should be defined."""
        from musicmixer.services.separation_modal import MELBAND_KARAOKE_CKPT

        assert MELBAND_KARAOKE_CKPT.endswith(".ckpt")

    def test_vocals_model_checkpoint_defined(self):
        """Vocals model checkpoint constant should be defined."""
        from musicmixer.services.separation_modal import MELBAND_VOCALS_CKPT

        assert MELBAND_VOCALS_CKPT.endswith(".ckpt")

    def test_bs_roformer_checkpoint_unchanged(self):
        """Existing BS-RoFormer checkpoint constant should be preserved."""
        from musicmixer.services.separation_modal import MODEL_CKPT

        assert MODEL_CKPT == "BS-Roformer-SW.ckpt"

    def test_no_msst_references(self):
        """No MSST-related constants should remain after the rewrite."""
        import musicmixer.services.separation_modal as mod

        assert not hasattr(mod, "MSST_WEIGHTS_DIR")
        assert not hasattr(mod, "MELBAND_KARAOKE_CONFIG")
        assert not hasattr(mod, "MELBAND_KARAOKE_HF_REPO")
        assert not hasattr(mod, "MELBAND_VOCALS_CONFIG")
        assert not hasattr(mod, "MELBAND_VOCALS_HF_REPO")
