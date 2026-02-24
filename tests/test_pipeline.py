"""Tests for musicmixer.services.pipeline_day1 - Day 1 synchronous pipeline."""

from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest


class TestRunPipelineSync:
    """Test the Day 1 synchronous pipeline."""

    @patch("musicmixer.services.pipeline_day1.overlay_and_export")
    @patch("musicmixer.services.pipeline_day1.separate_stems")
    def test_separates_both_songs(self, mock_separate, mock_overlay, tmp_path):
        """Pipeline should call separate_stems for both Song A and Song B."""
        from musicmixer.services.pipeline_day1 import run_pipeline_sync

        # Mock separation to return fake stem paths
        fake_stems = {
            "vocals": tmp_path / "vocals.wav",
            "drums": tmp_path / "drums.wav",
            "bass": tmp_path / "bass.wav",
            "guitar": tmp_path / "guitar.wav",
            "piano": tmp_path / "piano.wav",
            "other": tmp_path / "other.wav",
        }
        mock_separate.return_value = fake_stems
        mock_overlay.return_value = tmp_path / "remix.mp3"

        song_a = tmp_path / "song_a.wav"
        song_b = tmp_path / "song_b.wav"
        song_a.touch()
        song_b.touch()

        with patch("musicmixer.services.pipeline_day1.settings") as mock_settings:
            mock_settings.data_dir = tmp_path / "data"

            run_pipeline_sync("test-session", song_a, song_b)

        # Should have been called twice (once for each song)
        assert mock_separate.call_count == 2

    @patch("musicmixer.services.pipeline_day1.overlay_and_export")
    @patch("musicmixer.services.pipeline_day1.separate_stems")
    def test_passes_correct_stems_to_mixer(self, mock_separate, mock_overlay, tmp_path):
        """Pipeline should pass Song A vocals + Song B instrumentals to mixer."""
        from musicmixer.services.pipeline_day1 import run_pipeline_sync

        # Different paths for song A and song B stems
        song_a_stems = {
            "vocals": tmp_path / "a_vocals.wav",
            "drums": tmp_path / "a_drums.wav",
            "bass": tmp_path / "a_bass.wav",
            "guitar": tmp_path / "a_guitar.wav",
            "piano": tmp_path / "a_piano.wav",
            "other": tmp_path / "a_other.wav",
        }
        song_b_stems = {
            "vocals": tmp_path / "b_vocals.wav",
            "drums": tmp_path / "b_drums.wav",
            "bass": tmp_path / "b_bass.wav",
            "guitar": tmp_path / "b_guitar.wav",
            "piano": tmp_path / "b_piano.wav",
            "other": tmp_path / "b_other.wav",
        }

        # Return different stems for each call
        mock_separate.side_effect = [song_a_stems, song_b_stems]
        mock_overlay.return_value = tmp_path / "remix.mp3"

        song_a = tmp_path / "song_a.wav"
        song_b = tmp_path / "song_b.wav"
        song_a.touch()
        song_b.touch()

        with patch("musicmixer.services.pipeline_day1.settings") as mock_settings:
            mock_settings.data_dir = tmp_path / "data"

            run_pipeline_sync("test-session", song_a, song_b)

        # Verify overlay_and_export was called with correct stems
        mock_overlay.assert_called_once()
        call_kwargs = mock_overlay.call_args

        # Vocals should come from Song A
        assert call_kwargs.kwargs["vocal_stems"] == {"vocals": song_a_stems["vocals"]}

        # Instrumentals should come from Song B
        expected_instrumentals = {
            "drums": song_b_stems["drums"],
            "bass": song_b_stems["bass"],
            "guitar": song_b_stems["guitar"],
            "piano": song_b_stems["piano"],
            "other": song_b_stems["other"],
        }
        assert call_kwargs.kwargs["instrumental_stems"] == expected_instrumentals

    @patch("musicmixer.services.pipeline_day1.overlay_and_export")
    @patch("musicmixer.services.pipeline_day1.separate_stems")
    def test_returns_output_path(self, mock_separate, mock_overlay, tmp_path):
        """Pipeline should return the path to the exported remix."""
        from musicmixer.services.pipeline_day1 import run_pipeline_sync

        fake_stems = {
            "vocals": tmp_path / "vocals.wav",
            "drums": tmp_path / "drums.wav",
            "bass": tmp_path / "bass.wav",
            "guitar": tmp_path / "guitar.wav",
            "piano": tmp_path / "piano.wav",
            "other": tmp_path / "other.wav",
        }
        mock_separate.return_value = fake_stems
        mock_overlay.return_value = tmp_path / "data" / "remixes" / "test-session" / "remix.mp3"

        song_a = tmp_path / "song_a.wav"
        song_b = tmp_path / "song_b.wav"
        song_a.touch()
        song_b.touch()

        with patch("musicmixer.services.pipeline_day1.settings") as mock_settings:
            mock_settings.data_dir = tmp_path / "data"

            result = run_pipeline_sync("test-session", song_a, song_b)

        assert result is not None
        assert "remix.mp3" in str(result)

    @patch("musicmixer.services.pipeline_day1.overlay_and_export")
    @patch("musicmixer.services.pipeline_day1.separate_stems")
    def test_creates_output_directories(self, mock_separate, mock_overlay, tmp_path):
        """Pipeline should create stems and remix directories."""
        from musicmixer.services.pipeline_day1 import run_pipeline_sync

        fake_stems = {
            "vocals": tmp_path / "vocals.wav",
            "drums": tmp_path / "drums.wav",
            "bass": tmp_path / "bass.wav",
            "guitar": tmp_path / "guitar.wav",
            "piano": tmp_path / "piano.wav",
            "other": tmp_path / "other.wav",
        }
        mock_separate.return_value = fake_stems
        mock_overlay.return_value = tmp_path / "remix.mp3"

        song_a = tmp_path / "song_a.wav"
        song_b = tmp_path / "song_b.wav"
        song_a.touch()
        song_b.touch()

        data_dir = tmp_path / "data"

        with patch("musicmixer.services.pipeline_day1.settings") as mock_settings:
            mock_settings.data_dir = data_dir

            run_pipeline_sync("test-session", song_a, song_b)

        # Remix directory should have been created
        assert (data_dir / "remixes" / "test-session").exists()
