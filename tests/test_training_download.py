"""Tests for musicmixer.training.download -- mashup download pipeline.

Covers:
- Manifest loading and validation
- Download skipping for existing files
- Error handling (failed downloads continue)
- Rate limiting between downloads
- Sync wrapper around async download_youtube_audio
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from musicmixer.services.youtube import YouTubeAudioResult, YouTubeDownloadError
from musicmixer.training.download import (
    _RATE_LIMIT_DELAY,
    _download_one_sync,
    download_mashups,
    load_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest_data() -> list[dict]:
    """Minimal valid manifest with 3 entries."""
    return [
        {
            "id": "mashup_001",
            "url": "https://www.youtube.com/watch?v=abc123",
            "title": "Test Mashup 1",
            "artist": "Test Artist",
            "genre_hint": "pop",
            "source_a_url": None,
            "source_b_url": None,
            "notes": "",
        },
        {
            "id": "mashup_002",
            "url": "https://www.youtube.com/watch?v=def456",
            "title": "Test Mashup 2",
            "artist": "Test Artist 2",
            "genre_hint": "hiphop",
            "source_a_url": None,
            "source_b_url": None,
            "notes": "",
        },
        {
            "id": "mashup_003",
            "url": "https://www.youtube.com/watch?v=ghi789",
            "title": "Test Mashup 3",
            "artist": "Test Artist 3",
            "genre_hint": "edm",
            "source_a_url": None,
            "source_b_url": None,
            "notes": "",
        },
    ]


@pytest.fixture
def manifest_file(tmp_path: Path, manifest_data: list[dict]) -> Path:
    """Write manifest data to a temporary JSON file."""
    manifest_path = tmp_path / "mashup_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data))
    return manifest_path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    out = tmp_path / "raw"
    out.mkdir()
    return out


def _make_yt_result(wav_path: Path) -> YouTubeAudioResult:
    """Build a YouTubeAudioResult pointing at the given path."""
    return YouTubeAudioResult(
        wav_path=wav_path,
        title="Test Title",
        duration_seconds=180.0,
        source_codec="opus",
        source_bitrate=128,
    )


# ===========================================================================
# load_manifest tests
# ===========================================================================


class TestLoadManifest:
    """Validate manifest loading and error cases."""

    def test_loads_valid_manifest(
        self, manifest_file: Path, manifest_data: list[dict]
    ) -> None:
        result = load_manifest(manifest_file)
        assert len(result) == len(manifest_data)
        assert result[0]["id"] == "mashup_001"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            load_manifest(tmp_path / "nonexistent.json")

    def test_raises_on_non_array(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"not": "an array"}')
        with pytest.raises(ValueError, match="must be a JSON array"):
            load_manifest(path)

    def test_raises_on_missing_required_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"id": "ok"}]))  # missing 'url'
        with pytest.raises(ValueError, match="missing required fields"):
            load_manifest(path)

    def test_minimal_entry_accepted(self, tmp_path: Path) -> None:
        """Entries with only 'id' and 'url' are valid."""
        path = tmp_path / "minimal.json"
        path.write_text(
            json.dumps([{"id": "m1", "url": "https://youtube.com/watch?v=x"}])
        )
        result = load_manifest(path)
        assert len(result) == 1


# ===========================================================================
# _download_one_sync tests
# ===========================================================================


class TestDownloadOneSync:
    """Test the sync wrapper around async download_youtube_audio."""

    @patch("musicmixer.training.download.download_youtube_audio")
    def test_calls_async_function(
        self, mock_download: MagicMock, tmp_path: Path
    ) -> None:
        """Wrapper invokes the async function via asyncio.run."""
        expected_path = tmp_path / "output.wav"
        expected_path.touch()
        mock_download.return_value = _make_yt_result(expected_path)

        result = _download_one_sync(
            "https://www.youtube.com/watch?v=test", tmp_path
        )

        mock_download.assert_called_once_with(
            "https://www.youtube.com/watch?v=test", tmp_path
        )
        assert result.wav_path == expected_path

    @patch("musicmixer.training.download.download_youtube_audio")
    def test_propagates_youtube_errors(
        self, mock_download: MagicMock, tmp_path: Path
    ) -> None:
        """YouTubeDownloadError propagates through the sync wrapper."""
        mock_download.side_effect = YouTubeDownloadError("Video unavailable")

        with pytest.raises(YouTubeDownloadError, match="Video unavailable"):
            _download_one_sync("https://www.youtube.com/watch?v=bad", tmp_path)


# ===========================================================================
# download_mashups tests
# ===========================================================================


class TestDownloadMashups:
    """Test the batch download pipeline."""

    @patch("musicmixer.training.download._download_one_sync")
    @patch("musicmixer.training.download.time.sleep")
    def test_downloads_all_entries(
        self,
        mock_sleep: MagicMock,
        mock_download: MagicMock,
        manifest_file: Path,
        output_dir: Path,
    ) -> None:
        """All entries are downloaded and results returned."""

        def _side_effect(url: str, out: Path) -> YouTubeAudioResult:
            # Simulate yt-dlp creating a file with a UUID name
            fake_path = out / "fakeuuid.wav"
            fake_path.write_bytes(b"RIFF" + b"\x00" * 100)
            return _make_yt_result(fake_path)

        mock_download.side_effect = _side_effect

        results = download_mashups(manifest_file, output_dir)

        assert len(results) == 3
        assert "mashup_001" in results
        assert "mashup_002" in results
        assert "mashup_003" in results

        # All result paths should use {id}.wav naming
        for mashup_id, path in results.items():
            assert path.name == f"{mashup_id}.wav"

        # Rate limiting: sleep called between downloads (not after last)
        assert mock_sleep.call_count == 2

    @patch("musicmixer.training.download._download_one_sync")
    @patch("musicmixer.training.download.time.sleep")
    def test_skips_existing_files(
        self,
        mock_sleep: MagicMock,
        mock_download: MagicMock,
        manifest_file: Path,
        output_dir: Path,
    ) -> None:
        """Already-downloaded files are skipped."""
        # Pre-create mashup_001.wav
        existing = output_dir / "mashup_001.wav"
        existing.write_bytes(b"RIFF" + b"\x00" * 100)

        def _side_effect(url: str, out: Path) -> YouTubeAudioResult:
            fake_path = out / "fakeuuid.wav"
            fake_path.write_bytes(b"RIFF" + b"\x00" * 100)
            return _make_yt_result(fake_path)

        mock_download.side_effect = _side_effect

        results = download_mashups(manifest_file, output_dir)

        # mashup_001 was skipped, but still in results
        assert len(results) == 3
        assert results["mashup_001"] == existing

        # Only 2 downloads actually happened
        assert mock_download.call_count == 2

    @patch("musicmixer.training.download._download_one_sync")
    @patch("musicmixer.training.download.time.sleep")
    def test_continues_on_failure(
        self,
        mock_sleep: MagicMock,
        mock_download: MagicMock,
        manifest_file: Path,
        output_dir: Path,
    ) -> None:
        """Failed downloads are logged and skipped; remaining continue."""
        call_count = 0

        def _side_effect(url: str, out: Path) -> YouTubeAudioResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise YouTubeDownloadError("Video unavailable")
            fake_path = out / "fakeuuid.wav"
            fake_path.write_bytes(b"RIFF" + b"\x00" * 100)
            return _make_yt_result(fake_path)

        mock_download.side_effect = _side_effect

        results = download_mashups(manifest_file, output_dir)

        # First entry failed, 2 succeeded
        assert len(results) == 2
        assert "mashup_001" not in results
        assert "mashup_002" in results
        assert "mashup_003" in results

    @patch("musicmixer.training.download._download_one_sync")
    @patch("musicmixer.training.download.time.sleep")
    def test_handles_unexpected_errors(
        self,
        mock_sleep: MagicMock,
        mock_download: MagicMock,
        manifest_file: Path,
        output_dir: Path,
    ) -> None:
        """Non-YouTube errors are caught and logged."""
        call_count = 0

        def _side_effect(url: str, out: Path) -> YouTubeAudioResult:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Unexpected network issue")
            fake_path = out / "fakeuuid.wav"
            fake_path.write_bytes(b"RIFF" + b"\x00" * 100)
            return _make_yt_result(fake_path)

        mock_download.side_effect = _side_effect

        results = download_mashups(manifest_file, output_dir)

        # Second entry failed, 2 succeeded
        assert len(results) == 2
        assert "mashup_002" not in results

    @patch("musicmixer.training.download._download_one_sync")
    @patch("musicmixer.training.download.time.sleep")
    def test_creates_output_dir_if_missing(
        self,
        mock_sleep: MagicMock,
        mock_download: MagicMock,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        """Output directory is created if it does not exist."""
        new_dir = tmp_path / "new" / "nested" / "dir"

        def _side_effect(url: str, out: Path) -> YouTubeAudioResult:
            fake_path = out / "fakeuuid.wav"
            fake_path.write_bytes(b"RIFF" + b"\x00" * 100)
            return _make_yt_result(fake_path)

        mock_download.side_effect = _side_effect

        results = download_mashups(manifest_file, new_dir)

        assert new_dir.exists()
        assert len(results) == 3

    @patch("musicmixer.training.download._download_one_sync")
    @patch("musicmixer.training.download.time.sleep")
    def test_skips_empty_existing_files(
        self,
        mock_sleep: MagicMock,
        mock_download: MagicMock,
        manifest_file: Path,
        output_dir: Path,
    ) -> None:
        """Zero-byte existing files are re-downloaded (not treated as valid)."""
        empty = output_dir / "mashup_001.wav"
        empty.touch()  # 0 bytes

        def _side_effect(url: str, out: Path) -> YouTubeAudioResult:
            fake_path = out / "fakeuuid.wav"
            fake_path.write_bytes(b"RIFF" + b"\x00" * 100)
            return _make_yt_result(fake_path)

        mock_download.side_effect = _side_effect

        results = download_mashups(manifest_file, output_dir)

        # All 3 downloaded (empty file was not skipped)
        assert mock_download.call_count == 3
        assert len(results) == 3
