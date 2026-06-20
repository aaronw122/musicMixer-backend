"""Unit tests for the pure helpers in services.remix_stages."""

import pytest

from musicmixer.services.remix_stages import (
    UploadTooLargeError,
    extension_allowed,
    probe_duration,
    thumbnail_from_youtube_url,
    upload_extension,
    write_upload_file,
)


class TestThumbnailFromYouTubeUrl:
    def test_watch_url(self):
        assert (
            thumbnail_from_youtube_url("https://www.youtube.com/watch?v=abc123")
            == "https://i.ytimg.com/vi/abc123/hqdefault.jpg"
        )

    def test_short_url(self):
        assert (
            thumbnail_from_youtube_url("https://youtu.be/xyz789")
            == "https://i.ytimg.com/vi/xyz789/hqdefault.jpg"
        )

    def test_shorts_url(self):
        assert (
            thumbnail_from_youtube_url("https://www.youtube.com/shorts/short01")
            == "https://i.ytimg.com/vi/short01/hqdefault.jpg"
        )

    def test_embed_url(self):
        assert (
            thumbnail_from_youtube_url("https://www.youtube.com/embed/emb42")
            == "https://i.ytimg.com/vi/emb42/hqdefault.jpg"
        )

    def test_no_video_id_returns_none(self):
        assert thumbnail_from_youtube_url("https://www.youtube.com/") is None


class TestUploadExtension:
    def test_lowercases_suffix(self):
        assert upload_extension("Song.MP3") == ".mp3"

    def test_none_filename(self):
        assert upload_extension(None) == ""

    def test_no_suffix(self):
        assert upload_extension("song") == ""


class TestExtensionAllowed:
    ALLOWED = {".mp3", ".wav"}

    def test_allowed(self):
        assert extension_allowed("a.mp3", self.ALLOWED) is True
        assert extension_allowed("a.WAV", self.ALLOWED) is True

    def test_rejected(self):
        assert extension_allowed("a.flac", self.ALLOWED) is False
        assert extension_allowed(None, self.ALLOWED) is False


class TestWriteUploadFile:
    def test_writes_full_payload(self, tmp_path):
        import io

        dest = tmp_path / "out.bin"
        payload = b"x" * (3 * 1024 * 1024 + 7)  # spans multiple chunks
        write_upload_file(io.BytesIO(payload), dest, max_bytes=10 * 1024 * 1024)
        assert dest.read_bytes() == payload

    def test_rejects_oversize(self, tmp_path):
        import io

        dest = tmp_path / "out.bin"
        payload = b"y" * (2 * 1024 * 1024)
        with pytest.raises(UploadTooLargeError):
            write_upload_file(io.BytesIO(payload), dest, max_bytes=1024 * 1024)

    def test_seeks_to_start(self, tmp_path):
        import io

        dest = tmp_path / "out.bin"
        buf = io.BytesIO(b"abcdef")
        buf.read(3)  # advance the cursor
        write_upload_file(buf, dest, max_bytes=1024)
        assert dest.read_bytes() == b"abcdef"


class TestProbeDuration:
    def test_returns_none_for_missing_file(self, tmp_path):
        # ffprobe on a nonexistent path returns non-zero -> None
        assert probe_duration(tmp_path / "does_not_exist.mp3") is None
