"""Unit tests for the pure helpers in services.remix_stages."""

from pathlib import Path
from types import SimpleNamespace

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
            thumbnail_from_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            == "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
        )

    def test_short_url(self):
        assert (
            thumbnail_from_youtube_url("https://youtu.be/dQw4w9WgXcQ")
            == "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
        )

    def test_shorts_url(self):
        assert (
            thumbnail_from_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")
            == "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
        )

    def test_embed_url(self):
        assert (
            thumbnail_from_youtube_url("https://www.youtube.com/embed/dQw4w9WgXcQ")
            == "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
        )

    def test_no_video_id_returns_none(self):
        assert thumbnail_from_youtube_url("https://www.youtube.com/") is None

    def test_invalid_video_id_returns_none(self):
        assert (
            thumbnail_from_youtube_url("https://www.youtube.com/watch?v=../../evil")
            is None
        )


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
        assert not dest.exists()

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

    def test_pre_trims_uploads_before_processing(self, monkeypatch, tmp_path):
        import subprocess

        from musicmixer.config import settings

        audio = tmp_path / "long.mp3"
        audio.write_bytes(b"audio")
        monkeypatch.setattr(settings, "max_upload_duration_seconds", 900)
        monkeypatch.setattr(settings, "processing_max_duration_seconds", 210)

        def _fake_run(*_args, **_kwargs):
            return subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="300.0\n",
                stderr="",
            )

        calls: list[tuple[Path, int]] = []

        def _fake_pre_trim(path, max_duration_seconds):
            calls.append((path, max_duration_seconds))
            return path

        monkeypatch.setattr("musicmixer.services.remix_stages.subprocess.run", _fake_run)
        monkeypatch.setattr(
            "musicmixer.services.processor.pre_trim_for_processing",
            _fake_pre_trim,
        )

        assert probe_duration(audio) == 300.0
        assert calls == [(audio, 210)]

    def test_does_not_pre_trim_uploads_over_upload_limit(self, monkeypatch, tmp_path):
        import subprocess

        from musicmixer.config import settings

        audio = tmp_path / "too_long.mp3"
        audio.write_bytes(b"audio")
        monkeypatch.setattr(settings, "max_upload_duration_seconds", 900)
        monkeypatch.setattr(settings, "processing_max_duration_seconds", 210)

        def _fake_run(*_args, **_kwargs):
            return subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="901.0\n",
                stderr="",
            )

        calls: list[tuple[Path, int]] = []

        monkeypatch.setattr("musicmixer.services.remix_stages.subprocess.run", _fake_run)
        monkeypatch.setattr(
            "musicmixer.services.processor.pre_trim_for_processing",
            lambda path, max_duration_seconds: calls.append((path, max_duration_seconds)),
        )

        assert probe_duration(audio) == 901.0
        assert calls == []


class TestRestoreFullyCachedYouTubeRemix:
    def test_get_cached_stems_false_falls_back(self, monkeypatch, tmp_path):
        from musicmixer.services import song_cache
        from musicmixer.services.remix_stages import (
            FullyCachedCallbacks,
            FullyCachedInputs,
            restore_fully_cached_youtube_remix,
        )

        calls = {"restore": 0}

        def _fake_get_cached_stems(*_args, **_kwargs):
            calls["restore"] += 1
            return False

        monkeypatch.setattr(song_cache, "stem_cache_ready", lambda video_id, role: True)
        monkeypatch.setattr(song_cache, "get_cached_stems", _fake_get_cached_stems)

        settings = SimpleNamespace(data_dir=tmp_path / "data")
        inputs = FullyCachedInputs(
            session_id="s1",
            cached_song_a=SimpleNamespace(video_id="dQw4w9WgXcQ"),
            cached_song_b=SimpleNamespace(video_id="9bZkp7q19f0"),
        )
        callbacks = FullyCachedCallbacks(
            check_cancelled=lambda: None,
            on_cache_skip=lambda: None,
        )

        result = restore_fully_cached_youtube_remix(
            inputs,
            callbacks=callbacks,
            settings=settings,
        )

        assert result.used_cache is False
        assert calls["restore"] == 2
        assert not (settings.data_dir / "stems" / "s1").exists()


class TestDownloadYouTubePair:
    def test_skips_download_for_song_with_cached_stems(self, monkeypatch, tmp_path):
        from musicmixer.services import remix_stages, song_cache
        from musicmixer.services.remix_stages import (
            DownloadPairCallbacks,
            download_youtube_pair,
        )
        from musicmixer.services.youtube import YouTubeAudioResult

        monkeypatch.setattr(song_cache, "stem_cache_ready", lambda video_id, role: True)

        downloaded: list[str] = []

        async def _fake_download(url, output_dir, progress_callback, video_id):
            downloaded.append(video_id)
            wav = output_dir / f"{video_id}.wav"
            wav.write_bytes(b"")
            return YouTubeAudioResult(
                wav_path=wav, title="fresh", duration_seconds=100.0,
                source_codec="opus", source_bitrate=128,
            )

        monkeypatch.setattr(
            "musicmixer.services.youtube.download_youtube_audio", _fake_download,
        )
        monkeypatch.setattr(
            remix_stages, "_pre_trim_youtube_download",
            lambda result, max_duration_seconds: None,
        )

        progress_b: list[tuple[float, str]] = []
        callbacks = DownloadPairCallbacks(
            check_cancelled=lambda: None,
            tag_failed_song=lambda exc, slot: None,
            on_download_pair_started=lambda: None,
            on_song_a_download_progress=lambda pct, msg: None,
            on_song_b_download_progress=lambda pct, msg: progress_b.append((pct, msg)),
            on_download_pair_finished=lambda: None,
        )
        cached_b = SimpleNamespace(
            video_id="9bZkp7q19f0", title="Cached B", has_stems=True,
            meta=SimpleNamespace(duration_seconds=210.0, source_quality="youtube-opus-128kbps"),
        )
        settings = SimpleNamespace(
            data_dir=tmp_path / "data", processing_max_duration_seconds=210,
        )

        pair = download_youtube_pair(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=9bZkp7q19f0",
            session_id="s1",
            callbacks=callbacks,
            cached_song_a=None,
            cached_song_b=cached_b,
            settings=settings,
        )

        assert downloaded == ["dQw4w9WgXcQ"]
        assert pair.result_b.title == "Cached B"
        assert pair.result_b.duration_seconds == 210.0
        assert not pair.result_b.wav_path.exists()
        assert pair.source_quality_b == "youtube-opus-128kbps"
        assert progress_b == [(1.0, "Already had this one!")]

    def test_metadata_only_cache_still_downloads(self, monkeypatch, tmp_path):
        from musicmixer.services import remix_stages
        from musicmixer.services.remix_stages import (
            DownloadPairCallbacks,
            download_youtube_pair,
        )
        from musicmixer.services.youtube import YouTubeAudioResult

        downloaded: list[str] = []

        async def _fake_download(url, output_dir, progress_callback, video_id):
            downloaded.append(video_id)
            return YouTubeAudioResult(
                wav_path=output_dir / f"{video_id}.wav", title="fresh",
                duration_seconds=100.0, source_codec="opus", source_bitrate=128,
            )

        monkeypatch.setattr(
            "musicmixer.services.youtube.download_youtube_audio", _fake_download,
        )
        monkeypatch.setattr(
            remix_stages, "_pre_trim_youtube_download",
            lambda result, max_duration_seconds: None,
        )
        callbacks = DownloadPairCallbacks(
            check_cancelled=lambda: None,
            tag_failed_song=lambda exc, slot: None,
            on_download_pair_started=lambda: None,
            on_song_a_download_progress=lambda pct, msg: None,
            on_song_b_download_progress=lambda pct, msg: None,
            on_download_pair_finished=lambda: None,
        )
        cached_b = SimpleNamespace(
            video_id="9bZkp7q19f0", title="Cached B", has_stems=False,
            meta=SimpleNamespace(duration_seconds=210.0),
        )
        settings = SimpleNamespace(
            data_dir=tmp_path / "data", processing_max_duration_seconds=210,
        )

        download_youtube_pair(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=9bZkp7q19f0",
            session_id="s1",
            callbacks=callbacks,
            cached_song_a=None,
            cached_song_b=cached_b,
            settings=settings,
        )

        assert sorted(downloaded) == ["9bZkp7q19f0", "dQw4w9WgXcQ"]
