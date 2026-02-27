"""Tests for YouTube download service.

Covers:
- URL validation (SSRF prevention)
- Duration validation
- Error mapping (yt-dlp errors -> user-friendly messages)
- Download function (mocked yt-dlp)
- Progress callback throttling
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from musicmixer.services.youtube import (
    YouTubeAudioResult,
    YouTubeDownloadError,
    validate_youtube_url,
    _is_ip_literal,
    _map_ytdlp_error,
    download_youtube_audio,
)


# ===========================================================================
# URL validation tests (SSRF prevention)
# ===========================================================================


class TestValidateYouTubeUrl:
    """SSRF prevention: validate URLs before passing to yt-dlp."""

    # --- Valid URLs ---

    def test_standard_youtube_url(self) -> None:
        """Standard youtube.com watch URL passes validation."""
        validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_short_youtube_url(self) -> None:
        """Short youtu.be URL passes validation."""
        validate_youtube_url("https://youtu.be/dQw4w9WgXcQ")

    def test_mobile_youtube_url(self) -> None:
        """Mobile m.youtube.com URL passes validation."""
        validate_youtube_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_music_youtube_url(self) -> None:
        """YouTube Music URL passes validation."""
        validate_youtube_url("https://music.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_bare_youtube_url(self) -> None:
        """youtube.com (no www) passes validation."""
        validate_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")

    def test_youtube_url_with_extra_params(self) -> None:
        """URL with extra query params passes validation."""
        validate_youtube_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42&list=ignored"
        )

    def test_youtube_shorts_url(self) -> None:
        """YouTube Shorts URL passes validation."""
        validate_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")

    # --- Scheme validation ---

    def test_reject_http_scheme(self) -> None:
        """HTTP (non-HTTPS) scheme is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("http://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_reject_ftp_scheme(self) -> None:
        """FTP scheme is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("ftp://youtube.com/watch?v=dQw4w9WgXcQ")

    def test_reject_javascript_scheme(self) -> None:
        """javascript: scheme is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("javascript:alert(1)")

    def test_reject_file_scheme(self) -> None:
        """file:// scheme is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("file:///etc/passwd")

    def test_reject_data_scheme(self) -> None:
        """data: scheme is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("data:text/html,<h1>hi</h1>")

    # --- Userinfo bypass (@) ---

    def test_reject_at_sign_in_netloc(self) -> None:
        """@ in netloc is rejected (userinfo bypass attack)."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("https://youtube.com@evil.com/watch?v=abc")

    def test_reject_at_sign_with_credentials(self) -> None:
        """user:pass@ in netloc is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url(
                "https://user:pass@youtube.com/watch?v=abc"
            )

    # --- IP literal rejection ---

    def test_reject_ipv4_literal(self) -> None:
        """IPv4 address is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("https://192.168.1.1/watch?v=abc")

    def test_reject_localhost_ip(self) -> None:
        """127.0.0.1 is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("https://127.0.0.1/watch?v=abc")

    def test_reject_ipv6_literal(self) -> None:
        """IPv6 literal is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("https://[::1]/watch?v=abc")

    # --- Non-standard port ---

    def test_reject_non_standard_port(self) -> None:
        """Non-443 port is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url(
                "https://www.youtube.com:8080/watch?v=abc"
            )

    def test_allow_port_443(self) -> None:
        """Port 443 (default HTTPS) is allowed."""
        validate_youtube_url("https://www.youtube.com:443/watch?v=dQw4w9WgXcQ")

    # --- Hostname allowlist ---

    def test_reject_non_youtube_host(self) -> None:
        """Non-YouTube hostname is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("https://evil.com/watch?v=abc")

    def test_reject_youtube_subdomain_attack(self) -> None:
        """Subdomain impersonation is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url(
                "https://youtube.com.evil.com/watch?v=abc"
            )

    def test_reject_similar_domain(self) -> None:
        """Similar-looking domain is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("https://y0utube.com/watch?v=abc")

    def test_reject_localhost(self) -> None:
        """localhost is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("https://localhost/watch?v=abc")

    def test_reject_internal_network(self) -> None:
        """Internal network hostname is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("https://internal.corp.net/api")

    def test_reject_empty_url(self) -> None:
        """Empty string is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("")

    def test_reject_no_scheme(self) -> None:
        """URL without scheme is rejected."""
        with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
            validate_youtube_url("www.youtube.com/watch?v=abc")


# ===========================================================================
# IP literal detection
# ===========================================================================


class TestIsIpLiteral:
    def test_ipv4(self) -> None:
        assert _is_ip_literal("192.168.1.1") is True

    def test_ipv4_localhost(self) -> None:
        assert _is_ip_literal("127.0.0.1") is True

    def test_ipv6_brackets(self) -> None:
        assert _is_ip_literal("[::1]") is True

    def test_ipv6_colons(self) -> None:
        assert _is_ip_literal("::1") is True

    def test_hostname(self) -> None:
        assert _is_ip_literal("youtube.com") is False

    def test_www_hostname(self) -> None:
        assert _is_ip_literal("www.youtube.com") is False


# ===========================================================================
# Error mapping
# ===========================================================================


class TestMapYtdlpError:
    def test_video_unavailable(self) -> None:
        err = Exception("ERROR: Video unavailable")
        assert "unavailable or has been removed" in _map_ytdlp_error(err)

    def test_video_not_available(self) -> None:
        err = Exception("This video is not available")
        assert "unavailable or has been removed" in _map_ytdlp_error(err)

    def test_age_restricted(self) -> None:
        err = Exception("Sign in to confirm your age. This video may be restricted.")
        assert "Age-restricted" in _map_ytdlp_error(err)

    def test_age_gate(self) -> None:
        err = Exception("ERROR: age gate detected")
        assert "Age-restricted" in _map_ytdlp_error(err)

    def test_geo_restricted(self) -> None:
        err = Exception("Video is geo-restricted in your country")
        assert "not available in the server's region" in _map_ytdlp_error(err)

    def test_region_blocked(self) -> None:
        err = Exception("This content is blocked in your region")
        assert "not available in the server's region" in _map_ytdlp_error(err)

    def test_live_stream(self) -> None:
        err = Exception("This live stream recording is not available")
        assert "Live streams" in _map_ytdlp_error(err)

    def test_live_event(self) -> None:
        err = Exception("This live event has not started")
        assert "Live streams" in _map_ytdlp_error(err)

    def test_no_audio_stream(self) -> None:
        err = Exception("No audio streams found")
        assert "No audio track" in _map_ytdlp_error(err)

    def test_requested_format_not_available(self) -> None:
        err = Exception("Requested format not available")
        assert "No audio track" in _map_ytdlp_error(err)

    def test_rate_limited_429(self) -> None:
        err = Exception("HTTP Error 429: Too Many Requests")
        assert "temporarily limiting" in _map_ytdlp_error(err)

    def test_rate_limited_too_many(self) -> None:
        err = Exception("Too many requests")
        assert "temporarily limiting" in _map_ytdlp_error(err)

    def test_network_error(self) -> None:
        err = Exception("urlopen error [Errno 8] nodename")
        assert "Failed to download" in _map_ytdlp_error(err)

    def test_connection_error(self) -> None:
        err = Exception("Connection timed out")
        assert "Failed to download" in _map_ytdlp_error(err)

    def test_unknown_error_fallback(self) -> None:
        err = Exception("some completely novel error xyzzy")
        assert "Failed to download" in _map_ytdlp_error(err)


# ===========================================================================
# Duration validation
# ===========================================================================


class TestDurationValidation:
    @pytest.mark.asyncio
    async def test_reject_video_over_max_duration(self, tmp_path: Path) -> None:
        """Videos exceeding max duration are rejected before download."""
        mock_info = {
            "duration": 1200,  # 20 minutes > 15 min default
            "title": "Long Video",
            "acodec": "opus",
            "abr": 128,
            "is_live": False,
        }

        with patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = mock_info

            with pytest.raises(YouTubeDownloadError, match="under 15 minutes"):
                await download_youtube_audio(
                    "https://www.youtube.com/watch?v=abc123",
                    tmp_path,
                )

    @pytest.mark.asyncio
    async def test_accept_video_under_max_duration(self, tmp_path: Path) -> None:
        """Videos under max duration proceed to download."""
        mock_info = {
            "duration": 240,  # 4 minutes -- under limit
            "title": "Short Video",
            "acodec": "opus",
            "abr": 128,
            "is_live": False,
        }

        with patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = mock_info
            mock_ydl.download.return_value = None

            with patch("musicmixer.services.youtube.uuid") as mock_uuid:
                mock_uuid.uuid4.return_value.hex = "abc123def456"
                wav_file = tmp_path / "abc123def456.wav"
                wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

                result = await download_youtube_audio(
                    "https://www.youtube.com/watch?v=abc123",
                    tmp_path,
                )

            assert result.duration_seconds == 240
            assert result.title == "Short Video"
            assert result.source_codec == "opus"
            assert result.source_bitrate == 128


# ===========================================================================
# Live stream rejection
# ===========================================================================


class TestLiveStreamRejection:
    @pytest.mark.asyncio
    async def test_reject_live_stream(self, tmp_path: Path) -> None:
        """Live streams are rejected before download."""
        mock_info = {
            "duration": 0,
            "title": "Live Now!",
            "acodec": "opus",
            "abr": 128,
            "is_live": True,
        }

        with patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = mock_info

            with pytest.raises(YouTubeDownloadError, match="Live streams"):
                await download_youtube_audio(
                    "https://www.youtube.com/watch?v=live123",
                    tmp_path,
                )


# ===========================================================================
# Download function -- mocked yt-dlp
# ===========================================================================


class TestDownloadYouTubeAudio:
    @pytest.mark.asyncio
    async def test_successful_download(self, tmp_path: Path) -> None:
        """Full successful download flow with metadata extraction."""
        mock_info = {
            "duration": 180,
            "title": "Test Song - Artist",
            "acodec": "opus",
            "abr": 128,
            "is_live": False,
        }

        with patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = mock_info
            mock_ydl.download.return_value = None

            with patch("musicmixer.services.youtube.uuid") as mock_uuid:
                mock_uuid.uuid4.return_value.hex = "deadbeef1234"
                # Pre-create the WAV file that yt-dlp would create
                wav_path = tmp_path / "deadbeef1234.wav"
                wav_path.write_bytes(b"RIFF" + b"\x00" * 100)

                result = await download_youtube_audio(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    tmp_path,
                )

        assert isinstance(result, YouTubeAudioResult)
        assert result.wav_path == wav_path
        assert result.title == "Test Song - Artist"
        assert result.duration_seconds == 180.0
        assert result.source_codec == "opus"
        assert result.source_bitrate == 128

    @pytest.mark.asyncio
    async def test_ssrf_rejected_before_ytdlp(self, tmp_path: Path) -> None:
        """SSRF validation happens before yt-dlp is invoked."""
        with patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls:
            with pytest.raises(YouTubeDownloadError, match="only YouTube links"):
                await download_youtube_audio(
                    "https://evil.com/malicious",
                    tmp_path,
                )

            # yt-dlp should never have been instantiated
            mock_ydl_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_unavailable_video_returns_none_info(self, tmp_path: Path) -> None:
        """When extract_info returns None, raise appropriate error."""
        with patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = None

            with pytest.raises(YouTubeDownloadError, match="unavailable"):
                await download_youtube_audio(
                    "https://www.youtube.com/watch?v=deleted",
                    tmp_path,
                )

    @pytest.mark.asyncio
    async def test_ytdlp_download_error_mapped(self, tmp_path: Path) -> None:
        """yt-dlp DownloadError is mapped to user-friendly message."""
        import yt_dlp

        mock_info = {
            "duration": 180,
            "title": "Test",
            "acodec": "opus",
            "abr": 128,
            "is_live": False,
        }

        with patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = mock_info
            mock_ydl.download.side_effect = yt_dlp.utils.DownloadError(
                "ERROR: Video unavailable"
            )

            with pytest.raises(
                YouTubeDownloadError, match="unavailable or has been removed"
            ):
                await download_youtube_audio(
                    "https://www.youtube.com/watch?v=gone",
                    tmp_path,
                )

    @pytest.mark.asyncio
    async def test_missing_wav_after_download(self, tmp_path: Path) -> None:
        """When WAV file doesn't exist after download, raise error."""
        mock_info = {
            "duration": 180,
            "title": "Test",
            "acodec": "opus",
            "abr": 128,
            "is_live": False,
        }

        with patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = mock_info
            mock_ydl.download.return_value = None  # No file created

            with patch("musicmixer.services.youtube.uuid") as mock_uuid:
                mock_uuid.uuid4.return_value.hex = "nofile123"

                with pytest.raises(YouTubeDownloadError, match="Failed to download"):
                    await download_youtube_audio(
                        "https://www.youtube.com/watch?v=abc",
                        tmp_path,
                    )

    @pytest.mark.asyncio
    async def test_aac_codec_detected(self, tmp_path: Path) -> None:
        """AAC codec metadata is correctly extracted."""
        mock_info = {
            "duration": 200,
            "title": "AAC Song",
            "acodec": "aac",
            "abr": 128,
            "is_live": False,
        }

        with patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = mock_info
            mock_ydl.download.return_value = None

            with patch("musicmixer.services.youtube.uuid") as mock_uuid:
                mock_uuid.uuid4.return_value.hex = "aacfile123"
                wav_path = tmp_path / "aacfile123.wav"
                wav_path.write_bytes(b"RIFF" + b"\x00" * 100)

                result = await download_youtube_audio(
                    "https://www.youtube.com/watch?v=aac",
                    tmp_path,
                )

        assert result.source_codec == "aac"
        assert result.source_bitrate == 128

    @pytest.mark.asyncio
    async def test_noplaylist_option_set(self, tmp_path: Path) -> None:
        """Verify noplaylist: True is passed in yt-dlp options."""
        mock_info = {
            "duration": 180,
            "title": "Test",
            "acodec": "opus",
            "abr": 128,
            "is_live": False,
        }

        with patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = mock_info
            mock_ydl.download.return_value = None

            with patch("musicmixer.services.youtube.uuid") as mock_uuid:
                mock_uuid.uuid4.return_value.hex = "playlisttest"
                wav_path = tmp_path / "playlisttest.wav"
                wav_path.write_bytes(b"RIFF" + b"\x00" * 100)

                await download_youtube_audio(
                    "https://www.youtube.com/watch?v=test",
                    tmp_path,
                )

            # Check that noplaylist was set in the options
            call_args = mock_ydl_cls.call_args
            opts = call_args[0][0] if call_args[0] else call_args[1]
            assert opts.get("noplaylist") is True


# ===========================================================================
# Progress callback
# ===========================================================================


class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_progress_callback_invoked(self, tmp_path: Path) -> None:
        """Progress callback receives updates during download."""
        mock_info = {
            "duration": 180,
            "title": "Progress Test",
            "acodec": "opus",
            "abr": 128,
            "is_live": False,
        }

        callback_calls: list[tuple[float, str]] = []

        def track_progress(fraction: float, message: str) -> None:
            callback_calls.append((fraction, message))

        with patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = mock_info

            # Capture the progress_hooks to simulate callbacks
            captured_opts = {}

            def capture_opts(opts):
                captured_opts.update(opts)
                return MagicMock(
                    __enter__=MagicMock(return_value=mock_ydl),
                    __exit__=MagicMock(return_value=False),
                )

            mock_ydl_cls.side_effect = capture_opts

            def simulate_download(urls: list[str]) -> None:
                # Simulate progress hooks being called
                hooks = captured_opts.get("progress_hooks", [])
                for hook in hooks:
                    hook({
                        "status": "downloading",
                        "total_bytes": 1000,
                        "downloaded_bytes": 500,
                    })
                    hook({
                        "status": "finished",
                    })

                # Simulate postprocessor hooks
                pp_hooks = captured_opts.get("postprocessor_hooks", [])
                for hook in pp_hooks:
                    hook({"status": "started"})
                    hook({
                        "status": "finished",
                        "info_dict": {"filepath": str(tmp_path / "abc.wav")},
                    })

            mock_ydl.download.side_effect = simulate_download

            with patch("musicmixer.services.youtube.uuid") as mock_uuid:
                mock_uuid.uuid4.return_value.hex = "progtest123"
                wav_path = tmp_path / "progtest123.wav"
                wav_path.write_bytes(b"RIFF" + b"\x00" * 100)

                # Also create the path the postprocessor hook reports
                (tmp_path / "abc.wav").write_bytes(b"RIFF" + b"\x00" * 100)

                await download_youtube_audio(
                    "https://www.youtube.com/watch?v=prog",
                    tmp_path,
                    progress_callback=track_progress,
                )

        # Should have received progress updates
        assert len(callback_calls) > 0

        # Should contain download phase
        download_msgs = [m for _, m in callback_calls if "Downloading" in m or "Download" in m]
        assert len(download_msgs) > 0


# ===========================================================================
# Config integration
# ===========================================================================


class TestConfigIntegration:
    def test_youtube_config_defaults(self) -> None:
        """Config has YouTube settings with correct defaults."""
        from musicmixer.config import Settings

        s = Settings()
        assert s.youtube_enabled is True
        assert s.youtube_max_duration_seconds == 900

    @pytest.mark.asyncio
    async def test_custom_duration_limit(self, tmp_path: Path) -> None:
        """Custom youtube_max_duration_seconds is respected."""
        mock_info = {
            "duration": 600,  # 10 minutes
            "title": "Medium Video",
            "acodec": "opus",
            "abr": 128,
            "is_live": False,
        }

        with (
            patch("musicmixer.services.youtube.yt_dlp.YoutubeDL") as mock_ydl_cls,
            patch("musicmixer.services.youtube.settings") as mock_settings,
        ):
            mock_settings.youtube_max_duration_seconds = 300  # 5 minutes
            mock_ydl = MagicMock()
            mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = mock_info

            with pytest.raises(YouTubeDownloadError, match="under 5 minutes"):
                await download_youtube_audio(
                    "https://www.youtube.com/watch?v=medium",
                    tmp_path,
                )


# ===========================================================================
# YouTubeAudioResult dataclass
# ===========================================================================


class TestYouTubeAudioResult:
    def test_fields(self) -> None:
        """YouTubeAudioResult has all required fields."""
        result = YouTubeAudioResult(
            wav_path=Path("/tmp/test.wav"),
            title="Test Song",
            duration_seconds=180.0,
            source_codec="opus",
            source_bitrate=128,
        )
        assert result.wav_path == Path("/tmp/test.wav")
        assert result.title == "Test Song"
        assert result.duration_seconds == 180.0
        assert result.source_codec == "opus"
        assert result.source_bitrate == 128
