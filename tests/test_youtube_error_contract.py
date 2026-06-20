"""Tests for the structured-error wire contract on the SSE `error` event.

Producer side of the contract consumed by frontend PR #76:
    error_class: "transient" | "permanent"   (optional)
    failed_song: "A" | "B"                    (optional)

Covers:
- _build_error_event maps a YouTubeDownloadError's error_class + the tagged
  failed_song onto the SSE error event payload.
- _youtube_pipeline_wrapper emits the right failed_song ("A" vs "B") and
  error_class when song A vs song B download fails (downloads mocked).
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from musicmixer.api.remix import (
    _build_error_event,
    _tag_failed_song,
    _youtube_pipeline_wrapper,
)
from musicmixer.models import SessionState
from musicmixer.services.youtube import (
    ERROR_CLASS_PERMANENT,
    ERROR_CLASS_TRANSIENT,
    YouTubeDownloadError,
)


# ===========================================================================
# _build_error_event — payload shape
# ===========================================================================


class TestBuildErrorEvent:
    def test_base_fields_always_present(self) -> None:
        event = _build_error_event(RuntimeError("kaboom"))
        assert event["step"] == "error"
        assert event["detail"] == "kaboom"
        assert event["progress"] == 0

    def test_non_youtube_error_omits_error_class(self) -> None:
        event = _build_error_event(RuntimeError("kaboom"))
        assert "error_class" not in event
        assert "failed_song" not in event

    def test_youtube_transient_error_class_emitted(self) -> None:
        err = YouTubeDownloadError("403", error_class=ERROR_CLASS_TRANSIENT)
        event = _build_error_event(err)
        assert event["error_class"] == ERROR_CLASS_TRANSIENT

    def test_youtube_permanent_error_class_emitted(self) -> None:
        err = YouTubeDownloadError("gone", error_class=ERROR_CLASS_PERMANENT)
        event = _build_error_event(err)
        assert event["error_class"] == ERROR_CLASS_PERMANENT

    def test_failed_song_a_emitted_when_tagged(self) -> None:
        err = YouTubeDownloadError("403", error_class=ERROR_CLASS_TRANSIENT)
        _tag_failed_song(err, "A")
        event = _build_error_event(err)
        assert event["failed_song"] == "A"
        assert event["error_class"] == ERROR_CLASS_TRANSIENT

    def test_failed_song_b_emitted_when_tagged(self) -> None:
        err = YouTubeDownloadError("gone", error_class=ERROR_CLASS_PERMANENT)
        _tag_failed_song(err, "B")
        event = _build_error_event(err)
        assert event["failed_song"] == "B"

    def test_failed_song_omitted_when_not_attributable(self) -> None:
        err = YouTubeDownloadError("403", error_class=ERROR_CLASS_TRANSIENT)
        event = _build_error_event(err)
        assert "failed_song" not in event


# ===========================================================================
# _youtube_pipeline_wrapper — emits error_class + failed_song end-to-end
# ===========================================================================


def _run_wrapper_with_download(download_side_effect):
    """Run _youtube_pipeline_wrapper with download_youtube_audio mocked.

    Returns the SessionState so the caller can inspect session.last_event
    (the SSE error event payload that would be streamed to the client).
    """
    session = SessionState()
    processing_lock = MagicMock()
    app_state = MagicMock()

    with (
        patch("musicmixer.services.youtube.download_youtube_audio", side_effect=download_side_effect),
        patch("musicmixer.api.remix.settings") as mock_settings,
        patch("musicmixer.api.remix._process_next_queued"),
        patch("musicmixer.services.youtube.extract_video_id", return_value=None),
    ):
        # tmp data dir for the per-session upload dir mkdir
        import tempfile
        from pathlib import Path

        mock_settings.data_dir = Path(tempfile.mkdtemp())
        mock_settings.youtube_max_duration_seconds = 600
        mock_settings.processing_max_duration_seconds = 600

        _youtube_pipeline_wrapper(
            session_id="11111111-1111-1111-1111-111111111111",
            url_a="https://youtube.com/watch?v=aaa",
            url_b="https://youtube.com/watch?v=bbb",
            prompt="",
            session=session,
            processing_lock=processing_lock,
            app_state=app_state,
            cached_song_a=None,
            cached_song_b=None,
        )

    return session


class TestWrapperEmitsStructuredError:
    def test_song_a_failure_tags_failed_song_a(self) -> None:
        # Song A (url_a) is the vocal source. A transient 403 on A.
        def _side_effect(*args, **kwargs):
            url = kwargs.get("url", "")
            if "aaa" in url:
                raise YouTubeDownloadError("403 Forbidden", error_class=ERROR_CLASS_TRANSIENT)
            # Song B succeeds (or never completes) — block briefly so A fails first.
            raise YouTubeDownloadError("403 Forbidden", error_class=ERROR_CLASS_TRANSIENT)

        session = _run_wrapper_with_download(_side_effect)

        assert session.status == "error"
        assert session.last_event["step"] == "error"
        assert session.last_event["failed_song"] == "A"
        assert session.last_event["error_class"] == ERROR_CLASS_TRANSIENT

    def test_song_b_failure_tags_failed_song_b(self) -> None:
        # Song A succeeds, Song B (instrumental source) fails permanently.
        good_result = MagicMock()
        good_result.title = "Song A"
        good_result.duration_seconds = 120.0
        good_result.source_codec = "opus"
        good_result.source_bitrate = 128

        def _side_effect(*args, **kwargs):
            url = kwargs.get("url", "")
            if "aaa" in url:
                return good_result
            raise YouTubeDownloadError("Video unavailable", error_class=ERROR_CLASS_PERMANENT)

        session = _run_wrapper_with_download(_side_effect)

        assert session.status == "error"
        assert session.last_event["step"] == "error"
        assert session.last_event["failed_song"] == "B"
        assert session.last_event["error_class"] == ERROR_CLASS_PERMANENT
