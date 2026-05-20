"""Tests for the /api/stats endpoint."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from musicmixer.main import app


def _make_completion_entry(
    *,
    timestamp: str = "2026-05-20T12:00:00+00:00",
    session_id: str = "test-session-1",
    total_pipeline_time_s: float = 120.5,
    song_a_title: str = "Song A",
    song_b_title: str = "Song B",
    song_a_video_id: str = "vid_a",
    song_b_video_id: str = "vid_b",
    song_a_cache_hit: bool = False,
    song_b_cache_hit: bool = False,
    red_flags: list | None = None,
) -> dict:
    """Build a pipeline summary log entry."""
    return {
        "timestamp": timestamp,
        "level": "INFO",
        "message": "Pipeline summary",
        "session_id": session_id,
        "pipeline_step": "summary",
        "total_pipeline_time_s": total_pipeline_time_s,
        "song_a_title": song_a_title,
        "song_b_title": song_b_title,
        "song_a_video_id": song_a_video_id,
        "song_b_video_id": song_b_video_id,
        "song_a_cache_hit": song_a_cache_hit,
        "song_b_cache_hit": song_b_cache_hit,
        "red_flags": red_flags or [],
        "red_flags_count": len(red_flags or []),
    }


def _make_red_flag_entry(
    *,
    timestamp: str = "2026-05-20T12:00:00+00:00",
    session_id: str = "test-session-1",
    flag: str = "high_stretch_30%",
) -> dict:
    """Build a red-flag warning log entry."""
    return {
        "timestamp": timestamp,
        "level": "WARNING",
        "message": f"Red flag: {flag}",
        "session_id": session_id,
        "red_flag": flag,
    }


def _write_log(log_path: Path, entries: list[dict]) -> None:
    """Write entries as JSONL."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


@pytest.fixture
def client(tmp_path):
    """Create test client with temp data directory."""
    with patch("musicmixer.config.settings") as mock_settings:
        mock_settings.data_dir = tmp_path
        mock_settings.allowed_extensions = {".mp3", ".wav"}
        mock_settings.max_file_size_mb = 50
        mock_settings.cors_origins = ["http://localhost:5173"]
        mock_settings.max_concurrent_mixes = 1
        mock_settings.max_queue_depth = 10
        mock_settings.session_ttl_hours = 3
        mock_settings.queue_entry_ttl_minutes = 15
        mock_settings.max_upload_duration_seconds = 900
        mock_settings.distributed_limiter_enabled = False
        mock_settings.log_level = "INFO"

        for subdir in ("uploads", "stems", "remixes", "logs"):
            (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

        with patch("musicmixer.api.remix.settings", mock_settings), \
             patch("musicmixer.main.settings", mock_settings), \
             patch("musicmixer.api.stats._LOG_FILE", tmp_path / "logs" / "musicmixer.log"), \
             patch("musicmixer.api.remix.cleanup_expired_sessions"):
            with TestClient(app) as c:
                yield c


class TestStatsEndpointEmpty:
    """Tests for /api/stats with no log data."""

    def test_missing_log_file(self, client):
        """Should return zeroed stats when the log file does not exist."""
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["remixes"]["total"] == 0
        assert data["average_pipeline_time_s"] is None
        assert data["most_used_songs"] == []
        assert data["red_flags"] == []
        assert data["cache_hit_rate"]["total_remixes"] == 0

    def test_empty_log_file(self, client, tmp_path):
        """Should return zeroed stats when the log file is empty."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        log_path.write_text("")

        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["remixes"]["total"] == 0


class TestStatsEndpointBasic:
    """Tests for /api/stats with populated log data."""

    def test_single_completion(self, client, tmp_path):
        """Single completed remix should produce correct counts."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            _make_completion_entry(
                timestamp="2026-05-20T14:00:00+00:00",
                total_pipeline_time_s=95.3,
                song_a_title="Hypnotize",
                song_a_video_id="abc123",
                song_b_title="Althea",
                song_b_video_id="def456",
            ),
        ]
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()

        assert data["remixes"]["total"] == 1
        assert data["remixes"]["daily"]["2026-05-20"] == 1
        assert data["average_pipeline_time_s"] == 95.3
        assert len(data["most_used_songs"]) == 2  # song_a + song_b

    def test_multiple_completions_average(self, client, tmp_path):
        """Average pipeline time across multiple completions."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            _make_completion_entry(
                timestamp="2026-05-20T10:00:00+00:00",
                session_id="s1",
                total_pipeline_time_s=100.0,
            ),
            _make_completion_entry(
                timestamp="2026-05-20T11:00:00+00:00",
                session_id="s2",
                total_pipeline_time_s=200.0,
            ),
        ]
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        data = resp.json()

        assert data["remixes"]["total"] == 2
        assert data["average_pipeline_time_s"] == 150.0

    def test_daily_and_weekly_grouping(self, client, tmp_path):
        """Remixes on different days should group correctly."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            _make_completion_entry(
                timestamp="2026-05-18T10:00:00+00:00",
                session_id="s1",
            ),
            _make_completion_entry(
                timestamp="2026-05-19T10:00:00+00:00",
                session_id="s2",
            ),
            _make_completion_entry(
                timestamp="2026-05-19T14:00:00+00:00",
                session_id="s3",
            ),
        ]
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        data = resp.json()

        assert data["remixes"]["total"] == 3
        assert data["remixes"]["daily"]["2026-05-18"] == 1
        assert data["remixes"]["daily"]["2026-05-19"] == 2


class TestMostUsedSongs:
    """Tests for the most-used songs aggregation."""

    def test_songs_ranked_by_frequency(self, client, tmp_path):
        """Songs used more often should rank higher."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            _make_completion_entry(
                timestamp="2026-05-20T10:00:00+00:00",
                session_id="s1",
                song_a_title="Popular Song",
                song_a_video_id="pop1",
                song_b_title="Rare Song",
                song_b_video_id="rare1",
            ),
            _make_completion_entry(
                timestamp="2026-05-20T11:00:00+00:00",
                session_id="s2",
                song_a_title="Popular Song",
                song_a_video_id="pop1",
                song_b_title="Another Song",
                song_b_video_id="another1",
            ),
        ]
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        data = resp.json()

        songs = data["most_used_songs"]
        assert songs[0]["title"] == "Popular Song"
        assert songs[0]["count"] == 2

    def test_top_10_limit(self, client, tmp_path):
        """Should return at most 10 songs."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = []
        for i in range(8):
            entries.append(
                _make_completion_entry(
                    timestamp=f"2026-05-20T{10 + i}:00:00+00:00",
                    session_id=f"s{i}",
                    song_a_title=f"Song A{i}",
                    song_a_video_id=f"va{i}",
                    song_b_title=f"Song B{i}",
                    song_b_video_id=f"vb{i}",
                )
            )
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        data = resp.json()

        # 8 completions * 2 songs each = 16 unique songs, capped at 10
        assert len(data["most_used_songs"]) == 10


class TestRedFlagFrequency:
    """Tests for red flag frequency aggregation."""

    def test_flags_from_completion_summaries(self, client, tmp_path):
        """Red flags in completion summaries should be counted."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            _make_completion_entry(
                timestamp="2026-05-20T10:00:00+00:00",
                session_id="s1",
                red_flags=["high_stretch_30%", "ghost_stem"],
            ),
            _make_completion_entry(
                timestamp="2026-05-20T11:00:00+00:00",
                session_id="s2",
                red_flags=["high_stretch_30%"],
            ),
        ]
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        data = resp.json()

        flags = {f["flag"]: f["count"] for f in data["red_flags"]}
        assert flags["high_stretch_30%"] == 2
        assert flags["ghost_stem"] == 1

    def test_individual_warning_entries(self, client, tmp_path):
        """Individual red_flag warning entries should be counted."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            _make_red_flag_entry(flag="borderline_stem"),
            _make_red_flag_entry(flag="borderline_stem"),
            _make_red_flag_entry(flag="high_key_shift"),
        ]
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        data = resp.json()

        flags = {f["flag"]: f["count"] for f in data["red_flags"]}
        assert flags["borderline_stem"] == 2
        assert flags["high_key_shift"] == 1

    def test_no_double_counting(self, client, tmp_path):
        """When a summary entry has red_flags list, individual red_flag
        field in the same entry should not be double-counted."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        # A completion entry has red_flags list; it should NOT also count red_flag field
        entry = _make_completion_entry(
            timestamp="2026-05-20T10:00:00+00:00",
            session_id="s1",
            red_flags=["ghost_stem"],
        )
        # Even if red_flag is set on the same entry, it should be skipped
        entry["red_flag"] = "ghost_stem"
        _write_log(log_path, [entry])

        resp = client.get("/api/stats")
        data = resp.json()

        flags = {f["flag"]: f["count"] for f in data["red_flags"]}
        assert flags["ghost_stem"] == 1


class TestCacheHitRate:
    """Tests for cache hit rate calculation."""

    def test_no_cache_hits(self, client, tmp_path):
        """Zero cache hits should produce 0% rate."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            _make_completion_entry(
                session_id="s1",
                song_a_cache_hit=False,
                song_b_cache_hit=False,
            ),
        ]
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        data = resp.json()

        rate = data["cache_hit_rate"]
        assert rate["total_remixes"] == 1
        assert rate["song_a_cache_hits"] == 0
        assert rate["song_b_cache_hits"] == 0
        assert rate["overall_cache_hit_pct"] == 0.0

    def test_all_cache_hits(self, client, tmp_path):
        """All cache hits should produce 100% rate."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            _make_completion_entry(
                session_id="s1",
                song_a_cache_hit=True,
                song_b_cache_hit=True,
            ),
        ]
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        data = resp.json()

        rate = data["cache_hit_rate"]
        assert rate["overall_cache_hit_pct"] == 100.0

    def test_partial_cache_hits(self, client, tmp_path):
        """Partial hits should produce correct percentage."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            _make_completion_entry(
                session_id="s1",
                song_a_cache_hit=True,
                song_b_cache_hit=False,
            ),
            _make_completion_entry(
                session_id="s2",
                song_a_cache_hit=False,
                song_b_cache_hit=False,
            ),
        ]
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        data = resp.json()

        rate = data["cache_hit_rate"]
        assert rate["total_remixes"] == 2
        assert rate["song_a_cache_hits"] == 1
        assert rate["song_b_cache_hits"] == 0
        # 1 hit out of 4 slots = 25%
        assert rate["overall_cache_hit_pct"] == 25.0


class TestTimeRangeFilter:
    """Tests for the ?days= query parameter."""

    def test_days_filter(self, client, tmp_path):
        """Only entries within the time range should be included."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            _make_completion_entry(
                timestamp="2026-05-10T10:00:00+00:00",
                session_id="old",
                total_pipeline_time_s=50.0,
            ),
            _make_completion_entry(
                timestamp="2026-05-20T10:00:00+00:00",
                session_id="recent",
                total_pipeline_time_s=100.0,
            ),
        ]
        _write_log(log_path, entries)

        # With days=3, only the May 20 entry should be included
        resp = client.get("/api/stats?days=3")
        data = resp.json()

        assert data["remixes"]["total"] == 1
        assert data["average_pipeline_time_s"] == 100.0
        assert data["filter"]["days"] == 3

    def test_no_filter_returns_all(self, client, tmp_path):
        """Without days param, all entries should be included."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            _make_completion_entry(
                timestamp="2026-01-01T10:00:00+00:00",
                session_id="old",
            ),
            _make_completion_entry(
                timestamp="2026-05-20T10:00:00+00:00",
                session_id="recent",
            ),
        ]
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        data = resp.json()

        assert data["remixes"]["total"] == 2
        assert data["filter"]["days"] is None

    def test_invalid_days_param(self, client):
        """Invalid days param should return 422."""
        resp = client.get("/api/stats?days=0")
        assert resp.status_code == 422

        resp = client.get("/api/stats?days=-5")
        assert resp.status_code == 422

        resp = client.get("/api/stats?days=abc")
        assert resp.status_code == 422


class TestMalformedLogData:
    """Tests for graceful handling of bad log data."""

    def test_malformed_json_lines_skipped(self, client, tmp_path):
        """Lines that are not valid JSON should be silently skipped."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        content = (
            "this is not json\n"
            + json.dumps(_make_completion_entry(session_id="valid")) + "\n"
            + "{ broken json\n"
        )
        log_path.write_text(content)

        resp = client.get("/api/stats")
        data = resp.json()

        assert data["remixes"]["total"] == 1

    def test_entries_without_pipeline_step(self, client, tmp_path):
        """Regular log entries (not pipeline steps) should not count as completions."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        entries = [
            {"timestamp": "2026-05-20T10:00:00+00:00", "level": "INFO", "message": "Server started"},
            _make_completion_entry(session_id="real"),
        ]
        _write_log(log_path, entries)

        resp = client.get("/api/stats")
        data = resp.json()

        assert data["remixes"]["total"] == 1

    def test_blank_lines_skipped(self, client, tmp_path):
        """Blank lines in the log file should not cause errors."""
        log_path = tmp_path / "logs" / "musicmixer.log"
        content = (
            "\n"
            + json.dumps(_make_completion_entry(session_id="s1")) + "\n"
            + "\n\n"
            + json.dumps(_make_completion_entry(session_id="s2")) + "\n"
        )
        log_path.write_text(content)

        resp = client.get("/api/stats")
        data = resp.json()

        assert data["remixes"]["total"] == 2
