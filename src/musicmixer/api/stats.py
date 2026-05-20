"""Stats API endpoint — aggregate metrics from the JSONL log file."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Query

from musicmixer.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

_LOG_FILE = settings.data_dir / "logs" / "musicmixer.log"


def _parse_log_entries(log_path: Path, since: datetime | None = None) -> list[dict]:
    """Read and parse the JSONL log file, skipping malformed lines.

    If *since* is provided, only entries at or after that timestamp are
    included.
    """
    if not log_path.exists():
        return []

    entries: list[dict] = []
    try:
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                if since is not None:
                    ts_raw = entry.get("timestamp")
                    if ts_raw:
                        try:
                            ts = datetime.fromisoformat(ts_raw)
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            if ts < since:
                                continue
                        except (ValueError, TypeError):
                            pass  # keep entries with un-parseable timestamps

                entries.append(entry)
    except OSError:
        logger.warning("Could not read log file: %s", log_path)

    return entries


def _is_completion(entry: dict) -> bool:
    """Return True if this log entry represents a completed pipeline run."""
    return (
        entry.get("pipeline_step") == "summary"
        and "total_pipeline_time_s" in entry
    )


def _completion_date(entry: dict) -> str | None:
    """Extract the date string (YYYY-MM-DD) from a completion entry."""
    ts_raw = entry.get("timestamp")
    if not ts_raw:
        return None
    try:
        return datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _remixes_per_period(completions: list[dict]) -> dict:
    """Count remixes grouped by day and by ISO week."""
    daily: Counter[str] = Counter()
    weekly: Counter[str] = Counter()

    for entry in completions:
        date_str = _completion_date(entry)
        if date_str is None:
            continue
        daily[date_str] += 1
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            iso_year, iso_week, _ = dt.isocalendar()
            weekly[f"{iso_year}-W{iso_week:02d}"] += 1
        except ValueError:
            pass

    return {
        "daily": dict(sorted(daily.items())),
        "weekly": dict(sorted(weekly.items())),
        "total": sum(daily.values()),
    }


def _average_pipeline_time(completions: list[dict]) -> float | None:
    """Return the mean total_pipeline_time_s across completions."""
    times = [
        e["total_pipeline_time_s"]
        for e in completions
        if isinstance(e.get("total_pipeline_time_s"), (int, float))
    ]
    if not times:
        return None
    return round(sum(times) / len(times), 1)


def _most_used_songs(completions: list[dict], limit: int = 10) -> list[dict]:
    """Return the top songs by frequency (video ID + title)."""
    counter: Counter[tuple[str, str]] = Counter()

    for entry in completions:
        for suffix in ("a", "b"):
            vid = entry.get(f"song_{suffix}_video_id", "")
            title = entry.get(f"song_{suffix}_title", "")
            if vid or title:
                counter[(vid, title)] += 1

    return [
        {"video_id": vid, "title": title, "count": count}
        for (vid, title), count in counter.most_common(limit)
    ]


def _red_flag_frequency(entries: list[dict]) -> list[dict]:
    """Count red flag tags across all log entries."""
    counter: Counter[str] = Counter()

    for entry in entries:
        # Completion summaries have a red_flags list
        flags = entry.get("red_flags")
        if isinstance(flags, list):
            for flag in flags:
                if isinstance(flag, str):
                    counter[flag] += 1

        # Individual red-flag warning entries have a single red_flag field
        single_flag = entry.get("red_flag")
        if isinstance(single_flag, str) and not isinstance(flags, list):
            counter[single_flag] += 1

    return [
        {"flag": flag, "count": count}
        for flag, count in counter.most_common()
    ]


def _cache_hit_rate(completions: list[dict]) -> dict:
    """Compute cache hit rate from completion entries.

    A remix is considered a cache hit if either song_a_cache_hit or
    song_b_cache_hit is True.  Returns per-song and overall rates.
    """
    total = 0
    a_hits = 0
    b_hits = 0

    for entry in completions:
        if "song_a_cache_hit" not in entry and "song_b_cache_hit" not in entry:
            continue
        total += 1
        if entry.get("song_a_cache_hit"):
            a_hits += 1
        if entry.get("song_b_cache_hit"):
            b_hits += 1

    if total == 0:
        return {
            "total_remixes": 0,
            "song_a_cache_hits": 0,
            "song_b_cache_hits": 0,
            "overall_cache_hit_pct": 0.0,
        }

    # "overall" = percentage of individual song slots that were cached
    overall_pct = round((a_hits + b_hits) / (total * 2) * 100, 1)

    return {
        "total_remixes": total,
        "song_a_cache_hits": a_hits,
        "song_b_cache_hits": b_hits,
        "overall_cache_hit_pct": overall_pct,
    }


@router.get("/stats")
async def get_stats(
    days: int | None = Query(
        default=None,
        ge=1,
        le=365,
        description="Limit results to the last N days. Omit for all time.",
    ),
) -> dict:
    """Return aggregate pipeline metrics from the structured log file."""
    since: datetime | None = None
    if days is not None:
        since = datetime.now(timezone.utc) - timedelta(days=days)

    entries = _parse_log_entries(_LOG_FILE, since=since)
    completions = [e for e in entries if _is_completion(e)]

    return {
        "remixes": _remixes_per_period(completions),
        "average_pipeline_time_s": _average_pipeline_time(completions),
        "most_used_songs": _most_used_songs(completions),
        "red_flags": _red_flag_frequency(entries),
        "cache_hit_rate": _cache_hit_rate(completions),
        "filter": {"days": days},
    }
