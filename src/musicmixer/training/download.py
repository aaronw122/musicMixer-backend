"""Download mashups from a curated manifest.

Reads mashup_manifest.json and downloads each entry as a WAV file,
reusing the existing youtube.py download_youtube_audio() function.
Skips already-downloaded files, logs failures, and continues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from musicmixer.services.youtube import (
    YouTubeAudioResult,
    YouTubeDownloadError,
    download_youtube_audio,
)

logger = logging.getLogger(__name__)

# Minimum delay between downloads (seconds) to avoid rate limiting.
_RATE_LIMIT_DELAY: float = 2.0


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load and validate the mashup manifest JSON.

    Args:
        manifest_path: Path to mashup_manifest.json.

    Returns:
        List of manifest entry dicts, each with at least 'id' and 'url'.

    Raises:
        FileNotFoundError: If manifest_path does not exist.
        ValueError: If JSON is invalid or entries are missing required fields.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON array")

    required_fields = {"id", "url"}
    for i, entry in enumerate(data):
        missing = required_fields - set(entry.keys())
        if missing:
            raise ValueError(
                f"Manifest entry {i} missing required fields: {missing}"
            )

    return data


def _download_one_sync(
    url: str,
    output_dir: Path,
) -> YouTubeAudioResult:
    """Synchronous wrapper around the async download_youtube_audio().

    Creates a new event loop to run the async function, since the training
    pipeline is invoked synchronously from CLI scripts.
    """
    return asyncio.run(download_youtube_audio(url, output_dir))


def download_mashups(
    manifest_path: Path,
    output_dir: Path,
    max_concurrent: int = 3,
) -> dict[str, Path]:
    """Download mashups from manifest. Skips already-downloaded files.

    Downloads are performed sequentially with rate limiting to avoid
    YouTube throttling. The max_concurrent parameter is accepted for
    interface compatibility with the pipeline but downloads run one at
    a time (yt-dlp is not concurrency-safe within a single process).

    Args:
        manifest_path: Path to mashup_manifest.json.
        output_dir: Directory to write downloaded WAV files.
            Files are saved as {id}.wav.
        max_concurrent: Reserved for future concurrent download support.
            Currently downloads run sequentially.

    Returns:
        Dict mapping mashup ID to the Path of the downloaded WAV file.
        Only includes successfully downloaded entries.
    """
    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    skipped = 0
    failed = 0

    logger.info(
        "Starting mashup download: %d entries, output_dir=%s",
        len(manifest),
        output_dir,
    )

    for i, entry in enumerate(manifest):
        mashup_id = entry["id"]
        url = entry["url"]
        title = entry.get("title", mashup_id)

        # Check for existing download
        target_path = output_dir / f"{mashup_id}.wav"
        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info(
                "[%d/%d] Skipping %s (%s) — already downloaded",
                i + 1,
                len(manifest),
                mashup_id,
                title,
            )
            results[mashup_id] = target_path
            skipped += 1
            continue

        logger.info(
            "[%d/%d] Downloading %s (%s) from %s",
            i + 1,
            len(manifest),
            mashup_id,
            title,
            url,
        )

        try:
            result = _download_one_sync(url, output_dir)

            # Rename from yt-dlp's UUID-based filename to {id}.wav
            if result.wav_path != target_path:
                result.wav_path.rename(target_path)

            results[mashup_id] = target_path

            logger.info(
                "[%d/%d] Downloaded %s: duration=%.1fs, codec=%s",
                i + 1,
                len(manifest),
                mashup_id,
                result.duration_seconds,
                result.source_codec,
            )

        except YouTubeDownloadError as e:
            logger.warning(
                "[%d/%d] Failed to download %s (%s): %s",
                i + 1,
                len(manifest),
                mashup_id,
                title,
                e,
            )
            failed += 1
            continue

        except Exception as e:
            logger.error(
                "[%d/%d] Unexpected error downloading %s (%s): %s",
                i + 1,
                len(manifest),
                mashup_id,
                title,
                e,
                exc_info=True,
            )
            failed += 1
            continue

        # Rate limiting between downloads
        if i < len(manifest) - 1:
            time.sleep(_RATE_LIMIT_DELAY)

    logger.info(
        "Download complete: %d succeeded, %d skipped, %d failed (of %d total)",
        len(results) - skipped,
        skipped,
        failed,
        len(manifest),
    )

    return results
