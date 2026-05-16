"""Pre-compute and cache stems for all shelf songs.

Usage:
    cd backend
    uv run python scripts/warmup_shelf_stems.py

Reads data/shelf.json, downloads each song from YouTube, separates stems,
and caches them under data/shelf_stems/{song_id}/. Skips songs that already
have cached stems. Handles per-song failures gracefully (continues + reports).

Requires:
    - YouTube proxy configured (YOUTUBE_PROXY_SERVICE_URL in .env) OR
      direct YouTube access
    - Modal configured (STEM_BACKEND=modal) OR local separation available
"""

import asyncio
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

# Ensure the project source is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from musicmixer.config import settings
from musicmixer.services.stem_cache import (
    cache_shelf_stems,
    get_shelf_cached_stems,
    EXPECTED_4_STEMS,
    EXPECTED_6_STEMS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("warmup_shelf_stems")


def load_shelf_records() -> list[dict]:
    """Load all records from shelf.json."""
    shelf_path = settings.data_dir / "shelf.json"
    if not shelf_path.exists():
        logger.error("shelf.json not found at %s", shelf_path)
        sys.exit(1)

    with open(shelf_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    logger.info("Loaded %d records from shelf.json", len(records))
    return records


def is_already_cached(song_id: str) -> bool:
    """Check if stems are already cached for this song ID."""
    cache_dir = settings.shelf_stems_dir / song_id
    if not cache_dir.is_dir():
        return False

    wav_files = list(cache_dir.glob("*.wav"))
    if not wav_files:
        return False

    # Validate: all WAVs non-zero, recognized stem set
    for wav in wav_files:
        if wav.stat().st_size == 0:
            return False

    stem_names = {wav.stem for wav in wav_files}
    return stem_names >= EXPECTED_4_STEMS or stem_names >= EXPECTED_6_STEMS


async def download_song(youtube_url: str, output_dir: Path) -> Path:
    """Download a YouTube song to WAV. Returns the WAV path."""
    from musicmixer.services.youtube import download_youtube_audio

    def _progress(fraction: float, status: str) -> None:
        if fraction >= 1.0 or int(fraction * 10) % 3 == 0:
            logger.info("  Download progress: %.0f%% - %s", fraction * 100, status)

    result = await download_youtube_audio(
        url=youtube_url,
        output_dir=output_dir,
        progress_callback=_progress,
    )
    return result.wav_path


def separate_song(wav_path: Path, output_dir: Path, song_id: str) -> dict[str, Path]:
    """Separate a song into stems and cache them."""
    from musicmixer.services.separation import separate_stems

    def _progress(msg: str) -> None:
        logger.info("  Separation: %s", msg)

    # Run separation (shelf_song_id=None here -- we'll cache manually after)
    stem_paths = separate_stems(wav_path, output_dir, progress_callback=_progress)

    # Cache the stems under the shelf stems directory
    cache_shelf_stems(song_id, output_dir)

    return stem_paths


def process_song(record: dict) -> tuple[str, bool, str]:
    """Process a single shelf song. Returns (song_id, success, message)."""
    song_id = record["id"]
    youtube_url = record["youtube_url"]
    title = record.get("title", "Unknown")

    logger.info("Processing: %s (%s)", title, song_id[:12])

    # Check if already cached
    if is_already_cached(song_id):
        logger.info("  Already cached, skipping")
        return (song_id, True, "already cached")

    # Download
    with tempfile.TemporaryDirectory(prefix="warmup_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        download_dir = tmp_path / "download"
        download_dir.mkdir()
        stems_dir = tmp_path / "stems"
        stems_dir.mkdir()

        try:
            start = time.monotonic()
            wav_path = asyncio.run(download_song(youtube_url, download_dir))
            dl_elapsed = time.monotonic() - start
            logger.info("  Downloaded in %.1fs: %s", dl_elapsed, wav_path.name)
        except Exception as e:
            msg = f"Download failed: {e}"
            logger.error("  %s", msg)
            return (song_id, False, msg)

        # Separate
        try:
            start = time.monotonic()
            stem_paths = separate_song(wav_path, stems_dir, song_id)
            sep_elapsed = time.monotonic() - start
            logger.info("  Separated in %.1fs: %d stems", sep_elapsed, len(stem_paths))
        except Exception as e:
            msg = f"Separation failed: {e}"
            logger.error("  %s", msg)
            return (song_id, False, msg)

    return (song_id, True, f"cached ({len(stem_paths)} stems)")


def main():
    logger.info("=== Shelf Stems Warmup ===")
    logger.info("Shelf stems dir: %s", settings.shelf_stems_dir)
    logger.info("Stem backend: %s", settings.stem_backend)

    records = load_shelf_records()
    if not records:
        logger.warning("No records found in shelf.json")
        return

    results: list[tuple[str, bool, str]] = []
    for record in records:
        result = process_song(record)
        results.append(result)
        logger.info("")

    # Summary
    logger.info("=== Summary ===")
    successes = [r for r in results if r[1]]
    failures = [r for r in results if not r[1]]

    logger.info("Total: %d songs", len(results))
    logger.info("Success: %d", len(successes))
    logger.info("Failed: %d", len(failures))

    if failures:
        logger.info("")
        logger.info("Failures:")
        for song_id, _, msg in failures:
            logger.info("  %s: %s", song_id[:12], msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
