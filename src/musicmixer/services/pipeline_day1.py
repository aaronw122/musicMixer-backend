import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from musicmixer.config import settings
from musicmixer.services.separation import separate_stems
from musicmixer.services.mixer import overlay_and_export

logger = logging.getLogger(__name__)


def run_pipeline_sync(
    session_id: str,
    song_a_path: Path,
    song_b_path: Path,
) -> Path:
    """Day 1 synchronous pipeline.

    1. Separate both songs into stems (parallel via Modal)
    2. Overlay Song A vocals + Song B instrumentals
    3. Export MP3
    """
    stems_dir = settings.data_dir / "stems" / session_id
    remix_dir = settings.data_dir / "remixes" / session_id
    remix_dir.mkdir(parents=True, exist_ok=True)
    output_path = remix_dir / "remix.mp3"

    # Step 1: Separate both songs (in parallel)
    logger.info(f"[{session_id}] Separating stems...")

    song_a_stems_dir = stems_dir / "song_a"
    song_b_stems_dir = stems_dir / "song_b"

    # Run both separations concurrently -- Modal supports parallel calls
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(
            separate_stems, song_a_path, song_a_stems_dir
        )
        future_b = executor.submit(
            separate_stems, song_b_path, song_b_stems_dir
        )
        song_a_stems = future_a.result(timeout=300)
        song_b_stems = future_b.result(timeout=300)

    logger.info(
        f"[{session_id}] Separation complete. "
        f"Song A: {list(song_a_stems.keys())}, Song B: {list(song_b_stems.keys())}"
    )

    # Step 2+3: Overlay and export
    logger.info(f"[{session_id}] Mixing and exporting...")
    overlay_and_export(
        vocal_stems={"vocals": song_a_stems["vocals"]},
        instrumental_stems={
            k: song_b_stems[k]
            for k in ["drums", "bass", "guitar", "piano", "other"]
        },
        output_path=output_path,
    )

    logger.info(f"[{session_id}] Remix exported to {output_path}")
    return output_path
