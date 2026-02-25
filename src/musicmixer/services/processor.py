"""Audio processor: standardization, tempo matching, LUFS normalization, limiting, fades, export.

Steps 3 + 4 of the Day 2 implementation plan.

All audio stays float32 throughout. Never convert to int16 mid-pipeline.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyloudnorm
import soundfile as sf
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LUFS_FLOOR = -40.0  # Below this, audio is effectively silence/noise

# Module-level cache for rubberband version
_rubberband_version: int | None = None


# ---------------------------------------------------------------------------
# Step 3: Sample rate / channel standardization + tempo matching
# ---------------------------------------------------------------------------


def validate_stem(path: Path, expected_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load a stem WAV, ensure correct sample rate and stereo format.

    Returns (audio, sr) where audio is float32 stereo (N, 2).
    """
    audio, sr = sf.read(str(path), dtype="float32")

    if sr != expected_sr:
        import librosa

        logger.warning("Stem %s at %dHz, resampling to %dHz", path.name, sr, expected_sr)
        # librosa.resample expects (samples,) or (channels, samples) for multi-channel
        # but soundfile returns (samples, channels). Transpose for librosa, then back.
        if audio.ndim == 2:
            audio = librosa.resample(
                audio.T, orig_sr=sr, target_sr=expected_sr, res_type="soxr_hq"
            ).T
        else:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=expected_sr, res_type="soxr_hq"
            )
        sr = expected_sr

    # Mono -> stereo
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])

    return audio.astype(np.float32), sr


def trim_audio(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    """Trim audio to [start_sec, end_sec) range using sample-based slicing.

    Trimming happens BEFORE tempo stretch. Time ranges are in the original tempo domain.
    """
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    # Clamp to valid range
    start_sample = max(0, min(start_sample, len(audio)))
    end_sample = max(start_sample, min(end_sample, len(audio)))
    return audio[start_sample:end_sample]


def check_rubberband_version() -> int:
    """Check rubberband CLI version. Returns major version (2, 3, or 4).

    Caches result at module level for subsequent calls.
    """
    global _rubberband_version
    if _rubberband_version is not None:
        return _rubberband_version

    try:
        result = subprocess.run(
            ["rubberband", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # rubberband --version outputs just the version number, e.g. "4.0.0"
        # or may output "Rubber Band Library version X.Y.Z" on some builds
        version_text = result.stdout.strip() + result.stderr.strip()
        # Find first digit sequence that looks like a version
        for part in version_text.split():
            if part[0].isdigit():
                major = int(part.split(".")[0])
                _rubberband_version = major
                logger.info("Detected rubberband v%d (%s)", major, part)
                return major
        # Fallback: couldn't parse
        logger.warning("Could not parse rubberband version from: %s", version_text)
        _rubberband_version = 3
        return 3
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.warning("rubberband version check failed: %s", e)
        _rubberband_version = 3
        return 3


def rubberband_process(
    audio: np.ndarray,
    sr: int,
    source_bpm: float,
    target_bpm: float,
    semitones: float = 0,
    is_vocal: bool = False,
) -> np.ndarray:
    """Single-pass tempo + pitch adjustment via rubberband CLI.

    CRITICAL: -t takes a TIME RATIO (output_duration / input_duration), NOT speed ratio.
    Formula: time_ratio = source_bpm / target_bpm
      90 BPM -> 120 BPM: time_ratio = 90/120 = 0.75 (shorter = faster)
      120 BPM -> 90 BPM: time_ratio = 120/90 = 1.333 (longer = slower)
    INVERTING THIS PRODUCES THE OPPOSITE STRETCH.
    """
    # Skip-at-unity: no processing needed if ratio is effectively 1:1
    time_ratio = source_bpm / target_bpm
    if abs(time_ratio - 1.0) < 0.001 and abs(semitones) < 0.01:
        return audio

    # Create temp files with uuid prefix to prevent collision in parallel execution
    tmp_dir = Path(tempfile.gettempdir())
    uid = uuid4().hex[:8]
    in_path = tmp_dir / f"rb_in_{uid}.wav"
    out_path = tmp_dir / f"rb_out_{uid}.wav"

    try:
        sf.write(str(in_path), audio, sr, subtype="FLOAT")

        # Build rubberband command
        cmd = ["rubberband", "-t", str(time_ratio)]

        if abs(semitones) >= 0.01:
            cmd += ["-p", str(semitones)]

        # Engine flag depends on version
        rb_version = check_rubberband_version()
        if rb_version >= 3:
            cmd += ["-3"]  # R3 engine (fine mode)
        else:
            cmd += ["--crisp", "5"]  # Best R2 equivalent

        # Always preserve formants for vocals (tempo stretch also shifts formants)
        if is_vocal:
            cmd += ["--formant"]

        cmd += [str(in_path), str(out_path)]

        logger.info("Running rubberband: time_ratio=%.4f, semitones=%.2f, cmd=%s", time_ratio, semitones, " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)

        result, _ = sf.read(str(out_path), dtype="float32")
        return result.astype(np.float32)

    finally:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def compute_tempo_plan(
    vocal_bpm: float,
    instrumental_bpm: float,
    tempo_source: str = "song_b",
) -> tuple[float, bool, bool, list[str]]:
    """Decide target BPM and which stems to stretch.

    Returns: (target_bpm, stretch_vocals, stretch_instrumentals, warnings)

    Tiered stretch limits (asymmetric for slowdown vs speedup):
      < 10%: stretch either/both silently
      10-25%: stretch vocals only
      25-30% speedup / 25% slowdown: stretch vocals, strong warn
      30-45% speedup / 25-35% slowdown: vocals-only stretch, warn
      > 45% speedup / > 35% slowdown: skip stretching entirely
    """
    target_bpm = instrumental_bpm  # Default: match to instrumental

    if tempo_source == "average":
        gap_pct = abs(vocal_bpm - instrumental_bpm) / max(vocal_bpm, instrumental_bpm)
        if gap_pct < 0.15:
            target_bpm = (vocal_bpm + instrumental_bpm) / 2
        # else: keep instrumental as target

    # time_ratio for rubberband: source_bpm / target_bpm
    vocal_ratio = vocal_bpm / target_bpm
    inst_ratio = instrumental_bpm / target_bpm

    # Compute stretch percentage and direction
    vocal_stretch_pct = abs(1.0 - vocal_ratio)
    is_slowdown = vocal_ratio > 1.0  # Vocals need to be stretched longer = slowed down

    warnings: list[str] = []
    stretch_vocals = True
    stretch_instrumentals = abs(1.0 - inst_ratio) > 0.001

    # Apply tiered limits
    if is_slowdown:
        # Slowdown limits (tighter -- more artifacts when slowing)
        if vocal_stretch_pct > 0.35:
            stretch_vocals = False
            stretch_instrumentals = False
            warnings.append(
                "Tempo difference too large for stretching. Songs play at original tempos."
            )
        elif vocal_stretch_pct > 0.25:
            stretch_instrumentals = False
            warnings.append(
                "Vocals stretched significantly to match the beat -- they may sound different."
            )
        elif vocal_stretch_pct > 0.10:
            stretch_instrumentals = False
            warnings.append("Vocals adjusted to match the instrumental tempo.")
    else:
        # Speedup limits (more tolerant)
        if vocal_stretch_pct > 0.45:
            stretch_vocals = False
            stretch_instrumentals = False
            warnings.append(
                "Tempo difference too large for stretching. Songs play at original tempos."
            )
        elif vocal_stretch_pct > 0.30:
            stretch_instrumentals = False
            warnings.append(
                "Vocals stretched significantly to match the beat -- they may sound different."
            )
        elif vocal_stretch_pct > 0.25:
            stretch_instrumentals = False
            warnings.append("Vocals adjusted to match the instrumental tempo.")
        elif vocal_stretch_pct > 0.10:
            stretch_instrumentals = False
            # Silent for vocals-only at 10-25%

    return target_bpm, stretch_vocals, stretch_instrumentals, warnings


# ---------------------------------------------------------------------------
# Step 4: LUFS normalization + peak limiter + fades + export
# ---------------------------------------------------------------------------


def cross_song_level_match(
    vocal_audio: np.ndarray,
    instrumental_sum: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Match vocal loudness to instrumental level.

    Measures LUFS of vocal and instrumental via pyloudnorm.
    Safety cap: clip gain to [-12, +12] dB.
    """
    meter = pyloudnorm.Meter(sr)
    vocal_lufs = meter.integrated_loudness(vocal_audio)
    instrumental_lufs = meter.integrated_loudness(instrumental_sum)

    if vocal_lufs < LUFS_FLOOR or instrumental_lufs < LUFS_FLOOR:
        logger.warning(
            "Skipping level matching: vocal=%.1f LUFS, instrumental=%.1f LUFS",
            vocal_lufs,
            instrumental_lufs,
        )
        return vocal_audio

    # Vocals and instrumentals at equal LUFS. Per-section stem_gains in the
    # arrangement handle the artistic balance (vocals forward in chorus, etc.).
    # Professional mashups sit vocals at +0 to +1 dB relative to instrumental.
    vocal_offset_db = 0.0
    target_vocal_lufs = instrumental_lufs + vocal_offset_db
    gain_db = target_vocal_lufs - vocal_lufs
    gain_db = float(np.clip(gain_db, -12.0, 12.0))
    logger.info(
        "Level match: vocal=%.1f LUFS, inst=%.1f LUFS, gain=%.1f dB",
        vocal_lufs,
        instrumental_lufs,
        gain_db,
    )
    return vocal_audio * (10 ** (gain_db / 20.0))


def lufs_normalize(
    mixed: np.ndarray,
    sr: int,
    target_lufs: float = -14.0,
) -> np.ndarray:
    """Normalize the FINAL MIX to target LUFS (not individual stems!).

    Per-stem normalization to -14 LUFS causes the sum to land around -11 LUFS,
    forcing the limiter to crush the audio.

    Safety cap: clip gain to [-12, +12] dB.
    Skip if near-silent (< LUFS_FLOOR).
    """
    meter = pyloudnorm.Meter(sr)
    current_lufs = meter.integrated_loudness(mixed)

    if current_lufs < LUFS_FLOOR:
        logger.warning(
            "Final mix near-silent (%.1f LUFS), skipping normalization", current_lufs
        )
        return mixed

    gain_db = target_lufs - current_lufs
    gain_db = float(np.clip(gain_db, -12.0, 12.0))
    logger.info(
        "LUFS normalize: current=%.1f LUFS, target=%.1f LUFS, gain=%.1f dB",
        current_lufs,
        target_lufs,
        gain_db,
    )
    return mixed * (10 ** (gain_db / 20.0))


def true_peak(signal: np.ndarray) -> float:
    """4x oversampled true-peak measurement (practical ITU-R BS.1770-4 approximation).

    Uses scipy.signal.resample_poly with Hamming-windowed FIR. Can underestimate
    true peaks by ~0.3 dB (acceptable for MVP).

    Handles stereo: returns max across channels.
    """
    if signal.ndim == 2:
        return max(true_peak(signal[:, ch]) for ch in range(signal.shape[1]))
    upsampled = resample_poly(signal, 4, 1)
    return float(np.max(np.abs(upsampled)))


def soft_clip(
    signal: np.ndarray,
    ceiling: float,
    knee_db: float = 2.0,
) -> np.ndarray:
    """Soft-knee clipper at given ceiling.

    Below threshold: UNCHANGED (bit-identical).
    Knee region: quadratic compression (C1 continuous at both boundaries).
    Above ceiling: hard limit.

    ceiling = 10**(-1.0/20.0) ~ 0.891 for -1.0 dBTP
    """
    knee_linear = 10 ** (knee_db / 20.0)
    threshold = ceiling / knee_linear

    result = signal.copy()
    abs_signal = np.abs(signal)

    # Knee region: parabolic compression
    knee_mask = (abs_signal > threshold) & (abs_signal <= ceiling)
    if np.any(knee_mask):
        x = abs_signal[knee_mask]
        knee_width = ceiling - threshold
        t = (x - threshold) / knee_width  # 0 to 1
        compressed = threshold + knee_width * (2 * t - t * t)
        result[knee_mask] = np.sign(signal[knee_mask]) * compressed

    # Hard limit above ceiling
    over_mask = abs_signal > ceiling
    result[over_mask] = np.sign(signal[over_mask]) * ceiling

    return result


def apply_fades(
    audio: np.ndarray,
    sr: int,
    fade_in_sec: float = 2.0,
    fade_out_sec: float = 3.0,
    skip_fade_in: bool = False,
    skip_fade_out: bool = False,
) -> np.ndarray:
    """Apply equal-power cosine-squared fades.

    Handles both mono (N,) and stereo (N, 2) audio.
    Skip flags prevent double-fading with arrangement transitions.
    """
    result = audio.copy()

    if not skip_fade_in:
        n_in = min(int(fade_in_sec * sr), len(result))
        if n_in > 0:
            fade_in = np.cos(np.linspace(np.pi / 2, 0, n_in)).astype(np.float32) ** 2  # 0 -> 1
            if result.ndim == 2:
                result[:n_in] *= fade_in[:, np.newaxis]
            else:
                result[:n_in] *= fade_in

    if not skip_fade_out:
        n_out = min(int(fade_out_sec * sr), len(result))
        if n_out > 0:
            fade_out = np.cos(np.linspace(0, np.pi / 2, n_out)).astype(np.float32) ** 2  # 1 -> 0
            if result.ndim == 2:
                result[-n_out:] *= fade_out[:, np.newaxis]
            else:
                result[-n_out:] *= fade_out

    return result


def export_mp3(
    mixed: np.ndarray,
    sr: int,
    output_path: Path,
) -> Path:
    """Export float32 audio to MP3 via ffmpeg subprocess (NOT pydub).

    Writes float32 WAV via soundfile (subtype="FLOAT"), converts to MP3
    via ffmpeg with libmp3lame at 320kbps. 120-second timeout.
    Cleans up temp WAV file.
    """
    tmp_dir = Path(tempfile.gettempdir())
    uid = uuid4().hex[:8]
    tmp_wav = tmp_dir / f"export_{uid}.wav"

    try:
        sf.write(str(tmp_wav), mixed, sr, subtype="FLOAT")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp_wav),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "320k",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            logger.error("ffmpeg failed: %s", stderr[:500])
            raise RuntimeError(f"ffmpeg export failed: {stderr[:500]}")

        logger.info(
            "Exported MP3: %s (%d KB)", output_path, output_path.stat().st_size // 1024
        )
        return output_path

    finally:
        tmp_wav.unlink(missing_ok=True)
