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
GATE_FLOOR_DB = -45.0  # Below this RMS, assume silence (don't compress)

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

        # Formant preservation only when pitch-shifting vocals (not tempo-only)
        if is_vocal and abs(semitones) >= 0.01:
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
    tempo_source: str = "weighted_midpoint",
) -> tuple[float, bool, bool, list[str]]:
    """Decide target BPM and which stems to stretch.

    Returns: (target_bpm, stretch_vocals, stretch_instrumentals, warnings)

    Adaptive weighted midpoint splits the tempo stretch burden between
    vocals and instrumentals instead of forcing 100% onto vocals.

    Tiered stretch limits (asymmetric for slowdown vs speedup):
      < 10%: stretch either/both silently
      10-25%: stretch vocals only
      25-30% speedup / 25% slowdown: stretch vocals, strong warn
      30-45% speedup / 25-35% slowdown: vocals-only stretch, warn
      > 45% speedup / > 35% slowdown: skip stretching entirely
    """
    gap_pct = abs(vocal_bpm - instrumental_bpm) / max(vocal_bpm, instrumental_bpm)

    if tempo_source == "weighted_midpoint" or tempo_source == "average":
        if gap_pct <= 0.04:
            # Within DJ-transparent range -- match to instrumental
            target_bpm = instrumental_bpm
        elif gap_pct <= 0.10:
            # Safe mashup range -- 65/35 instrumental bias
            target_bpm = instrumental_bpm * 0.65 + vocal_bpm * 0.35
        elif gap_pct <= 0.20:
            # Extended range -- 70/30 bias, cap each side at 12%
            target_bpm = instrumental_bpm * 0.70 + vocal_bpm * 0.30
        else:
            # Beyond practical range -- clamp so neither side exceeds 12%
            # Bias toward instrumental
            target_bpm = instrumental_bpm * 0.70 + vocal_bpm * 0.30
            # Clamp: if vocal stretch > 12%, pull target toward vocal
            vocal_stretch = abs(vocal_bpm - target_bpm) / vocal_bpm
            if vocal_stretch > 0.12:
                # Solve: |vocal_bpm - target| / vocal_bpm = 0.12
                if vocal_bpm > target_bpm:
                    target_bpm = vocal_bpm * 0.88  # Max 12% slowdown
                else:
                    target_bpm = vocal_bpm * 1.12  # Max 12% speedup
    elif tempo_source == "song_a":
        target_bpm = vocal_bpm
    elif tempo_source == "song_b":
        target_bpm = instrumental_bpm
    else:
        target_bpm = instrumental_bpm

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
    # +2 dB compromise: +3 clipped with compressor makeup, 0 buried vocals. Revisit with spectral ducking (Day 4).
    vocal_offset_db = 2.0
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


def compress_dynamic_range(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -22.0,
    ratio: float = 3.0,
    attack_ms: float = 10.0,
    release_ms: float = 150.0,
    makeup_db: float = 0.0,
    gate_floor_db: float = GATE_FLOOR_DB,
) -> np.ndarray:
    """RMS-based feed-forward compressor with noise gate.

    Smooths out dynamic range so that loud passages are tamed and (with makeup
    gain) quiet active passages become more audible.  A noise gate prevents
    boosting silence between phrases.

    Parameters:
        threshold_db: RMS level above which compression kicks in (dBFS).
        ratio: Compression ratio (e.g. 3.0 means 3:1).
        attack_ms: How fast the compressor reacts to louder signal.
        release_ms: How fast the compressor releases after signal drops.
        makeup_db: Static gain added after compression to restore loudness.
        gate_floor_db: RMS below this is treated as silence (no gain change).
    """
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    # Frame-based RMS analysis (10ms frames for responsive tracking)
    frame_ms = 10
    frame_len = max(1, int(frame_ms * sr / 1000))
    n_frames = len(mono) // frame_len
    if n_frames == 0:
        return audio

    # Compute RMS per frame
    frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)
    rms_db = 20 * np.log10(rms + 1e-20)

    # Compute gain reduction per frame
    gain_reduction_db = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        if rms_db[i] < gate_floor_db:
            # Below noise gate: no gain change (don't boost silence)
            gain_reduction_db[i] = 0.0
        elif rms_db[i] > threshold_db:
            over_db = rms_db[i] - threshold_db
            gain_reduction_db[i] = -over_db * (1.0 - 1.0 / ratio)

    # Smooth with attack/release envelope (leaky integrator)
    attack_frames = max(1, attack_ms / frame_ms)
    release_frames = max(1, release_ms / frame_ms)
    attack_coeff = np.exp(-1.0 / attack_frames)
    release_coeff = np.exp(-1.0 / release_frames)

    smoothed = np.zeros(n_frames, dtype=np.float32)
    smoothed[0] = gain_reduction_db[0]
    for i in range(1, n_frames):
        if gain_reduction_db[i] < smoothed[i - 1]:
            # Signal getting louder -> compress (fast attack)
            coeff = attack_coeff
        else:
            # Signal getting quieter -> release (slow release)
            coeff = release_coeff
        smoothed[i] = coeff * smoothed[i - 1] + (1.0 - coeff) * gain_reduction_db[i]

    # Apply makeup gain only to active (non-gated) frames.
    # Smooth the gate mask with attack/release to avoid clicks at boundaries.
    gate_mask = (rms_db >= gate_floor_db).astype(np.float32)
    smoothed_mask = np.zeros(n_frames, dtype=np.float32)
    smoothed_mask[0] = gate_mask[0]
    for i in range(1, n_frames):
        if gate_mask[i] > smoothed_mask[i - 1]:
            smoothed_mask[i] = attack_coeff * smoothed_mask[i - 1] + (1.0 - attack_coeff) * gate_mask[i]
        else:
            smoothed_mask[i] = release_coeff * smoothed_mask[i - 1] + (1.0 - release_coeff) * gate_mask[i]
    smoothed += makeup_db * smoothed_mask

    # Convert to linear gain
    gain_linear = 10.0 ** (smoothed / 20.0)

    # Interpolate frame-level gain to sample-level
    frame_centers = np.arange(n_frames) * frame_len + frame_len // 2
    sample_indices = np.arange(len(audio))
    sample_gain = np.interp(sample_indices, frame_centers, gain_linear).astype(np.float32)

    if audio.ndim == 2:
        return audio * sample_gain[:, np.newaxis]
    return audio * sample_gain


def auto_level(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 2.0,
    max_boost_db: float = 6.0,
    max_cut_db: float = 4.0,
    target_percentile: float = 50.0,
    detector_audio: np.ndarray | None = None,
    active_floor_db: float = -45.0,
) -> np.ndarray:
    """Slow automatic gain control that maintains consistent RMS level.

    Unlike compression (which only reduces peaks), this can BOOST quiet
    sections to maintain overall consistency.  Uses a long analysis window
    (2s default) so gain changes are imperceptible — no "pumping".

    This specifically addresses the gap between vocal phrases: when the
    vocal drops to silence between bars, the instrumental-only mix is
    quieter.  The leveler gently boosts those moments so the overall
    volume feels consistent.

    active_floor_db: RMS floor in dBFS below which a window is considered
        inactive (tail/silence). Inactive windows are never boosted —
        only cuts are applied if they exceed threshold. Default -45 dBFS.
    """
    # Use a separate detector signal for RMS analysis if provided.
    # This lets the caller drive leveling from the instrumental bus so
    # vocal gain transitions don't trigger reactive cuts/boosts.
    det = detector_audio if detector_audio is not None else audio
    if det.ndim == 2:
        mono = np.mean(det, axis=1)
    else:
        mono = det

    # Convert dBFS floor to linear RMS threshold
    active_floor_linear = 10.0 ** (active_floor_db / 20.0)

    window = int(window_sec * sr)
    hop = window // 4  # 75% overlap for smooth gain curve
    n_windows = max(1, (len(mono) - window) // hop + 1)

    # Compute windowed RMS
    rms_values = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        start = i * hop
        chunk = mono[start : start + window]
        rms_values[i] = np.sqrt(np.mean(chunk ** 2) + 1e-20)

    # Target level = median of active windows (above the floor)
    active = rms_values[rms_values > active_floor_linear]
    if len(active) == 0:
        return audio
    target_rms = float(np.percentile(active, target_percentile))

    # Compute gain per window
    gains_db = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        if rms_values[i] < active_floor_linear:
            # Below activity floor: allow cuts but never boost.
            # This prevents near-silent tails/noise from being amplified.
            desired_db = 20.0 * np.log10(target_rms / max(rms_values[i], 1e-20))
            gains_db[i] = float(min(0.0, np.clip(desired_db, -max_cut_db, 0.0)))
        else:
            desired_db = 20.0 * np.log10(target_rms / rms_values[i])
            gains_db[i] = float(np.clip(desired_db, -max_cut_db, max_boost_db))

    gains_linear = (10.0 ** (gains_db / 20.0)).astype(np.float32)

    # Interpolate to sample rate (smooth transitions between windows)
    centers = np.arange(n_windows) * hop + window // 2
    sample_gain = np.interp(
        np.arange(len(audio)), centers, gains_linear
    ).astype(np.float32)

    logger.info(
        "Auto-level: target_rms=%.4f, gain_range=[%.1f, %.1f] dB",
        target_rms,
        float(np.min(gains_db)),
        float(np.max(gains_db)),
    )

    if audio.ndim == 2:
        return audio * sample_gain[:, np.newaxis]
    return audio * sample_gain


def lufs_normalize(
    mixed: np.ndarray,
    sr: int,
    target_lufs: float = -12.0,
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


def lufs_normalize_constrained(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -12.0,
    ceiling_dbtp: float = -1.0,
) -> np.ndarray:
    """LUFS normalize with peak constraint — never applies gain it can't keep.

    Computes both the LUFS gain needed and the maximum gain allowed by the
    peak ceiling, then applies whichever is smaller. This prevents the
    normalize-then-trim loop where LUFS boost is undone by peak limiting.

    Returns the gained signal. Logs a warning when peak-constrained.
    """
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio)

    if current_lufs == float("-inf"):
        logger.warning("Cannot normalize silent audio")
        return audio

    ceiling = 10 ** (ceiling_dbtp / 20.0)
    current_peak = true_peak(audio)

    lufs_gain_db = target_lufs - current_lufs
    # Maximum gain before peak exceeds ceiling
    peak_gain_db = 20 * np.log10(ceiling / max(current_peak, 1e-12))

    applied_gain_db = min(lufs_gain_db, peak_gain_db)
    applied_gain_db = float(np.clip(applied_gain_db, -12.0, 12.0))

    shortfall_db = lufs_gain_db - applied_gain_db
    if shortfall_db > 0.5:
        logger.warning(
            "LUFS constrained by peak ceiling: wanted %.1f dB, applied %.1f dB (%.1f dB shortfall). "
            "Output will be ~%.1f LUFS instead of %.1f LUFS",
            lufs_gain_db, applied_gain_db, shortfall_db,
            current_lufs + applied_gain_db, target_lufs,
        )
    else:
        logger.info(
            "LUFS normalize (constrained): current=%.1f LUFS, target=%.1f LUFS, gain=%.1f dB",
            current_lufs, target_lufs, applied_gain_db,
        )

    gain_linear = 10 ** (applied_gain_db / 20.0)
    return audio * gain_linear


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
    knee_db: float = 4.0,
) -> np.ndarray:
    """Soft-knee clipper at given ceiling using tanh waveshaper.

    Below knee onset: UNCHANGED (bit-identical).
    Knee region: tanh-shaped saturation curve — smooth, monotonic, C1 continuous.
    """
    knee_linear = 10 ** (knee_db / 20.0)
    threshold = ceiling / knee_linear  # knee onset

    result = signal.copy()
    abs_signal = np.abs(signal)

    # Mask: everything above threshold needs processing
    active = abs_signal > threshold
    if not np.any(active):
        return result

    x = abs_signal[active]
    # Map [threshold, inf) -> [0, inf), apply tanh saturation, map back
    knee_width = ceiling - threshold
    normalized = (x - threshold) / knee_width  # 0 at threshold, 1 at ceiling
    # tanh maps [0, inf) -> [0, 1) with smooth saturation
    compressed = threshold + knee_width * np.tanh(normalized)
    result[active] = np.sign(signal[active]) * compressed

    return result


def true_peak_limit(
    signal: np.ndarray,
    sr: int,
    ceiling_dbtp: float = -1.0,
    lookahead_ms: float = 5.0,
    release_ms: float = 50.0,
    attack_ms: float = 1.0,
) -> np.ndarray:
    """Look-ahead brickwall limiter targeting a true-peak ceiling.

    Block-based processing (64-sample blocks) with envelope smoothing.
    The lookahead shifts gain reduction backward in time so the limiter
    anticipates peaks before they arrive.

    Handles both mono (N,) and stereo (N, 2) signals. Returns float32.
    """
    from math import exp

    ceiling = 10 ** (ceiling_dbtp / 20.0)
    block_size = 64

    # Work with 2D internally: (N, channels)
    mono = signal.ndim == 1
    if mono:
        work = signal[:, np.newaxis].copy()
    else:
        work = signal.copy()

    n_samples = work.shape[0]
    n_blocks = (n_samples + block_size - 1) // block_size

    # --- Step 1: compute per-block gain reduction ---
    block_gains = np.ones(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n_samples)
        block = work[start:end]
        peak = np.max(np.abs(block))
        if peak > 1e-12:
            block_gains[i] = min(1.0, ceiling / peak)

    # --- Step 2: apply lookahead (shift gain reduction backward) ---
    lookahead_samples = int(lookahead_ms * sr / 1000)
    lookahead_blocks = max(1, lookahead_samples // block_size)
    # Shift the gain envelope left (earlier in time) by lookahead_blocks
    shifted_gains = np.ones(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        # Look ahead: take the minimum gain from current block to
        # lookahead_blocks into the future
        end_idx = min(i + lookahead_blocks + 1, n_blocks)
        shifted_gains[i] = np.min(block_gains[i:end_idx])

    # --- Step 3: smooth the envelope with attack/release ---
    # Coefficients must be computed for BLOCK-RATE processing (every block_size
    # samples), not sample-rate. If we use sample-rate coefficients but apply
    # them once per block, the attack is ~64x too slow and the limiter barely
    # reduces gain.
    alpha_a = exp(-block_size / (attack_ms * sr / 1000)) if attack_ms > 0 else 0.0
    alpha_r = exp(-block_size / (release_ms * sr / 1000)) if release_ms > 0 else 0.0

    smoothed = np.ones(n_blocks, dtype=np.float64)
    smoothed[0] = shifted_gains[0]
    for i in range(1, n_blocks):
        if shifted_gains[i] < smoothed[i - 1]:
            # Gain decreasing (attack): fast response
            alpha = alpha_a
        else:
            # Gain increasing (release): slow recovery
            alpha = alpha_r
        smoothed[i] = alpha * smoothed[i - 1] + (1.0 - alpha) * shifted_gains[i]

    # --- Step 4: expand block-level gains to sample-level and apply ---
    gain_envelope = np.ones(n_samples, dtype=np.float32)
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n_samples)
        gain_envelope[start:end] = smoothed[i]

    # Apply gain to all channels
    result = work * gain_envelope[:, np.newaxis]
    result = result.astype(np.float32)

    # Log max gain reduction applied
    min_gain = float(np.min(smoothed))
    if min_gain < 1.0:
        max_reduction_db = 20.0 * np.log10(min_gain) if min_gain > 0 else -np.inf
        logger.info(
            "true_peak_limit: max gain reduction %.1f dB (ceiling=%.1f dBTP)",
            max_reduction_db, ceiling_dbtp,
        )

    if mono:
        return result[:, 0]
    return result


def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 100.0, order: int = 2) -> np.ndarray:
    """Apply a Butterworth high-pass filter.

    Removes low-frequency bleed (bass rumble, kick drum artifacts) from
    separated vocal stems. Standard practice before layering vocals over
    different instrumentals.
    """
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, cutoff_hz, btype='high', fs=sr, output='sos')
    return sosfiltfilt(sos, audio, axis=0).astype(np.float32)


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
    use_s16_dither: bool = True,
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
        if use_s16_dither:
            cmd[4:4] = ["-af", "aresample=osf=s16:dither_method=triangular"]
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
