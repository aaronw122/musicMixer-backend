# A/B Test Suite for Sound Quality Enhancement (YouTube Links)

revision: 8

## Prerequisites

1. YouTube input feature must be implemented (`backend/src/musicmixer/services/youtube.py` + `/api/remix/youtube` endpoint). See `docs/impl/youtube-input-plan.md`.
2. Sound quality enhancement feature flags (`ab_per_stem_eq_v1`, `ab_resonance_detection_v1`, `ab_multiband_comp_v1`, `ab_static_mastering_v1`) must be added to `config.py`. The script asserts their existence at startup (see Step 0 below).

## Context

The sound quality enhancement plan adds 4 feature flags (EQ, resonance detection, multiband compression, reference mastering) but has no automated way to test them across multiple song pairs. We need a batch test script that runs all 5 mashup pairs with flags off and on, saves labeled outputs to `mashupTests/`, and clears previous results on each run.

**This version uses YouTube links instead of local files.** The script downloads audio via the YouTube service, eliminating the need to store ~500MB of example MP3s in the repo. Test pairs are defined as a simple list of URLs — anyone can reproduce the tests without needing the files locally.

## Test Matrix

| # | Vocals (Song A) | Instrumentals (Song B) | Mix Intent (documentation only) |
|---|-----------------|------------------------|----------------------------------|
| 1 | [Notorious B.I.G. - Hypnotize](https://www.youtube.com/watch?v=eaPzCHEQExs) | [Grateful Dead - Althea (2013 Remaster)](https://www.youtube.com/watch?v=ZZNZgtj26Fk) | Biggie rapping over Grateful Dead guitar |
| 2 | [Kanye West - Ghost Town](https://www.youtube.com/watch?v=qAsHVwl-MU4) | [Amanaz - Khala My Friend](https://www.youtube.com/watch?v=2QxeDecgNWg) | Ghost Town vocals over Zamrock groove |
| 3 | [JAY-Z - Encore](https://www.youtube.com/watch?v=7VksyVUAwi8) | [Linkin Park - Numb](https://www.youtube.com/watch?v=kXYiU_JCYtU) | Jay-Z rapping over Linkin Park instrumentals |
| 4 | [Coldplay - Adventure of a Lifetime](https://www.youtube.com/watch?v=QtXby3twMmI) | [Daft Punk - Give Life Back to Music](https://www.youtube.com/watch?v=IluRBvnYMoY) | Coldplay vocals over Daft Punk funk |
| 5 | [Dabrye - Air (feat. Doom)](https://www.youtube.com/watch?v=X-YaI5ZkRvw) | [Grateful Dead - Scarlet Begonias (2013 Remaster)](https://www.youtube.com/watch?v=xt4XAz2WZ3Y) | MF DOOM rapping over Grateful Dead jam |

### Genre coverage note

The current test matrix skews toward rap vocals (3/5 pairs) and Grateful Dead instrumentals (2/5 pairs). This limits confidence that enhancements generalize across vocal styles. **Recommendation for future expansion:** Replace one rap-vocal pair (e.g., pair #3 or #5) with sung vocals from a different genre (soul, R&B, or pop) to test EQ and resonance detection behavior on sustained pitch and vibrato, which stress these processors differently than percussive rap delivery.

### YouTube Source Details

All links verified 2026-02-27. Prefer official audio uploads (static image + audio only) over music videos where available — lower bandwidth, faster download, identical audio.

| Song | Video ID | Source Channel | Type |
|------|----------|---------------|------|
| Hypnotize | `eaPzCHEQExs` | The Notorious B.I.G. | Official Audio |
| Althea (2013 Remaster) | `ZZNZgtj26Fk` | Grateful Dead | Official (Grateful Dead/Rhino) |
| Ghost Town | `qAsHVwl-MU4` | Kanye West | Official Audio |
| Khala My Friend | `2QxeDecgNWg` | Amanaz - Topic | Topic channel (Virgin Music) |
| Encore | `7VksyVUAwi8` | JAY-Z - Topic | Topic channel (Universal) |
| Numb | `kXYiU_JCYtU` | Linkin Park | Official Music Video [4K] |
| Adventure of a Lifetime | `QtXby3twMmI` | Coldplay | Official Video |
| Give Life Back to Music | `IluRBvnYMoY` | Daft Punk | Official Audio |
| Air (feat. Doom) | `X-YaI5ZkRvw` | Dabrye - Topic | Topic channel (Ghostly Intl) |
| Scarlet Begonias (2013 Remaster) | `xt4XAz2WZ3Y` | Grateful Dead | Official (Grateful Dead/Rhino) |

> **Note:** All runs use an empty prompt (deterministic fallback). The "Mix Intent" column documents the intended mashup for human reference only — it is not passed to the pipeline. Prompt-driven mixing is out of scope for this sound-quality A/B test suite.

## Output Structure

**`--mode compare` (default):**

```
mashupTests/
├── 1-biggie-althea/
│   ├── control.mp3          # all new flags off
│   └── enhanced.mp3         # all new flags on
├── 2-ghosttown-khala/
│   ├── control.mp3
│   └── enhanced.mp3
├── 3-encore-numb/
│   ├── control.mp3
│   └── enhanced.mp3
├── 4-coldplay-daftpunk/
│   ├── control.mp3
│   └── enhanced.mp3
├── 5-doom-scarletbegonias/
│   ├── control.mp3
│   └── enhanced.mp3
└── results.txt              # pipe-delimited metrics (see format below)
```

**`--mode sweep` (recommended for diagnostic evaluation):**

```
mashupTests/
├── 1-biggie-althea/
│   ├── control.mp3                  # all new flags off
│   ├── sweep-per-stem-eq.mp3       # only AB_PER_STEM_EQ_V1 on
│   ├── sweep-resonance-detection.mp3  # AB_RESONANCE_DETECTION_V1 + AB_PER_STEM_EQ_V1 on
│   ├── sweep-multiband-comp.mp3    # only AB_MULTIBAND_COMP_V1 on
│   └── sweep-static-mastering.mp3  # only AB_STATIC_MASTERING_V1 on
├── ...                              # same structure for each pair
└── results.txt
```

### Results format (`results.txt`)

Machine-parseable, pipe-delimited, one row per run:

```
pair | mode | variant | status | duration_s | lufs | peak_dbtp | band_rms_low | band_rms_mid | band_rms_highmid | band_rms_high | spectral_centroid_hz | output_path | sha256
1-biggie-althea | compare | control | ok | 312.5 | -11.8 | -0.9 | -18.2 | -14.5 | -16.1 | -22.3 | 2841.7 | mashupTests/1-biggie-althea/control.mp3 | a1b2c3...
1-biggie-althea | compare | enhanced | ok | 328.1 | -12.1 | -1.0 | -17.9 | -14.1 | -15.8 | -21.9 | 2953.2 | mashupTests/1-biggie-althea/enhanced.mp3 | d4e5f6...
1-biggie-althea | sweep | sweep-per-stem-eq | ok | 315.2 | -12.0 | -0.9 | -18.0 | -14.3 | -15.9 | -22.1 | 2887.4 | mashupTests/1-biggie-althea/sweep-per-stem-eq.mp3 | 7a8b9c...
...
```

> **`mode` column** disambiguates compare-mode control (all new flags off) from sweep-mode control (Day 3 baseline on). Without this column, both would appear as "control" in results, making cross-mode comparisons ambiguous.

> **Spectral metrics** (`band_rms_*`, `spectral_centroid_hz`): Sound quality enhancements are frequency-domain operations (EQ, multiband compression), so loudness/dynamics metrics alone are insufficient. Per-band RMS uses crossover frequencies matching the multiband compressor (150, 600, 3000 Hz) to measure impact in each band. Spectral centroid tracks overall brightness shifts.

LUFS and true-peak are measured from the final output MP3 after each run using `pyloudnorm` + `librosa` (see Step 6 below).

## Implementation

### Files to create/modify

| File | Action |
|------|--------|
| `scripts/run_ab_test_suite.py` | **Create** — main batch test script |
| `backend/scripts/run_pipeline_phase.py` | **Modify** — add `--source-quality-a` / `--source-quality-b` CLI args via `argparse`, forward to `run_pipeline()` |
| `.gitignore` | **Modify** — add `mashupTests/` |
| `docs/impl/sound-quality-enhancement-plan.md` | **Modify** — add test matrix section |

### Flag matrix

**`--mode compare`** — exact env-var values passed to each subprocess invocation:

| Flag | Control | Enhanced |
|------|---------|----------|
| `AB_CONTROL_DAY3` | `True` | `True` |
| `AB_AUTOLVL_TUNE_V1` | `False` | `True` |
| `AB_VOCAL_MAKEUP_V1` | `False` | `True` |
| `AB_MP3_EXPORT_PATH_V1` | `False` | `True` |
| `AB_PER_STEM_EQ_V1` | `False` | `True` |
| `AB_RESONANCE_DETECTION_V1` | `False` | `True` |
| `AB_MULTIBAND_COMP_V1` | `False` | `True` |
| `AB_STATIC_MASTERING_V1` | `False` | `True` |
| `LYRICS_LOOKUP_ENABLED` | `false` | `false` |

**`--mode sweep`** — baseline flags held at production defaults; each sweep variant enables exactly one new flag:

| Flag | Control | sweep-per-stem-eq | sweep-resonance-detection | sweep-multiband-comp | sweep-static-mastering |
|------|---------|-------------------|--------------------------|---------------------|----------------------|
| `AB_CONTROL_DAY3` | `True` | `True` | `True` | `True` | `True` |
| `AB_AUTOLVL_TUNE_V1` | `True` | `True` | `True` | `True` | `True` |
| `AB_VOCAL_MAKEUP_V1` | `True` | `True` | `True` | `True` | `True` |
| `AB_MP3_EXPORT_PATH_V1` | `True` | `True` | `True` | `True` | `True` |
| `AB_PER_STEM_EQ_V1` | `False` | **`True`** | **`True`** | `False` | `False` |
| `AB_RESONANCE_DETECTION_V1` | `False` | `False` | **`True`** | `False` | `False` |
| `AB_MULTIBAND_COMP_V1` | `False` | `False` | `False` | **`True`** | `False` |
| `AB_STATIC_MASTERING_V1` | `False` | `False` | `False` | `False` | **`True`** |
| `LYRICS_LOOKUP_ENABLED` | `false` | `false` | `false` | `false` | `false` |

> **`AB_CONTROL_DAY3`** is `True` for both variants because it represents the Day 3 pipeline baseline, not a sound quality enhancement. The production default is `True`, and toggling it would confound results by changing non-audio-quality behavior between variants.

> **`LYRICS_LOOKUP_ENABLED`** is `false` for both variants to eliminate a confounding variable from the A/B comparison.

> **Sweep mode baseline flags:** In sweep mode, `AB_AUTOLVL_TUNE_V1`, `AB_VOCAL_MAKEUP_V1`, and `AB_MP3_EXPORT_PATH_V1` are set to `True` (production defaults) for all runs, including control. This isolates the effect of each new sound-quality flag against the production baseline rather than against a fully stripped-down pipeline.

> **Flag dependency: resonance detection requires per-stem EQ.** In `pipeline.py`, resonance detection is gated behind `if settings.ab_per_stem_eq_v1 and settings.ab_resonance_detection_v1` (line ~629). The `sweep-resonance-detection` variant therefore sets `AB_PER_STEM_EQ_V1=True` alongside `AB_RESONANCE_DETECTION_V1=True`. This measures the marginal effect of resonance detection on top of per-stem EQ, which is the only meaningful comparison since resonance detection is a no-op without it.

### Script design (`scripts/run_ab_test_suite.py`)

Based on existing patterns in `backend/scripts/run_pipeline_phase.py` and `scripts/run_modal_ab_phases.sh`.

**Important:** Each pipeline run MUST be invoked as a subprocess, not in-process. `config.py` instantiates `settings = Settings()` at module scope, so env vars set after import are ignored. Subprocess invocation ensures clean flag state per run.

**Step 0: Preflight validation**
   - Resolve `REPO_ROOT = Path(__file__).resolve().parent.parent`
   - Assert each expected flag exists on the `settings` object via `hasattr()` (import `settings` from `musicmixer.config`). If any flags are missing, abort with a message listing them and noting that the sound-quality-enhancement-plan prerequisites must be implemented first.
   - Verify `yt-dlp` is available: `import yt_dlp` — if import fails, abort with a message noting the YouTube input feature must be implemented first.
   - Parse a `--mode` CLI argument (default: `compare`):
     - **`compare`**: All-off (control) vs all-on (enhanced) — the existing behavior. Useful as a quick smoke test.
     - **`sweep`**: Runs each of the 4 new sound-quality flags individually against control to isolate per-flag attribution. For each flag, control has all 4 new flags off; the variant has only that one flag on. This produces 4 variant runs per pair (one per flag) plus 1 control, for a total of 5 runs per pair instead of 2. The 4 new flags swept are: `AB_PER_STEM_EQ_V1`, `AB_RESONANCE_DETECTION_V1`, `AB_MULTIBAND_COMP_V1`, `AB_STATIC_MASTERING_V1`. Baseline flags (`AB_CONTROL_DAY3`, `AB_AUTOLVL_TUNE_V1`, `AB_VOCAL_MAKEUP_V1`, `AB_MP3_EXPORT_PATH_V1`) are held constant at their production defaults for both control and variant.
   - This follows the incremental approach from `scripts/run_modal_ab_phases.sh`, which tests each flag one at a time to isolate attribution.
   - **Recommended mode for diagnostic evaluation is `sweep`**; `compare` is retained as a smoke test.

1. **Clear `mashupTests/`** at start of every run (`shutil.rmtree` + `mkdir`). Use `REPO_ROOT / "mashupTests"` for the output directory.
2. **Define test pairs** as a list of dicts with YouTube URLs, roles, and short names:
   ```python
   TEST_PAIRS = [
       {
           "name": "1-biggie-althea",
           "url_a": "https://www.youtube.com/watch?v=eaPzCHEQExs",
           "url_b": "https://www.youtube.com/watch?v=ZZNZgtj26Fk",
       },
       {
           "name": "2-ghosttown-khala",
           "url_a": "https://www.youtube.com/watch?v=qAsHVwl-MU4",
           "url_b": "https://www.youtube.com/watch?v=2QxeDecgNWg",
       },
       {
           "name": "3-encore-numb",
           "url_a": "https://www.youtube.com/watch?v=7VksyVUAwi8",
           "url_b": "https://www.youtube.com/watch?v=kXYiU_JCYtU",
       },
       {
           "name": "4-coldplay-daftpunk",
           "url_a": "https://www.youtube.com/watch?v=QtXby3twMmI",
           "url_b": "https://www.youtube.com/watch?v=IluRBvnYMoY",
       },
       {
           "name": "5-doom-scarletbegonias",
           "url_a": "https://www.youtube.com/watch?v=X-YaI5ZkRvw",
           "url_b": "https://www.youtube.com/watch?v=xt4XAz2WZ3Y",
       },
   ]
   ```
3. **Download phase:** For each pair, download both songs once using the YouTube service (`download_youtube_audio`). Cache the WAV files in a temp directory for reuse across control/enhanced runs. **Note:** `download_youtube_audio` is an `async` function returning a `YouTubeAudioResult` (not a bare path), so wrap calls in `asyncio.run()`:
   ```python
   import asyncio
   from musicmixer.services.youtube import download_youtube_audio, YouTubeAudioResult

   # Download once per pair, reuse for both variants
   dl_dir = REPO_ROOT / "backend" / "data" / "uploads" / f"ab-{pair['name']}"
   result_a: YouTubeAudioResult = asyncio.run(download_youtube_audio(pair["url_a"], dl_dir))
   result_b: YouTubeAudioResult = asyncio.run(download_youtube_audio(pair["url_b"], dl_dir))

   # Extract WAV paths for pipeline invocation
   wav_a = result_a.wav_path
   wav_b = result_b.wav_path

   # Build source quality strings (matches production format in api/remix.py)
   source_quality_a = f"youtube-{result_a.source_codec}-{result_a.source_bitrate}kbps"
   source_quality_b = f"youtube-{result_b.source_codec}-{result_b.source_bitrate}kbps"
   ```
   This avoids downloading each song twice (once for control, once for enhanced) and captures source quality metadata for propagation to the pipeline.
4. **For each pair, run pipeline variants** using the downloaded WAVs. The number of runs depends on `--mode`:
   - **`compare` mode (default):** 2 runs per pair:
     - **Control run:** all 4 new sound-quality flags off (see flag matrix)
     - **Enhanced run:** all 4 new sound-quality flags on
   - **`sweep` mode:** 5 runs per pair:
     - **Control run:** all 4 new sound-quality flags off
     - **sweep-per-stem-eq:** only `AB_PER_STEM_EQ_V1` on
     - **sweep-resonance-detection:** `AB_RESONANCE_DETECTION_V1` on + `AB_PER_STEM_EQ_V1` on (required dependency; see flag matrix note)
     - **sweep-multiband-comp:** only `AB_MULTIBAND_COMP_V1` on
     - **sweep-static-mastering:** only `AB_STATIC_MASTERING_V1` on
   - In both modes, baseline flags (`AB_CONTROL_DAY3`, `AB_AUTOLVL_TUNE_V1`, `AB_VOCAL_MAKEUP_V1`, `AB_MP3_EXPORT_PATH_V1`) are held at their production defaults (`True`) for all runs.
5. **Pipeline invocation** — spawn a subprocess per run, passing source quality via CLI args:
   ```python
   result = subprocess.run(
       [
           "uv", "run", "python", "scripts/run_pipeline_phase.py",
           phase_name, str(wav_a), str(wav_b),
           "--source-quality-a", source_quality_a,
           "--source-quality-b", source_quality_b,
       ],
       cwd=str(REPO_ROOT / "backend"),
       env={**os.environ, **flag_env_vars, "LYRICS_LOOKUP_ENABLED": "false"},
       capture_output=True,
       text=True,
       timeout=600,
   )
   ```
   - Use unique `phase_name` per run (e.g., `1-biggie-althea-control`)
   - Parse `result.stdout` for status and output path (run_pipeline_phase.py prints these to stdout)
   - Copy the output MP3 to `REPO_ROOT / "mashupTests" / pair_dir / "{control|enhanced}.mp3"`
   - Catch `subprocess.TimeoutExpired` and log as a timed-out failure; continue to the next run

   **Required change to `run_pipeline_phase.py`:** **FULL REPLACEMENT** of the existing `sys.argv` parsing. The current script uses bare `sys.argv` with a `if len(sys.argv) != 4: return 2` guard and positional `sys.argv[1]`, `sys.argv[2]`, `sys.argv[3]` references. All of that must be removed and replaced with `argparse`:
   ```python
   import argparse

   parser = argparse.ArgumentParser()
   parser.add_argument("phase")
   parser.add_argument("song_a")
   parser.add_argument("song_b")
   parser.add_argument("--source-quality-a", default=None)
   parser.add_argument("--source-quality-b", default=None)
   args = parser.parse_args()

   run_pipeline(
       session_id=f"ab-{args.phase}",
       song_a_path=args.song_a,
       song_b_path=args.song_b,
       prompt="",
       event_queue=events,
       session=session,
       source_quality_a=args.source_quality_a,
       source_quality_b=args.source_quality_b,
   )
   ```
   This matches production behavior in `/api/remix/youtube`, which builds quality strings as `f"youtube-{result.source_codec}-{result.source_bitrate}kbps"` and passes them to `run_pipeline(source_quality_a=..., source_quality_b=...)`.
6. **Measure LUFS, true-peak, and spectral metrics** after each successful run using `pyloudnorm` + `librosa` (NOT `soundfile` — `libsndfile` cannot decode MP3):
   ```python
   import hashlib
   import librosa
   import numpy as np
   import pyloudnorm as pyln
   from scipy.signal import butter, sosfilt
   from musicmixer.services.processor import true_peak

   # librosa.load returns (channels, samples); pyloudnorm expects (samples, channels)
   audio, sr = librosa.load(output_mp3_path, sr=None, mono=False)
   audio = audio.T  # transpose to (samples, channels) for pyloudnorm
   meter = pyln.Meter(sr)
   lufs = meter.integrated_loudness(audio)
   # true_peak() uses 4x oversampling per ITU-R BS.1770-4; handles stereo (N, 2)
   peak_dbtp = 20 * np.log10(true_peak(audio) + 1e-10)

   # Per-band RMS (crossover freqs match multiband compressor: 150, 600, 3000 Hz)
   mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
   crossovers = [150, 600, 3000]
   bands = []
   for i, fc in enumerate(crossovers + [None]):
       if i == 0:
           sos = butter(4, crossovers[0], btype='low', fs=sr, output='sos')
       elif fc is None:
           sos = butter(4, crossovers[-1], btype='high', fs=sr, output='sos')
       else:
           sos = butter(4, [crossovers[i-1], fc], btype='band', fs=sr, output='sos')
       band_signal = sosfilt(sos, mono)
       rms_db = 20 * np.log10(np.sqrt(np.mean(band_signal**2)) + 1e-10)
       bands.append(rms_db)
   band_rms_low, band_rms_mid, band_rms_highmid, band_rms_high = bands

   # Spectral centroid (mean across frames)
   spectral_centroid_hz = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))

   # SHA-256 of output file for reproducibility
   sha256 = hashlib.sha256(Path(output_mp3_path).read_bytes()).hexdigest()
   ```
7. **Validate control outputs** before continuing. After each control run, assert:
   - File size > 0 bytes
   - Duration > 30 seconds
   - LUFS > -40 (i.e., not near-silent)
   - True-peak < -0.1 dBTP (i.e., not clipped)

   If any check fails, log a warning and mark the pair as `control-invalid` in results.txt. Do not skip the enhanced/sweep variants — they may still produce useful data for comparison, but flag the control as suspect.
8. **Log results** to `REPO_ROOT / "mashupTests" / "results.txt"` in pipe-delimited format: `pair | mode | variant | status | duration_s | lufs | peak_dbtp | band_rms_low | band_rms_mid | band_rms_highmid | band_rms_high | spectral_centroid_hz | output_path | sha256`
9. **Log YouTube download metadata** for reproducibility. After each pair's download phase, write to `REPO_ROOT / "mashupTests" / "downloads.txt"`:
   ```
   pair | role | video_id | source_codec | source_bitrate_kbps | file_hash_sha256
   1-biggie-althea | song_a | eaPzCHEQExs | opus | 128 | abc123...
   1-biggie-althea | song_b | ZZNZgtj26Fk | opus | 128 | def456...
   ```
   The `file_hash_sha256` is computed from the downloaded WAV file before any pipeline processing. This allows verifying whether YouTube has re-encoded a video between test runs (which would change the hash and invalidate A/B comparisons).
10. **Clean up pipeline data** (`REPO_ROOT / "backend" / "data" / "stems" / session_id` and corresponding remixes dir) after copying output to save disk space (~500MB per session). Also clean up downloaded WAVs after both variants complete for a pair.

### Key implementation details

- **YouTube download caching:** Each pair downloads its two songs once and reuses the WAVs for both control and enhanced runs. This halves download time (10 songs downloaded instead of 20).
- **Flag toggling:** Each run is a subprocess with env vars set in the `subprocess.run(env=...)` argument. This is the ONLY correct approach — `config.py` instantiates `settings = Settings()` at module scope, so env vars set after import within the same process are ignored.
- **Path anchoring:** All paths are anchored to `REPO_ROOT = Path(__file__).resolve().parent.parent`. Output goes to `REPO_ROOT / "mashupTests"`, cleanup targets `REPO_ROOT / "backend" / "data" / "stems" / session_id`.
- **No local song files required:** The `examples/` directory is no longer needed for testing. Test pairs are fully defined by YouTube URLs.
- **Prompt:** All runs use an empty prompt (deterministic fallback). The test matrix's "Mix Intent" column is for human documentation only.
- **Timeouts:** Each `subprocess.run()` call uses `timeout=600` (10 minutes). `subprocess.TimeoutExpired` is caught, logged as a timed-out failure, and the suite continues to the next run. YouTube downloads have a separate timeout of 120 seconds per song.
- **Error handling:** If a download fails, log the error and skip the entire pair. If a pipeline run fails (non-zero exit or timeout), log the error and continue to the next run.
- **Flag inventory:** The complete set of `ab_*` flags to toggle (see flag matrix above):
  - Existing in `config.py`: `ab_control_day3`, `ab_autolvl_tune_v1`, `ab_vocal_makeup_v1`, `ab_mp3_export_path_v1`
  - From sound-quality-enhancement-plan (must be added to `config.py` first): `ab_per_stem_eq_v1`, `ab_resonance_detection_v1`, `ab_multiband_comp_v1`, `ab_static_mastering_v1`
- **Audio metrics:** After each successful run, measure integrated LUFS, true-peak (via `true_peak()` from `processor.py`), per-band RMS (crossover frequencies: 150, 600, 3000 Hz — matching the multiband compressor), and spectral centroid from the output MP3 using `pyloudnorm` + `librosa` + `scipy.signal`. Use `librosa.load(path, sr=None, mono=False)` and transpose the result before passing to `pyloudnorm` (librosa returns `(channels, samples)`, pyloudnorm expects `(samples, channels)`). Do NOT use `soundfile` — `libsndfile` cannot decode MP3. Compute SHA-256 hash of each output file for reproducibility. Log to `results.txt` in pipe-delimited format.
- **Control validation:** After each control run, validate: file size > 0, duration > 30s, LUFS > -40, peak < -0.1 dBTP. Failures are logged as `control-invalid` but do not skip subsequent variants for the pair.
- **Reproducibility:** Log YouTube download metadata (video ID, codec, bitrate, WAV file hash) to `downloads.txt`. This captures whether YouTube re-encoded a source between runs, which would invalidate A/B comparisons.

### .gitignore change

Add `mashupTests/` to the root `.gitignore` (alongside existing `notes/` and `examples/` entries).

## Verification

**`--mode compare` (default):**

1. Run from repo root: `uv run python scripts/run_ab_test_suite.py` (the script resolves all paths relative to its own location via `REPO_ROOT`)
2. Check `mashupTests/` has 5 subdirs, each with `control.mp3` and `enhanced.mp3`
3. Check `mashupTests/results.txt` shows 10 rows with pipe-delimited metrics (pair, mode, variant, status, duration, LUFS, peak, band RMS x4, spectral centroid, path, sha256)
4. Verify LUFS values are in a reasonable range (typically -20 to -8 LUFS; YouTube source material can produce lower values than CD-mastered content)
5. Verify control outputs pass sanity checks (file size > 0, duration > 30s, LUFS > -40, peak < -0.1 dBTP)
6. Verify files play correctly
7. Check `mashupTests/downloads.txt` contains download metadata for all 10 songs
8. Run again -- confirm previous outputs are cleared before new ones are written

**`--mode sweep` (recommended for diagnostic evaluation):**

1. Run from repo root: `uv run python scripts/run_ab_test_suite.py --mode sweep`
2. Check `mashupTests/` has 5 subdirs, each with `control.mp3` plus 4 `sweep-*.mp3` files
3. Check `mashupTests/results.txt` shows 25 rows (5 pairs x 5 runs each) with `mode` column set to `sweep`
4. Verify LUFS values are in a reasonable range (typically -20 to -8 LUFS)
5. Verify control outputs pass sanity checks (file size > 0, duration > 30s, LUFS > -40, peak < -0.1 dBTP)
6. Verify that sweep variants differ from control (different LUFS/peak/spectral values indicate the flag had an effect)
7. Compare per-band RMS and spectral centroid across sweep variants to identify which flags affect which frequency ranges
8. Check `mashupTests/downloads.txt` contains download metadata for all 10 songs
