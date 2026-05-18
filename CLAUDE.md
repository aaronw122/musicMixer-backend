# musicMixer -- Backend Service

Python API and audio processing pipeline. Accepts two songs, separates them into stems via cloud GPU, combines selected stems into a remix, and serves the result.

Parent workspace CLAUDE.md (`../CLAUDE.md`) covers shared conventions (safety rules, documentation hierarchy, testing philosophy, self-improvement). This file covers backend-specific details only.

## Repository Structure

```
backend/
  pyproject.toml
  .python-version          # Pin: Python 3.11
  .env                     # Local config overrides (gitignored)
  .gitignore
  CLAUDE.md
  src/
    musicmixer/
      __init__.py
      main.py              # FastAPI app, CORS, lifespan, static mount
      config.py            # Pydantic BaseSettings + .env loading
      api/
        __init__.py
        health.py          # GET /health
        remix.py           # POST /api/remix, GET /api/remix/{id}/audio
      services/
        __init__.py
        separation.py      # Backend-agnostic dispatcher (modal vs local)
        separation_modal.py  # Modal cloud GPU (BS-RoFormer 6-stem)
        separation_local.py  # Local fallback (htdemucs_ft 4-stem)
        mixer.py           # Stem overlay + MP3 export via ffmpeg
        pipeline_day1.py   # Synchronous pipeline (Day 1 only)
  static/
    index.html             # Minimal test UI (replaced by React frontend later)
  data/                    # Gitignored, created at runtime
    uploads/               # Raw uploaded MP3/WAV files
    stems/                 # Separated stem WAVs (~240MB per song)
    remixes/               # Final mixed MP3 output
```

## Tech Stack

- **Python 3.11** -- pinned in `.python-version`
- **FastAPI** -- web framework
- **Pydantic Settings** -- config via `.env` file
- **Modal** -- cloud GPU for stem separation (BS-RoFormer)
- **audio-separator** -- stem separation library (runs inside Modal container)
- **numpy + soundfile** -- audio processing (float32 throughout)
- **ffmpeg** (subprocess) -- MP3 export
- **No database** -- file-based storage in `data/`

## Package Manager

**Use `uv`, NOT pip, NOT poetry.**

```bash
uv add <package>           # Add dependency
uv add -d <package>        # Add dev dependency
uv remove <package>        # Remove dependency
uv run <command>           # Run command in project venv
uv sync                    # Install all dependencies from lockfile
```

## Running the Dev Server

```bash
cd /Users/aaron/Projects/musicMixer/backend
uv run uvicorn musicmixer.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health` should return `{"status":"ok"}`

## System Dependencies

These must be installed on the host (not managed by uv):

```bash
brew install ffmpeg libsndfile rubberband
```

- **ffmpeg** -- required for MP3 export. Verify: `ffmpeg -version`
- **libsndfile** -- required by soundfile. Verify: `uv run python -c "import soundfile"`
- **rubberband** -- required for tempo stretching (Day 2+). Verify: `rubberband --version` (need v3.x)

## Config

All config lives in `config.py` via Pydantic `BaseSettings`. Override any setting in `.env`:

| Setting | Default | Notes |
|---------|---------|-------|
| `stem_backend` | `"modal"` | `"modal"` for cloud GPU, `"local"` for htdemucs_ft |
| `data_dir` | `"data"` | Runtime storage root |
| `max_file_size_mb` | `50` | Per-file upload limit |
| `allowed_extensions` | `.mp3, .wav` | Validated on upload |
| `output_bitrate` | `"320k"` | MP3 export bitrate |
| `cors_origins` | `["http://localhost:5173"]` | Allowed CORS origins |

## API Patterns

- **Health:** `GET /health` returns `{"status":"ok"}`
- **API prefix:** all business endpoints under `/api/`
- **Create remix:** `POST /api/remix` -- multipart form with `song_a`, `song_b` (files), `prompt` (text). Returns `{"session_id": "<uuid>"}`
- **Get audio:** `GET /api/remix/{session_id}/audio` -- serves rendered MP3
- **No auth** -- single-user proof of concept
- **Day 1 is synchronous** -- POST blocks until remix completes. Day 2 adds async + SSE progress.

## Stem Separation Models

Two backends, two models. Controlled by `STEM_BACKEND` in `.env`:

| | Modal (cloud GPU) | Local (CPU) |
|---|---|---|
| **Env var** | `STEM_BACKEND=modal` (default) | `STEM_BACKEND=local` |
| **Model** | BS-Roformer-SW (`BS-Roformer-SW.ckpt` by jarredou) | htdemucs_ft |
| **Stems** | 6: vocals, drums, bass, guitar, piano, other | 4: vocals, drums, bass, other |
| **Speed** | ~1 min/song (~2 min for 2 songs in parallel) | 10-20 min/song |
| **Requires** | Modal account + token (`uv run modal setup`) | Just CPU + RAM |

Day 1 separates **both songs sequentially**, so double the single-song time.

**Switching:** Set `STEM_BACKEND=local` in `backend/.env` for local fallback. No code changes needed — `separation.py` dispatches automatically.

**Important:** These are *stem separation* models (splitting a song into parts). Audio *analysis* (BPM, key, energy) uses different libraries (librosa/essentia) and is not implemented until Day 2+.

## Expected Processing Times

| Operation | Modal | Local CPU | Notes |
|-----------|-------|-----------|-------|
| Stem separation (1 song) | 3-5 min | 10-20 min | First Modal run adds 60-90s cold start |
| Stem separation (2 songs) | 6-10 min | 20-40 min | Sequential in Day 1 |
| Mixing + export | <10 sec | <10 sec | CPU-bound, fast |
| Full pipeline (Day 1) | ~7-12 min | ~20-40 min | Upload → stems → mix → MP3 |

If processing seems stuck, check logs for progress. Stem separation produces no output until complete — long silences are normal.

## LLM Integration Status

**Day 1: No LLM.** The `prompt` field is accepted by the API but ignored. Stem selection is hardcoded: vocals from Song A, instrumentals from Song B. LLM-driven mix decisions come in Day 3.

## Modal Setup

1. Create a Modal account at modal.com
2. `uv add modal` (already in dependencies)
3. `uv run modal setup` (authenticates via browser)
4. Verify: `uv run modal token list`
5. Ensure `STEM_BACKEND=modal` in `.env` (or remove the line — modal is default)

**Fallback:** If Modal is not configured, set `STEM_BACKEND=local` in `.env` to use local CPU separation.

## Key Conventions

- All audio processing uses **float32** throughout. Never convert to int16 mid-pipeline.
- Stem separation returns `dict[str, Path]` mapping stem name to WAV file path.
- The separation dispatcher (`separation.py`) abstracts away the backend choice -- downstream code does not care whether Modal or local was used.
- Logging uses stdlib `logging` (structured logging via `structlog` comes later).

## Testing

No test suite yet (coming Day 4). For now, manual testing:

```bash
# Health check
curl http://localhost:8000/health

# Upload and remix (synchronous, takes 1-2 min)
curl -X POST http://localhost:8000/api/remix \
  -F "song_a=@/path/to/song_a.mp3" \
  -F "song_b=@/path/to/song_b.mp3" \
  -F "prompt=test"

# Fetch the remix audio
curl http://localhost:8000/api/remix/<session_id>/audio --output remix.mp3
```

## Background Jobs

Coming in Day 2. Day 1 pipeline is fully synchronous (POST blocks until done).

## File Watcher (--reload)

`uv run dev` starts uvicorn with `--reload`, which restarts the server on any file change in `backend/`. This is useful for development but dangerous when agents are editing files:

- Agent edits trigger restarts mid-request, dropping active connections
- Multiple rapid edits cause restart loops
- This is the root cause of the "zombie agent" problem

**When agents are working on backend files:** Use `uv run uvicorn musicmixer.main:app --port 8000` (without `--reload`) for stable operation. Or run `/dev --stop` first, let agents finish, then restart.

## Common Gotchas

1. **Do NOT use pydub for mixing or export.** Pydub quantizes to 16-bit integers internally, destroying float32 headroom. Sum stems in numpy, export via ffmpeg subprocess.

2. **StaticFiles mount must come LAST in `main.py`.** If mounted before API routes, it swallows `/api/*` requests. The mount order in `main.py` is: include routers first, then `app.mount("/", StaticFiles(...))`.

3. **BS-RoFormer checkpoint matters.** Use `BS-Roformer-SW.ckpt` (by jarredou) for 6-stem separation. The `Viperx-1297` / `12.9755` checkpoints are 2-stem only (vocals + instrumental).

4. **Validate float32 WAV from separation.** Use `subtype='FLOAT'` when writing with soundfile. Default may produce PCM_16.

5. **ffmpeg must be on PATH.** The mixer calls ffmpeg as a subprocess. If missing, you get a cryptic error. Check with `ffmpeg -version`.

6. **Each session produces ~500MB of stem data** (6 stems x 40MB x 2 songs + uploads + remix). Clean `data/` between test runs: `rm -rf data/stems/* data/uploads/* data/remixes/*`

7. **Local fallback (htdemucs_ft) returns 4 stems, not 6.** Missing `guitar` and `piano` are set to `None`. Downstream code (mixer) must skip `None` stems.

8. **Modal container cold starts** can take 60-90s on first run. Subsequent runs are faster. The pipeline timeout is 300s to accommodate this.

## PulseMap Analysis Setup

WhisperX (word alignment) requires a one-time model pre-download:

```bash
cd backend
uv run python -c "import whisperx; whisperx.load_model('base', 'cpu', compute_type='int8', language='en')"
```

This downloads the Whisper base model, pyannote VAD model, and wav2vec2 alignment model (~1.5GB total). Without this, word alignment will fail on first run.

**Torch 2.6+ compatibility:** The pyannote VAD checkpoint uses `omegaconf` globals that torch's `weights_only=True` default blocks. This is handled automatically in `pulsemap.py` via `torch.serialization.add_safe_globals()`.

## Lessons Learned

- **Kill background agents before `uv run dev`.** Agents writing files in the backend dir trigger `--reload` restart loops. See "File Watcher" section above. Quick check: `pgrep -lf 'claude -p|codex exec'`
- **Verify port is free before starting server.** Zombie processes hold ports after `Ctrl+C`. Check: `lsof -i :8000`. Kill: `kill $(lsof -i :8000 -t)`
- **No `.env` = Modal default.** Without `STEM_BACKEND=local` in `.env`, the server tries Modal (hangs if unconfigured).
- **Long silences during separation are normal.** Stem separation produces no intermediate output. Don't assume it's stuck until you've exceeded the expected times (see "Expected Processing Times" above).
- **Each session produces ~500MB of stem data.** Clean between test runs: `rm -rf data/stems/* data/uploads/* data/remixes/*`
