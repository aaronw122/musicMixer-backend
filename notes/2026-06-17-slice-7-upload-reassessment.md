# Slice 7 — Upload Pipeline Extraction Reassessment

**Date:** 2026-06-17
**Branch:** `feat/remix-orchestration-refactor/slice-7-upload-reassess` (off integration, Slices 0–6)
**Decision: No further extraction warranted. Note-only deliverable.**

## What was reviewed

Read end to end in `backend/src/musicmixer/api/remix.py`:

- `_pipeline_wrapper` (lines ~87–142) — the uploaded-files background wrapper
- `create_remix` (lines ~708–806) — the `POST /remix` upload route
- For context: `create_youtube_remix` (lines ~599–704) and the Slice 2 helper module `services/remix_stages.py`

## Why no further extraction is warranted

The plan (Slice 7) hypothesizes that the only likely useful extraction on the upload
path is route-side upload validation/write/probe helpers — and notes those were
**already extracted in Slice 2** into `services/remix_stages.py`. That hypothesis holds.

### `create_remix` already uses every Slice 2 helper

The upload route imports and uses all five Slice 2 helpers, with no duplicated inline
logic remaining:

| Helper | Use site in `create_remix` |
|---|---|
| `extension_allowed` | extension validation loop |
| `upload_extension` | error detail + dest filename suffix |
| `write_upload_file` | streaming write with size cap |
| `UploadTooLargeError` | caught → translated to HTTP 413 |
| `probe_duration` | ffprobe duration validation |

What remains inline in `create_remix` is exactly route-owned orchestration that the
plan's Non-Goals say must stay in `api/remix.py`:

- `SessionState` creation + registration under `sessions_lock`
- `_enqueue_or_start` (queue/processing-lock ownership)
- HTTP exception shaping (422 invalid type, 413 too-large / too-long)
- the `run_fn` closure binding `_pipeline_wrapper`

Moving any of that would relocate session/queue ownership out of the route — explicitly
forbidden by the plan.

### `_pipeline_wrapper` is already the minimal orchestration shell

It is short and already delegates the expensive work to `run_pipeline`. Everything it
still owns is what the plan says it must keep:

- `session.status` transitions (`processing` / `cancelled` / `error`)
- terminal `cancelled` / `error` SSE event emission (via `emit_progress`)
- processing-lock release + `_process_next_queued` in `finally`

The only "DRY" opportunities visible here are the hand-built `{step, detail, progress}`
error/cancel dicts and a plain-error helper — but those are **Slice 8** work
(C7 plain-error helper + C15 `progress_event` builder), explicitly out of scope for
Slice 7. Manufacturing that change now would also create a needless rebase conflict
with Slice 8.

C3 (atomic writes), C4 (stem constants), and C14 (docstring cleanup) were not touched.

## Confirmation

- The upload route already uses the Slice 2 helpers; no inline upload
  validation/write/probe logic was duplicated.
- Queue / SSE / session / processing-lock ownership remains entirely in `api/remix.py`
  and was untouched.
- No code changed. This note is the deliverable.

## Tests

`uv run pytest tests/test_api.py tests/test_pipeline_sse.py tests/test_youtube_endpoint.py`
→ **65 passed, 1 failed**.

The single failure is `tests/test_pipeline_sse.py::TestPostRemixAsync::test_mixed_endpoints_share_global_capacity_gate`,
the known brittle/order-dependent test that fails on `main` and in isolation as well —
unrelated to this slice (no code was changed). Not fixed, per plan guidance.
