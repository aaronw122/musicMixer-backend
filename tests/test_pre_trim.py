"""Tests for pre_trim_for_processing() in musicmixer.services.processor.

All subprocess calls (ffprobe, ffmpeg) are mocked -- no real binaries needed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from musicmixer.services.processor import pre_trim_for_processing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_file(tmp_path: Path, name: str = "song.mp3") -> Path:
    """Create a small dummy file for testing."""
    p = tmp_path / name
    p.write_bytes(b"\x00" * 128)
    return p


def _ffprobe_ok(duration: float) -> MagicMock:
    """Return a CompletedProcess mock simulating successful ffprobe."""
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = 0
    result.stdout = f"{duration}\n"
    result.stderr = ""
    return result


def _ffprobe_fail(rc: int = 1) -> MagicMock:
    """Return a CompletedProcess mock simulating failed ffprobe."""
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = rc
    result.stdout = ""
    result.stderr = "error"
    return result


def _silence_detect_result(silence_end: float | None = None) -> MagicMock:
    """Return a CompletedProcess mock simulating ffmpeg silencedetect."""
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = 0
    result.stdout = ""
    if silence_end is not None:
        result.stderr = (
            f"[silencedetect @ 0x1234] silence_end: {silence_end} "
            f"| silence_duration: {silence_end}\n"
        )
    else:
        result.stderr = ""
    return result


def _trim_ok() -> MagicMock:
    """Return a CompletedProcess mock simulating successful ffmpeg trim."""
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = 0
    result.stdout = ""
    result.stderr = ""
    return result


def _trim_fail(rc: int = 1) -> MagicMock:
    """Return a CompletedProcess mock simulating failed ffmpeg trim."""
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = rc
    result.stdout = ""
    result.stderr = "ffmpeg error output"
    return result


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestShortFileNoOp:
    """File under max duration -- function returns without trimming."""

    @patch("musicmixer.services.processor.subprocess.run")
    def test_short_file_returns_original_path(self, mock_run: MagicMock, tmp_path: Path) -> None:
        audio = _make_dummy_file(tmp_path)
        mock_run.return_value = _ffprobe_ok(duration=120.0)

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        assert result == audio
        # Only ffprobe should be called -- no silence detect, no trim
        assert mock_run.call_count == 1

    @patch("musicmixer.services.processor.subprocess.run")
    def test_exact_max_duration_is_noop(self, mock_run: MagicMock, tmp_path: Path) -> None:
        audio = _make_dummy_file(tmp_path)
        mock_run.return_value = _ffprobe_ok(duration=210.0)

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        assert result == audio
        assert mock_run.call_count == 1


class TestLongFileGetsTrimmed:
    """File over max duration -- verify ffmpeg is called with correct args."""

    @patch("shutil.move")
    @patch("musicmixer.services.processor.subprocess.run")
    def test_long_file_triggers_trim(
        self, mock_run: MagicMock, mock_move: MagicMock, tmp_path: Path
    ) -> None:
        audio = _make_dummy_file(tmp_path)

        # Call sequence: ffprobe -> silencedetect -> trim
        mock_run.side_effect = [
            _ffprobe_ok(duration=300.0),
            _silence_detect_result(silence_end=None),  # no silence
            _trim_ok(),
        ]

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        assert result == audio
        assert mock_run.call_count == 3

        # Verify the trim call (3rd call) has correct args
        trim_call_args = mock_run.call_args_list[2][0][0]
        assert trim_call_args[0] == "ffmpeg"
        assert "-y" in trim_call_args
        assert "-ss" in trim_call_args
        assert "-t" in trim_call_args
        assert "-c" in trim_call_args
        assert "copy" in trim_call_args

        # Verify -t value is the max_duration_seconds
        t_idx = trim_call_args.index("-t")
        assert trim_call_args[t_idx + 1] == "210.0"

        # Verify -ss offset is 0.0 (no silence detected)
        ss_idx = trim_call_args.index("-ss")
        assert trim_call_args[ss_idx + 1] == "0.0"

    @patch("shutil.move")
    @patch("musicmixer.services.processor.subprocess.run")
    def test_trim_replaces_original_via_shutil_move(
        self, mock_run: MagicMock, mock_move: MagicMock, tmp_path: Path
    ) -> None:
        audio = _make_dummy_file(tmp_path)

        mock_run.side_effect = [
            _ffprobe_ok(duration=400.0),
            _silence_detect_result(silence_end=None),
            _trim_ok(),
        ]

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        # shutil.move should be called to replace original with trimmed file
        assert mock_move.call_count == 1
        move_args = mock_move.call_args[0]
        # Second arg should be the original audio path
        assert move_args[1] == str(audio)


class TestFfprobeFailure:
    """ffprobe returns non-zero or raises -- function returns original path gracefully."""

    @patch("musicmixer.services.processor.subprocess.run")
    def test_ffprobe_nonzero_returns_original(self, mock_run: MagicMock, tmp_path: Path) -> None:
        audio = _make_dummy_file(tmp_path)
        mock_run.return_value = _ffprobe_fail(rc=1)

        result = pre_trim_for_processing(audio)

        assert result == audio
        assert mock_run.call_count == 1

    @patch("musicmixer.services.processor.subprocess.run")
    def test_ffprobe_timeout_returns_original(self, mock_run: MagicMock, tmp_path: Path) -> None:
        audio = _make_dummy_file(tmp_path)
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

        result = pre_trim_for_processing(audio)

        assert result == audio

    @patch("musicmixer.services.processor.subprocess.run")
    def test_ffprobe_not_found_returns_original(self, mock_run: MagicMock, tmp_path: Path) -> None:
        audio = _make_dummy_file(tmp_path)
        mock_run.side_effect = FileNotFoundError("ffprobe not found")

        result = pre_trim_for_processing(audio)

        assert result == audio

    @patch("musicmixer.services.processor.subprocess.run")
    def test_ffprobe_invalid_output_returns_original(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """ffprobe returns rc=0 but non-numeric output -> ValueError -> returns original."""
        audio = _make_dummy_file(tmp_path)
        result_mock = MagicMock(spec=subprocess.CompletedProcess)
        result_mock.returncode = 0
        result_mock.stdout = "N/A\n"
        result_mock.stderr = ""
        mock_run.return_value = result_mock

        result = pre_trim_for_processing(audio)

        assert result == audio


class TestFfmpegTrimFailure:
    """ffmpeg trim subprocess fails -- returns original path gracefully."""

    @patch("musicmixer.services.processor.subprocess.run")
    def test_trim_nonzero_returns_original(self, mock_run: MagicMock, tmp_path: Path) -> None:
        audio = _make_dummy_file(tmp_path)

        mock_run.side_effect = [
            _ffprobe_ok(duration=300.0),
            _silence_detect_result(silence_end=None),
            _trim_fail(rc=1),
        ]

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        assert result == audio
        assert mock_run.call_count == 3

    @patch("musicmixer.services.processor.subprocess.run")
    def test_trim_timeout_returns_original(self, mock_run: MagicMock, tmp_path: Path) -> None:
        audio = _make_dummy_file(tmp_path)

        mock_run.side_effect = [
            _ffprobe_ok(duration=300.0),
            _silence_detect_result(silence_end=None),
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120),
        ]

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        assert result == audio

    @patch("musicmixer.services.processor.subprocess.run")
    def test_trim_file_not_found_returns_original(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        audio = _make_dummy_file(tmp_path)

        mock_run.side_effect = [
            _ffprobe_ok(duration=300.0),
            _silence_detect_result(silence_end=None),
            FileNotFoundError("ffmpeg not found"),
        ]

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        assert result == audio


class TestLeadingSilenceSkipped:
    """Verify silence detection output is parsed and offset is applied."""

    @patch("shutil.move")
    @patch("musicmixer.services.processor.subprocess.run")
    def test_silence_end_used_as_offset(
        self, mock_run: MagicMock, mock_move: MagicMock, tmp_path: Path
    ) -> None:
        audio = _make_dummy_file(tmp_path)

        mock_run.side_effect = [
            _ffprobe_ok(duration=300.0),
            _silence_detect_result(silence_end=3.5),
            _trim_ok(),
        ]

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        assert result == audio

        # Verify the trim call uses 3.5 as the -ss offset
        trim_call_args = mock_run.call_args_list[2][0][0]
        ss_idx = trim_call_args.index("-ss")
        assert trim_call_args[ss_idx + 1] == "3.5"

    @patch("shutil.move")
    @patch("musicmixer.services.processor.subprocess.run")
    def test_no_silence_detected_uses_zero_offset(
        self, mock_run: MagicMock, mock_move: MagicMock, tmp_path: Path
    ) -> None:
        audio = _make_dummy_file(tmp_path)

        mock_run.side_effect = [
            _ffprobe_ok(duration=300.0),
            _silence_detect_result(silence_end=None),  # no silence_end in output
            _trim_ok(),
        ]

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        # Verify -ss is 0.0
        trim_call_args = mock_run.call_args_list[2][0][0]
        ss_idx = trim_call_args.index("-ss")
        assert trim_call_args[ss_idx + 1] == "0.0"

    @patch("shutil.move")
    @patch("musicmixer.services.processor.subprocess.run")
    def test_silence_detect_failure_still_trims_from_start(
        self, mock_run: MagicMock, mock_move: MagicMock, tmp_path: Path
    ) -> None:
        """If silence detection raises, we still trim but from offset 0."""
        audio = _make_dummy_file(tmp_path)

        mock_run.side_effect = [
            _ffprobe_ok(duration=300.0),
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120),  # silence detect fails
            _trim_ok(),
        ]

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        assert result == audio
        # Trim should still happen (3rd call) with offset 0
        assert mock_run.call_count == 3
        trim_call_args = mock_run.call_args_list[2][0][0]
        ss_idx = trim_call_args.index("-ss")
        assert trim_call_args[ss_idx + 1] == "0.0"


class TestOffsetCappedAt10s:
    """If silence_end is >10s, offset is capped at 10."""

    @patch("shutil.move")
    @patch("musicmixer.services.processor.subprocess.run")
    def test_large_silence_end_capped_at_10(
        self, mock_run: MagicMock, mock_move: MagicMock, tmp_path: Path
    ) -> None:
        audio = _make_dummy_file(tmp_path)

        mock_run.side_effect = [
            _ffprobe_ok(duration=300.0),
            _silence_detect_result(silence_end=25.0),  # way beyond 10s
            _trim_ok(),
        ]

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        # Verify -ss is capped at 10.0, not 25.0
        trim_call_args = mock_run.call_args_list[2][0][0]
        ss_idx = trim_call_args.index("-ss")
        assert trim_call_args[ss_idx + 1] == "10.0"

    @patch("shutil.move")
    @patch("musicmixer.services.processor.subprocess.run")
    def test_exactly_10s_silence_not_capped(
        self, mock_run: MagicMock, mock_move: MagicMock, tmp_path: Path
    ) -> None:
        audio = _make_dummy_file(tmp_path)

        mock_run.side_effect = [
            _ffprobe_ok(duration=300.0),
            _silence_detect_result(silence_end=10.0),
            _trim_ok(),
        ]

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        trim_call_args = mock_run.call_args_list[2][0][0]
        ss_idx = trim_call_args.index("-ss")
        assert trim_call_args[ss_idx + 1] == "10.0"

    @patch("shutil.move")
    @patch("musicmixer.services.processor.subprocess.run")
    def test_silence_under_10s_used_as_is(
        self, mock_run: MagicMock, mock_move: MagicMock, tmp_path: Path
    ) -> None:
        audio = _make_dummy_file(tmp_path)

        mock_run.side_effect = [
            _ffprobe_ok(duration=300.0),
            _silence_detect_result(silence_end=7.5),
            _trim_ok(),
        ]

        result = pre_trim_for_processing(audio, max_duration_seconds=210.0)

        trim_call_args = mock_run.call_args_list[2][0][0]
        ss_idx = trim_call_args.index("-ss")
        assert trim_call_args[ss_idx + 1] == "7.5"
