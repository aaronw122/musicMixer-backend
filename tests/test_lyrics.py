"""Tests for lyrics lookup, LRC parsing, and bar mapping.

Step 8 of the lyrics lookup plan. Covers:
- Filename parsing (standard + edge cases)
- LRC parsing (timestamps, metadata filtering, offset)
- Bar mapping (beat_frames-based + BPM fallback)
- Plain lyrics section distribution
- Mocked syncedlyrics integration test
- Top-level lookup_lyrics_for_song function
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from musicmixer.models import LyricLine, LyricsData
from musicmixer.services.lyrics import (
    parse_filename,
    read_id3_tags,
    resolve_song_identity,
    fetch_lyrics,
    parse_lrc,
    parse_plain_lyrics,
    lookup_lyrics_for_song,
    MAX_LYRIC_LINES,
)


# ===========================================================================
# Filename parsing tests
# ===========================================================================

class TestParseFilename:
    def test_standard_format(self) -> None:
        """Standard 'Artist - Title.mp3' format."""
        result = parse_filename("The Notorious B.I.G. - Hypnotize.mp3")
        assert result == ("The Notorious B.I.G.", "Hypnotize")

    def test_with_official_audio_suffix(self) -> None:
        """Strip (Official Audio) suffix."""
        result = parse_filename("The Notorious B.I.G. - Hypnotize (Official Audio).mp3")
        assert result == ("The Notorious B.I.G.", "Hypnotize")

    def test_with_remaster_suffix(self) -> None:
        """Strip (2013 Remaster) suffix."""
        result = parse_filename("Grateful Dead - Althea (2013 Remaster).mp3")
        assert result == ("Grateful Dead", "Althea")

    def test_with_hd_bracket_suffix(self) -> None:
        """Strip [HD] bracket suffix."""
        result = parse_filename("Artist - Title [HD].mp3")
        assert result == ("Artist", "Title")

    def test_with_official_video_suffix(self) -> None:
        """Strip (Official Music Video) suffix."""
        result = parse_filename("Daft Punk - Around The World (Official Music Video).mp3")
        assert result == ("Daft Punk", "Around The World")

    def test_multiple_suffixes(self) -> None:
        """Strip multiple suffixes."""
        result = parse_filename("Artist - Title (Official Audio) [HD].mp3")
        assert result == ("Artist", "Title")

    def test_wav_extension(self) -> None:
        """Works with .wav files too."""
        result = parse_filename("Artist - Title.wav")
        assert result == ("Artist", "Title")

    def test_no_dash_returns_none(self) -> None:
        """No ' - ' separator returns None."""
        result = parse_filename("JustASongTitle.mp3")
        assert result is None

    def test_multiple_dashes(self) -> None:
        """Multiple dashes: split on first ' - '."""
        result = parse_filename("AC-DC - Back In Black - Remastered.mp3")
        assert result is not None
        assert result[0] == "AC-DC"
        assert result[1] == "Back In Black - Remastered"

    def test_unicode_characters(self) -> None:
        """Unicode characters in artist/title."""
        result = parse_filename("Bjork - Joga.mp3")
        assert result == ("Bjork", "Joga")

    def test_empty_string(self) -> None:
        """Empty string returns None."""
        result = parse_filename("")
        assert result is None

    def test_track_number_prefix(self) -> None:
        """Strip track number prefix."""
        result = parse_filename("01 - Artist - Title.mp3")
        assert result == ("Artist", "Title")

    def test_track_number_dot_prefix(self) -> None:
        """Strip '01. Artist - Title' prefix."""
        result = parse_filename("01. Artist - Title.mp3")
        assert result == ("Artist", "Title")

    def test_youtube_video_id(self) -> None:
        """Strip YouTube video ID at end of filename."""
        result = parse_filename("Artist - Title-dQw4w9WgXcQ.mp3")
        assert result == ("Artist", "Title")

    def test_underscore_separators(self) -> None:
        """Underscores treated as spaces."""
        result = parse_filename("The_Artist - Some_Title.mp3")
        assert result == ("The Artist", "Some Title")

    def test_lyrics_video_suffix(self) -> None:
        """Strip (Lyric Video) and (Lyrics Video) suffixes."""
        result = parse_filename("Artist - Title (Lyric Video).mp3")
        assert result == ("Artist", "Title")

    def test_live_suffix(self) -> None:
        """Strip (Live) and (Live at ...) suffixes."""
        result = parse_filename("Artist - Title (Live).mp3")
        assert result == ("Artist", "Title")


# ===========================================================================
# LRC parsing tests
# ===========================================================================

class TestParseLrc:
    def test_standard_format(self) -> None:
        """Standard [mm:ss.cc] format."""
        lrc = "[00:12.34]Hello world\n[00:15.67]Second line"
        lines = parse_lrc(lrc)
        assert len(lines) == 2
        assert lines[0].text == "Hello world"
        assert lines[0].timestamp_seconds == pytest.approx(12.34, abs=0.01)
        assert lines[1].text == "Second line"
        assert lines[1].timestamp_seconds == pytest.approx(15.67, abs=0.01)

    def test_millisecond_variant(self) -> None:
        """[mm:ss.ccc] millisecond format."""
        lrc = "[01:23.456]Three digit frac"
        lines = parse_lrc(lrc)
        assert len(lines) == 1
        assert lines[0].timestamp_seconds == pytest.approx(83.456, abs=0.001)

    def test_no_fractional_seconds(self) -> None:
        """[mm:ss] without fractional part."""
        lrc = "[02:30]No fraction"
        lines = parse_lrc(lrc)
        assert len(lines) == 1
        assert lines[0].timestamp_seconds == pytest.approx(150.0)

    def test_single_digit_minutes(self) -> None:
        """[m:ss.cc] single-digit minutes."""
        lrc = "[3:45.12]Single digit min"
        lines = parse_lrc(lrc)
        assert len(lines) == 1
        assert lines[0].timestamp_seconds == pytest.approx(225.12, abs=0.01)

    def test_metadata_lines_filtered(self) -> None:
        """Metadata lines like [ar:], [ti:], [al:] are filtered out."""
        lrc = (
            "[ar:Artist Name]\n"
            "[ti:Song Title]\n"
            "[al:Album Name]\n"
            "[offset:0]\n"
            "[00:05.00]Actual lyric line"
        )
        lines = parse_lrc(lrc)
        assert len(lines) == 1
        assert lines[0].text == "Actual lyric line"

    def test_offset_applied(self) -> None:
        """[offset:] value applied to all timestamps."""
        lrc = "[offset:500]\n[00:10.00]Shifted line"
        lines = parse_lrc(lrc)
        assert len(lines) == 1
        # offset 500ms = 0.5s, so 10.0 + 0.5 = 10.5
        assert lines[0].timestamp_seconds == pytest.approx(10.5)

    def test_negative_offset(self) -> None:
        """Negative [offset:] applied correctly."""
        lrc = "[offset:-200]\n[00:01.00]Early line"
        lines = parse_lrc(lrc)
        assert len(lines) == 1
        # offset -200ms = -0.2s, so 1.0 - 0.2 = 0.8
        assert lines[0].timestamp_seconds == pytest.approx(0.8)

    def test_offset_clamps_to_zero(self) -> None:
        """Offset that would make timestamp negative gets clamped to 0."""
        lrc = "[offset:-5000]\n[00:02.00]Should clamp"
        lines = parse_lrc(lrc)
        assert len(lines) == 1
        assert lines[0].timestamp_seconds == pytest.approx(0.0)

    def test_empty_lines_filtered(self) -> None:
        """Lines with only timestamps (no text) are filtered out."""
        lrc = "[00:05.00]\n[00:10.00]Has text"
        lines = parse_lrc(lrc)
        assert len(lines) == 1
        assert lines[0].text == "Has text"

    def test_multiple_timestamps_per_line(self) -> None:
        """Multiple timestamps on one line produce multiple LyricLine objects."""
        lrc = "[00:05.00][01:05.00]Repeated chorus line"
        lines = parse_lrc(lrc)
        assert len(lines) == 2
        assert all(l.text == "Repeated chorus line" for l in lines)
        assert lines[0].timestamp_seconds == pytest.approx(5.0)
        assert lines[1].timestamp_seconds == pytest.approx(65.0)

    def test_sorted_by_timestamp(self) -> None:
        """Output is sorted by timestamp even if input is not."""
        lrc = "[00:30.00]Later\n[00:05.00]Earlier"
        lines = parse_lrc(lrc)
        assert lines[0].text == "Earlier"
        assert lines[1].text == "Later"

    def test_empty_input(self) -> None:
        """Empty string returns empty list."""
        assert parse_lrc("") == []

    def test_single_digit_fractional(self) -> None:
        """[mm:ss.c] single-digit fractional."""
        lrc = "[00:10.5]Single frac digit"
        lines = parse_lrc(lrc)
        assert len(lines) == 1
        assert lines[0].timestamp_seconds == pytest.approx(10.5)


class TestParsePlainLyrics:
    def test_basic_plain_lyrics(self) -> None:
        """Basic plain lyrics parsing."""
        text = "Line one\nLine two\nLine three"
        lines = parse_plain_lyrics(text)
        assert len(lines) == 3
        assert lines[0].text == "Line one"
        assert lines[0].timestamp_seconds is None

    def test_blank_lines_filtered(self) -> None:
        """Blank lines are filtered out."""
        text = "Line one\n\n\nLine two\n   \nLine three"
        lines = parse_plain_lyrics(text)
        assert len(lines) == 3

    def test_empty_input(self) -> None:
        """Empty string returns empty list."""
        assert parse_plain_lyrics("") == []


# ===========================================================================
# Mocked syncedlyrics integration tests
# ===========================================================================

class TestFetchLyrics:
    @patch("musicmixer.services.lyrics.syncedlyrics")
    def test_synced_lyrics_found(self, mock_syncedlyrics: MagicMock) -> None:
        """Synced lyrics returned from syncedlyrics."""
        mock_syncedlyrics.search.return_value = (
            "[00:05.00]First line\n[00:10.00]Second line"
        )
        result = fetch_lyrics("Artist", "Title")
        assert result is not None
        raw_text, is_synced = result
        assert is_synced is True
        assert "[00:05.00]" in raw_text
        mock_syncedlyrics.search.assert_called_once_with("Artist Title")

    @patch("musicmixer.services.lyrics.syncedlyrics")
    def test_plain_lyrics_found(self, mock_syncedlyrics: MagicMock) -> None:
        """Plain lyrics (no timestamps) returned."""
        mock_syncedlyrics.search.return_value = "First line\nSecond line"
        result = fetch_lyrics("Artist", "Title")
        assert result is not None
        raw_text, is_synced = result
        assert is_synced is False

    @patch("musicmixer.services.lyrics.syncedlyrics")
    def test_no_lyrics_found(self, mock_syncedlyrics: MagicMock) -> None:
        """None returned when no lyrics found."""
        mock_syncedlyrics.search.return_value = None
        result = fetch_lyrics("Artist", "Title")
        assert result is None

    @patch("musicmixer.services.lyrics.syncedlyrics")
    def test_empty_string_result(self, mock_syncedlyrics: MagicMock) -> None:
        """Empty string treated as no lyrics (review note: use 'if not result:')."""
        mock_syncedlyrics.search.return_value = ""
        result = fetch_lyrics("Artist", "Title")
        assert result is None

    @patch("musicmixer.services.lyrics.syncedlyrics")
    def test_exception_handled(self, mock_syncedlyrics: MagicMock) -> None:
        """Exception from syncedlyrics is caught gracefully."""
        mock_syncedlyrics.search.side_effect = Exception("Network error")
        result = fetch_lyrics("Artist", "Title")
        assert result is None

    def test_empty_query(self) -> None:
        """Empty artist and title returns None without calling syncedlyrics."""
        result = fetch_lyrics("", "")
        assert result is None

    @patch("musicmixer.services.lyrics.syncedlyrics")
    def test_artist_only_query(self, mock_syncedlyrics: MagicMock) -> None:
        """Title-only query (empty artist) works."""
        mock_syncedlyrics.search.return_value = "Some lyrics"
        result = fetch_lyrics("", "Just a Title")
        assert result is not None
        mock_syncedlyrics.search.assert_called_once_with("Just a Title")


# ===========================================================================
# Top-level function tests
# ===========================================================================

class TestLookupLyricsForSong:
    @patch("musicmixer.services.lyrics.fetch_lyrics")
    def test_with_parseable_filename(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        """Successful lookup with parseable filename."""
        audio_file = tmp_path / "song.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_fetch.return_value = (
            "[00:05.00]Hello\n[00:10.00]World",
            True,
        )

        result = lookup_lyrics_for_song(
            audio_path=audio_file,
            original_filename="Biggie - Hypnotize (Official Audio).mp3",
        )

        assert result is not None
        assert result.artist == "Biggie"
        assert result.title == "Hypnotize"
        assert result.is_synced is True
        assert len(result.lines) == 2
        assert result.source == "filename"
        assert result.lookup_duration_ms > 0
        mock_fetch.assert_called_once_with("Biggie", "Hypnotize")

    @patch("musicmixer.services.lyrics.fetch_lyrics")
    def test_no_lyrics_found(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        """Returns None when no lyrics found."""
        audio_file = tmp_path / "song.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_fetch.return_value = None

        result = lookup_lyrics_for_song(
            audio_path=audio_file,
            original_filename="Artist - Title.mp3",
        )
        assert result is None

    @patch("musicmixer.services.lyrics.fetch_lyrics")
    def test_plain_lyrics(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        """Plain lyrics parsed correctly."""
        audio_file = tmp_path / "song.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_fetch.return_value = ("Line one\nLine two\nLine three", False)

        result = lookup_lyrics_for_song(
            audio_path=audio_file,
            original_filename="Artist - Title.mp3",
        )

        assert result is not None
        assert result.is_synced is False
        assert len(result.lines) == 3
        assert all(l.timestamp_seconds is None for l in result.lines)

    def test_unidentifiable_song(self, tmp_path: Path) -> None:
        """Returns None if song cannot be identified."""
        audio_file = tmp_path / "song.mp3"
        audio_file.write_bytes(b"fake audio")

        result = lookup_lyrics_for_song(
            audio_path=audio_file,
            original_filename="",
        )
        assert result is None

    @patch("musicmixer.services.lyrics.fetch_lyrics")
    def test_caps_lines_at_max(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        """Lines capped at MAX_LYRIC_LINES."""
        audio_file = tmp_path / "song.mp3"
        audio_file.write_bytes(b"fake audio")

        # Generate 100 synced lyric lines
        lrc_lines = [f"[00:{i:02d}.00]Line {i}" for i in range(100)]
        mock_fetch.return_value = ("\n".join(lrc_lines), True)

        result = lookup_lyrics_for_song(
            audio_path=audio_file,
            original_filename="Artist - Title.mp3",
        )

        assert result is not None
        assert len(result.lines) == MAX_LYRIC_LINES

    @patch("musicmixer.services.lyrics.fetch_lyrics")
    def test_exception_returns_none(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        """Any unexpected exception returns None gracefully."""
        audio_file = tmp_path / "song.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_fetch.side_effect = RuntimeError("Unexpected error")

        result = lookup_lyrics_for_song(
            audio_path=audio_file,
            original_filename="Artist - Title.mp3",
        )
        assert result is None


# ===========================================================================
# Identity resolver tests
# ===========================================================================

class TestResolveSongIdentity:
    def test_filename_priority(self, tmp_path: Path) -> None:
        """Filename parsing takes priority over ID3."""
        audio_file = tmp_path / "song.mp3"
        audio_file.write_bytes(b"fake audio")

        result = resolve_song_identity(audio_file, "Artist - Title.mp3")
        assert result is not None
        assert result == ("Artist", "Title", "filename")

    @patch("musicmixer.services.lyrics.read_id3_tags")
    def test_id3_fallback(self, mock_id3: MagicMock, tmp_path: Path) -> None:
        """Falls back to ID3 if filename unparseable."""
        audio_file = tmp_path / "song.mp3"
        audio_file.write_bytes(b"fake audio")
        mock_id3.return_value = ("ID3 Artist", "ID3 Title")

        result = resolve_song_identity(audio_file, "unparseable_filename.mp3")
        assert result is not None
        assert result == ("ID3 Artist", "ID3 Title", "id3")

    @patch("musicmixer.services.lyrics.read_id3_tags")
    def test_stem_fallback(self, mock_id3: MagicMock, tmp_path: Path) -> None:
        """Falls back to cleaned filename stem if no ID3 and no dash."""
        audio_file = tmp_path / "song.mp3"
        audio_file.write_bytes(b"fake audio")
        mock_id3.return_value = None

        result = resolve_song_identity(audio_file, "SomeSongName.mp3")
        assert result is not None
        assert result == ("", "SomeSongName", "filename")

    @patch("musicmixer.services.lyrics.read_id3_tags")
    def test_empty_filename_no_id3(self, mock_id3: MagicMock, tmp_path: Path) -> None:
        """Returns None if no filename and no ID3 tags."""
        audio_file = tmp_path / "song.mp3"
        audio_file.write_bytes(b"fake audio")
        mock_id3.return_value = None

        result = resolve_song_identity(audio_file, "")
        assert result is None
