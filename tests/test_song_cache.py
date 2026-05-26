"""Tests for musicmixer.services.song_cache — Redis-backed per-song cache."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicmixer.models import (
    AudioMetadata,
    ChordEvent,
    ChordProgression,
    DrumPattern,
    EnergyBuckets,
    LyricLine,
    LyricsData,
    PolyphonyInfo,
    SectionInfo,
    SongStructure,
    StemAnalysis,
    VocalGap,
    WordAlignment,
    WordEvent,
)
from musicmixer.services.song_cache import (
    ROLE_INSTRUMENTAL,
    ROLE_VOCAL,
    _MIN_INSTRUMENTAL_STEMS,
    _MIN_VOCAL_STEMS,
    _deserialize_audio_metadata,
    _deserialize_lyrics,
    _get_redis,
    _meta_key,
    _serialize_audio_metadata,
    _serialize_lyrics,
    _stems_key,
    cache_song_metadata,
    cache_song_stems,
    get_cached_song,
    get_cached_stems,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STEM_NAMES = ["vocals", "drums", "bass", "guitar", "piano", "other"]
VOCAL_STEM_NAMES = ["lead_vocals", "backing_vocals", "instrumental"]


def _make_wav(path: Path, duration: float = 0.1, sr: int = 44100) -> None:
    """Write a minimal float32 WAV file."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sr, format="WAV", subtype="FLOAT")


def _make_stem_dir(base: Path) -> Path:
    """Create a directory with 6 minimal stem WAVs (instrumental role)."""
    base.mkdir(parents=True, exist_ok=True)
    for name in STEM_NAMES:
        _make_wav(base / f"{name}.wav")
    return base


def _make_vocal_stem_dir(base: Path) -> Path:
    """Create a directory with 3 minimal stem WAVs (vocal role)."""
    base.mkdir(parents=True, exist_ok=True)
    for name in VOCAL_STEM_NAMES:
        _make_wav(base / f"{name}.wav")
    return base


def _make_audio_metadata() -> AudioMetadata:
    """Build a fully-populated AudioMetadata for testing serialization."""
    return AudioMetadata(
        bpm=120.0,
        bpm_confidence=0.92,
        beat_frames=np.array([1024, 2048, 3072, 4096]),
        duration_seconds=180.0,
        total_beats=360,
        beat_times=np.array([0.046, 0.691, 1.337, 1.982]),
        downbeat_times=np.array([0.046, 2.628]),
        key="C",
        scale="major",
        key_confidence=0.85,
        has_modulation=False,
        source_quality="youtube-opus-128kbps",
        mean_rms=0.075,
        stem_analysis=StemAnalysis(
            bar_rms={
                "vocals": np.array([0.01, 0.05, 0.09]),
                "drums": np.array([0.03, 0.06, 0.07]),
                "bass": np.array([0.02, 0.04, 0.05]),
                "guitar": np.array([0.005, 0.01, 0.01]),
                "piano": np.array([0.02, 0.03, 0.03]),
                "other": np.array([0.001, 0.002, 0.003]),
            },
            combined_energy=np.array([0.15, 0.42, 0.68]),
            vocal_active=np.array([False, True, True]),
            vocal_gaps=[VocalGap(start_bar=0, end_bar=1, length_bars=1)],
            bucket_thresholds=EnergyBuckets(noise_floor=0.02, p10=0.18, p50=0.52, p85=0.81),
        ),
        song_structure=SongStructure(
            sections=[
                SectionInfo(
                    start_bar=0, end_bar=8, bar_count=8,
                    start_time=0.0, end_time=16.0,
                    label="intro", energy_level="low",
                    energy_trajectory="low->medium", density="sparse",
                    vocal_status="vox:no", vocal_prominence_db=None,
                    annotations=["GOOD INSTRUMENTAL SOURCE"],
                    section_source="ml",
                ),
            ],
            vocal_gaps=[VocalGap(start_bar=0, end_bar=1, length_bars=1)],
            total_bars=48,
        ),
        chord_progression=ChordProgression(
            chords=[ChordEvent(start_ms=0, end_ms=2000, chord="C")],
            unique_chords=["C", "G", "Am"],
            most_common_chord="C",
            progression_summary="I-V-vi in C major",
        ),
        polyphony_info=PolyphonyInfo(
            polyphonic=False, method="mid_side",
            gate1_ratio=0.1, gate2_ratio=0.05,
        ),
        drum_pattern=DrumPattern(
            kick_count=100, snare_count=50, hihat_count=200,
            total_hits=350, duration_ms=180000,
            style_hint="four_on_floor",
        ),
        word_alignment=WordAlignment(
            words=[WordEvent(start_ms=5000, text="hello", end=5300)],
            source="whisperx",
            lrclib_validated=True,
            lrclib_offset_ms=-20,
        ),
    )


def _make_lyrics_data() -> LyricsData:
    """Build a LyricsData for testing serialization."""
    return LyricsData(
        artist="Test Artist",
        title="Test Song",
        source="lrclib",
        is_synced=True,
        lines=[
            LyricLine(text="Hello world", timestamp_seconds=5.0),
            LyricLine(text="Goodbye world", timestamp_seconds=10.0),
        ],
        raw_text="Hello world\nGoodbye world",
        lookup_duration_ms=123.4,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_redis():
    """Provide a fresh Redis client and clean up test keys after each test."""
    import musicmixer.services.song_cache as _mod
    _mod._redis_client = None  # reset singleton to force fresh connection
    r = _get_redis()
    yield r
    # Clean up any test keys (both meta and stems patterns)
    for key in r.scan_iter("song:test_*"):
        r.delete(key)
    _mod._redis_client = None


@pytest.fixture
def tmp_stems(tmp_path: Path) -> Path:
    """Create a temp directory with 6 stem WAVs."""
    return _make_stem_dir(tmp_path / "stems")


# ---------------------------------------------------------------------------
# Serialization round-trip tests
# ---------------------------------------------------------------------------

class TestSerialization:
    """Verify AudioMetadata and LyricsData survive JSON round-trip."""

    def test_audio_metadata_round_trip(self):
        meta = _make_audio_metadata()
        json_str = _serialize_audio_metadata(meta)
        restored = _deserialize_audio_metadata(json_str)

        assert restored.bpm == meta.bpm
        assert restored.bpm_confidence == meta.bpm_confidence
        assert restored.duration_seconds == meta.duration_seconds
        assert restored.total_beats == meta.total_beats
        assert restored.key == meta.key
        assert restored.scale == meta.scale
        assert restored.key_confidence == meta.key_confidence
        assert restored.has_modulation == meta.has_modulation
        assert restored.source_quality == meta.source_quality
        assert restored.mean_rms == meta.mean_rms

    def test_numpy_arrays_round_trip(self):
        meta = _make_audio_metadata()
        restored = _deserialize_audio_metadata(_serialize_audio_metadata(meta))

        np.testing.assert_array_almost_equal(restored.beat_frames, meta.beat_frames)
        np.testing.assert_array_almost_equal(restored.beat_times, meta.beat_times)
        np.testing.assert_array_almost_equal(restored.downbeat_times, meta.downbeat_times)

    def test_stem_analysis_round_trip(self):
        meta = _make_audio_metadata()
        restored = _deserialize_audio_metadata(_serialize_audio_metadata(meta))

        assert restored.stem_analysis is not None
        for stem_name in STEM_NAMES:
            np.testing.assert_array_almost_equal(
                restored.stem_analysis.bar_rms[stem_name],
                meta.stem_analysis.bar_rms[stem_name],
            )
        np.testing.assert_array_almost_equal(
            restored.stem_analysis.combined_energy,
            meta.stem_analysis.combined_energy,
        )
        np.testing.assert_array_equal(
            restored.stem_analysis.vocal_active,
            meta.stem_analysis.vocal_active,
        )
        assert len(restored.stem_analysis.vocal_gaps) == 1
        assert restored.stem_analysis.bucket_thresholds.p50 == meta.stem_analysis.bucket_thresholds.p50

    def test_song_structure_round_trip(self):
        meta = _make_audio_metadata()
        restored = _deserialize_audio_metadata(_serialize_audio_metadata(meta))

        assert restored.song_structure is not None
        assert len(restored.song_structure.sections) == 1
        sec = restored.song_structure.sections[0]
        assert sec.label == "intro"
        assert sec.energy_level == "low"
        assert sec.annotations == ["GOOD INSTRUMENTAL SOURCE"]
        assert sec.section_source == "ml"
        assert restored.song_structure.total_bars == 48

    def test_pulsemap_round_trip(self):
        meta = _make_audio_metadata()
        restored = _deserialize_audio_metadata(_serialize_audio_metadata(meta))

        assert restored.chord_progression is not None
        assert restored.chord_progression.chords[0].chord == "C"
        assert restored.chord_progression.progression_summary == "I-V-vi in C major"

        assert restored.polyphony_info is not None
        assert restored.polyphony_info.polyphonic is False

        assert restored.drum_pattern is not None
        assert restored.drum_pattern.style_hint == "four_on_floor"

        assert restored.word_alignment is not None
        assert restored.word_alignment.words[0].text == "hello"
        assert restored.word_alignment.lrclib_validated is True

    def test_lyrics_round_trip(self):
        lyrics = _make_lyrics_data()
        restored = _deserialize_lyrics(_serialize_lyrics(lyrics))

        assert restored.artist == lyrics.artist
        assert restored.title == lyrics.title
        assert restored.source == lyrics.source
        assert restored.is_synced == lyrics.is_synced
        assert len(restored.lines) == 2
        assert restored.lines[0].text == "Hello world"
        assert restored.lines[0].timestamp_seconds == 5.0
        assert restored.raw_text == lyrics.raw_text

    def test_none_optional_fields(self):
        """AudioMetadata with all optional fields as None should round-trip."""
        meta = AudioMetadata(
            bpm=100.0,
            bpm_confidence=0.5,
            beat_frames=np.array([512, 1024]),
            duration_seconds=60.0,
            total_beats=100,
        )
        restored = _deserialize_audio_metadata(_serialize_audio_metadata(meta))

        assert restored.bpm == 100.0
        assert restored.stem_analysis is None
        assert restored.song_structure is None
        assert restored.chord_progression is None
        assert restored.word_alignment is None
        assert restored.key is None


# ---------------------------------------------------------------------------
# Redis integration tests
# ---------------------------------------------------------------------------

class TestCacheReadWrite:
    """Test writing to and reading from Redis."""

    def test_cache_miss(self):
        result = get_cached_song("test_nonexistent_video_id", ROLE_VOCAL)
        assert result is None

    def test_cache_hit_metadata_only(self, clean_redis):
        meta = _make_audio_metadata()
        lyrics = _make_lyrics_data()

        cache_song_metadata(
            video_id="test_vid_001",
            title="Test Song",
            artist="Test Artist",
            meta=meta,
            lyrics=lyrics,
        )
        result = get_cached_song("test_vid_001", ROLE_VOCAL)

        assert result is not None
        assert result.video_id == "test_vid_001"
        assert result.title == "Test Song"
        assert result.artist == "Test Artist"
        assert result.meta.bpm == 120.0
        assert result.lyrics is not None
        assert result.lyrics.artist == "Test Artist"
        assert result.has_stems is False  # no stems cached yet
        assert result.stems_path is None

    def test_cache_hit_with_stems(self, clean_redis, tmp_stems, tmp_path):
        meta = _make_audio_metadata()

        cache_song_metadata(
            video_id="test_vid_002",
            title="Test Song",
            artist="Test Artist",
            meta=meta,
            lyrics=None,
        )
        cache_song_stems("test_vid_002", ROLE_INSTRUMENTAL, tmp_stems)

        result = get_cached_song("test_vid_002", ROLE_INSTRUMENTAL)

        assert result is not None
        assert result.has_stems is True
        assert result.stems_path is not None

    def test_cache_no_lyrics(self, clean_redis):
        meta = _make_audio_metadata()
        cache_song_metadata(
            video_id="test_vid_003",
            title="Test Song",
            artist="Test Artist",
            meta=meta,
            lyrics=None,
        )

        result = get_cached_song("test_vid_003", ROLE_VOCAL)
        assert result is not None
        assert result.lyrics is None

    def test_get_cached_stems_copies_files(self, clean_redis, tmp_stems, tmp_path):
        cache_song_stems("test_vid_004", ROLE_INSTRUMENTAL, tmp_stems)

        output_dir = tmp_path / "restored"
        success = get_cached_stems("test_vid_004", ROLE_INSTRUMENTAL, output_dir)

        assert success is True
        for name in STEM_NAMES:
            assert (output_dir / f"{name}.wav").exists()

    def test_get_cached_stems_miss(self, tmp_path):
        output_dir = tmp_path / "restored"
        success = get_cached_stems("test_nonexistent", ROLE_VOCAL, output_dir)
        assert success is False

    def test_overwrite_existing_cache(self, clean_redis):
        meta1 = AudioMetadata(
            bpm=100.0, bpm_confidence=0.5,
            beat_frames=np.array([512]), duration_seconds=60.0, total_beats=100,
        )
        meta2 = AudioMetadata(
            bpm=140.0, bpm_confidence=0.9,
            beat_frames=np.array([1024]), duration_seconds=120.0, total_beats=200,
        )

        cache_song_metadata(
            video_id="test_vid_005",
            title="Song V1",
            artist="",
            meta=meta1,
            lyrics=None,
        )
        cache_song_metadata(
            video_id="test_vid_005",
            title="Song V2",
            artist="",
            meta=meta2,
            lyrics=None,
        )

        result = get_cached_song("test_vid_005", ROLE_VOCAL)
        assert result is not None
        assert result.meta.bpm == 140.0  # second write wins
        assert result.title == "Song V2"


# ---------------------------------------------------------------------------
# Role-aware cache tests
# ---------------------------------------------------------------------------

class TestRoleAwareCache:
    """Test split cache keys, filesystem paths, and validation."""

    def test_meta_key_format(self):
        """_meta_key produces song:{video_id}:meta."""
        assert _meta_key("abc123") == "song:abc123:meta"

    def test_stems_key_format(self):
        """_stems_key produces song:{video_id}:{role}:stems."""
        assert _stems_key("abc123", ROLE_VOCAL) == "song:abc123:vocal:stems"
        assert _stems_key("abc123", ROLE_INSTRUMENTAL) == "song:abc123:instrumental:stems"

    def test_stems_key_rejects_invalid_role(self):
        """_stems_key raises ValueError for unknown roles."""
        with pytest.raises(ValueError, match="Invalid role"):
            _stems_key("abc123", "unknown")

    def test_same_video_different_roles_shares_metadata(self, clean_redis, tmp_path):
        """Same video cached under vocal and instrumental shares ONE metadata key
        but has TWO separate stems keys."""
        meta = _make_audio_metadata()

        # Cache metadata once (shared)
        cache_song_metadata(
            video_id="test_vid_dual",
            title="Song Title",
            artist="",
            meta=meta,
            lyrics=None,
        )

        # Cache stems for both roles
        vocal_stems = _make_vocal_stem_dir(tmp_path / "vocal_stems")
        cache_song_stems("test_vid_dual", ROLE_VOCAL, vocal_stems)

        inst_stems = _make_stem_dir(tmp_path / "inst_stems")
        cache_song_stems("test_vid_dual", ROLE_INSTRUMENTAL, inst_stems)

        # Verify ONE shared metadata key
        r = clean_redis
        assert r.exists("song:test_vid_dual:meta") == 1

        # Verify TWO separate stems keys (plain strings, not hashes)
        assert r.exists("song:test_vid_dual:vocal:stems") == 1
        assert r.exists("song:test_vid_dual:instrumental:stems") == 1
        assert r.type("song:test_vid_dual:vocal:stems") == "string"
        assert r.type("song:test_vid_dual:instrumental:stems") == "string"

        # Both roles return the same metadata (title) but different stems paths
        vocal_result = get_cached_song("test_vid_dual", ROLE_VOCAL)
        inst_result = get_cached_song("test_vid_dual", ROLE_INSTRUMENTAL)
        assert vocal_result is not None
        assert inst_result is not None
        assert vocal_result.title == "Song Title"
        assert inst_result.title == "Song Title"

        # Verify separate filesystem directories
        assert vocal_result.stems_path != inst_result.stems_path
        assert "vocal" in vocal_result.stems_path
        assert "instrumental" in inst_result.stems_path

    def test_cache_miss_for_wrong_role_returns_metadata_no_stems(self, clean_redis, tmp_path):
        """Caching as vocal, querying as instrumental returns metadata but has_stems=False."""
        meta = _make_audio_metadata()
        vocal_stems = _make_vocal_stem_dir(tmp_path / "vocal_stems")
        cache_song_metadata(
            video_id="test_vid_role_miss",
            title="Song",
            artist="",
            meta=meta,
            lyrics=None,
        )
        cache_song_stems("test_vid_role_miss", ROLE_VOCAL, vocal_stems)

        # Querying as instrumental: metadata IS found (shared), but stems are not
        result = get_cached_song("test_vid_role_miss", ROLE_INSTRUMENTAL)
        assert result is not None
        assert result.title == "Song"
        assert result.meta.bpm == 120.0
        assert result.has_stems is False
        assert result.stems_path is None

    def test_metadata_only_no_stems(self, clean_redis):
        """Cache metadata only (no stems) returns CachedSong with has_stems=False."""
        meta = _make_audio_metadata()
        cache_song_metadata(
            video_id="test_vid_meta_only",
            title="Meta Only Song",
            artist="",
            meta=meta,
            lyrics=None,
        )

        result = get_cached_song("test_vid_meta_only", ROLE_VOCAL)
        assert result is not None
        assert result.title == "Meta Only Song"
        assert result.has_stems is False
        assert result.stems_path is None

        # Same for instrumental
        result_inst = get_cached_song("test_vid_meta_only", ROLE_INSTRUMENTAL)
        assert result_inst is not None
        assert result_inst.has_stems is False

    def test_metadata_shared_vocal_stems_query_instrumental(self, clean_redis, tmp_path):
        """Cache metadata + vocal stems, query as instrumental: metadata present, has_stems=False."""
        meta = _make_audio_metadata()
        vocal_stems = _make_vocal_stem_dir(tmp_path / "vocal_stems")
        cache_song_metadata(
            video_id="test_vid_cross_role",
            title="Cross Role Song",
            artist="",
            meta=meta,
            lyrics=None,
        )
        cache_song_stems("test_vid_cross_role", ROLE_VOCAL, vocal_stems)

        # Vocal query: full hit
        vocal_result = get_cached_song("test_vid_cross_role", ROLE_VOCAL)
        assert vocal_result is not None
        assert vocal_result.has_stems is True

        # Instrumental query: metadata hit, stems miss
        inst_result = get_cached_song("test_vid_cross_role", ROLE_INSTRUMENTAL)
        assert inst_result is not None
        assert inst_result.title == "Cross Role Song"
        assert inst_result.has_stems is False

    def test_both_roles_have_stems(self, clean_redis, tmp_path):
        """Cache metadata + vocal stems + instrumental stems: both roles return has_stems=True."""
        meta = _make_audio_metadata()
        cache_song_metadata(
            video_id="test_vid_both_roles",
            title="Both Roles Song",
            artist="",
            meta=meta,
            lyrics=None,
        )

        vocal_stems = _make_vocal_stem_dir(tmp_path / "vocal_stems")
        cache_song_stems("test_vid_both_roles", ROLE_VOCAL, vocal_stems)

        inst_stems = _make_stem_dir(tmp_path / "inst_stems")
        cache_song_stems("test_vid_both_roles", ROLE_INSTRUMENTAL, inst_stems)

        vocal_result = get_cached_song("test_vid_both_roles", ROLE_VOCAL)
        inst_result = get_cached_song("test_vid_both_roles", ROLE_INSTRUMENTAL)

        assert vocal_result is not None
        assert vocal_result.has_stems is True
        assert inst_result is not None
        assert inst_result.has_stems is True

    def test_vocal_role_stem_validation(self, clean_redis, tmp_path):
        """Vocal role requires >= 3 stems to set has_stems=True."""
        meta = _make_audio_metadata()
        vocal_stems = _make_vocal_stem_dir(tmp_path / "vocal_stems")
        cache_song_metadata(
            video_id="test_vid_voc_valid",
            title="Song",
            artist="",
            meta=meta,
            lyrics=None,
        )
        cache_song_stems("test_vid_voc_valid", ROLE_VOCAL, vocal_stems)

        result = get_cached_song("test_vid_voc_valid", ROLE_VOCAL)
        assert result is not None
        assert result.has_stems is True

    def test_vocal_role_insufficient_stems(self, clean_redis, tmp_path):
        """Vocal role with < 3 stems sets has_stems=False."""
        meta = _make_audio_metadata()
        sparse_dir = tmp_path / "sparse_stems"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        _make_wav(sparse_dir / "lead_vocals.wav")
        _make_wav(sparse_dir / "backing_vocals.wav")
        # Only 2 stems — below _MIN_VOCAL_STEMS threshold
        cache_song_metadata(
            video_id="test_vid_voc_sparse",
            title="Song",
            artist="",
            meta=meta,
            lyrics=None,
        )
        cache_song_stems("test_vid_voc_sparse", ROLE_VOCAL, sparse_dir)

        result = get_cached_song("test_vid_voc_sparse", ROLE_VOCAL)
        assert result is not None
        assert result.has_stems is False

    def test_instrumental_role_stem_validation(self, clean_redis, tmp_stems):
        """Instrumental role requires >= 4 stems to set has_stems=True."""
        meta = _make_audio_metadata()
        cache_song_metadata(
            video_id="test_vid_inst_valid",
            title="Song",
            artist="",
            meta=meta,
            lyrics=None,
        )
        cache_song_stems("test_vid_inst_valid", ROLE_INSTRUMENTAL, tmp_stems)

        result = get_cached_song("test_vid_inst_valid", ROLE_INSTRUMENTAL)
        assert result is not None
        assert result.has_stems is True

    def test_instrumental_role_insufficient_stems(self, clean_redis, tmp_path):
        """Instrumental role with < 4 stems sets has_stems=False."""
        meta = _make_audio_metadata()
        sparse_dir = tmp_path / "sparse_inst"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        _make_wav(sparse_dir / "vocals.wav")
        _make_wav(sparse_dir / "drums.wav")
        _make_wav(sparse_dir / "bass.wav")
        # Only 3 stems — below _MIN_INSTRUMENTAL_STEMS threshold
        cache_song_metadata(
            video_id="test_vid_inst_sparse",
            title="Song",
            artist="",
            meta=meta,
            lyrics=None,
        )
        cache_song_stems("test_vid_inst_sparse", ROLE_INSTRUMENTAL, sparse_dir)

        result = get_cached_song("test_vid_inst_sparse", ROLE_INSTRUMENTAL)
        assert result is not None
        assert result.has_stems is False

    def test_backward_compatibility_old_key_is_miss(self, clean_redis):
        """Old-format song:{video_id}:{role} key in Redis returns None (no :meta suffix)."""
        r = clean_redis
        # Simulate old-format entries (pre-split keys)
        r.hset("song:test_vid_old_format:vocal", mapping={"title": "Old Song", "meta": "{}"})

        result_vocal = get_cached_song("test_vid_old_format", ROLE_VOCAL)
        result_inst = get_cached_song("test_vid_old_format", ROLE_INSTRUMENTAL)
        assert result_vocal is None
        assert result_inst is None

        # Clean up old-format key
        r.delete("song:test_vid_old_format:vocal")

    def test_get_cached_stems_vocal_role(self, clean_redis, tmp_path):
        """get_cached_stems with vocal role copies files and uses vocal threshold."""
        vocal_stems = _make_vocal_stem_dir(tmp_path / "vocal_stems")
        cache_song_stems("test_vid_vocal_restore", ROLE_VOCAL, vocal_stems)

        output_dir = tmp_path / "restored"
        success = get_cached_stems("test_vid_vocal_restore", ROLE_VOCAL, output_dir)

        assert success is True
        for name in VOCAL_STEM_NAMES:
            assert (output_dir / f"{name}.wav").exists()

    def test_get_cached_stems_wrong_role_miss(self, clean_redis, tmp_path):
        """get_cached_stems returns False for wrong role (no directory)."""
        vocal_stems = _make_vocal_stem_dir(tmp_path / "vocal_stems")
        cache_song_stems("test_vid_stems_role", ROLE_VOCAL, vocal_stems)

        output_dir = tmp_path / "restored"
        success = get_cached_stems("test_vid_stems_role", ROLE_INSTRUMENTAL, output_dir)
        assert success is False

    def test_constants_values(self):
        """Verify module-level constants have expected values."""
        assert _MIN_VOCAL_STEMS == 3
        assert _MIN_INSTRUMENTAL_STEMS == 4
