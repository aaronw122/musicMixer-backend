"""Tests for musicmixer.services.song_cache — Redis-backed per-song cache."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicmixer.config import settings
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
    SongRole,
    _MIN_INSTRUMENTAL_STEMS,
    _VALID_STEM_SETS_BY_ROLE,
    _is_degenerate_energy,
    _stems_valid_for_role,
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


def _make_role_stem_dir(base: Path, role: SongRole) -> Path:
    """Create a complete stem directory for a cache role."""
    if role == ROLE_VOCAL:
        return _make_vocal_stem_dir(base)
    if role == ROLE_INSTRUMENTAL:
        return _make_stem_dir(base)
    raise ValueError(f"Invalid role {role!r}")


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


def _cache_test_metadata(
    video_id: str,
    title: str = "Song",
    artist: str = "",
    lyrics: LyricsData | None = None,
) -> AudioMetadata:
    """Cache standard test metadata and return it for assertions."""
    meta = _make_audio_metadata()
    cache_song_metadata(
        video_id=video_id,
        title=title,
        artist=artist,
        meta=meta,
        lyrics=lyrics,
    )
    return meta


def _cache_test_stems(video_id: str, role: SongRole, tmp_path: Path) -> Path:
    """Cache a complete role-specific stem directory and return its source path."""
    stems_dir = _make_role_stem_dir(tmp_path / f"{video_id}_{role}_stems", role)
    cache_song_stems(video_id, role, stems_dir)
    return stems_dir


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
        _cache_test_metadata("test_vid_dual", title="Song Title")
        _cache_test_stems("test_vid_dual", ROLE_VOCAL, tmp_path)
        _cache_test_stems("test_vid_dual", ROLE_INSTRUMENTAL, tmp_path)

        r = clean_redis
        assert r.exists(_meta_key("test_vid_dual")) == 1
        assert r.exists(_stems_key("test_vid_dual", ROLE_VOCAL)) == 1
        assert r.exists(_stems_key("test_vid_dual", ROLE_INSTRUMENTAL)) == 1
        assert r.type(_stems_key("test_vid_dual", ROLE_VOCAL)) == "string"
        assert r.type(_stems_key("test_vid_dual", ROLE_INSTRUMENTAL)) == "string"

        vocal_result = get_cached_song("test_vid_dual", ROLE_VOCAL)
        inst_result = get_cached_song("test_vid_dual", ROLE_INSTRUMENTAL)
        assert vocal_result is not None
        assert inst_result is not None
        assert vocal_result.title == "Song Title"
        assert inst_result.title == "Song Title"

        assert vocal_result.stems_path != inst_result.stems_path
        assert vocal_result.stems_path is not None
        assert inst_result.stems_path is not None
        assert Path(vocal_result.stems_path).name == ROLE_VOCAL
        assert Path(inst_result.stems_path).name == ROLE_INSTRUMENTAL

    def test_cache_miss_for_wrong_role_returns_metadata_no_stems(self, clean_redis, tmp_path):
        """Caching as vocal, querying as instrumental returns metadata but has_stems=False."""
        _cache_test_metadata("test_vid_role_miss")
        _cache_test_stems("test_vid_role_miss", ROLE_VOCAL, tmp_path)

        result = get_cached_song("test_vid_role_miss", ROLE_INSTRUMENTAL)
        assert result is not None
        assert result.title == "Song"
        assert result.meta.bpm == 120.0
        assert result.has_stems is False
        assert result.stems_path is None

    def test_metadata_only_no_stems(self, clean_redis):
        """Cache metadata only (no stems) returns CachedSong with has_stems=False."""
        _cache_test_metadata("test_vid_meta_only", title="Meta Only Song")

        result = get_cached_song("test_vid_meta_only", ROLE_VOCAL)
        assert result is not None
        assert result.title == "Meta Only Song"
        assert result.has_stems is False
        assert result.stems_path is None

        result_inst = get_cached_song("test_vid_meta_only", ROLE_INSTRUMENTAL)
        assert result_inst is not None
        assert result_inst.has_stems is False

    def test_metadata_shared_vocal_stems_query_instrumental(self, clean_redis, tmp_path):
        """Cache metadata + vocal stems, query as instrumental: metadata present, has_stems=False."""
        _cache_test_metadata("test_vid_cross_role", title="Cross Role Song")
        _cache_test_stems("test_vid_cross_role", ROLE_VOCAL, tmp_path)

        vocal_result = get_cached_song("test_vid_cross_role", ROLE_VOCAL)
        assert vocal_result is not None
        assert vocal_result.has_stems is True

        inst_result = get_cached_song("test_vid_cross_role", ROLE_INSTRUMENTAL)
        assert inst_result is not None
        assert inst_result.title == "Cross Role Song"
        assert inst_result.has_stems is False

    def test_both_roles_have_stems(self, clean_redis, tmp_path):
        """Cache metadata + vocal stems + instrumental stems: both roles return has_stems=True."""
        _cache_test_metadata("test_vid_both_roles", title="Both Roles Song")
        _cache_test_stems("test_vid_both_roles", ROLE_VOCAL, tmp_path)
        _cache_test_stems("test_vid_both_roles", ROLE_INSTRUMENTAL, tmp_path)

        vocal_result = get_cached_song("test_vid_both_roles", ROLE_VOCAL)
        inst_result = get_cached_song("test_vid_both_roles", ROLE_INSTRUMENTAL)

        assert vocal_result is not None
        assert vocal_result.has_stems is True
        assert inst_result is not None
        assert inst_result.has_stems is True

    def test_vocal_role_stem_validation(self, clean_redis, tmp_path):
        """Vocal role with the full modal vocal shape sets has_stems=True."""
        _cache_test_metadata("test_vid_voc_valid")
        _cache_test_stems("test_vid_voc_valid", ROLE_VOCAL, tmp_path)

        result = get_cached_song("test_vid_voc_valid", ROLE_VOCAL)
        assert result is not None
        assert result.has_stems is True

    def test_vocal_role_insufficient_stems(self, clean_redis, tmp_path):
        """Vocal role with a non-accepted partial shape sets has_stems=False."""
        sparse_dir = tmp_path / "sparse_stems"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        _make_wav(sparse_dir / "lead_vocals.wav")
        _make_wav(sparse_dir / "backing_vocals.wav")
        _cache_test_metadata("test_vid_voc_sparse")
        cache_song_stems("test_vid_voc_sparse", ROLE_VOCAL, sparse_dir)

        result = get_cached_song("test_vid_voc_sparse", ROLE_VOCAL)
        assert result is not None
        assert result.has_stems is False

    def test_instrumental_role_stem_validation(self, clean_redis, tmp_stems):
        """Instrumental role with the full modal 6-stem shape sets has_stems=True."""
        _cache_test_metadata("test_vid_inst_valid")
        cache_song_stems("test_vid_inst_valid", ROLE_INSTRUMENTAL, tmp_stems)

        result = get_cached_song("test_vid_inst_valid", ROLE_INSTRUMENTAL)
        assert result is not None
        assert result.has_stems is True

    def test_instrumental_role_insufficient_stems(self, clean_redis, tmp_path):
        """Instrumental role with a non-accepted partial shape sets has_stems=False."""
        sparse_dir = tmp_path / "sparse_inst"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        _make_wav(sparse_dir / "vocals.wav")
        _make_wav(sparse_dir / "drums.wav")
        _make_wav(sparse_dir / "bass.wav")
        _cache_test_metadata("test_vid_inst_sparse")
        cache_song_stems("test_vid_inst_sparse", ROLE_INSTRUMENTAL, sparse_dir)

        result = get_cached_song("test_vid_inst_sparse", ROLE_INSTRUMENTAL)
        assert result is not None
        assert result.has_stems is False

    def test_backward_compatibility_old_key_is_miss(self, clean_redis):
        """Old-format song:{video_id}:{role} key in Redis returns None (no :meta suffix)."""
        r = clean_redis
        r.hset("song:test_vid_old_format:vocal", mapping={"title": "Old Song", "meta": "{}"})

        result_vocal = get_cached_song("test_vid_old_format", ROLE_VOCAL)
        result_inst = get_cached_song("test_vid_old_format", ROLE_INSTRUMENTAL)
        assert result_vocal is None
        assert result_inst is None

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

    def test_overwrite_clears_stale_lyrics(self, clean_redis):
        """Overwriting metadata without lyrics clears previously cached lyrics."""
        meta = _make_audio_metadata()
        lyrics = _make_lyrics_data()

        cache_song_metadata("test_vid_stale", "Song With Lyrics", "Artist A", meta, lyrics)
        cache_song_metadata("test_vid_stale", "Song Without Lyrics", "Artist B", meta, None)

        result = get_cached_song("test_vid_stale", role=ROLE_VOCAL)
        assert result is not None
        assert result.title == "Song Without Lyrics"
        assert result.artist == "Artist B"
        assert result.lyrics is None

    def test_stems_without_metadata_returns_none(self, clean_redis, tmp_path):
        """Stems key without metadata key returns None (metadata is required)."""
        r = clean_redis
        r.set(_stems_key("test_vid_orphan", ROLE_VOCAL), str(tmp_path))

        result = get_cached_song("test_vid_orphan", role=ROLE_VOCAL)
        assert result is None

    def test_constants_values(self):
        """Verify module-level validation constants have expected values."""
        assert _MIN_INSTRUMENTAL_STEMS == 4
        assert _VALID_STEM_SETS_BY_ROLE[ROLE_VOCAL] == (
            frozenset({"lead_vocals", "backing_vocals", "instrumental"}),
            frozenset({"lead_vocals", "instrumental"}),
        )
        assert _VALID_STEM_SETS_BY_ROLE[ROLE_INSTRUMENTAL] == (
            frozenset({"vocals", "drums", "bass", "guitar", "piano", "other"}),
            frozenset({"vocals", "drums", "bass", "other"}),
        )


# ---------------------------------------------------------------------------
# Name-aware stem validation (self-healing role cache)
# ---------------------------------------------------------------------------

# Legacy 6-stem instrumental blob, which the regression window wrongly cached
# under the vocal role.
LEGACY_BLOB_NAMES = ["vocals", "drums", "bass", "guitar", "piano", "other"]
LOCAL_INSTRUMENTAL_NAMES = ["vocals", "drums", "bass", "other"]
LOCAL_VOCAL_NAMES = ["lead_vocals", "instrumental"]


def _make_named_dir(base: Path, names: list[str]) -> Path:
    """Create a directory containing one minimal WAV per provided stem name."""
    base.mkdir(parents=True, exist_ok=True)
    for name in names:
        _make_wav(base / f"{name}.wav")
    return base


def _seed_corrupt_cache(video_id: str, role: SongRole, names: list[str]) -> Path:
    """Seed an on-disk role cache dir + Redis stems key directly.

    Bypasses cache_song_stems() so we can simulate a pre-existing corrupt entry
    (e.g. a legacy blob written before name-aware validation existed).
    """
    cache_dir = settings.song_cache_dir / video_id / role
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    _make_named_dir(cache_dir, names)
    r = _get_redis()
    r.set(_stems_key(video_id, role), str(cache_dir))
    return cache_dir


@pytest.fixture
def clean_disk_cache():
    """Remove on-disk cache dirs for test video IDs created during a test."""
    created: list[str] = []
    yield created
    for video_id in created:
        cache_root = settings.song_cache_dir / video_id
        if cache_root.exists():
            shutil.rmtree(cache_root)


class TestNameAwareStemValidation:
    """Validation is stem-name aware, not count-only (self-healing role cache)."""

    def test_helper_accepts_modal_vocal_shape(self, tmp_path):
        d = _make_named_dir(tmp_path / "v", VOCAL_STEM_NAMES)
        assert _stems_valid_for_role(ROLE_VOCAL, list(d.glob("*.wav"))) is True

    def test_helper_accepts_local_vocal_fallback_shape(self, tmp_path):
        d = _make_named_dir(tmp_path / "v", LOCAL_VOCAL_NAMES)
        assert _stems_valid_for_role(ROLE_VOCAL, list(d.glob("*.wav"))) is True

    def test_helper_rejects_partial_vocal_lead_only(self, tmp_path):
        d = _make_named_dir(tmp_path / "v", ["lead_vocals"])
        assert _stems_valid_for_role(ROLE_VOCAL, list(d.glob("*.wav"))) is False

    def test_helper_rejects_partial_vocal_lead_plus_backing(self, tmp_path):
        d = _make_named_dir(tmp_path / "v", ["lead_vocals", "backing_vocals"])
        assert _stems_valid_for_role(ROLE_VOCAL, list(d.glob("*.wav"))) is False

    def test_helper_rejects_legacy_blob_under_vocal(self, tmp_path):
        d = _make_named_dir(tmp_path / "v", LEGACY_BLOB_NAMES)
        assert _stems_valid_for_role(ROLE_VOCAL, list(d.glob("*.wav"))) is False

    def test_helper_accepts_modal_instrumental_shape(self, tmp_path):
        d = _make_named_dir(tmp_path / "i", STEM_NAMES)
        assert _stems_valid_for_role(ROLE_INSTRUMENTAL, list(d.glob("*.wav"))) is True

    def test_helper_accepts_local_instrumental_shape(self, tmp_path):
        d = _make_named_dir(tmp_path / "i", LOCAL_INSTRUMENTAL_NAMES)
        assert _stems_valid_for_role(ROLE_INSTRUMENTAL, list(d.glob("*.wav"))) is True

    def test_helper_rejects_partial_instrumental_drums_bass(self, tmp_path):
        d = _make_named_dir(tmp_path / "i", ["drums", "bass"])
        assert _stems_valid_for_role(ROLE_INSTRUMENTAL, list(d.glob("*.wav"))) is False

    def test_reject_corrupt_vocal_entry_blob(self, clean_redis, clean_disk_cache, tmp_path):
        """A 6-stem blob seeded under vocal/ is rejected by BOTH validation sites."""
        video_id = "test_vid_corrupt_vocal"
        clean_disk_cache.append(video_id)
        _cache_test_metadata(video_id)
        _seed_corrupt_cache(video_id, ROLE_VOCAL, LEGACY_BLOB_NAMES)

        result = get_cached_song(video_id, ROLE_VOCAL)
        assert result is not None
        assert result.has_stems is False

        output_dir = tmp_path / "restored"
        assert get_cached_stems(video_id, ROLE_VOCAL, output_dir) is False

    def test_accept_valid_vocal_modal(self, clean_redis, clean_disk_cache, tmp_path):
        """Modal vocal shape (lead, backing, instrumental) is accepted."""
        video_id = "test_vid_voc_modal"
        clean_disk_cache.append(video_id)
        _cache_test_metadata(video_id)
        stems = _make_named_dir(tmp_path / "src", VOCAL_STEM_NAMES)
        cache_song_stems(video_id, ROLE_VOCAL, stems)

        result = get_cached_song(video_id, ROLE_VOCAL)
        assert result is not None
        assert result.has_stems is True
        assert get_cached_stems(video_id, ROLE_VOCAL, tmp_path / "out") is True

    def test_accept_valid_vocal_local_fallback(self, clean_redis, clean_disk_cache, tmp_path):
        """Local fallback vocal shape (lead, instrumental — 2 files) is accepted."""
        video_id = "test_vid_voc_local"
        clean_disk_cache.append(video_id)
        _cache_test_metadata(video_id)
        stems = _make_named_dir(tmp_path / "src", LOCAL_VOCAL_NAMES)
        cache_song_stems(video_id, ROLE_VOCAL, stems)

        result = get_cached_song(video_id, ROLE_VOCAL)
        assert result is not None
        assert result.has_stems is True

    def test_reject_partial_vocal_entries(self, clean_redis, clean_disk_cache, tmp_path):
        """Partial / non-accepted vocal shapes never set has_stems=True."""
        for suffix, names in [
            ("lead_only", ["lead_vocals"]),
            ("lead_backing", ["lead_vocals", "backing_vocals"]),
            ("backing_only", ["backing_vocals"]),
        ]:
            video_id = f"test_vid_voc_partial_{suffix}"
            clean_disk_cache.append(video_id)
            _cache_test_metadata(video_id)
            _seed_corrupt_cache(video_id, ROLE_VOCAL, names)

            result = get_cached_song(video_id, ROLE_VOCAL)
            assert result is not None, suffix
            assert result.has_stems is False, suffix
            assert get_cached_stems(video_id, ROLE_VOCAL, tmp_path / f"out_{suffix}") is False, suffix

    def test_instrumental_unaffected_six_stem(self, clean_redis, clean_disk_cache, tmp_path):
        """The 6-stem instrumental set remains valid."""
        video_id = "test_vid_inst_six"
        clean_disk_cache.append(video_id)
        _cache_test_metadata(video_id)
        stems = _make_named_dir(tmp_path / "src", STEM_NAMES)
        cache_song_stems(video_id, ROLE_INSTRUMENTAL, stems)

        result = get_cached_song(video_id, ROLE_INSTRUMENTAL)
        assert result is not None
        assert result.has_stems is True

    def test_reject_partial_instrumental_keeps_known_sets(
        self, clean_redis, clean_disk_cache, tmp_path
    ):
        """drums+bass alone invalid; local 4-stem and modal 6-stem remain valid."""
        # drums + bass alone → invalid
        bad_id = "test_vid_inst_drums_bass"
        clean_disk_cache.append(bad_id)
        _cache_test_metadata(bad_id)
        _seed_corrupt_cache(bad_id, ROLE_INSTRUMENTAL, ["drums", "bass"])
        bad = get_cached_song(bad_id, ROLE_INSTRUMENTAL)
        assert bad is not None
        assert bad.has_stems is False
        assert get_cached_stems(bad_id, ROLE_INSTRUMENTAL, tmp_path / "bad_out") is False

        # local 4-stem → valid
        local_id = "test_vid_inst_local4"
        clean_disk_cache.append(local_id)
        _cache_test_metadata(local_id)
        cache_song_stems(
            local_id,
            ROLE_INSTRUMENTAL,
            _make_named_dir(tmp_path / "local_src", LOCAL_INSTRUMENTAL_NAMES),
        )
        local = get_cached_song(local_id, ROLE_INSTRUMENTAL)
        assert local is not None
        assert local.has_stems is True

        # modal 6-stem → valid
        modal_id = "test_vid_inst_modal6"
        clean_disk_cache.append(modal_id)
        _cache_test_metadata(modal_id)
        cache_song_stems(
            modal_id,
            ROLE_INSTRUMENTAL,
            _make_named_dir(tmp_path / "modal_src", STEM_NAMES),
        )
        modal = get_cached_song(modal_id, ROLE_INSTRUMENTAL)
        assert modal is not None
        assert modal.has_stems is True

    def test_self_heal_prunes_stale_wavs(self, clean_redis, clean_disk_cache, tmp_path):
        """Recaching a corrupt vocal/ entry replaces the dir with only valid stems.

        Simulates the regression-window state: a 6-stem blob cached under vocal/.
        Validation rejects it, re-separation runs, and cache_song_stems() with
        valid vocal output must leave ONLY the valid vocal shape — no legacy
        WAVs lingering.
        """
        video_id = "test_vid_self_heal"
        clean_disk_cache.append(video_id)
        _cache_test_metadata(video_id)

        # 1. Seed the corrupt 6-stem blob under vocal/.
        cache_dir = _seed_corrupt_cache(video_id, ROLE_VOCAL, LEGACY_BLOB_NAMES)
        assert (cache_dir / "drums.wav").exists()

        # 2. Validation rejects it → re-separation would run.
        rejected = get_cached_song(video_id, ROLE_VOCAL)
        assert rejected is not None
        assert rejected.has_stems is False

        # 3. Re-run "separation" (fresh valid vocal output) and recache.
        fresh = _make_named_dir(tmp_path / "fresh_vocal", VOCAL_STEM_NAMES)
        cache_song_stems(video_id, ROLE_VOCAL, fresh)

        # 4. The cache dir now holds ONLY the valid vocal shape.
        remaining = {f.stem for f in cache_dir.glob("*.wav")}
        assert remaining == set(VOCAL_STEM_NAMES)
        for stale in LEGACY_BLOB_NAMES:
            assert not (cache_dir / f"{stale}.wav").exists(), stale

        # 5. The healed entry is now valid.
        healed = get_cached_song(video_id, ROLE_VOCAL)
        assert healed is not None
        assert healed.has_stems is True


# ---------------------------------------------------------------------------
# Self-heal: degenerate (zeroed) energy metadata on a medium-cache hit
# ---------------------------------------------------------------------------

def _make_degenerate_metadata() -> AudioMetadata:
    """AudioMetadata whose stem_analysis carries zeroed-out energy.

    Mirrors what a pre-Phase-1 vocal-source analysis wrote: empty/all-zero
    combined_energy + all-false vocal_active in the shared meta. beat_frames is
    short (<4) so analyze_stems treats the whole stem as a single bar — enough to
    recompute real energy from short test WAVs.
    """
    meta = _make_audio_metadata()
    meta.beat_frames = np.array([0, 1])  # <4 beats → one bar over the full stem
    meta.stem_analysis = StemAnalysis(
        bar_rms={},
        combined_energy=np.array([]),
        vocal_active=np.array([], dtype=bool),
        vocal_gaps=[],
        bucket_thresholds=EnergyBuckets(noise_floor=0.0, p10=0.0, p50=0.0, p85=0.0),
    )
    return meta


class TestDegenerateEnergyPredicate:
    """_is_degenerate_energy flags zeroed energy, passes healthy analyses."""

    def test_empty_combined_energy_is_degenerate(self):
        sa = StemAnalysis(
            bar_rms={},
            combined_energy=np.array([]),
            vocal_active=np.array([], dtype=bool),
            vocal_gaps=[],
            bucket_thresholds=EnergyBuckets(noise_floor=0.0, p10=0.0, p50=0.0, p85=0.0),
        )
        assert _is_degenerate_energy(sa) is True

    def test_all_zero_energy_all_false_vocals_is_degenerate(self):
        sa = StemAnalysis(
            bar_rms={},
            combined_energy=np.array([0.0, 0.0, 0.0]),
            vocal_active=np.array([False, False, False]),
            vocal_gaps=[],
            bucket_thresholds=EnergyBuckets(noise_floor=0.0, p10=0.0, p50=0.0, p85=0.0),
        )
        assert _is_degenerate_energy(sa) is True

    def test_healthy_energy_is_not_degenerate(self):
        sa = _make_audio_metadata().stem_analysis
        assert _is_degenerate_energy(sa) is False

    def test_zero_energy_but_active_vocals_is_not_degenerate(self):
        # Energy flat but vocals detected → not the zeroed-meta defect.
        sa = StemAnalysis(
            bar_rms={},
            combined_energy=np.array([0.0, 0.0, 0.0]),
            vocal_active=np.array([False, True, False]),
            vocal_gaps=[],
            bucket_thresholds=EnergyBuckets(noise_floor=0.0, p10=0.0, p50=0.0, p85=0.0),
        )
        assert _is_degenerate_energy(sa) is False

    def test_none_analysis_is_not_degenerate(self):
        assert _is_degenerate_energy(None) is False


class TestSelfHealDegenerateEnergy:
    """A degenerate medium-cache hit with cached stems recomputes + rewrites meta."""

    def test_self_heal_recomputes_and_rewrites_meta(
        self, clean_redis, clean_disk_cache, tmp_path
    ):
        """Degenerate meta + valid on-disk instrumental stems → energy recomputed."""
        video_id = "test_vid_selfheal_energy"
        clean_disk_cache.append(video_id)

        # Seed a degenerate meta (zeroed energy) in Redis.
        cache_song_metadata(
            video_id=video_id,
            title="Song",
            artist="Artist",
            meta=_make_degenerate_metadata(),
            lyrics=None,
        )
        # Seed valid instrumental stems on disk (real sine-wave audio).
        cache_song_stems(
            video_id,
            ROLE_INSTRUMENTAL,
            _make_named_dir(tmp_path / "inst_src", STEM_NAMES),
        )

        # Sanity: what's in Redis right now is degenerate.
        r = clean_redis
        raw = r.hget(_meta_key(video_id), "meta")
        before = _deserialize_audio_metadata(raw)
        assert _is_degenerate_energy(before.stem_analysis) is True

        # The seeded meta carries an ML-derived song_structure (SongFormer-style
        # sections). Section labels were never part of the energy defect, so the
        # heal must NOT downgrade them to a heuristic recompute.
        assert before.song_structure is not None
        assert len(before.song_structure.sections) == 1
        assert before.song_structure.sections[0].label == "intro"
        assert before.song_structure.sections[0].section_source == "ml"

        # Medium-cache load triggers the self-heal.
        result = get_cached_song(video_id, ROLE_INSTRUMENTAL)
        assert result is not None
        assert result.has_stems is True

        # Returned meta now carries real energy.
        assert _is_degenerate_energy(result.meta.stem_analysis) is False
        assert result.meta.stem_analysis.combined_energy.size > 0
        assert np.any(result.meta.stem_analysis.combined_energy)

        # ...but the existing ML song_structure is preserved, not replaced by the
        # recompute's heuristic structure.
        assert result.meta.song_structure is not None
        assert len(result.meta.song_structure.sections) == 1
        assert result.meta.song_structure.sections[0].label == "intro"
        assert result.meta.song_structure.sections[0].section_source == "ml"
        assert result.meta.song_structure.total_bars == before.song_structure.total_bars

        # And :meta was rewritten in Redis (persisted, not just the returned copy).
        healed_raw = r.hget(_meta_key(video_id), "meta")
        healed = _deserialize_audio_metadata(healed_raw)
        assert _is_degenerate_energy(healed.stem_analysis) is False
        # Persisted structure is the preserved ML one, too.
        assert healed.song_structure is not None
        assert healed.song_structure.sections[0].label == "intro"
        assert healed.song_structure.sections[0].section_source == "ml"

    def test_self_heal_falls_back_to_recomputed_structure_when_absent(
        self, clean_redis, clean_disk_cache, tmp_path
    ):
        """Degenerate meta with NO existing song_structure → use recomputed one."""
        video_id = "test_vid_selfheal_no_structure"
        clean_disk_cache.append(video_id)

        degenerate = _make_degenerate_metadata()
        degenerate.song_structure = None  # nothing to preserve
        cache_song_metadata(
            video_id=video_id,
            title="Song",
            artist="Artist",
            meta=degenerate,
            lyrics=None,
        )
        cache_song_stems(
            video_id,
            ROLE_INSTRUMENTAL,
            _make_named_dir(tmp_path / "inst_src", STEM_NAMES),
        )

        result = get_cached_song(video_id, ROLE_INSTRUMENTAL)
        assert result is not None
        assert result.has_stems is True
        # Energy healed...
        assert _is_degenerate_energy(result.meta.stem_analysis) is False
        # ...and since there was no cached structure, the recomputed one is used
        # (i.e. structure is now populated rather than left None).
        assert result.meta.song_structure is not None

    def test_no_recompute_when_stems_absent(self, clean_redis, clean_disk_cache):
        """Degenerate meta but NO cached stems → fall through, meta left degenerate."""
        video_id = "test_vid_selfheal_no_stems"
        clean_disk_cache.append(video_id)
        cache_song_metadata(
            video_id=video_id,
            title="Song",
            artist="Artist",
            meta=_make_degenerate_metadata(),
            lyrics=None,
        )

        result = get_cached_song(video_id, ROLE_INSTRUMENTAL)
        assert result is not None
        assert result.has_stems is False
        # No stems to recompute from → meta stays degenerate (full fresh analysis
        # happens later in the pipeline, not here).
        assert _is_degenerate_energy(result.meta.stem_analysis) is True
        healed = _deserialize_audio_metadata(clean_redis.hget(_meta_key(video_id), "meta"))
        assert _is_degenerate_energy(healed.stem_analysis) is True

    def test_healthy_meta_untouched(self, clean_redis, clean_disk_cache, tmp_path):
        """A healthy cached meta is returned as-is; no recompute, no rewrite."""
        video_id = "test_vid_selfheal_healthy"
        clean_disk_cache.append(video_id)
        cache_song_metadata(
            video_id=video_id,
            title="Song",
            artist="Artist",
            meta=_make_audio_metadata(),
            lyrics=None,
        )
        cache_song_stems(
            video_id,
            ROLE_INSTRUMENTAL,
            _make_named_dir(tmp_path / "inst_src", STEM_NAMES),
        )

        cached_at_before = clean_redis.hget(_meta_key(video_id), "cached_at")
        expected = _make_audio_metadata().stem_analysis.combined_energy

        result = get_cached_song(video_id, ROLE_INSTRUMENTAL)
        assert result is not None
        # Energy preserved exactly — not recomputed from the (different) stems.
        np.testing.assert_array_equal(
            result.meta.stem_analysis.combined_energy, expected
        )
        # :meta was not rewritten (cached_at unchanged).
        assert clean_redis.hget(_meta_key(video_id), "cached_at") == cached_at_before
