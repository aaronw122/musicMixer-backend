"""Tests for musicmixer.services.remix_cache — content-hash caching."""

import io

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(freq: float = 440.0, duration: float = 0.1, sr: int = 44100) -> bytes:
    """Create minimal float32 WAV bytes (stereo sine wave)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests: get_cache_key
# ---------------------------------------------------------------------------

class TestGetCacheKey:
    """SHA-256 content hashing."""

    def test_deterministic(self, tmp_path):
        """Same file content -> same hash."""
        from musicmixer.services.remix_cache import get_cache_key

        audio = tmp_path / "song.wav"
        audio.write_bytes(_make_wav_bytes(440.0))

        key1 = get_cache_key(audio)
        key2 = get_cache_key(audio)
        assert key1 == key2

    def test_different_content_different_key(self, tmp_path):
        """Different file content -> different hash."""
        from musicmixer.services.remix_cache import get_cache_key

        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        a.write_bytes(_make_wav_bytes(440.0))
        b.write_bytes(_make_wav_bytes(880.0))

        assert get_cache_key(a) \!= get_cache_key(b)

    def test_returns_hex_string(self, tmp_path):
        """Key should be a 64-char hex string (SHA-256)."""
        from musicmixer.services.remix_cache import get_cache_key

        audio = tmp_path / "song.wav"
        audio.write_bytes(_make_wav_bytes())

        key = get_cache_key(audio)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)
