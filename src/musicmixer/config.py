from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8880
    cors_origins: list[str] = ["http://localhost:5173", "https://mixer.awill.co"]

    # Logging
    log_level: str = "INFO"

    # File limits
    max_file_size_mb: int = 50
    max_upload_duration_seconds: int = 900  # 15 minutes
    allowed_extensions: set[str] = {".mp3", ".wav"}

    # Storage
    data_dir: Path = Path("data")
    max_concurrent_mixes: int = Field(default=1, ge=1, le=8)
    max_queue_depth: int = 10
    session_ttl_hours: int = 3
    queue_entry_ttl_minutes: int = 150
    processing_timeout_minutes: int = 20
    processing_max_duration_seconds: int = 210  # trim input audio longer than 3:30 before processing
    distributed_limiter_enabled: bool = False

    # Stem separation
    stem_backend: str = "modal"  # "modal" or "local"

    # Section detection
    section_detection_backend: str = "auto"  # "auto" (try ML, fall back to heuristic) | "ml" | "heuristic"

    # Stem cache
    stem_cache_enabled: bool = False
    stem_cache_max_gb: float = 10.0
    stem_cache_dir: Path = Path("data/stem_cache")

    # Shelf stems cache (pre-computed stems for default shelf songs)
    # Always enabled for shelf songs, independent of stem_cache_enabled.
    # Separate directory to avoid LRU eviction.
    shelf_stems_dir: Path = Path("data/shelf_stems")

    # Remix output cache
    remix_cache_enabled: bool = False
    remix_cache_max_gb: float = 5.0
    remix_cache_dir: Path = Path("data/remix_cache")

    # Output
    output_format: str = "mp3"
    output_bitrate: str = "320k"

    # Lyrics lookup (Day 3)
    lyrics_lookup_enabled: bool = True

    # LLM (Day 3)
    anthropic_api_key: str = ""  # ANTHROPIC_API_KEY env var
    llm_model: str = "claude-sonnet-4-20250514"
    llm_timeout_seconds: int = 20
    llm_max_retries: int = 1

    # YouTube
    youtube_enabled: bool = True
    youtube_max_duration_seconds: int = 900  # 15 minutes
    youtube_proxy: str = ""  # SOCKS5 proxy for YouTube downloads (bypasses datacenter IP blocks)
    youtube_proxy_service_url: str = ""  # URL of yt-proxy microservice (e.g. https://yt-proxy.awill.co)
    youtube_proxy_api_key: str = ""  # API key for yt-proxy service

    # PulseMap analysis toggles
    pulsemap_chords_enabled: bool = True
    pulsemap_polyphony_enabled: bool = True
    pulsemap_drums_enabled: bool = True
    pulsemap_word_alignment_enabled: bool = False  # disabled: loads WhisperX + wav2vec2 (~1.5GB), OOMs on 4GB hosts

    # Taste training (candidate generation + scoring)
    ab_taste_model_v1: bool = False

    #Redis
    redis_url: str = "redis://localhost:6379"
    song_cache_dir: Path = Path("data/song_cache")

    # Twilio SMS notifications
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    sms_enabled: bool = False
    app_base_url: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
