from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8880
    cors_origins: list[str] = ["http://localhost:5173", "https://mixer.awill.co"]

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
    distributed_limiter_enabled: bool = False

    # Stem separation
    stem_backend: str = "modal"  # "modal" or "local"

    # Stem cache
    stem_cache_enabled: bool = True
    stem_cache_max_gb: float = 10.0
    stem_cache_dir: Path = Path("data/stem_cache")

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

    # SMS notifications
    sms_enabled: bool = False
    app_base_url: str = "http://localhost:5173"

    # Taste training (candidate generation + scoring)
    ab_taste_model_v1: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
