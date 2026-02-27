from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]

    # File limits
    max_file_size_mb: int = 50
    allowed_extensions: set[str] = {".mp3", ".wav"}

    # Storage
    data_dir: Path = Path("data")

    # Stem separation
    stem_backend: str = "modal"  # "modal" or "local"

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

    # Day 3.5 audio-quality A/B flags (default-off)
    ab_control_day3: bool = True
    ab_autolvl_tune_v1: bool = False
    ab_vocal_makeup_v1: bool = False
    ab_mp3_export_path_v1: bool = False

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
