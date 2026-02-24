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

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
