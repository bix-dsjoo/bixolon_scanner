from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BIXOLON_", extra="ignore")

    package_dir: Path = Path("models/current")
    catalog_dir: Path | None = None
    catalog_store_id: str | None = None
    catalog_key_id: str | None = None
    catalog_signing_key: SecretStr | None = None
    provider: Literal["auto", "cuda", "cpu"] = "auto"
    cuda_dll_dir: Path | None = None
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=50_000_000, ge=1)
    jpeg_draft_size: int = Field(default=1500, gt=0)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
