"""
Application configuration.

Loads settings from environment variables / .env file using Pydantic Settings.

This module MUST remain a pure configuration layer:
- No business logic.
- No database access.
- No AI logic.

Later milestones (Auth, Storage, Async Processing) will extend this settings
object with additional fields as those features are implemented.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings, sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "room-visualizer-api"
    environment: str = "development"
    log_level: str = "info"

    # Database (connection string only — models arrive in Milestone 3)
    database_url: str = "postgresql+psycopg://roomviz:changeme@postgres:5432/roomviz"

    # --- Authentication ---
    # Server-side secret ("pepper") used to key the HMAC-SHA256 hash of raw
    # API keys. This value is NEVER stored in the database — it only lives
    # in the environment. Losing/rotating it invalidates every existing
    # API key's ability to verify, since verification re-derives the hash
    # using this secret. The default below is safe only for local dev; it
    # MUST be overridden with a long random value in any real deployment.
    api_key_hash_secret: str = "dev-only-insecure-default-change-me"

    # NOTE: Redis / Celery and Storage settings are intentionally omitted here.
    # They will be added in the Async Processing and Storage milestones,
    # per the decision to defer those dependencies until their implementation
    # milestones.


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using lru_cache avoids re-parsing environment variables on every call
    while still allowing dependency injection in FastAPI routes later.
    """
    return Settings()
