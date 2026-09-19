"""
Baseline logging configuration.

Kept intentionally minimal for the project foundation milestone.
No request tracing, correlation IDs, or structured logging yet —
those can be layered in later without changing this module's role.
"""

import logging

from app.core.config import get_settings


def configure_logging() -> None:
    """Configure root logging based on application settings."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
