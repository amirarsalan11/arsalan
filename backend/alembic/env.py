"""
Alembic environment script.

Milestone 2 scope: `target_metadata` now points at the real
`Base.metadata` populated by `app.db.models`, so `alembic revision
--autogenerate` can detect the six core tables (Tenant, ApiKey,
AllowedDomain, Material, RenderJob, UsageRecord).
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings
from app.db.base import Base

# Importing app.db.models registers every model class on Base's registry,
# which is required for target_metadata below to be complete. The import
# is only needed for its side effect, hence the noqa.
import app.db.models  # noqa: F401

# Alembic Config object, provides access to values within alembic.ini
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override the sqlalchemy.url from alembic.ini with the application's
# configured DATABASE_URL so there is a single source of truth.
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
