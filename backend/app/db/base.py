"""
Declarative base and shared mixins for all ORM models.

Scope (Milestone 2 — Database Models):
- `Base`: the single SQLAlchemy 2.x declarative base every model inherits
  from. Alembic's `env.py` points at `Base.metadata` for autogeneration.
- `GUID`: a UUID column type. PostgreSQL is the only supported production
  database (per architecture lock), so this uses PostgreSQL's native UUID
  type there. It transparently falls back to CHAR(36) only so the model
  unit tests in this milestone can run against SQLite without a live
  Postgres instance — production migrations always target PostgreSQL.
- `TimestampMixin` / `CreatedAtMixin`: shared created_at/updated_at columns
  so every model doesn't redeclare them.

No repository logic, no session-scoped business logic, and no
tenant-filtering logic live here — this module only defines shared
structure that every model depends on.
"""

import uuid
from datetime import datetime

from sqlalchemy import CHAR, DateTime, TypeDecorator, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class GUID(TypeDecorator):
    """Platform-independent UUID column type.

    - On PostgreSQL: stored as native UUID (`gen_random_uuid()` provides
      server-side defaults; the `pgcrypto` extension is enabled by the
      Milestone 2 migration to guarantee that function's availability).
    - On any other dialect (SQLite, used only by this milestone's unit
      tests): stored as CHAR(36) holding the string form of the UUID.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return str(value)
        if not isinstance(value, uuid.UUID):
            return str(uuid.UUID(str(value)))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            return uuid.UUID(str(value))
        return value


class TimestampMixin:
    """created_at / updated_at columns for mutable entities.

    Both are server-side defaults (`func.now()`), so the timestamp is set
    by the database itself rather than the application clock.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class CreatedAtMixin:
    """created_at only, for immutable append-only records (e.g. UsageRecord)."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
