"""
Database session/engine setup.

Defines the SQLAlchemy engine and session factory using the configured
DATABASE_URL. Models now exist (see app.db.models / app.db.base.Base),
but this module still does not contain any query or business logic —
that is introduced by the repository layer in a later milestone.

Explicitly OUT of scope for this milestone:
- No session usage inside API routes yet.
- No repository layer.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.db.base import Base  # re-exported for convenience: `from app.db.session import Base`

settings = get_settings()

# `future=True` is implicit in SQLAlchemy 2.x; kept explicit engine creation
# minimal on purpose. Connection pooling tuning is deferred to a later
# milestone once real query patterns exist.
engine = create_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
