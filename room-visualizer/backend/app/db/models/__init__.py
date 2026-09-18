"""
ORM model package.

Importing this package registers every model on the shared declarative
`Base`. This is required for two reasons:

1. Alembic's `env.py` imports this package so `Base.metadata` is fully
   populated before `autogenerate` runs.
2. SQLAlchemy's string-based `relationship()` forward references (e.g.
   `Mapped["ApiKey"]`) are only resolved once every referenced class has
   been imported into the same registry.

No other code should construct models by importing individual model
modules directly in preference to this package, to keep that guarantee
intact.
"""

from app.db.models.allowed_domain import AllowedDomain
from app.db.models.api_key import ApiKey
from app.db.models.material import Material
from app.db.models.render_job import RenderJob, RenderJobStatus
from app.db.models.tenant import Tenant
from app.db.models.usage_record import UsageRecord

__all__ = [
    "AllowedDomain",
    "ApiKey",
    "Material",
    "RenderJob",
    "RenderJobStatus",
    "Tenant",
    "UsageRecord",
]
