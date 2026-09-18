"""
Repository layer package.

Encapsulates all direct database access (SQLAlchemy queries). Services
call repositories only; no other layer issues raw queries against the
models.

`BaseRepository` provides only the operations that are genuinely
identical across models (get_by_id, list_by_tenant, update). Anything
domain-specific (e.g. `get_by_slug`, `get_by_key_hash`, `get_by_domain`)
lives on the owning repository instead of being generalized into the
base.
"""

from app.repositories.allowed_domain_repository import AllowedDomainRepository
from app.repositories.api_key_repository import ApiKeyRepository
from app.repositories.base import BaseRepository
from app.repositories.material_repository import MaterialRepository
from app.repositories.render_job_repository import RenderJobRepository
from app.repositories.tenant_repository import TenantRepository
from app.repositories.usage_record_repository import UsageRecordRepository

__all__ = [
    "AllowedDomainRepository",
    "ApiKeyRepository",
    "BaseRepository",
    "MaterialRepository",
    "RenderJobRepository",
    "TenantRepository",
    "UsageRecordRepository",
]
