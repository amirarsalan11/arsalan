"""
Service layer package.

Contains business logic orchestration and is the only layer that API
routes call into directly. Services never talk to the database
directly — they call into the repository layer only. Services may
call other services for orchestration (e.g. TenantService calling
ApiKeyService for the tenant-bootstrap key issuance), but never skip
down to another service's repository directly.

Services raise the exceptions in `app.services.exceptions`
(NotFoundError, ConflictError, AuthenticationError, ForbiddenError),
never HTTPException — the API layer translates those into HTTP
responses.
"""

from app.services.allowed_domain_service import AllowedDomainService
from app.services.api_key_service import ApiKeyService
from app.services.auth_service import AuthService
from app.services.exceptions import (
    AuthenticationError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ServiceError,
)
from app.services.material_service import MaterialService
from app.services.render_job_service import RenderJobService
from app.services.tenant_service import TenantService
from app.services.usage_record_service import UsageRecordService

__all__ = [
    "AllowedDomainService",
    "ApiKeyService",
    "AuthenticationError",
    "AuthService",
    "ConflictError",
    "ForbiddenError",
    "MaterialService",
    "NotFoundError",
    "RenderJobService",
    "ServiceError",
    "TenantService",
    "UsageRecordService",
]
