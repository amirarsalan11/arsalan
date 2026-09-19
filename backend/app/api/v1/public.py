"""
Public API routes — SDK + Widget milestone.

These are the ONLY endpoints in the entire backend that do not require
`Authorization: Bearer <key>`. They exist specifically because the
browser-side SDK/widget has no secret key and must never be given one.

Authentication substitute: `get_public_tenant_id` (app/api/deps.py)
verifies the request's Origin header against the tenant's
AllowedDomain whitelist instead of a bearer token. See that
dependency's docstring for the full behavior, including how CORS
headers are set only *after* the domain check passes.

Scope, deliberately minimal:
- GET /public/tenants/{tenant_id}/config — tenant display name +
  active status only.
- GET /public/tenants/{tenant_id}/materials — that tenant's ACTIVE
  materials only, via dedicated public schemas that never include
  `texture_url` or any other storage-location detail.

Nothing here creates, modifies, or deletes anything — every route in
this router is read-only. Mutating operations (render job creation,
material management, etc.) remain exclusively behind
`get_current_tenant_id` / Bearer auth, untouched by this milestone.
"""

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_public_tenant_id
from app.schemas.public import PublicMaterialRead, PublicTenantConfig
from app.services import MaterialService, TenantService

router = APIRouter(prefix="/public/tenants", tags=["public"])


@router.get("/{tenant_id}/config", response_model=PublicTenantConfig)
def get_public_tenant_config(
    tenant_id: uuid.UUID = Depends(get_public_tenant_id),
    db: Session = Depends(get_db),
) -> PublicTenantConfig:
    service = TenantService(db)
    tenant = service.get_tenant(tenant_id)

    return PublicTenantConfig(
        tenant_id=tenant.id,
        name=tenant.name,
        is_active=tenant.is_active,
    )


@router.get("/{tenant_id}/materials", response_model=list[PublicMaterialRead])
def list_public_materials(
    tenant_id: uuid.UUID = Depends(get_public_tenant_id),
    db: Session = Depends(get_db),
) -> list[PublicMaterialRead]:
    service = MaterialService(db)
    materials = service.list_active_materials(tenant_id)
    return [PublicMaterialRead.model_validate(m) for m in materials]
