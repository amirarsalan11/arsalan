"""
Tenant API routes.

Authentication milestone changes:
- `create_tenant` (POST /tenants) is the ONE remaining unauthenticated
  endpoint — it's the bootstrap flow: a brand-new tenant cannot present
  an API key it doesn't have yet. It now returns both the tenant and
  its automatically issued first API key (raw value, shown once).
- Every other route now requires `Authorization: Bearer <key>` via
  `get_current_tenant_id`, AND verifies the authenticated tenant_id
  matches the `{tenant_id}` path parameter. A mismatch returns 404
  (not 403) — consistent with how cross-tenant Material/RenderJob
  lookups already behave, and avoids confirming whether a given
  tenant_id even exists to a caller who isn't it.

Per approval: still NO list-all-tenants endpoint (would leak
cross-tenant data even with auth, since no admin role concept exists
yet).
"""

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant_id, get_db
from app.schemas.api_key import ApiKeyIssuedRead, ApiKeyRead
from app.schemas.tenant import TenantCreate, TenantCreateResponse, TenantRead, TenantUpdate
from app.services import NotFoundError, TenantService

router = APIRouter(prefix="/tenants", tags=["tenants"])


def _require_self(authenticated_tenant_id: uuid.UUID, path_tenant_id: uuid.UUID) -> None:
    """Enforce that an authenticated caller may only act on their own
    tenant record. Raises NotFoundError (-> 404) rather than a
    forbidden-style error on mismatch, so a caller can never learn
    whether a given tenant_id exists by trying to act on it.
    """
    if authenticated_tenant_id != path_tenant_id:
        raise NotFoundError(f"Tenant '{path_tenant_id}' not found.")


@router.post("", response_model=TenantCreateResponse, status_code=status.HTTP_201_CREATED)
def create_tenant(payload: TenantCreate, db: Session = Depends(get_db)) -> TenantCreateResponse:
    """Bootstrap endpoint — intentionally unauthenticated. Creates the
    tenant and issues its first API key in the same call, since a new
    tenant has no key to authenticate with beforehand."""
    service = TenantService(db)
    tenant, api_key, raw_key = service.create_tenant(name=payload.name, slug=payload.slug)

    return TenantCreateResponse(
        tenant=TenantRead.model_validate(tenant),
        api_key=ApiKeyIssuedRead(
            **ApiKeyRead.model_validate(api_key).model_dump(),
            raw_key=raw_key,
        ),
    )


@router.get("/{tenant_id}", response_model=TenantRead)
def get_tenant(
    tenant_id: uuid.UUID,
    authenticated_tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> TenantRead:
    _require_self(authenticated_tenant_id, tenant_id)
    service = TenantService(db)
    tenant = service.get_tenant(tenant_id)
    return TenantRead.model_validate(tenant)


@router.patch("/{tenant_id}", response_model=TenantRead)
def update_tenant(
    tenant_id: uuid.UUID,
    payload: TenantUpdate,
    authenticated_tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> TenantRead:
    _require_self(authenticated_tenant_id, tenant_id)
    service = TenantService(db)
    tenant = service.update_tenant(tenant_id, name=payload.name)
    return TenantRead.model_validate(tenant)


@router.post("/{tenant_id}/deactivate", response_model=TenantRead)
def deactivate_tenant(
    tenant_id: uuid.UUID,
    authenticated_tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> TenantRead:
    _require_self(authenticated_tenant_id, tenant_id)
    service = TenantService(db)
    tenant = service.deactivate_tenant(tenant_id)
    return TenantRead.model_validate(tenant)
