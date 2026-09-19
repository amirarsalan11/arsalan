"""
ApiKey API routes.

Authentication milestone changes:
- Tenant is now resolved via `get_current_tenant_id` (real
  Authorization: Bearer <key> verification) instead of the removed
  X-Tenant-Id placeholder.
- `create_api_key` no longer accepts a caller-supplied hash — the raw
  key is generated server-side and returned exactly once in this
  response (`ApiKeyIssuedRead.raw_key`). It is never persisted or
  retrievable again afterward.
"""

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant_id, get_db
from app.schemas.api_key import ApiKeyCreate, ApiKeyIssuedRead, ApiKeyRead
from app.services import ApiKeyService

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


@router.post("", response_model=ApiKeyIssuedRead, status_code=status.HTTP_201_CREATED)
def create_api_key(
    payload: ApiKeyCreate,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> ApiKeyIssuedRead:
    service = ApiKeyService(db)
    api_key, raw_key = service.create_api_key(tenant_id=tenant_id, label=payload.label)
    return ApiKeyIssuedRead(
        **ApiKeyRead.model_validate(api_key).model_dump(),
        raw_key=raw_key,
    )


@router.get("", response_model=list[ApiKeyRead])
def list_api_keys(
    tenant_id: uuid.UUID = Depends(get_current_tenant_id), db: Session = Depends(get_db)
) -> list[ApiKeyRead]:
    service = ApiKeyService(db)
    api_keys = service.list_api_keys(tenant_id)
    return [ApiKeyRead.model_validate(k) for k in api_keys]


@router.get("/{api_key_id}", response_model=ApiKeyRead)
def get_api_key(
    api_key_id: uuid.UUID,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> ApiKeyRead:
    service = ApiKeyService(db)
    api_key = service.get_api_key(tenant_id, api_key_id)
    return ApiKeyRead.model_validate(api_key)


@router.post("/{api_key_id}/revoke", response_model=ApiKeyRead)
def revoke_api_key(
    api_key_id: uuid.UUID,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> ApiKeyRead:
    service = ApiKeyService(db)
    api_key = service.revoke_api_key(tenant_id, api_key_id)
    return ApiKeyRead.model_validate(api_key)
