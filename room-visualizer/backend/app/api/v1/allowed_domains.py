"""
AllowedDomain API routes — CRUD foundation only.

Lets a tenant manage its own domain whitelist. Per approved scope,
this milestone does NOT enforce domain validation on any route
(including these) — that enforcement is future infrastructure for
public/widget-facing endpoints that don't exist yet. These routes
themselves are still protected by API-key authentication like every
other tenant-scoped resource.
"""

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant_id, get_db
from app.schemas.allowed_domain import AllowedDomainCreate, AllowedDomainRead
from app.services import AllowedDomainService

router = APIRouter(prefix="/allowed-domains", tags=["allowed-domains"])


@router.post("", response_model=AllowedDomainRead, status_code=status.HTTP_201_CREATED)
def register_domain(
    payload: AllowedDomainCreate,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> AllowedDomainRead:
    service = AllowedDomainService(db)
    allowed_domain = service.register_domain(tenant_id=tenant_id, domain=payload.domain)
    return AllowedDomainRead.model_validate(allowed_domain)


@router.get("", response_model=list[AllowedDomainRead])
def list_domains(
    tenant_id: uuid.UUID = Depends(get_current_tenant_id), db: Session = Depends(get_db)
) -> list[AllowedDomainRead]:
    service = AllowedDomainService(db)
    domains = service.list_domains(tenant_id)
    return [AllowedDomainRead.model_validate(d) for d in domains]


@router.post("/{allowed_domain_id}/deactivate", response_model=AllowedDomainRead)
def deactivate_domain(
    allowed_domain_id: uuid.UUID,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> AllowedDomainRead:
    service = AllowedDomainService(db)
    allowed_domain = service.deactivate_domain(tenant_id, allowed_domain_id)
    return AllowedDomainRead.model_validate(allowed_domain)
