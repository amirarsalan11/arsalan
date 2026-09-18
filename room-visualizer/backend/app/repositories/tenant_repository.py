"""
TenantRepository.

Tenant is the aggregate root and has no tenant_id of its own, so it
does NOT inherit from BaseRepository (whose get_by_id/update assume a
tenant_id column). This repository implements its own get_by_id/update,
plus the domain-specific get_by_slug lookup.
"""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Tenant


class TenantRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, name: str, slug: str) -> Tenant:
        tenant = Tenant(name=name, slug=slug)
        self.session.add(tenant)
        self.session.flush()
        return tenant

    def get_by_id(self, tenant_id: uuid.UUID) -> Tenant | None:
        stmt = select(Tenant).where(Tenant.id == tenant_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_slug(self, slug: str) -> Tenant | None:
        """Domain-specific lookup — kept on this repository, not the
        shared base, since it has no equivalent on any other model."""
        stmt = select(Tenant).where(Tenant.slug == slug)
        return self.session.execute(stmt).scalar_one_or_none()

    def update(self, tenant_id: uuid.UUID, **fields: Any) -> Tenant | None:
        tenant = self.get_by_id(tenant_id)
        if tenant is None:
            return None

        for field_name, value in fields.items():
            setattr(tenant, field_name, value)

        self.session.flush()
        return tenant
