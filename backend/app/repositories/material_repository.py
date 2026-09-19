"""
MaterialRepository.

Tenant-scoped material catalog access. No global/shared catalog per the
Milestone 2 adjustment — every query here is tenant-filtered via
BaseRepository.
"""

import uuid
from typing import Any

from sqlalchemy import select

from app.db.models import Material
from app.repositories.base import BaseRepository


class MaterialRepository(BaseRepository[Material]):
    model = Material

    def create(self, tenant_id: uuid.UUID, **fields: Any) -> Material:
        material = Material(tenant_id=tenant_id, **fields)
        self.session.add(material)
        self.session.flush()
        return material

    def list_active_by_tenant(self, tenant_id: uuid.UUID) -> list[Material]:
        """Domain-specific query needed by the public materials
        endpoint (SDK + Widget milestone): unlike `list_by_tenant`
        (inherited from BaseRepository, returns everything), this
        filters to `is_active=True` only. A public, unauthenticated
        caller must never see deactivated materials — those may be
        deactivated precisely because they're being retired/hidden."""
        stmt = select(Material).where(
            Material.tenant_id == tenant_id,
            Material.is_active.is_(True),
        )
        return list(self.session.execute(stmt).scalars().all())
