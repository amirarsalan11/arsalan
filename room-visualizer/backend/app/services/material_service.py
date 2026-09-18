"""
MaterialService.

Tenant-scoped material catalog CRUD foundation. No storage/S3 logic —
`texture_url` is accepted as a plain string supplied by the caller.
"""

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Material
from app.repositories import MaterialRepository
from app.services.exceptions import NotFoundError


class MaterialService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = MaterialRepository(session)

    def create_material(
        self,
        tenant_id: uuid.UUID,
        name: str,
        texture_url: str,
        category: str,
        width_mm: int | None = None,
        height_mm: int | None = None,
    ) -> Material:
        material = self.repository.create(
            tenant_id=tenant_id,
            name=name,
            texture_url=texture_url,
            category=category,
            width_mm=width_mm,
            height_mm=height_mm,
        )
        self.session.commit()
        return material

    def list_materials(self, tenant_id: uuid.UUID) -> list[Material]:
        return self.repository.list_by_tenant(tenant_id)

    def list_active_materials(self, tenant_id: uuid.UUID) -> list[Material]:
        """Used by the public materials endpoint — active-only,
        tenant-scoped. See MaterialRepository.list_active_by_tenant."""
        return self.repository.list_active_by_tenant(tenant_id)

    def get_material(self, tenant_id: uuid.UUID, material_id: uuid.UUID) -> Material:
        material = self.repository.get_by_id(tenant_id, material_id)
        if material is None:
            raise NotFoundError(f"Material '{material_id}' not found.")
        return material

    def update_material(
        self, tenant_id: uuid.UUID, material_id: uuid.UUID, **fields: Any
    ) -> Material:
        # Only pass through fields that were actually provided (non-None)
        # by the API layer's schema — see schemas/material.py.
        material = self.repository.update(tenant_id, material_id, **fields)
        if material is None:
            raise NotFoundError(f"Material '{material_id}' not found.")

        self.session.commit()
        return material

    def deactivate_material(self, tenant_id: uuid.UUID, material_id: uuid.UUID) -> Material:
        material = self.repository.update(tenant_id, material_id, is_active=False)
        if material is None:
            raise NotFoundError(f"Material '{material_id}' not found.")

        self.session.commit()
        return material
