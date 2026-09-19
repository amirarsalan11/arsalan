"""Material API routes — tenant-scoped CRUD foundation."""

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant_id, get_db
from app.schemas.material import MaterialCreate, MaterialRead, MaterialUpdate
from app.services import MaterialService

router = APIRouter(prefix="/materials", tags=["materials"])


@router.post("", response_model=MaterialRead, status_code=status.HTTP_201_CREATED)
def create_material(
    payload: MaterialCreate,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> MaterialRead:
    service = MaterialService(db)
    material = service.create_material(
        tenant_id=tenant_id,
        name=payload.name,
        texture_url=payload.texture_url,
        category=payload.category,
        width_mm=payload.width_mm,
        height_mm=payload.height_mm,
    )
    return MaterialRead.model_validate(material)


@router.get("", response_model=list[MaterialRead])
def list_materials(
    tenant_id: uuid.UUID = Depends(get_current_tenant_id), db: Session = Depends(get_db)
) -> list[MaterialRead]:
    service = MaterialService(db)
    materials = service.list_materials(tenant_id)
    return [MaterialRead.model_validate(m) for m in materials]


@router.get("/{material_id}", response_model=MaterialRead)
def get_material(
    material_id: uuid.UUID,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> MaterialRead:
    service = MaterialService(db)
    material = service.get_material(tenant_id, material_id)
    return MaterialRead.model_validate(material)


@router.patch("/{material_id}", response_model=MaterialRead)
def update_material(
    material_id: uuid.UUID,
    payload: MaterialUpdate,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> MaterialRead:
    service = MaterialService(db)
    # Only pass fields the caller actually set, so an omitted field never
    # overwrites existing data with None.
    fields = payload.model_dump(exclude_unset=True)
    material = service.update_material(tenant_id, material_id, **fields)
    return MaterialRead.model_validate(material)


@router.post("/{material_id}/deactivate", response_model=MaterialRead)
def deactivate_material(
    material_id: uuid.UUID,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> MaterialRead:
    service = MaterialService(db)
    material = service.deactivate_material(tenant_id, material_id)
    return MaterialRead.model_validate(material)
