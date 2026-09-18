import uuid

import pytest
from sqlalchemy.orm import Session

from app.db.models import RenderJobStatus
from app.services import (
    MaterialService,
    NotFoundError,
    RenderJobService,
    TenantService,
)


def _tenant_id(db_session: Session) -> uuid.UUID:
    tenant, _, _ = TenantService(db_session).create_tenant(name="Acme", slug="acme")
    return tenant.id


def test_create_render_job_defaults_pending(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = RenderJobService(db_session)

    render_job = service.create_render_job(
        tenant_id=tenant_id, input_image_url="https://cdn.example.com/in.jpg"
    )

    assert render_job.status == RenderJobStatus.PENDING


def test_create_render_job_with_invalid_material_raises(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = RenderJobService(db_session)

    with pytest.raises(NotFoundError):
        service.create_render_job(
            tenant_id=tenant_id,
            input_image_url="https://cdn.example.com/in.jpg",
            material_id=uuid.uuid4(),
        )


def test_create_render_job_with_valid_material_succeeds(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    material = MaterialService(db_session).create_material(
        tenant_id=tenant_id, name="Oak", texture_url="u1", category="hardwood"
    )
    service = RenderJobService(db_session)

    render_job = service.create_render_job(
        tenant_id=tenant_id,
        input_image_url="https://cdn.example.com/in.jpg",
        material_id=material.id,
    )
    assert render_job.material_id == material.id


def test_update_render_job_status(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = RenderJobService(db_session)
    render_job = service.create_render_job(
        tenant_id=tenant_id, input_image_url="https://cdn.example.com/in.jpg"
    )

    updated = service.update_render_job_status(
        tenant_id,
        render_job.id,
        status=RenderJobStatus.COMPLETED,
        output_image_url="https://cdn.example.com/out.jpg",
    )
    assert updated.status == RenderJobStatus.COMPLETED
    assert updated.output_image_url == "https://cdn.example.com/out.jpg"
