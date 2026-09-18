import uuid

from sqlalchemy.orm import Session

from app.db.models import RenderJobStatus
from app.repositories import RenderJobRepository, TenantRepository


def _tenant_id(db_session: Session) -> uuid.UUID:
    tenant = TenantRepository(db_session).create(name="Acme", slug="acme")
    db_session.commit()
    return tenant.id


def test_render_job_create_defaults_to_pending(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    repo = RenderJobRepository(db_session)

    render_job = repo.create(tenant_id=tenant_id, input_image_url="https://cdn.example.com/in.jpg")
    db_session.commit()

    assert render_job.status == RenderJobStatus.PENDING
    assert render_job.material_id is None


def test_render_job_update_transitions_status(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    repo = RenderJobRepository(db_session)
    render_job = repo.create(tenant_id=tenant_id, input_image_url="https://cdn.example.com/in.jpg")
    db_session.commit()

    updated = repo.update(
        tenant_id,
        render_job.id,
        status=RenderJobStatus.COMPLETED,
        output_image_url="https://cdn.example.com/out.jpg",
    )
    assert updated is not None
    assert updated.status == RenderJobStatus.COMPLETED
    assert updated.output_image_url == "https://cdn.example.com/out.jpg"
