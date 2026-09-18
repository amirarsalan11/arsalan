"""
RenderJob API routes — CRUD foundation only.

`create_render_job` only writes a `pending` row; no Celery dispatch, no
AI pipeline invocation happens here or anywhere else in this milestone.
`update_render_job_status` is exposed now as the foundation hook a
future Celery worker will call — nothing calls it automatically yet.
"""

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant_id, get_db
from app.schemas.render_job import RenderJobCreate, RenderJobRead, RenderJobStatusUpdate
from app.services import RenderJobService

router = APIRouter(prefix="/render-jobs", tags=["render-jobs"])


@router.post("", response_model=RenderJobRead, status_code=status.HTTP_201_CREATED)
def create_render_job(
    payload: RenderJobCreate,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> RenderJobRead:
    service = RenderJobService(db)
    render_job = service.create_render_job(
        tenant_id=tenant_id,
        input_image_url=payload.input_image_url,
        material_id=payload.material_id,
    )
    return RenderJobRead.model_validate(render_job)


@router.get("", response_model=list[RenderJobRead])
def list_render_jobs(
    tenant_id: uuid.UUID = Depends(get_current_tenant_id), db: Session = Depends(get_db)
) -> list[RenderJobRead]:
    service = RenderJobService(db)
    render_jobs = service.list_render_jobs(tenant_id)
    return [RenderJobRead.model_validate(r) for r in render_jobs]


@router.get("/{render_job_id}", response_model=RenderJobRead)
def get_render_job(
    render_job_id: uuid.UUID,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> RenderJobRead:
    service = RenderJobService(db)
    render_job = service.get_render_job(tenant_id, render_job_id)
    return RenderJobRead.model_validate(render_job)


@router.patch("/{render_job_id}/status", response_model=RenderJobRead)
def update_render_job_status(
    render_job_id: uuid.UUID,
    payload: RenderJobStatusUpdate,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> RenderJobRead:
    service = RenderJobService(db)
    render_job = service.update_render_job_status(
        tenant_id,
        render_job_id,
        status=payload.status,
        output_image_url=payload.output_image_url,
        error_message=payload.error_message,
    )
    return RenderJobRead.model_validate(render_job)
