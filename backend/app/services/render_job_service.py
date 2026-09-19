"""
RenderJobService.

CRUD foundation only. `create_render_job` writes a `pending` row and
does nothing else — no Celery dispatch, no AI pipeline invocation.
`update_render_job_status` exists as the hook a future Celery worker
will call to transition status/populate output_image_url/error_message,
but nothing calls it automatically in this milestone.
"""

import uuid

from sqlalchemy.orm import Session

from app.db.models import Material, RenderJob, RenderJobStatus
from app.repositories import MaterialRepository, RenderJobRepository
from app.services.exceptions import NotFoundError


class RenderJobService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = RenderJobRepository(session)
        self._material_repository = MaterialRepository(session)

    def create_render_job(
        self,
        tenant_id: uuid.UUID,
        input_image_url: str,
        material_id: uuid.UUID | None = None,
    ) -> RenderJob:
        if material_id is not None:
            material: Material | None = self._material_repository.get_by_id(
                tenant_id, material_id
            )
            if material is None:
                raise NotFoundError(
                    f"Material '{material_id}' not found for this tenant."
                )

        render_job = self.repository.create(
            tenant_id=tenant_id,
            input_image_url=input_image_url,
            material_id=material_id,
        )
        self.session.commit()
        return render_job

    def list_render_jobs(self, tenant_id: uuid.UUID) -> list[RenderJob]:
        return self.repository.list_by_tenant(tenant_id)

    def get_render_job(self, tenant_id: uuid.UUID, render_job_id: uuid.UUID) -> RenderJob:
        render_job = self.repository.get_by_id(tenant_id, render_job_id)
        if render_job is None:
            raise NotFoundError(f"RenderJob '{render_job_id}' not found.")
        return render_job

    def update_render_job_status(
        self,
        tenant_id: uuid.UUID,
        render_job_id: uuid.UUID,
        status: RenderJobStatus,
        output_image_url: str | None = None,
        error_message: str | None = None,
    ) -> RenderJob:
        fields: dict = {"status": status}
        if output_image_url is not None:
            fields["output_image_url"] = output_image_url
        if error_message is not None:
            fields["error_message"] = error_message

        render_job = self.repository.update(tenant_id, render_job_id, **fields)
        if render_job is None:
            raise NotFoundError(f"RenderJob '{render_job_id}' not found.")

        self.session.commit()
        return render_job
