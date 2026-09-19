"""
RenderJobRepository.

CRUD-foundation only. Creating a row here does NOT enqueue anything —
no Celery/Redis interaction exists in this milestone. `update` (inherited
from BaseRepository) is the hook a future Celery worker will call to
transition status/populate output_image_url, but nothing calls it yet
except this milestone's own API endpoint.
"""

import uuid

from app.db.models import RenderJob, RenderJobStatus
from app.repositories.base import BaseRepository


class RenderJobRepository(BaseRepository[RenderJob]):
    model = RenderJob

    def create(
        self,
        tenant_id: uuid.UUID,
        input_image_url: str,
        material_id: uuid.UUID | None = None,
    ) -> RenderJob:
        render_job = RenderJob(
            tenant_id=tenant_id,
            input_image_url=input_image_url,
            material_id=material_id,
            status=RenderJobStatus.PENDING,
        )
        self.session.add(render_job)
        self.session.flush()
        return render_job
