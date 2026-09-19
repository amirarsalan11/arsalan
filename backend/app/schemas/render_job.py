"""
Pydantic schemas for the RenderJob resource.

`RenderJobStatusUpdate` exists as the request shape for the foundation
status-update endpoint — nothing calls this automatically yet; it's the
hook a future Celery worker milestone will use.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import RenderJobStatus


class RenderJobCreate(BaseModel):
    input_image_url: str = Field(min_length=1, max_length=1024)
    material_id: uuid.UUID | None = None


class RenderJobStatusUpdate(BaseModel):
    status: RenderJobStatus
    output_image_url: str | None = Field(default=None, max_length=1024)
    error_message: str | None = Field(default=None, max_length=2048)


class RenderJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    material_id: uuid.UUID | None
    status: RenderJobStatus
    input_image_url: str
    output_image_url: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
