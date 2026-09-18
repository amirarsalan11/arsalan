"""
Pydantic schemas for the UsageRecord resource.

Read-only schema surface — no Create schema, matching the "read
foundation" scope (no public create endpoint exists).
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UsageRecordRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    render_job_id: uuid.UUID | None
    event_type: str
    created_at: datetime
