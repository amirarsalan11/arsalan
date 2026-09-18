"""Pydantic schemas for the AllowedDomain resource (CRUD foundation only)."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AllowedDomainCreate(BaseModel):
    domain: str = Field(min_length=1, max_length=255)


class AllowedDomainRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    domain: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
