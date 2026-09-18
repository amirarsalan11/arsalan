"""Pydantic schemas for the Material resource."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MaterialCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    texture_url: str = Field(min_length=1, max_length=1024)
    category: str = Field(min_length=1, max_length=100)
    width_mm: int | None = Field(default=None, gt=0)
    height_mm: int | None = Field(default=None, gt=0)


class MaterialUpdate(BaseModel):
    """All fields optional — a PATCH only touches what's provided."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    texture_url: str | None = Field(default=None, min_length=1, max_length=1024)
    category: str | None = Field(default=None, min_length=1, max_length=100)
    width_mm: int | None = Field(default=None, gt=0)
    height_mm: int | None = Field(default=None, gt=0)


class MaterialRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    texture_url: str
    category: str
    width_mm: int | None
    height_mm: int | None
    is_active: bool
    created_at: datetime
    updated_at: datetime
