"""Pydantic schemas for the Tenant resource."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.api_key import ApiKeyIssuedRead


class TenantCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    slug: str = Field(min_length=1, max_length=100)


class TenantUpdate(BaseModel):
    """All fields optional — a PATCH only touches what's provided.

    Note: `slug` is intentionally not included. See TenantService for
    the reasoning (slug changes are a deliberate future decision, not a
    generic PATCH field).
    """

    name: str | None = Field(default=None, min_length=1, max_length=255)


class TenantRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class TenantCreateResponse(BaseModel):
    """Response shape for the tenant bootstrap flow (POST /tenants).

    Bundles the newly created tenant with its automatically issued
    first API key. `api_key.raw_key` is transmitted here and ONLY
    here — it is not persisted and cannot be retrieved again after
    this response.
    """

    tenant: TenantRead
    api_key: ApiKeyIssuedRead
