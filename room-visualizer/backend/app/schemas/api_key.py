"""
Pydantic schemas for the ApiKey resource.

Authentication milestone change: `ApiKeyCreate` no longer accepts a
caller-supplied `key_hash`. The raw key is now generated server-side
(app/core/security.py) and its hash is what gets persisted — the
caller only supplies a human-readable `label`. See `ApiKeyIssuedRead`
for the one-time response shape that includes the raw key.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ApiKeyCreate(BaseModel):
    label: str = Field(min_length=1, max_length=255)


class ApiKeyRead(BaseModel):
    """Standard read shape. Never includes the raw key or the hash —
    only used for list/get/revoke responses, where the raw key is
    already gone (it only ever existed at creation time)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    label: str
    is_active: bool
    last_used_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ApiKeyIssuedRead(ApiKeyRead):
    """Returned ONLY from the creation endpoint (and the tenant
    bootstrap flow). Includes `raw_key` — the one and only moment this
    value is ever transmitted. It is never persisted and can never be
    retrieved again after this response; if lost, the caller must
    revoke the key and issue a new one.
    """

    raw_key: str
