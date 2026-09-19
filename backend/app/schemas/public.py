"""
Public-facing Pydantic schemas.

Deliberately NOT reusing `TenantRead`/`MaterialRead` (the
authenticated, internal-facing schemas) here. Those schemas are
maintained for internal/authenticated consumers and could grow
internal-only fields over time (e.g. billing status, internal notes)
without anyone thinking to check whether a public schema also
re-exports them. Keeping fully separate public schemas means a future
field added to `Tenant`/`Material` never leaks to the public router by
accident — someone has to deliberately add it here too.

Notably absent from `PublicMaterialRead`: `texture_url` and any other
storage-location detail. Per the approved scope ("never expose
internal storage information"), the public materials endpoint returns
just enough for a selection UI to render a catalog entry, not the raw
asset location. Once real storage/CDN delivery exists, a deliberate
decision should be made about what (if anything) becomes safe to
expose here — this milestone does not make that call.
"""

import uuid

from pydantic import BaseModel, ConfigDict


class PublicTenantConfig(BaseModel):
    """Response for GET /public/tenants/{tenant_id}/config.

    Only what a public embed needs to render itself sensibly: is this
    tenant real and active, and what's its display name. No slug (used
    in internal URL contexts, not meant for public disclosure by
    design choice, even though it isn't a secret), no timestamps, no
    counts, nothing else.
    """

    model_config = ConfigDict(from_attributes=True)

    tenant_id: uuid.UUID
    name: str
    is_active: bool


class PublicMaterialRead(BaseModel):
    """Response for GET /public/tenants/{tenant_id}/materials.

    Deliberately excludes: `texture_url` (storage location detail),
    `width_mm`/`height_mm` (not yet needed publicly, and are a
    property of the internal Material Engine's future scale-estimation
    use, not the selection UI), `created_at`/`updated_at`, and
    `is_active` itself (the endpoint already filters to active-only —
    exposing the flag would just invite a client to try passing
    inactive materials through some other path).
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    category: str
