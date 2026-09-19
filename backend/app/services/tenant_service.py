"""
TenantService.

Authentication milestone change: `create_tenant` now also issues the
tenant's first API key as part of the same bootstrap call (see
Authentication milestone plan for why this is the chosen bootstrap
mechanism — a brand-new tenant has no key yet, so key creation cannot
itself require authentication). The raw key is returned once, bundled
with the tenant, and never persisted anywhere.

Per the approved adjustment, there is no `list_tenants` — an unprotected
"list all tenants" endpoint would leak cross-tenant data. Only create /
get / update / deactivate exist, matching the approved API surface.
"""

import uuid

from sqlalchemy.orm import Session

from app.db.models import ApiKey, Tenant
from app.repositories import TenantRepository
from app.services.api_key_service import ApiKeyService
from app.services.exceptions import ConflictError, NotFoundError

_BOOTSTRAP_API_KEY_LABEL = "Default API Key"


class TenantService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = TenantRepository(session)

    def create_tenant(self, name: str, slug: str) -> tuple[Tenant, ApiKey, str]:
        """Create a tenant and automatically issue its first API key.

        Returns (tenant, api_key, raw_key). `raw_key` must be surfaced
        to the caller in this response and never again — see
        ApiKeyService.create_api_key.
        """
        if self.repository.get_by_slug(slug) is not None:
            raise ConflictError(f"A tenant with slug '{slug}' already exists.")

        tenant = self.repository.create(name=name, slug=slug)
        # Flush (not commit) so `tenant.id` is assigned and visible within
        # this same transaction before the API key references it as a
        # foreign key. The API key service's own commit below finalizes
        # both inserts atomically.
        self.session.flush()

        api_key_service = ApiKeyService(self.session)
        api_key, raw_key = api_key_service.create_api_key(
            tenant_id=tenant.id, label=_BOOTSTRAP_API_KEY_LABEL
        )

        return tenant, api_key, raw_key

    def get_tenant(self, tenant_id: uuid.UUID) -> Tenant:
        tenant = self.repository.get_by_id(tenant_id)
        if tenant is None:
            raise NotFoundError(f"Tenant '{tenant_id}' not found.")
        return tenant

    def update_tenant(self, tenant_id: uuid.UUID, *, name: str | None = None) -> Tenant:
        # Only `name` is updatable here. `slug` is intentionally not
        # editable via this foundation-level update — changing a tenant's
        # slug has wider implications (URLs, integrations) that belong to
        # a deliberate future decision, not a generic PATCH field.
        fields = {}
        if name is not None:
            fields["name"] = name

        tenant = self.repository.update(tenant_id, **fields)
        if tenant is None:
            raise NotFoundError(f"Tenant '{tenant_id}' not found.")

        self.session.commit()
        return tenant

    def deactivate_tenant(self, tenant_id: uuid.UUID) -> Tenant:
        tenant = self.repository.update(tenant_id, is_active=False)
        if tenant is None:
            raise NotFoundError(f"Tenant '{tenant_id}' not found.")

        self.session.commit()
        return tenant
