"""
AllowedDomainRepository.

CRUD foundation for a tenant's domain whitelist, plus `get_by_domain`
— the lookup a future request-time validation dependency will need
to check "is this Origin/Referer allowed for this tenant". Nothing in
this milestone wires that check into any existing route (see
AllowedDomainService.is_domain_allowed for the matching primitive).
"""

import uuid

from sqlalchemy import select

from app.db.models import AllowedDomain
from app.repositories.base import BaseRepository


class AllowedDomainRepository(BaseRepository[AllowedDomain]):
    model = AllowedDomain

    def create(self, tenant_id: uuid.UUID, domain: str) -> AllowedDomain:
        allowed_domain = AllowedDomain(tenant_id=tenant_id, domain=domain)
        self.session.add(allowed_domain)
        self.session.flush()
        return allowed_domain

    def get_by_domain(self, tenant_id: uuid.UUID, domain: str) -> AllowedDomain | None:
        """Domain-specific lookup: is `domain` registered for this tenant
        (active or not — callers decide what to do with an inactive
        match)."""
        stmt = select(AllowedDomain).where(
            AllowedDomain.tenant_id == tenant_id,
            AllowedDomain.domain == domain,
        )
        return self.session.execute(stmt).scalar_one_or_none()
