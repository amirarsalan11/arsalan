"""
AllowedDomainService.

CRUD foundation for a tenant's domain whitelist, plus
`is_domain_allowed` — the matching primitive a future request-time
dependency (protecting the not-yet-built public widget/embed
endpoints) will call. Per the approved scope, this milestone does NOT
wire that check into any existing route; it exists as tested, ready
infrastructure.
"""

import uuid

from sqlalchemy.orm import Session

from app.db.models import AllowedDomain
from app.repositories import AllowedDomainRepository
from app.services.exceptions import ConflictError, NotFoundError


class AllowedDomainService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = AllowedDomainRepository(session)

    def register_domain(self, tenant_id: uuid.UUID, domain: str) -> AllowedDomain:
        normalized = _normalize_domain(domain)

        existing = self.repository.get_by_domain(tenant_id, normalized)
        if existing is not None:
            raise ConflictError(f"Domain '{normalized}' is already registered for this tenant.")

        allowed_domain = self.repository.create(tenant_id=tenant_id, domain=normalized)
        self.session.commit()
        return allowed_domain

    def list_domains(self, tenant_id: uuid.UUID) -> list[AllowedDomain]:
        return self.repository.list_by_tenant(tenant_id)

    def deactivate_domain(self, tenant_id: uuid.UUID, allowed_domain_id: uuid.UUID) -> AllowedDomain:
        allowed_domain = self.repository.update(tenant_id, allowed_domain_id, is_active=False)
        if allowed_domain is None:
            raise NotFoundError(f"AllowedDomain '{allowed_domain_id}' not found.")

        self.session.commit()
        return allowed_domain

    def is_domain_allowed(self, tenant_id: uuid.UUID, origin_or_domain: str) -> bool:
        """Domain-matching primitive for future request-time enforcement.

        Not called by any route in this milestone. Extracts the
        hostname from a full origin/URL if one is passed (e.g.
        "https://shop.example.com:443/path" -> "shop.example.com"),
        normalizes it the same way `register_domain` does, and checks
        for an *active* match on the tenant's whitelist.

        Exact-match only for the MVP — no wildcard/subdomain matching
        (e.g. registering "example.com" does NOT allow
        "shop.example.com"). Wildcard support, if ever needed, is a
        deliberate future decision, not an accidental byproduct of this
        primitive.
        """
        normalized = _normalize_domain(origin_or_domain)
        match = self.repository.get_by_domain(tenant_id, normalized)
        return match is not None and match.is_active


def _normalize_domain(raw: str) -> str:
    """Extract and normalize a bare hostname from a domain or full origin.

    Handles being passed either a bare domain ("shop.example.com") or a
    full Origin/Referer-style URL ("https://shop.example.com:8443/foo").
    Lowercases the result and strips a trailing port, since ports are
    not part of the identity we're validating.
    """
    value = raw.strip().lower()

    if "://" in value:
        value = value.split("://", 1)[1]

    # Strip path/query if a full URL slipped through.
    value = value.split("/", 1)[0]

    # Strip port, if present.
    value = value.split(":", 1)[0]

    return value
