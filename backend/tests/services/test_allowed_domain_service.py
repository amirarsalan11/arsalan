import uuid

import pytest
from sqlalchemy.orm import Session

from app.services import AllowedDomainService, ConflictError, NotFoundError, TenantService


def _tenant_id(db_session: Session) -> uuid.UUID:
    tenant, _, _ = TenantService(db_session).create_tenant(name="Acme", slug="acme")
    return tenant.id


def test_register_domain(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = AllowedDomainService(db_session)

    allowed_domain = service.register_domain(tenant_id, "shop.example.com")

    assert allowed_domain.domain == "shop.example.com"
    assert allowed_domain.is_active is True


def test_register_domain_normalizes_full_origin_to_bare_hostname(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = AllowedDomainService(db_session)

    allowed_domain = service.register_domain(tenant_id, "https://Shop.Example.com:8443/path")

    assert allowed_domain.domain == "shop.example.com"


def test_register_duplicate_domain_raises_conflict(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = AllowedDomainService(db_session)

    service.register_domain(tenant_id, "shop.example.com")
    with pytest.raises(ConflictError):
        service.register_domain(tenant_id, "shop.example.com")


def test_list_domains_is_tenant_scoped(db_session: Session) -> None:
    tenant_a = _tenant_id(db_session)
    tenant_b, _, _ = TenantService(db_session).create_tenant(name="Other", slug="other")
    tenant_b = tenant_b.id

    service = AllowedDomainService(db_session)
    service.register_domain(tenant_a, "a.example.com")
    service.register_domain(tenant_b, "b.example.com")

    assert [d.domain for d in service.list_domains(tenant_a)] == ["a.example.com"]
    assert [d.domain for d in service.list_domains(tenant_b)] == ["b.example.com"]


def test_deactivate_domain(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = AllowedDomainService(db_session)
    allowed_domain = service.register_domain(tenant_id, "shop.example.com")

    deactivated = service.deactivate_domain(tenant_id, allowed_domain.id)
    assert deactivated.is_active is False


def test_deactivate_unknown_domain_raises_not_found(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = AllowedDomainService(db_session)

    with pytest.raises(NotFoundError):
        service.deactivate_domain(tenant_id, uuid.uuid4())


# --- Domain-matching primitive (foundation only, not wired to any route) ---


def test_is_domain_allowed_true_for_registered_active_domain(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = AllowedDomainService(db_session)
    service.register_domain(tenant_id, "shop.example.com")

    assert service.is_domain_allowed(tenant_id, "shop.example.com") is True


def test_is_domain_allowed_matches_full_origin_against_bare_registration(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = AllowedDomainService(db_session)
    service.register_domain(tenant_id, "shop.example.com")

    assert service.is_domain_allowed(tenant_id, "https://shop.example.com/checkout") is True


def test_is_domain_allowed_false_for_unregistered_domain(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = AllowedDomainService(db_session)

    assert service.is_domain_allowed(tenant_id, "unregistered.example.com") is False


def test_is_domain_allowed_false_after_deactivation(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = AllowedDomainService(db_session)
    allowed_domain = service.register_domain(tenant_id, "shop.example.com")
    service.deactivate_domain(tenant_id, allowed_domain.id)

    assert service.is_domain_allowed(tenant_id, "shop.example.com") is False


def test_is_domain_allowed_does_not_match_subdomain(db_session: Session) -> None:
    """Exact-match only for the MVP — registering the parent domain
    must NOT implicitly allow a subdomain (or vice versa)."""
    tenant_id = _tenant_id(db_session)
    service = AllowedDomainService(db_session)
    service.register_domain(tenant_id, "example.com")

    assert service.is_domain_allowed(tenant_id, "shop.example.com") is False


def test_is_domain_allowed_is_tenant_scoped(db_session: Session) -> None:
    tenant_a = _tenant_id(db_session)
    tenant_b, _, _ = TenantService(db_session).create_tenant(name="Other", slug="other")
    tenant_b = tenant_b.id

    service = AllowedDomainService(db_session)
    service.register_domain(tenant_a, "shop.example.com")

    assert service.is_domain_allowed(tenant_a, "shop.example.com") is True
    assert service.is_domain_allowed(tenant_b, "shop.example.com") is False
