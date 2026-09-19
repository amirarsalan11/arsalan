import uuid

from sqlalchemy.orm import Session

from app.repositories import AllowedDomainRepository, TenantRepository


def _tenant_id(db_session: Session) -> uuid.UUID:
    tenant = TenantRepository(db_session).create(name="Acme", slug="acme")
    db_session.commit()
    return tenant.id


def test_create_and_get_by_domain(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    repo = AllowedDomainRepository(db_session)

    repo.create(tenant_id=tenant_id, domain="shop.example.com")
    db_session.commit()

    fetched = repo.get_by_domain(tenant_id, "shop.example.com")
    assert fetched is not None
    assert fetched.domain == "shop.example.com"
    assert fetched.is_active is True


def test_get_by_domain_returns_none_for_unregistered_domain(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    repo = AllowedDomainRepository(db_session)

    assert repo.get_by_domain(tenant_id, "unregistered.example.com") is None


def test_get_by_domain_is_tenant_scoped(db_session: Session) -> None:
    tenant_a = _tenant_id(db_session)
    tenant_b = TenantRepository(db_session).create(name="Other", slug="other").id
    db_session.commit()

    repo = AllowedDomainRepository(db_session)
    repo.create(tenant_id=tenant_a, domain="shop.example.com")
    db_session.commit()

    # Same domain string, but registered for tenant A — tenant B must
    # not see it as its own.
    assert repo.get_by_domain(tenant_a, "shop.example.com") is not None
    assert repo.get_by_domain(tenant_b, "shop.example.com") is None


def test_list_by_tenant(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    repo = AllowedDomainRepository(db_session)

    repo.create(tenant_id=tenant_id, domain="a.example.com")
    repo.create(tenant_id=tenant_id, domain="b.example.com")
    db_session.commit()

    domains = repo.list_by_tenant(tenant_id)
    assert {d.domain for d in domains} == {"a.example.com", "b.example.com"}


def test_update_deactivates_domain(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    repo = AllowedDomainRepository(db_session)
    allowed_domain = repo.create(tenant_id=tenant_id, domain="shop.example.com")
    db_session.commit()

    updated = repo.update(tenant_id, allowed_domain.id, is_active=False)
    db_session.commit()

    assert updated is not None
    assert updated.is_active is False
