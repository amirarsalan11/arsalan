import uuid

from sqlalchemy.orm import Session

from app.repositories import TenantRepository


def test_create_and_get_by_id(db_session: Session) -> None:
    repo = TenantRepository(db_session)
    tenant = repo.create(name="Acme Inc", slug="acme")
    db_session.commit()

    fetched = repo.get_by_id(tenant.id)
    assert fetched is not None
    assert fetched.slug == "acme"


def test_get_by_id_returns_none_when_missing(db_session: Session) -> None:
    repo = TenantRepository(db_session)
    assert repo.get_by_id(uuid.uuid4()) is None


def test_get_by_slug(db_session: Session) -> None:
    repo = TenantRepository(db_session)
    repo.create(name="Acme Inc", slug="acme")
    db_session.commit()

    fetched = repo.get_by_slug("acme")
    assert fetched is not None
    assert fetched.name == "Acme Inc"
    assert repo.get_by_slug("does-not-exist") is None


def test_update_returns_none_when_missing(db_session: Session) -> None:
    repo = TenantRepository(db_session)
    assert repo.update(uuid.uuid4(), name="New Name") is None


def test_update_applies_fields(db_session: Session) -> None:
    repo = TenantRepository(db_session)
    tenant = repo.create(name="Old Name", slug="acme")
    db_session.commit()

    updated = repo.update(tenant.id, name="New Name")
    assert updated is not None
    assert updated.name == "New Name"
