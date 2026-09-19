import uuid

from sqlalchemy.orm import Session

from app.repositories import ApiKeyRepository, TenantRepository


def _tenant_id(db_session: Session) -> uuid.UUID:
    tenant = TenantRepository(db_session).create(name="Acme", slug="acme")
    db_session.commit()
    return tenant.id


def test_api_key_create_and_get_by_key_hash(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    repo = ApiKeyRepository(db_session)

    api_key = repo.create(tenant_id=tenant_id, label="Prod key", key_hash="hash-abc")
    db_session.commit()

    fetched = repo.get_by_key_hash("hash-abc")
    assert fetched is not None
    assert fetched.id == api_key.id

    assert repo.get_by_key_hash("does-not-exist") is None


def test_api_key_get_by_id_is_tenant_scoped(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    other_tenant_id = uuid.uuid4()
    repo = ApiKeyRepository(db_session)
    api_key = repo.create(tenant_id=tenant_id, label="Prod key", key_hash="hash-abc")
    db_session.commit()

    assert repo.get_by_id(tenant_id, api_key.id) is not None
    assert repo.get_by_id(other_tenant_id, api_key.id) is None
