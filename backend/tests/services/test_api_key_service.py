import uuid

import pytest
from sqlalchemy.orm import Session

from app.services import ApiKeyService, NotFoundError, TenantService


def _tenant_id(db_session: Session) -> uuid.UUID:
    tenant, _, _ = TenantService(db_session).create_tenant(name="Acme", slug="acme")
    return tenant.id


def test_create_api_key_generates_raw_key_and_stores_only_hash(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = ApiKeyService(db_session)

    api_key, raw_key = service.create_api_key(tenant_id, label="Prod key")

    assert raw_key.startswith("rv_live_")
    # The persisted record never carries the raw value — only a hash,
    # and that hash must not equal the raw key itself.
    assert api_key.key_hash != raw_key
    assert not hasattr(api_key, "raw_key")

    fetched = service.get_api_key(tenant_id, api_key.id)
    assert fetched.id == api_key.id
    assert fetched.is_active is True


def test_create_api_key_raw_keys_are_unique(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = ApiKeyService(db_session)

    _, raw_key_1 = service.create_api_key(tenant_id, label="Key 1")
    _, raw_key_2 = service.create_api_key(tenant_id, label="Key 2")

    assert raw_key_1 != raw_key_2


def test_get_api_key_not_found_raises(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = ApiKeyService(db_session)

    with pytest.raises(NotFoundError):
        service.get_api_key(tenant_id, uuid.uuid4())


def test_revoke_api_key(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = ApiKeyService(db_session)
    api_key, _ = service.create_api_key(tenant_id, label="Prod key")

    revoked = service.revoke_api_key(tenant_id, api_key.id)
    assert revoked.is_active is False


def test_list_api_keys_is_tenant_scoped(db_session: Session) -> None:
    tenant_a = _tenant_id(db_session)
    tenant_b, _, _ = TenantService(db_session).create_tenant(name="Other", slug="other")
    tenant_b = tenant_b.id
    service = ApiKeyService(db_session)

    service.create_api_key(tenant_a, label="A key")
    service.create_api_key(tenant_b, label="B key")

    # Each tenant already has its bootstrap key plus the one created here.
    assert len(service.list_api_keys(tenant_a)) == 2
    assert len(service.list_api_keys(tenant_b)) == 2
