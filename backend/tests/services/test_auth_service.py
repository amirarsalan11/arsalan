import uuid

import pytest
from sqlalchemy.orm import Session

from app.services import ApiKeyService, AuthenticationError, AuthService, TenantService


def _bootstrap(db_session: Session):
    """Create a tenant via the service layer and return (tenant, raw_key)
    for its auto-issued first API key."""
    tenant, _, raw_key = TenantService(db_session).create_tenant(name="Acme", slug="acme")
    return tenant, raw_key


def test_verify_api_key_resolves_correct_tenant(db_session: Session) -> None:
    tenant, raw_key = _bootstrap(db_session)

    resolved_tenant_id = AuthService(db_session).verify_api_key(raw_key)

    assert resolved_tenant_id == tenant.id


def test_verify_api_key_rejects_unknown_key(db_session: Session) -> None:
    with pytest.raises(AuthenticationError):
        AuthService(db_session).verify_api_key("rv_live_totally-unknown-value")


def test_verify_api_key_rejects_wrong_prefix(db_session: Session) -> None:
    _, raw_key = _bootstrap(db_session)
    # Same suffix, wrong prefix — must not verify even though the
    # underlying random portion is a real issued value.
    tampered = "not_" + raw_key

    with pytest.raises(AuthenticationError):
        AuthService(db_session).verify_api_key(tampered)


def test_verify_api_key_rejects_empty_string(db_session: Session) -> None:
    with pytest.raises(AuthenticationError):
        AuthService(db_session).verify_api_key("")


def test_verify_api_key_rejects_revoked_key(db_session: Session) -> None:
    tenant, raw_key = _bootstrap(db_session)

    api_key_service = ApiKeyService(db_session)
    bootstrap_key = api_key_service.list_api_keys(tenant.id)[0]
    api_key_service.revoke_api_key(tenant.id, bootstrap_key.id)

    with pytest.raises(AuthenticationError):
        AuthService(db_session).verify_api_key(raw_key)


def test_verify_api_key_rejects_key_for_deactivated_tenant(db_session: Session) -> None:
    tenant, raw_key = _bootstrap(db_session)

    TenantService(db_session).deactivate_tenant(tenant.id)

    with pytest.raises(AuthenticationError):
        AuthService(db_session).verify_api_key(raw_key)


def test_verify_api_key_updates_last_used_at(db_session: Session) -> None:
    tenant, raw_key = _bootstrap(db_session)

    api_key_service = ApiKeyService(db_session)
    bootstrap_key = api_key_service.list_api_keys(tenant.id)[0]
    assert bootstrap_key.last_used_at is None

    AuthService(db_session).verify_api_key(raw_key)

    refreshed = api_key_service.get_api_key(tenant.id, bootstrap_key.id)
    assert refreshed.last_used_at is not None


def test_different_tenants_keys_do_not_cross_resolve(db_session: Session) -> None:
    tenant_a, raw_key_a = _bootstrap(db_session)
    tenant_b, _, raw_key_b = TenantService(db_session).create_tenant(name="Other", slug="other")

    auth_service = AuthService(db_session)
    assert auth_service.verify_api_key(raw_key_a) == tenant_a.id
    assert auth_service.verify_api_key(raw_key_b) == tenant_b.id
