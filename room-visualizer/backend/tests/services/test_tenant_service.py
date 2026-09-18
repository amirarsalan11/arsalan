import uuid

import pytest
from sqlalchemy.orm import Session

from app.services import ConflictError, NotFoundError, TenantService


def test_create_tenant_also_issues_first_api_key(db_session: Session) -> None:
    service = TenantService(db_session)
    tenant, api_key, raw_key = service.create_tenant(name="Acme", slug="acme")

    assert tenant.name == "Acme"
    assert tenant.is_active is True

    # Bootstrap flow: a first API key is issued automatically, scoped to
    # the new tenant, and the raw value is a real generated key.
    assert api_key.tenant_id == tenant.id
    assert api_key.is_active is True
    assert raw_key.startswith("rv_live_")


def test_create_tenant_duplicate_slug_raises_conflict(db_session: Session) -> None:
    service = TenantService(db_session)
    service.create_tenant(name="Acme", slug="acme")

    with pytest.raises(ConflictError):
        service.create_tenant(name="Acme Duplicate", slug="acme")


def test_get_tenant_not_found_raises(db_session: Session) -> None:
    service = TenantService(db_session)
    with pytest.raises(NotFoundError):
        service.get_tenant(uuid.uuid4())


def test_update_tenant_name(db_session: Session) -> None:
    service = TenantService(db_session)
    tenant, _, _ = service.create_tenant(name="Old Name", slug="acme")

    updated = service.update_tenant(tenant.id, name="New Name")
    assert updated.name == "New Name"


def test_deactivate_tenant(db_session: Session) -> None:
    service = TenantService(db_session)
    tenant, _, _ = service.create_tenant(name="Acme", slug="acme")

    deactivated = service.deactivate_tenant(tenant.id)
    assert deactivated.is_active is False
