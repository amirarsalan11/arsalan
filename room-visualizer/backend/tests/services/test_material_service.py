import uuid

import pytest
from sqlalchemy.orm import Session

from app.services import MaterialService, NotFoundError, TenantService


def _tenant_id(db_session: Session) -> uuid.UUID:
    tenant, _, _ = TenantService(db_session).create_tenant(name="Acme", slug="acme")
    return tenant.id


def test_create_and_get_material(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = MaterialService(db_session)

    material = service.create_material(
        tenant_id=tenant_id,
        name="Oak Hardwood",
        texture_url="https://cdn.example.com/oak.png",
        category="hardwood",
    )
    fetched = service.get_material(tenant_id, material.id)

    assert fetched.id == material.id
    assert fetched.is_active is True


def test_get_material_not_found_raises(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = MaterialService(db_session)

    with pytest.raises(NotFoundError):
        service.get_material(tenant_id, uuid.uuid4())


def test_update_material(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = MaterialService(db_session)
    material = service.create_material(
        tenant_id=tenant_id,
        name="Oak",
        texture_url="u1",
        category="hardwood",
    )

    updated = service.update_material(tenant_id, material.id, name="Oak Premium")
    assert updated.name == "Oak Premium"


def test_deactivate_material(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = MaterialService(db_session)
    material = service.create_material(
        tenant_id=tenant_id, name="Oak", texture_url="u1", category="hardwood"
    )

    deactivated = service.deactivate_material(tenant_id, material.id)
    assert deactivated.is_active is False


def test_material_cross_tenant_isolation(db_session: Session) -> None:
    tenant_a = _tenant_id(db_session)
    tenant_b_record, _, _ = TenantService(db_session).create_tenant(name="Other", slug="other")
    tenant_b = tenant_b_record.id
    service = MaterialService(db_session)

    material = service.create_material(
        tenant_id=tenant_a, name="Oak", texture_url="u1", category="hardwood"
    )

    with pytest.raises(NotFoundError):
        service.get_material(tenant_b, material.id)
