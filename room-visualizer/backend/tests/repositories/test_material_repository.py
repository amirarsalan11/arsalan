import uuid

from sqlalchemy.orm import Session

from app.repositories import MaterialRepository, TenantRepository


def _tenant_id(db_session: Session) -> uuid.UUID:
    tenant = TenantRepository(db_session).create(name="Acme", slug="acme")
    db_session.commit()
    return tenant.id


def test_material_create_and_get_is_tenant_scoped(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    other_tenant_id = uuid.uuid4()  # simulate a different tenant

    repo = MaterialRepository(db_session)
    material = repo.create(
        tenant_id=tenant_id,
        name="Oak Hardwood",
        texture_url="https://cdn.example.com/oak.png",
        category="hardwood",
    )
    db_session.commit()

    assert repo.get_by_id(tenant_id, material.id) is not None
    # Same material id, wrong tenant -> must not resolve.
    assert repo.get_by_id(other_tenant_id, material.id) is None


def test_material_list_by_tenant(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    repo = MaterialRepository(db_session)
    repo.create(tenant_id=tenant_id, name="A", texture_url="u1", category="tile")
    repo.create(tenant_id=tenant_id, name="B", texture_url="u2", category="tile")
    db_session.commit()

    materials = repo.list_by_tenant(tenant_id)
    assert len(materials) == 2


def test_material_update_is_tenant_scoped(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    other_tenant_id = uuid.uuid4()
    repo = MaterialRepository(db_session)
    material = repo.create(
        tenant_id=tenant_id, name="A", texture_url="u1", category="tile"
    )
    db_session.commit()

    # Attempting to update using the wrong tenant_id must fail to find it.
    assert repo.update(other_tenant_id, material.id, name="Hacked") is None

    updated = repo.update(tenant_id, material.id, name="Renamed")
    assert updated is not None
    assert updated.name == "Renamed"
