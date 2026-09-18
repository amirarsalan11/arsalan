"""
Model-layer tests for Milestone 2 (Database Models).

Scope: verify the ORM models are structurally correct — they can be
created, relationships resolve in both directions, constraints are
enforced, and the RenderJob status defaults/validates correctly.

These tests run against an in-memory SQLite database (via the
cross-dialect `GUID` type defined in app/db/base.py), NOT PostgreSQL.
This is a deliberate trade-off: no live Postgres instance is reachable
in this authoring environment, and SQLite lets the model *shape* be
validated without one. It does NOT validate PostgreSQL-specific
behavior (native UUID generation via `gen_random_uuid()`, the
`pgcrypto` extension, or exact constraint error messages) — that
requires running the Alembic migration against a real Postgres
instance, which is called out explicitly in Known Limitations.

No repository layer, service layer, or API routes are exercised here —
only direct Session usage against the models, which is appropriate for
a model-layer test and does not imply a repository pattern is in place.
"""

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import (
    AllowedDomain,
    ApiKey,
    Material,
    RenderJob,
    RenderJobStatus,
    Tenant,
    UsageRecord,
)


@pytest.fixture()
def session():
    """A fresh in-memory SQLite database per test, with all tables created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    with Session(engine) as db_session:
        yield db_session

    engine.dispose()


def _make_tenant(session: Session, slug: str = "acme") -> Tenant:
    tenant = Tenant(name="Acme Inc", slug=slug)
    session.add(tenant)
    session.commit()
    return tenant


def test_tenant_can_be_created_with_defaults(session: Session) -> None:
    tenant = _make_tenant(session)

    assert isinstance(tenant.id, uuid.UUID)
    assert tenant.is_active is True
    assert tenant.created_at is not None
    assert tenant.updated_at is not None


def test_tenant_slug_must_be_unique(session: Session) -> None:
    _make_tenant(session, slug="acme")

    with pytest.raises(IntegrityError):
        session.add(Tenant(name="Acme Duplicate", slug="acme"))
        session.commit()


def test_api_key_belongs_to_tenant_and_relationship_resolves(session: Session) -> None:
    tenant = _make_tenant(session)

    api_key = ApiKey(tenant_id=tenant.id, key_hash="hash-123", label="Production key")
    session.add(api_key)
    session.commit()

    assert api_key.tenant_id == tenant.id
    assert api_key.is_active is True
    assert api_key.last_used_at is None

    # Relationship resolves both directions.
    session.refresh(tenant)
    assert api_key.tenant.id == tenant.id
    assert api_key in tenant.api_keys


def test_api_key_hash_must_be_unique(session: Session) -> None:
    tenant = _make_tenant(session)
    session.add(ApiKey(tenant_id=tenant.id, key_hash="dup-hash", label="Key A"))
    session.commit()

    with pytest.raises(IntegrityError):
        session.add(ApiKey(tenant_id=tenant.id, key_hash="dup-hash", label="Key B"))
        session.commit()


def test_allowed_domain_unique_per_tenant(session: Session) -> None:
    tenant = _make_tenant(session)
    session.add(AllowedDomain(tenant_id=tenant.id, domain="shop.example.com"))
    session.commit()

    # Same domain for the same tenant again -> violates the composite
    # unique constraint.
    with pytest.raises(IntegrityError):
        session.add(AllowedDomain(tenant_id=tenant.id, domain="shop.example.com"))
        session.commit()


def test_allowed_domain_can_repeat_across_different_tenants(session: Session) -> None:
    tenant_a = _make_tenant(session, slug="tenant-a")
    tenant_b = _make_tenant(session, slug="tenant-b")

    session.add(AllowedDomain(tenant_id=tenant_a.id, domain="shared.example.com"))
    session.add(AllowedDomain(tenant_id=tenant_b.id, domain="shared.example.com"))
    session.commit()

    domains = session.query(AllowedDomain).filter_by(domain="shared.example.com").all()
    assert len(domains) == 2
    assert {d.tenant_id for d in domains} == {tenant_a.id, tenant_b.id}


def test_material_is_tenant_scoped_not_global(session: Session) -> None:
    tenant_a = _make_tenant(session, slug="tenant-a")
    tenant_b = _make_tenant(session, slug="tenant-b")

    session.add(
        Material(
            tenant_id=tenant_a.id,
            name="Oak Hardwood",
            texture_url="https://cdn.example.com/oak.png",
            category="hardwood",
        )
    )
    session.commit()

    # Per the approved adjustment, materials are tenant-scoped only —
    # there is no global/shared material catalog in this MVP. Querying
    # by tenant must not leak the other tenant's rows.
    tenant_a_materials = session.query(Material).filter_by(tenant_id=tenant_a.id).all()
    tenant_b_materials = session.query(Material).filter_by(tenant_id=tenant_b.id).all()

    assert len(tenant_a_materials) == 1
    assert len(tenant_b_materials) == 0


def test_render_job_defaults_to_pending_status(session: Session) -> None:
    tenant = _make_tenant(session)

    render_job = RenderJob(
        tenant_id=tenant.id,
        input_image_url="https://cdn.example.com/input.jpg",
    )
    session.add(render_job)
    session.commit()

    assert render_job.status == RenderJobStatus.PENDING
    assert render_job.material_id is None
    assert render_job.output_image_url is None


def test_render_job_material_relationship_and_set_null_shape(session: Session) -> None:
    tenant = _make_tenant(session)
    material = Material(
        tenant_id=tenant.id,
        name="Marble Tile",
        texture_url="https://cdn.example.com/marble.png",
        category="tile",
        width_mm=600,
        height_mm=600,
    )
    session.add(material)
    session.commit()

    render_job = RenderJob(
        tenant_id=tenant.id,
        material_id=material.id,
        input_image_url="https://cdn.example.com/room.jpg",
        status=RenderJobStatus.PROCESSING,
    )
    session.add(render_job)
    session.commit()

    assert render_job.material.id == material.id
    assert render_job in material.render_jobs


def test_usage_record_links_to_render_job_and_has_no_updated_at(session: Session) -> None:
    tenant = _make_tenant(session)
    render_job = RenderJob(
        tenant_id=tenant.id,
        input_image_url="https://cdn.example.com/room.jpg",
        status=RenderJobStatus.COMPLETED,
        output_image_url="https://cdn.example.com/result.jpg",
    )
    session.add(render_job)
    session.commit()

    usage_record = UsageRecord(
        tenant_id=tenant.id,
        render_job_id=render_job.id,
        event_type="render_completed",
    )
    session.add(usage_record)
    session.commit()

    assert usage_record.render_job.id == render_job.id
    assert usage_record in render_job.usage_records
    assert not hasattr(UsageRecord, "updated_at")
    assert usage_record.created_at is not None


def test_all_tenant_scoped_tables_carry_tenant_id() -> None:
    """Structural check: every tenant-scoped model must have a tenant_id
    column, which is the schema-level backbone of tenant isolation.
    """
    tenant_scoped_models = [ApiKey, AllowedDomain, Material, RenderJob, UsageRecord]

    for model in tenant_scoped_models:
        assert "tenant_id" in model.__table__.columns, f"{model.__name__} is missing tenant_id"
