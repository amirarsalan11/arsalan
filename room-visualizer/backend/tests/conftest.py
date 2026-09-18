"""
Shared pytest fixtures.

`db_session` gives repository/service tests a fresh in-memory SQLite
database per test (same cross-dialect approach as Milestone 2's model
tests, via the GUID type).

`client` gives API-layer tests a FastAPI TestClient with `get_db`
overridden to use that same in-memory session, so API tests don't
require a live Postgres instance either.

`bootstrap_tenant` (Authentication milestone addition) drives the real
tenant-creation bootstrap flow through the HTTP layer and returns the
tenant id, the one-time raw API key, and ready-to-use auth headers —
so every API test that needs an authenticated tenant can get one in a
single call instead of re-implementing the bootstrap dance.
"""

import uuid
from collections.abc import Generator
from typing import TypedDict

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.db.base import Base
from app.main import app


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def _override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class BootstrappedTenant(TypedDict):
    tenant_id: str
    raw_key: str
    headers: dict[str, str]


def _bootstrap_tenant(
    client: TestClient, *, name: str | None = None, slug: str | None = None
) -> BootstrappedTenant:
    """Create a tenant via the real HTTP bootstrap flow and return its
    id, one-time raw API key, and ready-to-use Authorization header.

    `name`/`slug` default to a fresh random value per call (not a fixed
    "acme") specifically so that calling this twice in the same test —
    e.g. to get two distinct tenants for a cross-tenant isolation check
    — never collides on a duplicate slug and 409s. Pass explicit
    name/slug when a test wants to assert on a recognizable value.
    """
    unique_suffix = uuid.uuid4().hex[:8]
    resolved_name = name if name is not None else f"Tenant {unique_suffix}"
    resolved_slug = slug if slug is not None else f"tenant-{unique_suffix}"

    response = client.post("/api/v1/tenants", json={"name": resolved_name, "slug": resolved_slug})
    assert response.status_code == 201, response.text
    body = response.json()

    tenant_id = body["tenant"]["id"]
    raw_key = body["api_key"]["raw_key"]

    return {
        "tenant_id": tenant_id,
        "raw_key": raw_key,
        "headers": {"Authorization": f"Bearer {raw_key}"},
    }


@pytest.fixture()
def bootstrap_tenant(client: TestClient):
    """Factory fixture: call with optional name/slug to bootstrap a
    tenant and get back {tenant_id, raw_key, headers}. Each call
    creates a genuinely new tenant with an auto-generated unique
    slug by default, so tests needing two tenants (e.g. cross-tenant
    isolation checks) can call it twice with no collision risk, with
    or without explicit distinct slugs.
    """

    def _factory(*, name: str | None = None, slug: str | None = None) -> BootstrappedTenant:
        return _bootstrap_tenant(client, name=name, slug=slug)

    return _factory
