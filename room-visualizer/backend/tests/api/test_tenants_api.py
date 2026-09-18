import uuid

from fastapi.testclient import TestClient


def test_create_tenant_returns_tenant_and_raw_api_key(client: TestClient) -> None:
    response = client.post("/api/v1/tenants", json={"name": "Acme", "slug": "acme"})
    assert response.status_code == 201
    body = response.json()

    assert body["tenant"]["slug"] == "acme"
    assert body["tenant"]["is_active"] is True

    # The raw key is only ever shown here — confirm it's present, looks
    # like a real generated key (has the required prefix), and that no
    # hash/internal field leaks alongside it.
    assert body["api_key"]["raw_key"].startswith("rv_live_")
    assert "key_hash" not in body["api_key"]
    assert body["api_key"]["label"] == "Default API Key"


def test_create_tenant_duplicate_slug_returns_409(client: TestClient) -> None:
    client.post("/api/v1/tenants", json={"name": "Acme", "slug": "acme"})
    response = client.post("/api/v1/tenants", json={"name": "Acme 2", "slug": "acme"})
    assert response.status_code == 409


def test_get_tenant_requires_auth(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()

    unauthenticated = client.get(f"/api/v1/tenants/{bootstrapped['tenant_id']}")
    assert unauthenticated.status_code == 401

    authenticated = client.get(
        f"/api/v1/tenants/{bootstrapped['tenant_id']}", headers=bootstrapped["headers"]
    )
    assert authenticated.status_code == 200
    assert authenticated.json()["id"] == bootstrapped["tenant_id"]


def test_get_tenant_not_found_returns_404_for_unknown_id(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()
    response = client.get(f"/api/v1/tenants/{uuid.uuid4()}", headers=bootstrapped["headers"])
    assert response.status_code == 404


def test_tenant_cannot_access_another_tenants_record(client: TestClient, bootstrap_tenant) -> None:
    tenant_a = bootstrap_tenant(name="Acme", slug="acme")
    tenant_b = bootstrap_tenant(name="Other", slug="other")

    # Tenant A's valid, authenticated key trying to read Tenant B's record.
    response = client.get(
        f"/api/v1/tenants/{tenant_b['tenant_id']}", headers=tenant_a["headers"]
    )
    # 404, not 403 — a caller must not learn whether a given tenant_id
    # exists by probing with someone else's key.
    assert response.status_code == 404


def test_update_tenant(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()

    response = client.patch(
        f"/api/v1/tenants/{bootstrapped['tenant_id']}",
        json={"name": "New Name"},
        headers=bootstrapped["headers"],
    )
    assert response.status_code == 200
    assert response.json()["name"] == "New Name"


def test_deactivate_tenant(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()

    response = client.post(
        f"/api/v1/tenants/{bootstrapped['tenant_id']}/deactivate",
        headers=bootstrapped["headers"],
    )
    assert response.status_code == 200
    assert response.json()["is_active"] is False


def test_no_list_tenants_endpoint(client: TestClient) -> None:
    """Per approval: GET /api/v1/tenants (list all) must not exist."""
    response = client.get("/api/v1/tenants")
    assert response.status_code in (404, 405)
