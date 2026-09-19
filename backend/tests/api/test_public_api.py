"""
Tests for the public router (SDK + Widget milestone).

These endpoints are the ONLY ones in the backend that don't require
Authorization: Bearer — they're protected by Origin-based
AllowedDomain validation instead. Tests here specifically exercise
that substitute mechanism, plus confirm the public schemas never leak
internal fields (texture_url, key_hash-adjacent details, etc.).
"""

import uuid

from fastapi.testclient import TestClient


def _register_domain(client: TestClient, headers: dict, domain: str) -> None:
    response = client.post("/api/v1/allowed-domains", json={"domain": domain}, headers=headers)
    assert response.status_code == 201, response.text


def test_public_config_requires_origin_header(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()
    _register_domain(client, bootstrapped["headers"], "shop.example.com")

    # No Origin header at all -> must be rejected, not silently allowed.
    response = client.get(f"/api/v1/public/tenants/{bootstrapped['tenant_id']}/config")
    assert response.status_code == 403


def test_public_config_rejects_non_allowlisted_origin(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()
    _register_domain(client, bootstrapped["headers"], "shop.example.com")

    response = client.get(
        f"/api/v1/public/tenants/{bootstrapped['tenant_id']}/config",
        headers={"Origin": "https://attacker.example.com"},
    )
    assert response.status_code == 403


def test_public_config_succeeds_for_allowlisted_origin(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant(name="Acme Corp")
    _register_domain(client, bootstrapped["headers"], "shop.example.com")

    response = client.get(
        f"/api/v1/public/tenants/{bootstrapped['tenant_id']}/config",
        headers={"Origin": "https://shop.example.com"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Acme Corp"
    assert body["is_active"] is True
    assert body["tenant_id"] == bootstrapped["tenant_id"]


def test_public_config_sets_cors_header_only_on_success(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()
    _register_domain(client, bootstrapped["headers"], "shop.example.com")

    allowed = client.get(
        f"/api/v1/public/tenants/{bootstrapped['tenant_id']}/config",
        headers={"Origin": "https://shop.example.com"},
    )
    assert allowed.headers.get("access-control-allow-origin") == "https://shop.example.com"

    rejected = client.get(
        f"/api/v1/public/tenants/{bootstrapped['tenant_id']}/config",
        headers={"Origin": "https://attacker.example.com"},
    )
    # Critically: no CORS header at all on a rejected request — the
    # header must never be set before the domain check passes.
    assert "access-control-allow-origin" not in rejected.headers


def test_public_config_does_not_require_bearer_auth(client: TestClient, bootstrap_tenant) -> None:
    """The whole point of this router: no Authorization header at all,
    just an allowed Origin."""
    bootstrapped = bootstrap_tenant()
    _register_domain(client, bootstrapped["headers"], "shop.example.com")

    response = client.get(
        f"/api/v1/public/tenants/{bootstrapped['tenant_id']}/config",
        headers={"Origin": "https://shop.example.com"},
        # Explicitly no Authorization header.
    )
    assert response.status_code == 200


def test_public_config_unknown_tenant_returns_404(client: TestClient) -> None:
    response = client.get(
        f"/api/v1/public/tenants/{uuid.uuid4()}/config",
        headers={"Origin": "https://shop.example.com"},
    )
    assert response.status_code == 404


def test_public_config_deactivated_tenant_returns_404(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()
    _register_domain(client, bootstrapped["headers"], "shop.example.com")
    client.post(
        f"/api/v1/tenants/{bootstrapped['tenant_id']}/deactivate", headers=bootstrapped["headers"]
    )

    response = client.get(
        f"/api/v1/public/tenants/{bootstrapped['tenant_id']}/config",
        headers={"Origin": "https://shop.example.com"},
    )
    assert response.status_code == 404


def _create_material(client: TestClient, headers: dict, name: str, category: str = "hardwood") -> dict:
    response = client.post(
        "/api/v1/materials",
        json={"name": name, "texture_url": "https://internal-cdn.example.com/secret-path.png", "category": category},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_public_materials_only_returns_active(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()
    _register_domain(client, bootstrapped["headers"], "shop.example.com")

    active = _create_material(client, bootstrapped["headers"], "Active Oak")
    inactive = _create_material(client, bootstrapped["headers"], "Retired Tile", category="tile")
    client.post(
        f"/api/v1/materials/{inactive['id']}/deactivate", headers=bootstrapped["headers"]
    )

    response = client.get(
        f"/api/v1/public/tenants/{bootstrapped['tenant_id']}/materials",
        headers={"Origin": "https://shop.example.com"},
    )
    assert response.status_code == 200
    names = [m["name"] for m in response.json()]
    assert names == ["Active Oak"]


def test_public_materials_never_expose_texture_url(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()
    _register_domain(client, bootstrapped["headers"], "shop.example.com")
    _create_material(client, bootstrapped["headers"], "Oak Hardwood")

    response = client.get(
        f"/api/v1/public/tenants/{bootstrapped['tenant_id']}/materials",
        headers={"Origin": "https://shop.example.com"},
    )
    body = response.json()
    assert len(body) == 1
    assert "texture_url" not in body[0]
    assert "width_mm" not in body[0]
    assert "height_mm" not in body[0]
    assert "is_active" not in body[0]
    assert set(body[0].keys()) == {"id", "name", "category"}


def test_public_materials_are_tenant_scoped(client: TestClient, bootstrap_tenant) -> None:
    tenant_a = bootstrap_tenant()
    tenant_b = bootstrap_tenant()
    _register_domain(client, tenant_a["headers"], "a.example.com")
    _register_domain(client, tenant_b["headers"], "b.example.com")

    _create_material(client, tenant_a["headers"], "Tenant A Material")
    _create_material(client, tenant_b["headers"], "Tenant B Material")

    response = client.get(
        f"/api/v1/public/tenants/{tenant_a['tenant_id']}/materials",
        headers={"Origin": "https://a.example.com"},
    )
    names = [m["name"] for m in response.json()]
    assert names == ["Tenant A Material"]


def test_public_materials_rejects_non_allowlisted_origin(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()
    _register_domain(client, bootstrapped["headers"], "shop.example.com")
    _create_material(client, bootstrapped["headers"], "Oak")

    response = client.get(
        f"/api/v1/public/tenants/{bootstrapped['tenant_id']}/materials",
        headers={"Origin": "https://attacker.example.com"},
    )
    assert response.status_code == 403


def test_existing_authenticated_routes_are_unaffected(client: TestClient, bootstrap_tenant) -> None:
    """Sanity check that this milestone didn't disturb the existing
    Bearer-auth flow: an Origin header alone must NOT grant access to
    an authenticated route, and a valid Bearer key must still work
    exactly as before, with no Origin header at all."""
    bootstrapped = bootstrap_tenant()

    origin_only = client.get(
        "/api/v1/materials", headers={"Origin": "https://shop.example.com"}
    )
    assert origin_only.status_code == 401

    bearer_only = client.get("/api/v1/materials", headers=bootstrapped["headers"])
    assert bearer_only.status_code == 200
