import uuid

from fastapi.testclient import TestClient


def test_allowed_domains_require_auth(client: TestClient) -> None:
    response = client.get("/api/v1/allowed-domains")
    assert response.status_code == 401


def test_register_and_list_domain(client: TestClient, bootstrap_tenant) -> None:
    headers = bootstrap_tenant()["headers"]

    create_resp = client.post(
        "/api/v1/allowed-domains", json={"domain": "shop.example.com"}, headers=headers
    )
    assert create_resp.status_code == 201
    assert create_resp.json()["domain"] == "shop.example.com"
    assert create_resp.json()["is_active"] is True

    list_resp = client.get("/api/v1/allowed-domains", headers=headers)
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1


def test_register_duplicate_domain_returns_409(client: TestClient, bootstrap_tenant) -> None:
    headers = bootstrap_tenant()["headers"]
    client.post("/api/v1/allowed-domains", json={"domain": "shop.example.com"}, headers=headers)

    response = client.post(
        "/api/v1/allowed-domains", json={"domain": "shop.example.com"}, headers=headers
    )
    assert response.status_code == 409


def test_allowed_domains_are_isolated_per_tenant(client: TestClient, bootstrap_tenant) -> None:
    headers_a = bootstrap_tenant()["headers"]
    headers_b = bootstrap_tenant()["headers"]

    client.post("/api/v1/allowed-domains", json={"domain": "a.example.com"}, headers=headers_a)

    response = client.get("/api/v1/allowed-domains", headers=headers_b)
    assert response.status_code == 200
    assert response.json() == []


def test_deactivate_domain(client: TestClient, bootstrap_tenant) -> None:
    headers = bootstrap_tenant()["headers"]
    created = client.post(
        "/api/v1/allowed-domains", json={"domain": "shop.example.com"}, headers=headers
    ).json()

    response = client.post(
        f"/api/v1/allowed-domains/{created['id']}/deactivate", headers=headers
    )
    assert response.status_code == 200
    assert response.json()["is_active"] is False


def test_deactivate_unknown_domain_returns_404(client: TestClient, bootstrap_tenant) -> None:
    headers = bootstrap_tenant()["headers"]
    response = client.post(
        f"/api/v1/allowed-domains/{uuid.uuid4()}/deactivate", headers=headers
    )
    assert response.status_code == 404
