from fastapi.testclient import TestClient


def test_create_api_key_requires_auth(client: TestClient) -> None:
    response = client.post("/api/v1/api-keys", json={"label": "Prod"})
    assert response.status_code == 401


def test_create_and_list_api_keys(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()

    create_resp = client.post(
        "/api/v1/api-keys", json={"label": "Prod"}, headers=bootstrapped["headers"]
    )
    assert create_resp.status_code == 201
    created_body = create_resp.json()
    assert created_body["raw_key"].startswith("rv_live_")
    assert "key_hash" not in created_body

    list_resp = client.get("/api/v1/api-keys", headers=bootstrapped["headers"])
    assert list_resp.status_code == 200
    # The bootstrap flow already issued one key; this test creates a second.
    assert len(list_resp.json()) == 2
    # The list response must never include the raw key or hash.
    for key in list_resp.json():
        assert "raw_key" not in key
        assert "key_hash" not in key


def test_new_api_key_can_itself_authenticate(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()

    create_resp = client.post(
        "/api/v1/api-keys", json={"label": "Second Key"}, headers=bootstrapped["headers"]
    )
    second_raw_key = create_resp.json()["raw_key"]

    response = client.get(
        "/api/v1/api-keys", headers={"Authorization": f"Bearer {second_raw_key}"}
    )
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_api_keys_are_isolated_per_tenant(client: TestClient, bootstrap_tenant) -> None:
    tenant_a = bootstrap_tenant(name="Acme", slug="acme")
    tenant_b = bootstrap_tenant(name="Other", slug="other")

    client.post("/api/v1/api-keys", json={"label": "A key"}, headers=tenant_a["headers"])

    response = client.get("/api/v1/api-keys", headers=tenant_b["headers"])
    assert response.status_code == 200
    # Tenant B only sees its own bootstrap-issued key, never tenant A's.
    assert len(response.json()) == 1


def test_revoke_api_key(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()
    created = client.post(
        "/api/v1/api-keys", json={"label": "Prod"}, headers=bootstrapped["headers"]
    ).json()

    response = client.post(
        f"/api/v1/api-keys/{created['id']}/revoke", headers=bootstrapped["headers"]
    )
    assert response.status_code == 200
    assert response.json()["is_active"] is False


def test_revoked_key_can_no_longer_authenticate(client: TestClient, bootstrap_tenant) -> None:
    bootstrapped = bootstrap_tenant()
    second_key = client.post(
        "/api/v1/api-keys", json={"label": "Rotatable"}, headers=bootstrapped["headers"]
    ).json()

    client.post(
        f"/api/v1/api-keys/{second_key['id']}/revoke", headers=bootstrapped["headers"]
    )

    response = client.get(
        "/api/v1/api-keys",
        headers={"Authorization": f"Bearer {second_key['raw_key']}"},
    )
    assert response.status_code == 401


def test_malformed_authorization_header_returns_401(client: TestClient) -> None:
    response = client.get("/api/v1/api-keys", headers={"Authorization": "not-a-bearer-token"})
    assert response.status_code == 401


def test_unknown_api_key_returns_401(client: TestClient) -> None:
    response = client.get(
        "/api/v1/api-keys",
        headers={"Authorization": "Bearer rv_live_totally-made-up-key-value"},
    )
    assert response.status_code == 401
