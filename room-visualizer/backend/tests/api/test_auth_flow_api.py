"""
End-to-end authentication flow tests, driven entirely through the HTTP
layer (no direct service/repository access) — these exercise the full
API -> Auth Dependency -> Auth Service -> Repository -> Database chain
exactly as a real client would.
"""

from fastapi.testclient import TestClient


def test_full_bootstrap_and_authenticated_request_flow(client: TestClient) -> None:
    # 1. Bootstrap: create a tenant, unauthenticated, get a raw key back once.
    create_resp = client.post("/api/v1/tenants", json={"name": "Acme", "slug": "acme"})
    assert create_resp.status_code == 201
    body = create_resp.json()
    raw_key = body["api_key"]["raw_key"]
    tenant_id = body["tenant"]["id"]

    # 2. Use that raw key to authenticate a completely separate request.
    headers = {"Authorization": f"Bearer {raw_key}"}
    me_resp = client.get(f"/api/v1/tenants/{tenant_id}", headers=headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["id"] == tenant_id


def test_raw_key_never_appears_in_any_subsequent_response(client: TestClient) -> None:
    body = client.post("/api/v1/tenants", json={"name": "Acme", "slug": "acme"}).json()
    raw_key = body["api_key"]["raw_key"]
    headers = {"Authorization": f"Bearer {raw_key}"}

    # None of these later, authenticated responses should ever include
    # the raw key value anywhere in their body.
    list_resp = client.get("/api/v1/api-keys", headers=headers)
    assert raw_key not in list_resp.text

    get_tenant_resp = client.get(f"/api/v1/tenants/{body['tenant']['id']}", headers=headers)
    assert raw_key not in get_tenant_resp.text


def test_missing_authorization_header_returns_401_with_bearer_challenge(client: TestClient) -> None:
    response = client.get("/api/v1/materials")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_all_auth_failure_reasons_return_identical_generic_message(
    client: TestClient, bootstrap_tenant
) -> None:
    """Every distinct failure cause — missing header, malformed header,
    unknown key, revoked key, deactivated tenant — must produce the
    exact same response body, so a caller can never distinguish them.
    """
    bootstrapped = bootstrap_tenant()

    missing = client.get("/api/v1/materials")
    malformed = client.get("/api/v1/materials", headers={"Authorization": "not-bearer"})
    unknown = client.get(
        "/api/v1/materials", headers={"Authorization": "Bearer rv_live_unknown"}
    )

    assert missing.status_code == malformed.status_code == unknown.status_code == 401
    assert missing.json() == malformed.json() == unknown.json()

    # Now revoke the tenant's only key and confirm it produces the same body too.
    api_keys = client.get("/api/v1/api-keys", headers=bootstrapped["headers"]).json()
    client.post(
        f"/api/v1/api-keys/{api_keys[0]['id']}/revoke", headers=bootstrapped["headers"]
    )
    revoked = client.get("/api/v1/materials", headers=bootstrapped["headers"])
    assert revoked.status_code == 401
    assert revoked.json() == missing.json()


def test_deactivated_tenant_key_can_no_longer_authenticate(
    client: TestClient, bootstrap_tenant
) -> None:
    bootstrapped = bootstrap_tenant()

    deactivate_resp = client.post(
        f"/api/v1/tenants/{bootstrapped['tenant_id']}/deactivate",
        headers=bootstrapped["headers"],
    )
    assert deactivate_resp.status_code == 200

    # The same, still-active-looking key must now fail to authenticate
    # anything, because its owning tenant is deactivated.
    response = client.get("/api/v1/materials", headers=bootstrapped["headers"])
    assert response.status_code == 401
