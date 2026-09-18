from fastapi.testclient import TestClient


def test_usage_records_require_auth(client: TestClient) -> None:
    response = client.get("/api/v1/usage-records")
    assert response.status_code == 401


def test_list_usage_records_empty_by_default(client: TestClient, bootstrap_tenant) -> None:
    headers = bootstrap_tenant()["headers"]
    response = client.get("/api/v1/usage-records", headers=headers)
    assert response.status_code == 200
    assert response.json() == []


def test_no_create_usage_record_endpoint(client: TestClient, bootstrap_tenant) -> None:
    """Per the read-foundation scope, POST /usage-records must not exist."""
    headers = bootstrap_tenant()["headers"]
    response = client.post("/api/v1/usage-records", json={}, headers=headers)
    assert response.status_code == 405
