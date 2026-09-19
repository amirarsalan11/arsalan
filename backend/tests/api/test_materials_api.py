from fastapi.testclient import TestClient


def test_create_and_get_material(client: TestClient, bootstrap_tenant) -> None:
    headers = bootstrap_tenant()["headers"]
    create_resp = client.post(
        "/api/v1/materials",
        json={
            "name": "Oak Hardwood",
            "texture_url": "https://cdn.example.com/oak.png",
            "category": "hardwood",
        },
        headers=headers,
    )
    assert create_resp.status_code == 201
    material_id = create_resp.json()["id"]

    get_resp = client.get(f"/api/v1/materials/{material_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Oak Hardwood"


def test_material_requires_auth(client: TestClient) -> None:
    response = client.get("/api/v1/materials")
    assert response.status_code == 401


def test_material_not_visible_to_other_tenant(client: TestClient, bootstrap_tenant) -> None:
    headers_a = bootstrap_tenant()["headers"]
    headers_b = bootstrap_tenant()["headers"]

    material = client.post(
        "/api/v1/materials",
        json={"name": "Oak", "texture_url": "u1", "category": "hardwood"},
        headers=headers_a,
    ).json()

    response = client.get(f"/api/v1/materials/{material['id']}", headers=headers_b)
    assert response.status_code == 404


def test_update_material_partial(client: TestClient, bootstrap_tenant) -> None:
    headers = bootstrap_tenant()["headers"]
    material = client.post(
        "/api/v1/materials",
        json={"name": "Oak", "texture_url": "u1", "category": "hardwood"},
        headers=headers,
    ).json()

    response = client.patch(
        f"/api/v1/materials/{material['id']}", json={"name": "Oak Premium"}, headers=headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Oak Premium"
    assert body["category"] == "hardwood"  # untouched field preserved


def test_deactivate_material(client: TestClient, bootstrap_tenant) -> None:
    headers = bootstrap_tenant()["headers"]
    material = client.post(
        "/api/v1/materials",
        json={"name": "Oak", "texture_url": "u1", "category": "hardwood"},
        headers=headers,
    ).json()

    response = client.post(f"/api/v1/materials/{material['id']}/deactivate", headers=headers)
    assert response.status_code == 200
    assert response.json()["is_active"] is False
