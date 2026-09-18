import uuid

from fastapi.testclient import TestClient


def test_render_jobs_require_auth(client: TestClient) -> None:
    response = client.get("/api/v1/render-jobs")
    assert response.status_code == 401


def test_create_render_job_defaults_pending(client: TestClient, bootstrap_tenant) -> None:
    headers = bootstrap_tenant()["headers"]
    response = client.post(
        "/api/v1/render-jobs",
        json={"input_image_url": "https://cdn.example.com/in.jpg"},
        headers=headers,
    )
    assert response.status_code == 201
    assert response.json()["status"] == "pending"


def test_create_render_job_with_unknown_material_returns_404(
    client: TestClient, bootstrap_tenant
) -> None:
    headers = bootstrap_tenant()["headers"]
    response = client.post(
        "/api/v1/render-jobs",
        json={
            "input_image_url": "https://cdn.example.com/in.jpg",
            "material_id": str(uuid.uuid4()),
        },
        headers=headers,
    )
    assert response.status_code == 404


def test_create_render_job_with_other_tenants_material_returns_404(
    client: TestClient, bootstrap_tenant
) -> None:
    headers_a = bootstrap_tenant()["headers"]
    headers_b = bootstrap_tenant()["headers"]

    material = client.post(
        "/api/v1/materials",
        json={"name": "Oak", "texture_url": "u1", "category": "hardwood"},
        headers=headers_a,
    ).json()

    # Tenant B tries to create a render job referencing tenant A's material.
    response = client.post(
        "/api/v1/render-jobs",
        json={
            "input_image_url": "https://cdn.example.com/in.jpg",
            "material_id": material["id"],
        },
        headers=headers_b,
    )
    assert response.status_code == 404


def test_update_render_job_status(client: TestClient, bootstrap_tenant) -> None:
    headers = bootstrap_tenant()["headers"]
    render_job = client.post(
        "/api/v1/render-jobs",
        json={"input_image_url": "https://cdn.example.com/in.jpg"},
        headers=headers,
    ).json()

    response = client.patch(
        f"/api/v1/render-jobs/{render_job['id']}/status",
        json={"status": "completed", "output_image_url": "https://cdn.example.com/out.jpg"},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["output_image_url"] == "https://cdn.example.com/out.jpg"


def test_list_render_jobs_is_tenant_scoped(client: TestClient, bootstrap_tenant) -> None:
    headers_a = bootstrap_tenant()["headers"]
    headers_b = bootstrap_tenant()["headers"]

    client.post(
        "/api/v1/render-jobs",
        json={"input_image_url": "https://cdn.example.com/in.jpg"},
        headers=headers_a,
    )

    response = client.get("/api/v1/render-jobs", headers=headers_b)
    assert response.status_code == 200
    assert response.json() == []
