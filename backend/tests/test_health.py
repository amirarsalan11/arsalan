"""
Smoke test for Milestone 1: verifies the FastAPI app boots and the
/health endpoint responds correctly. No database, Redis, or AI
dependencies are exercised by this test.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert "service" in body
    assert "environment" in body
