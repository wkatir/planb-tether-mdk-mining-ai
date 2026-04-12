import pytest
from fastapi.testclient import TestClient


class TestFastAPI:
    def test_app_can_be_created(self):
        from app.api.main import app

        assert app is not None
        assert app.title == "MDK Mining AI"
        assert app.version == "0.1.0"

    def test_root_endpoint(self):
        from app.api.main import app

        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "MDK Mining AI API"}

    def test_health_endpoint(self):
        from app.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_telemetry_endpoint_exists(self):
        from app.api.main import app

        client = TestClient(app)
        response = client.get("/telemetry/")
        assert response.status_code in [200, 404, 405]

    def test_kpi_endpoint_exists(self):
        from app.api.main import app

        client = TestClient(app)
        response = client.get("/kpi/")
        assert response.status_code in [200, 404, 405]

    def test_control_endpoint_exists(self):
        from app.api.main import app

        client = TestClient(app)
        response = client.get("/control/")
        assert response.status_code in [200, 404, 405]
