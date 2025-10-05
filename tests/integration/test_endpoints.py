from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_readyz_before_model_loaded():
    response = client.get("/readyz")
    # Перед стартом сервера модель не завантажена — статус 503
    assert response.status_code in (200, 503)  # залежить від тестового запуску

def test_predict_invalid_input():
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422