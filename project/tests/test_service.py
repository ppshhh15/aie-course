import pytest
from fastapi.testclient import TestClient
from src.service.app import app

# Фикстура для автоматического управления жизненным циклом приложения в тестах
@pytest.fixture
def client():
    # Конструкция 'with' гарантирует запуск Lifespan (загрузку модели) перед тестом
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    """Тест корневого эндпоинта API на корректность ответа"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint(client):
    """Тест отправки валидных данных на предсказание"""
    test_car = {
        "year": 2015,
        "make": "Ford",
        "model": "Fusion",
        "trim": "SE",
        "body": "Sedan",
        "transmission": "automatic",
        "state": "ca",
        "condition": 4.5,
        "odometer": 60000,
        "color": "black",
        "interior": "black"
    }
    response = client.post("/predict", json=test_car)
    
    # Ожидаем успешный статус и наличие предсказанной цены > 0
    assert response.status_code == 200
    assert "predicted_price_usd" in response.json()
    assert response.json()["predicted_price_usd"] > 0