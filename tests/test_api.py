import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_valid_nominal_case():
    """Valid peak season demand estimation."""
    payload = {
        "branch_id": "Sucursal_Norte",
        "product_id": "PROD_GLIFO",
        "product_category": "Herbicida",
        "week_of_year": 44,
        "is_rainy_season": 1,
        "promo_applied": 0,
        "lag_1_demand": 120.0,
        "lag_2_demand": 115.0,
        "lag_4_demand": 110.0,
        "rolling_mean_4w": 115.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "estimated_demand_liters" in data
    assert data["estimated_demand_liters"] > 0

def test_difficult_edge_case():
    """Edge case: Zero sales history and promotion active off-season."""
    payload = {
        "branch_id": "Sucursal_Sur",
        "product_id": "PROD_CLORPI",
        "product_category": "Insecticida",
        "week_of_year": 25,
        "is_rainy_season": 0,
        "promo_applied": 1,
        "lag_1_demand": 0.0,
        "lag_2_demand": 0.0,
        "lag_4_demand": 0.0,
        "rolling_mean_4w": 0.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["estimated_demand_liters"] >= 0.0

def test_invalid_input_case():
    """Invalid case: Out-of-bounds week and negative lag demand."""
    payload = {
        "branch_id": "Sucursal_Norte",
        "product_id": "PROD_GLIFO",
        "product_category": "Herbicida",
        "week_of_year": 99,  # Invalid: max 52
        "is_rainy_season": 1,
        "promo_applied": 0,
        "lag_1_demand": -10.0,  # Invalid: negative
        "lag_2_demand": 10.0,
        "lag_4_demand": 10.0,
        "rolling_mean_4w": 10.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error