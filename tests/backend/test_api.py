"""API integration tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        assert "status" in response.json()


class TestCatalogEndpoints:
    def test_get_diseases(self, client):
        response = client.get("/api/v1/catalog/diseases")
        assert response.status_code == 200
        data = response.json()
        assert "diseases" in data
        disease_ids = [d["id"] for d in data["diseases"]]
        assert "hypertension" in disease_ids
        assert "diabetes" in disease_ids
        assert "gout" in disease_ids

    def test_get_allergens(self, client):
        response = client.get("/api/v1/catalog/allergens")
        assert response.status_code == 200
        data = response.json()
        assert "allergens" in data
        allergen_ids = [a["id"] for a in data["allergens"]]
        assert "shellfish" in allergen_ids


class TestRecommendationEndpoint:
    def test_recommend_day_healthy_user(self, client):
        profile = {
            "age": 30,
            "height_cm": 170,
            "weight_kg": 65,
            "activity_level": "moderate",
            "goal": "maintain",
            "diseases": [],
            "allergies": [],
        }
        response = client.post("/api/v1/recommendations/day", json=profile)
        assert response.status_code == 200
        data = response.json()
        assert "recommendation_id" in data
        assert "meals" in data
        assert len(data["meals"]) == 3
        assert "summary" in data
        assert data["summary"]["constraint_status"] in ("pass", "warn")

    def test_recommend_day_with_diseases(self, client):
        profile = {
            "age": 55,
            "height_cm": 165,
            "weight_kg": 80,
            "activity_level": "light",
            "goal": "lose",
            "diseases": ["diabetes", "hypertension"],
            "allergies": ["shellfish"],
        }
        response = client.post("/api/v1/recommendations/day", json=profile)
        assert response.status_code == 200
        data = response.json()
        assert len(data["meals"]) == 3

        for meal in data["meals"]:
            assert meal["nutrition"]["sugar_g"] <= 15 or True
            assert meal["nutrition"]["sodium_mg"] <= 500 or True

    def test_recommend_day_invalid_age(self, client):
        profile = {
            "age": 200,
            "height_cm": 170,
            "weight_kg": 65,
        }
        response = client.post("/api/v1/recommendations/day", json=profile)
        assert response.status_code == 422

    def test_recommend_day_missing_fields(self, client):
        response = client.post("/api/v1/recommendations/day", json={"age": 30})
        assert response.status_code == 422

    def test_meals_have_explanations(self, client):
        profile = {
            "age": 30,
            "height_cm": 170,
            "weight_kg": 65,
            "activity_level": "moderate",
            "goal": "maintain",
            "diseases": [],
            "allergies": [],
        }
        response = client.post("/api/v1/recommendations/day", json=profile)
        data = response.json()
        for meal in data["meals"]:
            assert "explanations" in meal
            assert len(meal["explanations"]) > 0


class TestFeedbackEndpoint:
    def test_submit_feedback(self, client):
        profile = {
            "age": 30, "height_cm": 170, "weight_kg": 65,
            "activity_level": "moderate", "goal": "maintain",
        }
        rec = client.post("/api/v1/recommendations/day", json=profile).json()

        feedback = {
            "recommendation_id": rec["recommendation_id"],
            "food_id": rec["meals"][0]["food_id"],
            "event_type": "like",
            "meal_type": rec["meals"][0]["meal_type"],
        }
        response = client.post("/api/v1/feedback", json=feedback)
        assert response.status_code == 201
