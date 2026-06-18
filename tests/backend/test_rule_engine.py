"""Tests for Disease Rule Engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from backend.services.rule_engine import DiseaseRuleEngine, RuleStatus


@pytest.fixture
def engine():
    return DiseaseRuleEngine()


@pytest.fixture
def sample_food():
    return {
        "food_id": "F001",
        "name": "Phở bò",
        "ingredients": ["rice_noodle", "beef", "onion", "scallion", "ginger", "fish_sauce"],
        "meal_types": ["breakfast", "lunch", "dinner"],
        "calories_kcal": 420,
        "protein_g": 28,
        "carb_g": 52,
        "fat_g": 12,
        "sugar_g": 3,
        "fiber_g": 2,
        "sodium_mg": 980,
        "purine_level": "moderate",
        "cost_estimate": 45000,
        "diet_tags": ["vietnamese"],
    }


@pytest.fixture
def high_sodium_food():
    return {
        "food_id": "F049",
        "name": "Mì gói",
        "ingredients": ["instant_noodle", "egg", "scallion"],
        "sodium_mg": 1500,
        "sugar_g": 2,
        "purine_level": "low",
        "carb_g": 48,
        "calories_kcal": 380,
        "protein_g": 10,
        "fat_g": 16,
        "fiber_g": 2,
        "cost_estimate": 12000,
        "meal_types": ["breakfast"],
        "diet_tags": [],
    }


class TestRuleEngine:
    def test_healthy_food_passes_no_disease(self, engine, sample_food):
        result = engine.filter_food(sample_food, diseases=[], allergies=[])
        assert result.is_safe is True

    def test_hypertension_rejects_high_sodium(self, engine, high_sodium_food):
        result = engine.filter_food(
            high_sodium_food, diseases=["hypertension"], allergies=[]
        )
        assert result.is_safe is False
        assert any("natri" in r.lower() or "sodium" in r.lower() for r in result.rejections)

    def test_diabetes_rejects_high_sugar(self, engine):
        sweet_food = {
            "food_id": "F026",
            "name": "Chè đậu xanh",
            "ingredients": ["mung_bean", "sugar", "coconut_milk"],
            "sugar_g": 28,
            "sodium_mg": 50,
            "purine_level": "low",
            "carb_g": 42,
            "calories_kcal": 250,
            "protein_g": 6,
            "fat_g": 8,
            "fiber_g": 2,
            "cost_estimate": 15000,
            "meal_types": ["snack"],
            "diet_tags": [],
        }
        result = engine.filter_food(sweet_food, diseases=["diabetes"], allergies=[])
        assert result.is_safe is False

    def test_gout_rejects_high_purine(self, engine):
        high_purine = {
            "food_id": "F043",
            "name": "Lẩu thái",
            "ingredients": ["shrimp", "squid", "mushroom"],
            "purine_level": "high",
            "sodium_mg": 1400,
            "sugar_g": 6,
            "calories_kcal": 480,
            "carb_g": 30,
            "protein_g": 30,
            "fat_g": 25,
            "fiber_g": 4,
            "cost_estimate": 80000,
            "meal_types": ["lunch"],
            "diet_tags": [],
        }
        result = engine.filter_food(high_purine, diseases=["gout"], allergies=[])
        assert result.is_safe is False

    def test_allergy_rejection(self, engine, sample_food):
        result = engine.filter_food(
            sample_food, diseases=[], allergies=["fish"]
        )
        assert result.is_safe is False
        assert any("fish_sauce" in r or "dị ứng" in r.lower() or "allergen" in r.lower() for r in result.rejections)

    def test_no_allergy_passes(self, engine, sample_food):
        result = engine.filter_food(
            sample_food, diseases=[], allergies=["peanut"]
        )
        assert result.is_safe is True

    def test_dislike_is_soft(self, engine, sample_food):
        result = engine.filter_food(
            sample_food, diseases=[], allergies=[],
            disliked_ingredients=["ginger"],
        )
        assert result.is_safe is True
        assert len(result.warnings) > 0

    def test_filter_all(self, engine, sample_food, high_sodium_food):
        foods = [sample_food, high_sodium_food]
        safe, trace = engine.filter_all(foods, ["hypertension"], [])
        assert len(safe) <= len(foods)
        assert all(fid in trace for fid in ["F001", "F049"])

    def test_multiple_diseases(self, engine, high_sodium_food):
        result = engine.filter_food(
            high_sodium_food,
            diseases=["hypertension", "diabetes"],
            allergies=[],
        )
        assert result.is_safe is False
        assert len(result.decisions) >= 2
