"""Tests for Target Calculator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from backend.services.target_calculator import TargetCalculator
from backend.schemas.user import UserProfileIn, ActivityLevel, Goal, Disease


@pytest.fixture
def calculator():
    return TargetCalculator()


@pytest.fixture
def healthy_user():
    return UserProfileIn(
        age=30, height_cm=170, weight_kg=65,
        activity_level=ActivityLevel.moderate,
        goal=Goal.maintain,
    )


@pytest.fixture
def diabetic_user():
    return UserProfileIn(
        age=55, height_cm=165, weight_kg=80,
        activity_level=ActivityLevel.light,
        goal=Goal.lose,
        diseases=[Disease.diabetes],
    )


@pytest.fixture
def htn_user():
    return UserProfileIn(
        age=50, height_cm=170, weight_kg=75,
        activity_level=ActivityLevel.sedentary,
        goal=Goal.maintain,
        diseases=[Disease.hypertension],
    )


class TestTargetCalculator:
    def test_healthy_user_calories(self, calculator, healthy_user):
        targets = calculator.calculate(healthy_user)
        assert 1500 < targets.calories_kcal < 3000

    def test_healthy_user_positive_nutrients(self, calculator, healthy_user):
        targets = calculator.calculate(healthy_user)
        assert targets.protein_g > 0
        assert targets.carb_g > 0
        assert targets.fat_g > 0

    def test_diabetes_limits_sugar(self, calculator, diabetic_user):
        targets = calculator.calculate(diabetic_user)
        assert targets.sugar_g <= 45

    def test_hypertension_limits_sodium(self, calculator, htn_user):
        targets = calculator.calculate(htn_user)
        assert targets.sodium_mg <= 1500

    def test_lose_goal_reduces_calories(self, calculator):
        maintain = UserProfileIn(
            age=30, height_cm=170, weight_kg=70,
            activity_level=ActivityLevel.moderate,
            goal=Goal.maintain,
        )
        lose = UserProfileIn(
            age=30, height_cm=170, weight_kg=70,
            activity_level=ActivityLevel.moderate,
            goal=Goal.lose,
        )
        m_targets = calculator.calculate(maintain)
        l_targets = calculator.calculate(lose)
        assert l_targets.calories_kcal < m_targets.calories_kcal

    def test_per_meal_targets(self, calculator, healthy_user):
        daily = calculator.calculate(healthy_user)
        meal_targets = calculator.get_per_meal_targets(daily)
        assert "breakfast" in meal_targets
        assert "lunch" in meal_targets
        assert "dinner" in meal_targets
        assert meal_targets["breakfast"].calories_kcal < daily.calories_kcal

    def test_bmi_calculation(self, healthy_user):
        bmi = healthy_user.bmi
        assert 18 < bmi < 30

    def test_minimum_calories(self, calculator):
        very_small = UserProfileIn(
            age=70, height_cm=150, weight_kg=45,
            activity_level=ActivityLevel.sedentary,
            goal=Goal.lose,
        )
        targets = calculator.calculate(very_small)
        assert targets.calories_kcal >= 1200
