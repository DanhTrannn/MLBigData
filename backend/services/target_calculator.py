"""
Target Calculator.
Computes daily and per-meal nutrient targets from user profile and config.
"""
import yaml
from pathlib import Path
from backend.schemas.user import UserProfileIn, NutrientTargets

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "nutrient_targets.yaml"


class TargetCalculator:
    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def calculate(self, profile: UserProfileIn) -> NutrientTargets:
        bmr = self._calculate_bmr(profile)
        activity_mult = self.config["activity_multipliers"].get(
            profile.activity_level.value, 1.375
        )
        tdee = bmr * activity_mult

        goal_adj = self.config["goal_adjustments"].get(profile.goal.value, 0)
        calories = tdee + goal_adj

        min_cal = self.config.get("min_calories", 1200)
        max_cal = self.config.get("max_calories", 4000)
        calories = max(min_cal, min(max_cal, calories))

        defaults = self.config["default_targets"]
        cal_ratio = calories / defaults["calories_kcal"]

        targets = NutrientTargets(
            calories_kcal=round(calories, 0),
            protein_g=round(defaults["protein_g"] * cal_ratio, 1),
            carb_g=round(defaults["carb_g"] * cal_ratio, 1),
            fat_g=round(defaults["fat_g"] * cal_ratio, 1),
            sugar_g=round(defaults["sugar_g"] * cal_ratio, 1),
            fiber_g=round(defaults["fiber_g"] * cal_ratio, 1),
            sodium_mg=defaults["sodium_mg"],
        )

        disease_overrides = self.config.get("disease_overrides", {})
        for disease in profile.diseases:
            overrides = disease_overrides.get(disease.value, {})
            for field, value in overrides.items():
                current = getattr(targets, field)
                setattr(targets, field, min(current, value))

        return targets

    def get_per_meal_targets(
        self, daily: NutrientTargets
    ) -> dict[str, NutrientTargets]:
        ratios = self.config.get("per_meal_ratios", {
            "breakfast": 0.30, "lunch": 0.40, "dinner": 0.30,
        })
        result = {}
        for meal_type, ratio in ratios.items():
            result[meal_type] = NutrientTargets(
                calories_kcal=round(daily.calories_kcal * ratio, 0),
                protein_g=round(daily.protein_g * ratio, 1),
                carb_g=round(daily.carb_g * ratio, 1),
                fat_g=round(daily.fat_g * ratio, 1),
                sugar_g=round(daily.sugar_g * ratio, 1),
                fiber_g=round(daily.fiber_g * ratio, 1),
                sodium_mg=round(daily.sodium_mg * ratio, 0),
            )
        return result

    def _calculate_bmr(self, profile: UserProfileIn) -> float:
        sex = (profile.sex or self.config["bmr"].get("default_sex", "male")).lower()
        w = profile.weight_kg
        h = profile.height_cm
        a = profile.age

        if sex == "female":
            return 10 * w + 6.25 * h - 5 * a - 161
        else:
            return 10 * w + 6.25 * h - 5 * a + 5
