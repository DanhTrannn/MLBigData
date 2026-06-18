"""
Feature schema definition.
Ensures consistent features between training and inference.
"""

NUMERIC_FEATURES = [
    "user_age", "user_height_cm", "user_weight_kg", "user_bmi",
    "user_activity_level", "user_goal",
    "user_has_hypertension", "user_has_diabetes", "user_has_gout",
    "user_num_allergies", "user_budget_normalized",
    "food_calories_kcal", "food_protein_g", "food_carb_g", "food_fat_g",
    "food_sugar_g", "food_fiber_g", "food_sodium_mg",
    "food_serving_size_g", "food_cost_estimate",
    "food_purine_level", "food_num_ingredients", "food_num_meal_types",
    "food_is_vegetarian", "food_is_seafood",
    "calorie_deviation", "protein_deviation", "sodium_ratio",
    "meal_breakfast", "meal_lunch", "meal_dinner",
]

FEATURE_SCHEMA_VERSION = "2026-01"


def get_feature_names() -> list[str]:
    return NUMERIC_FEATURES.copy()


def get_schema_info() -> dict:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "num_features": len(NUMERIC_FEATURES),
        "feature_names": NUMERIC_FEATURES,
    }
