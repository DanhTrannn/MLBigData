"""
Food feature extraction.
"""
import numpy as np


def extract_food_features(food: dict) -> dict:
    return {
        "calories_kcal": food.get("calories_kcal", 0),
        "protein_g": food.get("protein_g", 0),
        "carb_g": food.get("carb_g", 0),
        "fat_g": food.get("fat_g", 0),
        "sugar_g": food.get("sugar_g", 0),
        "fiber_g": food.get("fiber_g", 0),
        "sodium_mg": food.get("sodium_mg", 0),
        "serving_size_g": food.get("serving_size_g", 0),
        "cost_estimate": food.get("cost_estimate", 0),
        "purine_level": _purine_to_num(food.get("purine_level")),
        "num_ingredients": len(food.get("ingredients", [])),
        "num_meal_types": len(food.get("meal_types", [])),
        "is_vegetarian": 1.0 if "vegetarian" in food.get("diet_tags", []) else 0.0,
        "is_seafood": 1.0 if "seafood" in food.get("diet_tags", []) else 0.0,
    }


def _purine_to_num(level) -> float:
    mapping = {"low": 0.0, "moderate": 0.5, "high": 1.0}
    return mapping.get(level, 0.0) if level else 0.0
