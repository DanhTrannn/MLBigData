"""
User feature extraction.
"""
import numpy as np


def extract_user_features(user: dict) -> dict:
    bmi = user.get("bmi", 0)
    if bmi == 0 and user.get("height_cm") and user.get("weight_kg"):
        h = user["height_cm"] / 100
        bmi = user["weight_kg"] / (h * h)

    activity_map = {"sedentary": 0.0, "light": 0.33, "moderate": 0.66, "high": 1.0}
    goal_map = {"lose": 0.0, "maintain": 0.5, "gain": 1.0}

    return {
        "age": user.get("age", 0) / 100.0,
        "height_cm": user.get("height_cm", 0) / 200.0,
        "weight_kg": user.get("weight_kg", 0) / 150.0,
        "bmi": bmi / 40.0,
        "activity_level": activity_map.get(user.get("activity_level", "moderate"), 0.5),
        "goal": goal_map.get(user.get("goal", "maintain"), 0.5),
        "has_hypertension": 1.0 if "hypertension" in user.get("diseases", []) else 0.0,
        "has_diabetes": 1.0 if "diabetes" in user.get("diseases", []) else 0.0,
        "has_gout": 1.0 if "gout" in user.get("diseases", []) else 0.0,
        "num_allergies": len(user.get("allergies", [])),
        "budget_normalized": (user.get("budget_per_day") or 150000) / 300000,
    }
