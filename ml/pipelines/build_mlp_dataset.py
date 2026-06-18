"""
Build MLP training dataset from user-food interactions and features.
"""
import json
import numpy as np
from pathlib import Path
from ml.features.food_features import extract_food_features
from ml.features.user_features import extract_user_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
USERS_PATH = PROJECT_ROOT / "data" / "processed" / "users.json"
FOODS_PATH = PROJECT_ROOT / "data" / "processed" / "foods.json"
INTERACTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "interactions.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def build_dataset():
    with open(USERS_PATH, "r") as f:
        users = json.load(f)
    with open(FOODS_PATH, "r") as f:
        foods = json.load(f)
    with open(INTERACTIONS_PATH, "r") as f:
        interactions = json.load(f)

    user_map = {u["user_id"]: u for u in users}
    food_map = {f["food_id"]: f for f in foods}

    features_list = []
    labels = []

    for interaction in interactions:
        user = user_map.get(interaction["user_id"])
        food = food_map.get(interaction["food_id"])
        if not user or not food:
            continue

        event_type = interaction["event_type"]
        if event_type == "rating":
            label = (interaction.get("event_value", 3) - 1) / 4.0
        elif event_type in ("like", "eaten"):
            label = 1.0
        elif event_type == "dislike":
            label = 0.0
        elif event_type == "swap":
            label = 0.2
        else:
            continue

        uf = extract_user_features(user)
        ff = extract_food_features(food)

        meal_type = interaction.get("meal_type", "lunch")
        meal_breakfast = 1.0 if meal_type == "breakfast" else 0.0
        meal_lunch = 1.0 if meal_type == "lunch" else 0.0
        meal_dinner = 1.0 if meal_type == "dinner" else 0.0

        cal_deviation = abs(ff["calories_kcal"] - 400) / 400
        protein_deviation = abs(ff["protein_g"] - 20) / 20
        sodium_ratio = ff["sodium_mg"] / 500 if ff["sodium_mg"] > 0 else 0.0

        feature_vec = [
            uf["age"], uf["height_cm"], uf["weight_kg"], uf["bmi"],
            uf["activity_level"], uf["goal"],
            uf["has_hypertension"], uf["has_diabetes"], uf["has_gout"],
            uf["num_allergies"], uf["budget_normalized"],
            ff["calories_kcal"] / 1000, ff["protein_g"] / 50,
            ff["carb_g"] / 80, ff["fat_g"] / 50,
            ff["sugar_g"] / 30, ff["fiber_g"] / 10, ff["sodium_mg"] / 1500,
            ff["serving_size_g"] / 600, ff["cost_estimate"] / 80000,
            ff["purine_level"], ff["num_ingredients"] / 15,
            ff["num_meal_types"] / 4, ff["is_vegetarian"], ff["is_seafood"],
            cal_deviation, protein_deviation, min(sodium_ratio, 3.0),
            meal_breakfast, meal_lunch, meal_dinner,
        ]

        features_list.append(feature_vec)
        labels.append(label)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "mlp_features.npy", X)
    np.save(OUTPUT_DIR / "mlp_labels.npy", y)

    print(f"Built MLP dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Label distribution: mean={y.mean():.3f}, std={y.std():.3f}")

    return X, y


if __name__ == "__main__":
    build_dataset()
