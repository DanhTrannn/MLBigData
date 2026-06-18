"""
Generate synthetic user-food interactions.
Ratings are based on utility function, not random.
"""
import json
import random
import math
from datetime import datetime, timedelta
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "synthetic_generation.yaml"
FOODS_PATH = PROJECT_ROOT / "data" / "processed" / "foods.json"
USERS_PATH = PROJECT_ROOT / "data" / "processed" / "users.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

ALLERGEN_INGREDIENTS = {
    "shellfish": {"shrimp", "squid", "crab", "snail"},
    "peanut": {"peanut"},
    "milk": {"milk", "condensed_milk", "yogurt", "butter"},
    "egg": {"egg"},
    "soy": {"tofu", "soy_milk", "soy_sauce", "soy_pudding"},
    "wheat": {"wheat_flour", "baguette", "pasta", "instant_noodle", "wheat_noodle"},
    "fish": {"fish", "fish_sauce"},
}


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_hard_rule_violation(user: dict, food: dict) -> bool:
    food_ingredients = set(food.get("ingredients", []))
    for allergy in user.get("allergies", []):
        allergen_ingredients = ALLERGEN_INGREDIENTS.get(allergy, set())
        if food_ingredients & allergen_ingredients:
            return True

    diseases = user.get("diseases", [])
    if "hypertension" in diseases and food.get("sodium_mg", 0) > 500:
        return True
    if "diabetes" in diseases and food.get("sugar_g", 0) > 15:
        return True
    if "gout" in diseases and food.get("purine_level") == "high":
        return True

    return False


def compute_utility(user: dict, food: dict, rng: random.Random) -> float:
    utility = 0.0

    food_tags = set(t.lower() for t in food.get("diet_tags", []))
    preferred = set(t.lower() for t in user.get("preferred_tags", []))
    tag_overlap = len(food_tags & preferred)
    utility += tag_overlap * 0.5

    bmi = user.get("bmi", 22)
    goal = user.get("goal", "maintain")
    cal = food.get("calories_kcal", 300)

    if goal == "lose":
        utility += max(0, 1.0 - cal / 400) * 0.3
    elif goal == "gain":
        utility += min(1.0, cal / 500) * 0.3
    else:
        utility += max(0, 1.0 - abs(cal - 400) / 400) * 0.3

    budget = user.get("budget_per_day", 150000)
    cost = food.get("cost_estimate", 30000)
    per_meal_budget = budget / 3
    if cost <= per_meal_budget:
        utility += 0.2
    else:
        utility -= (cost - per_meal_budget) / per_meal_budget * 0.2

    disliked = set(user.get("disliked_ingredients", []))
    food_ingredients = set(food.get("ingredients", []))
    if disliked & food_ingredients:
        utility -= 0.5

    taste_seed = user.get("latent_taste_seed", 0)
    food_id_hash = hash(food.get("food_id", "")) % 10000
    latent = math.sin(taste_seed * food_id_hash * 0.001) * 0.3
    utility += latent

    noise_std = 0.5
    utility += rng.gauss(0, noise_std)

    return utility


def generate_interactions(config: dict, users: list[dict], foods: list[dict]) -> list[dict]:
    seed = config["users"]["seed"]
    rng = random.Random(seed)

    min_per_user = config["interactions"]["min_per_user"]
    max_per_user = config["interactions"]["max_per_user"]
    event_dist = config["interactions"]["event_distribution"]
    hard_policy = config["interactions"]["hard_rule_violation_policy"]

    event_types = list(event_dist.keys())
    event_weights = list(event_dist.values())

    interactions = []
    base_time = datetime(2026, 1, 1)

    for user in users:
        n_interactions = rng.randint(min_per_user, max_per_user)
        user_foods = list(foods)
        rng.shuffle(user_foods)

        count = 0
        for food in user_foods:
            if count >= n_interactions:
                break

            violation = is_hard_rule_violation(user, food)

            if violation and hard_policy == "no_positive_signal":
                event_type = "dislike"
                event_value = None
            else:
                utility = compute_utility(user, food, rng)
                event_type = rng.choices(event_types, weights=event_weights, k=1)[0]

                if event_type == "rating":
                    event_value = max(1, min(5, round(utility * 2 + 3, 1)))
                elif event_type == "like":
                    if utility < 0:
                        event_type = "dislike"
                    event_value = None
                elif event_type == "dislike":
                    event_value = None
                elif event_type == "eaten":
                    event_value = None
                elif event_type == "swap":
                    event_value = None
                else:
                    event_value = None

            if violation and hard_policy == "no_positive_signal":
                if event_type in ("like", "eaten", "rating"):
                    if event_type == "rating" and (event_value or 0) > 2:
                        continue

            meal_types = food.get("meal_types", ["lunch"])
            meal_type = rng.choice(meal_types)

            timestamp = base_time + timedelta(
                days=rng.randint(0, 180),
                hours=rng.randint(6, 22),
                minutes=rng.randint(0, 59),
            )

            interactions.append({
                "user_id": user["user_id"],
                "food_id": food["food_id"],
                "event_type": event_type,
                "event_value": event_value,
                "meal_type": meal_type,
                "timestamp": timestamp.isoformat(),
                "context": {
                    "diseases": user.get("diseases", []),
                    "allergies": user.get("allergies", []),
                },
            })
            count += 1

    interactions.sort(key=lambda x: x["timestamp"])
    return interactions


def generate_report(interactions: list[dict], users: list[dict]) -> dict:
    event_counts = {}
    for i in interactions:
        et = i["event_type"]
        event_counts[et] = event_counts.get(et, 0) + 1

    user_interaction_counts = {}
    for i in interactions:
        uid = i["user_id"]
        user_interaction_counts[uid] = user_interaction_counts.get(uid, 0) + 1

    food_interaction_counts = {}
    for i in interactions:
        fid = i["food_id"]
        food_interaction_counts[fid] = food_interaction_counts.get(fid, 0) + 1

    counts = list(user_interaction_counts.values())
    cold_start = sum(1 for c in counts if c <= 3)

    return {
        "total_interactions": len(interactions),
        "total_users_with_interactions": len(user_interaction_counts),
        "total_foods_interacted": len(food_interaction_counts),
        "event_distribution": event_counts,
        "interactions_per_user": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": round(sum(counts) / max(len(counts), 1), 2),
        },
        "cold_start_users": cold_start,
        "sparsity": round(
            1 - len(interactions) / max(len(users) * 100, 1), 4
        ),
    }


def main():
    config = load_config()

    print("Loading users and foods...")
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        users = json.load(f)
    with open(FOODS_PATH, "r", encoding="utf-8") as f:
        foods = json.load(f)

    print(f"Generating interactions for {len(users)} users x {len(foods)} foods...")
    interactions = generate_interactions(config, users, foods)
    report = generate_report(interactions, users)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "interactions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(interactions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(interactions)} interactions to {output_path}")

    report_path = REPORTS_DIR / "interactions_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report to {report_path}")

    print(f"\nSummary:")
    print(f"  Total interactions: {report['total_interactions']}")
    print(f"  Event distribution: {report['event_distribution']}")
    print(f"  Cold start users: {report['cold_start_users']}")
    print(f"  Sparsity: {report['sparsity']}")


if __name__ == "__main__":
    main()
