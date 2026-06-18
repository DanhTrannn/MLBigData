"""
Generate synthetic user profiles with correlated attributes.
"""
import json
import csv
import random
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "synthetic_generation.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_users(config: dict) -> list[dict]:
    seed = config["users"]["seed"]
    random.seed(seed)

    count = config["users"]["count"]
    age_range = config["users"]["age_range"]
    height_range = config["users"]["height_cm_range"]
    weight_range = config["users"]["weight_kg_range"]
    activity_dist = config["users"]["activity_distribution"]
    goal_dist = config["users"]["goal_distribution"]
    disease_prev = config["users"]["disease_prevalence"]
    allergy_prev = config["users"]["allergy_prevalence"]
    budget_range = config["users"]["budget_range_vnd"]
    comorbidity_corr = config["users"].get("comorbidity_correlation", 0.3)

    activities = list(activity_dist.keys())
    activity_weights = list(activity_dist.values())
    goals = list(goal_dist.keys())
    goal_weights = list(goal_dist.values())

    users = []
    for i in range(count):
        age = random.randint(age_range[0], age_range[1])
        height = round(random.uniform(height_range[0], height_range[1]), 1)
        weight = round(random.uniform(weight_range[0], weight_range[1]), 1)

        height_m = height / 100
        bmi = weight / (height_m * height_m)

        activity = random.choices(activities, weights=activity_weights, k=1)[0]
        goal = random.choices(goals, weights=goal_weights, k=1)[0]

        diseases = []
        base_prob_mult = 1.0 if bmi > 25 else 0.5
        for disease, base_prob in disease_prev.items():
            prob = min(1.0, base_prob * base_prob_mult)
            if diseases:
                prob = min(1.0, prob + comorbidity_corr * 0.5)
            if random.random() < prob:
                diseases.append(disease)

        allergies = []
        for allergen, prob in allergy_prev.items():
            if random.random() < prob:
                allergies.append(allergen)

        budget = round(random.uniform(budget_range[0], budget_range[1]) / 10000) * 10000

        cuisine_tags = ["vietnamese", "seafood", "vegetarian", "spicy", "western"]
        preferred_tags = random.sample(cuisine_tags, k=random.randint(1, 3))

        users.append({
            "user_id": f"U{i+1:04d}",
            "age": age,
            "height_cm": height,
            "weight_kg": weight,
            "bmi": round(bmi, 1),
            "activity_level": activity,
            "goal": goal,
            "diseases": diseases,
            "allergies": allergies,
            "budget_per_day": budget,
            "preferred_tags": preferred_tags,
            "latent_taste_seed": random.randint(0, 10000),
        })

    return users


def generate_report(users: list[dict]) -> dict:
    report = {
        "total_users": len(users),
        "age_stats": _stats([u["age"] for u in users]),
        "bmi_stats": _stats([u["bmi"] for u in users]),
        "disease_counts": {},
        "allergy_counts": {},
        "activity_counts": {},
        "goal_counts": {},
    }

    for u in users:
        for d in u["diseases"]:
            report["disease_counts"][d] = report["disease_counts"].get(d, 0) + 1
        for a in u["allergies"]:
            report["allergy_counts"][a] = report["allergy_counts"].get(a, 0) + 1
        report["activity_counts"][u["activity_level"]] = report["activity_counts"].get(u["activity_level"], 0) + 1
        report["goal_counts"][u["goal"]] = report["goal_counts"].get(u["goal"], 0) + 1

    return report


def _stats(values: list) -> dict:
    if not values:
        return {}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
        "count": len(values),
    }


def main():
    config = load_config()
    print(f"Generating {config['users']['count']} synthetic users...")

    users = generate_users(config)
    report = generate_report(users)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "users.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(users)} users to {output_path}")

    report_path = REPORTS_DIR / "synthetic_users_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report to {report_path}")

    print(f"\nDistribution:")
    print(f"  Diseases: {report['disease_counts']}")
    print(f"  Allergies: {report['allergy_counts']}")
    print(f"  Activity: {report['activity_counts']}")
    print(f"  Goals: {report['goal_counts']}")


if __name__ == "__main__":
    main()
