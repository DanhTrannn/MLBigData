"""
Food data preprocessing pipeline.
Reads raw CSV, normalizes columns/units, assigns meal types and diet tags,
detects duplicates, and exports cleaned data.
"""
import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "sample_foods.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

NUTRIENT_FIELDS = [
    "calories_kcal", "protein_g", "carb_g", "fat_g",
    "sugar_g", "fiber_g", "sodium_mg",
]

ALLERGEN_INGREDIENTS = {
    "shellfish": {"shrimp", "squid", "crab", "snail"},
    "peanut": {"peanut"},
    "milk": {"milk", "condensed_milk", "yogurt", "butter"},
    "egg": {"egg"},
    "soy": {"tofu", "soy_milk", "soy_sauce", "soy_pudding"},
    "wheat": {"wheat_flour", "baguette", "pasta", "instant_noodle", "wheat_noodle"},
    "fish": {"fish", "fish_sauce"},
}

VEGETARIAN_EXCLUDE = {
    "beef", "pork", "chicken", "fish", "shrimp", "squid", "crab",
    "snail", "pork_rib", "pork_belly", "pork_offal", "pork_skin",
    "catfish", "sausage", "pate",
}


def read_raw_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_list_field(value: str, sep: str = "|") -> list[str]:
    if not value or not value.strip():
        return []
    return [item.strip() for item in value.split(sep) if item.strip()]


def normalize_row(row: dict) -> dict:
    record = {}
    record["food_id"] = row["food_id"].strip()
    record["name"] = row["name"].strip()
    record["meal_types"] = parse_list_field(row.get("meal_types", ""))
    record["ingredients"] = parse_list_field(row.get("ingredients", ""))
    record["serving_size_g"] = _safe_float(row.get("serving_size_g", "0"))
    record["calories_kcal"] = _safe_float(row.get("calories_kcal", "0"))
    record["protein_g"] = _safe_float(row.get("protein_g", "0"))
    record["carb_g"] = _safe_float(row.get("carb_g", "0"))
    record["fat_g"] = _safe_float(row.get("fat_g", "0"))
    record["sugar_g"] = _safe_float(row.get("sugar_g", "0"))
    record["fiber_g"] = _safe_float(row.get("fiber_g", "0"))
    record["sodium_mg"] = _safe_float(row.get("sodium_mg", "0"))
    record["purine_level"] = row.get("purine_level", "").strip().lower() or None
    record["cost_estimate"] = _safe_float(row.get("cost_estimate", "0"))
    record["diet_tags"] = parse_list_field(row.get("diet_tags", ""))
    record["source"] = row.get("source", "").strip()
    record["source_version"] = row.get("source_version", "").strip()
    record["quality_flags"] = parse_list_field(row.get("quality_flags", ""))
    return record


def _safe_float(value: str) -> float:
    try:
        v = float(value.strip()) if value and value.strip() else 0.0
        return max(v, 0.0)
    except (ValueError, TypeError):
        return 0.0


def assign_allergen_tags(record: dict) -> dict:
    ingredients = set(record["ingredients"])
    allergens = []
    for allergen, triggers in ALLERGEN_INGREDIENTS.items():
        if ingredients & triggers:
            allergens.append(allergen)
    record["allergen_tags"] = allergens
    return record


def assign_diet_tags(record: dict) -> dict:
    ingredients = set(record["ingredients"])
    tags = set(record.get("diet_tags", []))

    if not (ingredients & VEGETARIAN_EXCLUDE):
        tags.add("vegetarian")

    seafood_ingredients = {"shrimp", "squid", "crab", "snail", "fish"}
    if ingredients & seafood_ingredients:
        tags.add("seafood")

    if record.get("sodium_mg", 0) <= 300:
        tags.add("low_sodium")

    if record.get("protein_g", 0) >= 20:
        tags.add("high_protein")

    if record.get("calories_kcal", 0) <= 250:
        tags.add("low_calorie")

    record["diet_tags"] = sorted(tags)
    return record


def detect_duplicates(records: list[dict], threshold: float = 0.95) -> list[tuple]:
    duplicates = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i]["name"].lower() == records[j]["name"].lower():
                duplicates.append((records[i]["food_id"], records[j]["food_id"], "exact_name"))
                continue
            nutrients_i = [records[i].get(f, 0) for f in NUTRIENT_FIELDS]
            nutrients_j = [records[j].get(f, 0) for f in NUTRIENT_FIELDS]
            if _nutrient_similarity(nutrients_i, nutrients_j) > threshold:
                ing_i = set(records[i]["ingredients"])
                ing_j = set(records[j]["ingredients"])
                if ing_i and ing_j and _jaccard(ing_i, ing_j) > 0.8:
                    duplicates.append((records[i]["food_id"], records[j]["food_id"], "similar_nutrient_ingredient"))
    return duplicates


def _nutrient_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    diffs = []
    for x, y in zip(a, b):
        max_val = max(abs(x), abs(y), 1e-6)
        diffs.append(1.0 - abs(x - y) / max_val)
    return sum(diffs) / len(diffs)


def _jaccard(set_a: set, set_b: set) -> float:
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def assign_quality_flags(record: dict) -> dict:
    flags = list(record.get("quality_flags", []))

    critical_fields = ["calories_kcal", "protein_g", "carb_g", "fat_g", "sodium_mg"]
    for field in critical_fields:
        if record.get(field, 0) == 0:
            if "missing_critical" not in flags:
                flags.append("missing_critical")
            break

    if record.get("calories_kcal", 0) > 1000:
        if "outlier_calories" not in flags:
            flags.append("outlier_calories")

    if record.get("sodium_mg", 0) > 2000:
        if "outlier_sodium" not in flags:
            flags.append("outlier_sodium")

    record["quality_flags"] = flags
    return record


def validate_records(records: list[dict]) -> dict:
    report = {
        "total_records": len(records),
        "issues": [],
        "field_coverage": {},
        "duplicate_pairs": [],
    }

    for record in records:
        for field in NUTRIENT_FIELDS:
            if record.get(field, 0) == 0:
                report["issues"].append({
                    "food_id": record["food_id"],
                    "field": field,
                    "issue": "zero_value",
                })
        if not record.get("meal_types"):
            report["issues"].append({
                "food_id": record["food_id"],
                "field": "meal_types",
                "issue": "missing",
            })
        if not record.get("ingredients"):
            report["issues"].append({
                "food_id": record["food_id"],
                "field": "ingredients",
                "issue": "missing",
            })

    for field in NUTRIENT_FIELDS + ["serving_size_g", "cost_estimate"]:
        values = [r.get(field, 0) for r in records]
        non_zero = [v for v in values if v > 0]
        report["field_coverage"][field] = {
            "non_zero_count": len(non_zero),
            "coverage_pct": round(len(non_zero) / max(len(records), 1) * 100, 1),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "mean": round(sum(values) / max(len(values), 1), 2),
        }

    return report


def preprocess():
    print(f"Reading raw data from {RAW_PATH}")
    raw_records = read_raw_csv(RAW_PATH)
    print(f"Read {len(raw_records)} records")

    records = []
    for row in raw_records:
        record = normalize_row(row)
        record = assign_allergen_tags(record)
        record = assign_diet_tags(record)
        record = assign_quality_flags(record)
        records.append(record)

    duplicates = detect_duplicates(records)
    if duplicates:
        print(f"Found {len(duplicates)} duplicate pairs: {duplicates}")

    report = validate_records(records)
    report["duplicate_pairs"] = duplicates

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_json = PROCESSED_DIR / "foods.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records to {output_json}")

    report_path = REPORTS_DIR / "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved quality report to {report_path}")

    return records, report


if __name__ == "__main__":
    records, report = preprocess()
    print(f"\nSummary:")
    print(f"  Total records: {report['total_records']}")
    print(f"  Issues: {len(report['issues'])}")
    print(f"  Duplicates: {len(report['duplicate_pairs'])}")
