"""
Validate processed food data against the expected schema.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "foods.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

REQUIRED_FIELDS = [
    "food_id", "name", "meal_types", "ingredients",
    "serving_size_g", "calories_kcal", "protein_g", "carb_g", "fat_g",
    "sugar_g", "fiber_g", "sodium_mg", "cost_estimate", "diet_tags",
    "source", "source_version",
]

VALID_MEAL_TYPES = {"breakfast", "lunch", "dinner", "snack"}
VALID_PURINE_LEVELS = {"low", "moderate", "high", None}


def validate_processed_data(path: Path = PROCESSED_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    report = {
        "total_records": len(records),
        "passed": 0,
        "failed": 0,
        "errors": [],
    }

    seen_ids = set()

    for i, record in enumerate(records):
        errors = []

        for field in REQUIRED_FIELDS:
            if field not in record:
                errors.append(f"Missing field: {field}")

        fid = record.get("food_id", f"index_{i}")

        if fid in seen_ids:
            errors.append(f"Duplicate food_id: {fid}")
        seen_ids.add(fid)

        if not isinstance(record.get("meal_types"), list):
            errors.append("meal_types must be a list")
        else:
            invalid = set(record["meal_types"]) - VALID_MEAL_TYPES
            if invalid:
                errors.append(f"Invalid meal_types: {invalid}")

        if not isinstance(record.get("ingredients"), list):
            errors.append("ingredients must be a list")

        numeric_fields = [
            "serving_size_g", "calories_kcal", "protein_g", "carb_g",
            "fat_g", "sugar_g", "fiber_g", "sodium_mg", "cost_estimate",
        ]
        for field in numeric_fields:
            val = record.get(field)
            if val is not None and not isinstance(val, (int, float)):
                errors.append(f"{field} must be numeric, got {type(val).__name__}")
            elif isinstance(val, (int, float)) and val < 0:
                errors.append(f"{field} must be non-negative, got {val}")

        purine = record.get("purine_level")
        if purine not in VALID_PURINE_LEVELS:
            errors.append(f"Invalid purine_level: {purine}")

        if errors:
            report["failed"] += 1
            report["errors"].append({"food_id": fid, "errors": errors})
        else:
            report["passed"] += 1

    report["unique_food_ids"] = len(seen_ids)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    report = validate_processed_data()
    print(f"Validation complete:")
    print(f"  Total: {report['total_records']}")
    print(f"  Passed: {report['passed']}")
    print(f"  Failed: {report['failed']}")
    if report["errors"]:
        for err in report["errors"][:10]:
            print(f"  {err['food_id']}: {err['errors']}")
