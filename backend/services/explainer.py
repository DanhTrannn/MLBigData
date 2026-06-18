"""
Explainer service.
Combines rule-based explanations, SHAP feature contributions for MLP,
and optimizer summary.
"""
import numpy as np
from backend.schemas.user import UserProfileIn, NutrientTargets
from backend.ml_runtime.model_bundle import ModelBundle
from backend.utils.logging import logger

from ml.features.food_features import extract_food_features
from ml.features.user_features import extract_user_features
from ml.features.feature_schema import get_feature_names


class Explainer:
    def __init__(self, bundle: ModelBundle | None = None):
        self.bundle = bundle
        self._shap_explainer = None
        self._feature_names = get_feature_names()

        if bundle and bundle.mlp_model is not None and bundle.shap_background is not None:
            self._init_shap(bundle.mlp_model, bundle.shap_background)

    def _init_shap(self, mlp_model, background):
        try:
            import shap
            import torch

            bg_tensor = torch.tensor(background, dtype=torch.float32)
            self._shap_explainer = shap.DeepExplainer(mlp_model, bg_tensor)
            logger.info("SHAP DeepExplainer initialized")
        except Exception as e:
            logger.warning(f"Failed to init SHAP explainer: {e}")
            self._shap_explainer = None

    def _get_shap_explanations(
        self,
        profile: UserProfileIn,
        food: dict,
        meal_type: str,
        targets: NutrientTargets,
        lang: str = "vi",
    ) -> list[str]:
        if self._shap_explainer is None:
            return []

        try:
            import torch

            user_feats = extract_user_features({
                "age": profile.age,
                "height_cm": profile.height_cm,
                "weight_kg": profile.weight_kg,
                "bmi": profile.bmi,
                "activity_level": profile.activity_level.value,
                "goal": profile.goal.value,
                "diseases": [d.value for d in profile.diseases],
                "allergies": [a.value for a in profile.allergies],
                "budget_per_day": profile.budget_per_day,
            })
            ff = extract_food_features(food)

            cal_dev = abs(ff["calories_kcal"] - targets.calories_kcal / 3) / max(targets.calories_kcal / 3, 1)
            protein_dev = abs(ff["protein_g"] - targets.protein_g / 3) / max(targets.protein_g / 3, 1)
            sodium_ratio = ff["sodium_mg"] / max(targets.sodium_mg / 3, 1)

            feature_vec = [
                user_feats["age"], user_feats["height_cm"], user_feats["weight_kg"],
                user_feats["bmi"], user_feats["activity_level"], user_feats["goal"],
                user_feats["has_hypertension"], user_feats["has_diabetes"],
                user_feats["has_gout"], user_feats["num_allergies"],
                user_feats["budget_normalized"],
                ff["calories_kcal"] / 1000, ff["protein_g"] / 50,
                ff["carb_g"] / 80, ff["fat_g"] / 50,
                ff["sugar_g"] / 30, ff["fiber_g"] / 10, ff["sodium_mg"] / 1500,
                ff["serving_size_g"] / 600, ff["cost_estimate"] / 80000,
                ff["purine_level"], ff["num_ingredients"] / 15,
                ff["num_meal_types"] / 4, ff["is_vegetarian"], ff["is_seafood"],
                cal_dev, protein_dev, min(sodium_ratio, 3.0),
                1.0 if meal_type == "breakfast" else 0.0,
                1.0 if meal_type == "lunch" else 0.0,
                1.0 if meal_type == "dinner" else 0.0,
            ]

            x = torch.tensor([feature_vec], dtype=torch.float32)
            shap_values = self._shap_explainer.shap_values(x)
            vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

            feature_labels = {
                "food_calories_kcal": ("Năng lượng", "Energy"),
                "food_protein_g": ("Chất đạm", "Protein"),
                "food_fiber_g": ("Chất xơ", "Fiber"),
                "food_sugar_g": ("Đường", "Sugar"),
                "food_sodium_mg": ("Natri", "Sodium"),
                "food_cost_estimate": ("Chi phí", "Cost"),
                "calorie_deviation": ("Độ lệch năng lượng", "Calorie deviation"),
            }

            explanations = []
            indexed = list(enumerate(vals))
            indexed.sort(key=lambda x: abs(x[1]), reverse=True)

            for idx, shap_val in indexed[:3]:
                if abs(shap_val) < 0.01:
                    continue
                fname = self._feature_names[idx] if idx < len(self._feature_names) else f"feature_{idx}"
                if fname in feature_labels:
                    label_vi, label_en = feature_labels[fname]
                    direction = "tích cực" if shap_val > 0 else "tiêu cực"
                    direction_en = "positive" if shap_val > 0 else "negative"
                    if lang == "vi":
                        explanations.append(f"SHAP: {label_vi} đóng góp {direction} ({shap_val:+.3f})")
                    else:
                        explanations.append(f"SHAP: {label_en} has {direction_en} contribution ({shap_val:+.3f})")

            return explanations

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return []

    def explain(
        self,
        profile: UserProfileIn,
        plan: list[dict],
        rule_trace: dict,
        daily_targets: NutrientTargets,
        lang: str = "vi",
    ) -> list[dict]:
        explained_meals = []
        for meal in plan:
            explanations = []
            fid = meal["food_id"]

            trace = rule_trace.get(fid, {})
            if trace.get("is_safe"):
                explanations.append(
                    "Không vi phạm quy tắc an toàn đã cấu hình."
                    if lang == "vi"
                    else "No safety rule violations."
                )

            for w in trace.get("warnings", []):
                explanations.append(f"⚠ {w}")

            if profile.preferred_tags:
                food_tags = set(t.lower() for t in meal.get("diet_tags", []))
                preferred = set(t.lower() for t in profile.preferred_tags)
                overlap = food_tags & preferred
                if overlap:
                    tag_names = ", ".join(overlap)
                    explanations.append(
                        f"Phù hợp với sở thích: {tag_names}."
                        if lang == "vi"
                        else f"Matches your preferences: {tag_names}."
                    )

            cal = meal.get("calories_kcal", 0)
            explanations.append(
                f"Năng lượng: {cal:.0f} kcal."
                if lang == "vi"
                else f"Energy: {cal:.0f} kcal."
            )

            shap_explanations = self._get_shap_explanations(
                profile, meal, meal.get("meal_type", "lunch"), daily_targets, lang
            )
            explanations.extend(shap_explanations)

            meal["explanations"] = explanations
            explained_meals.append(meal)

        return explained_meals

    def explain_plan_summary(
        self,
        plan: list[dict],
        daily_targets: NutrientTargets,
        lang: str = "vi",
    ) -> dict:
        total_cal = sum(m.get("calories_kcal", 0) for m in plan)
        total_sodium = sum(m.get("sodium_mg", 0) for m in plan)
        total_sugar = sum(m.get("sugar_g", 0) for m in plan)
        total_cost = sum(m.get("cost_estimate", 0) for m in plan)

        cal_dev = total_cal - daily_targets.calories_kcal
        status = "pass"
        if total_sodium > daily_targets.sodium_mg:
            status = "warn"
        if total_sugar > daily_targets.sugar_g:
            status = "warn"

        return {
            "total_calories_kcal": round(total_cal, 0),
            "total_protein_g": round(sum(m.get("protein_g", 0) for m in plan), 1),
            "total_carb_g": round(sum(m.get("carb_g", 0) for m in plan), 1),
            "total_fat_g": round(sum(m.get("fat_g", 0) for m in plan), 1),
            "total_sugar_g": round(total_sugar, 1),
            "total_fiber_g": round(sum(m.get("fiber_g", 0) for m in plan), 1),
            "total_sodium_mg": round(total_sodium, 0),
            "estimated_cost": round(total_cost, 0),
            "constraint_status": status,
            "target_calories_kcal": daily_targets.calories_kcal,
            "calorie_deviation": round(cal_dev, 0),
        }
