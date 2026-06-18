"""
MLP Scorer.
Uses trained MLP model to score user-food suitability.
Falls back to content-based scoring when MLP is not available.
"""
import numpy as np
import torch
from pathlib import Path

from backend.schemas.user import UserProfileIn, NutrientTargets
from ml.features.food_features import extract_food_features
from ml.features.user_features import extract_user_features

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


class SuitabilityScorer:
    def __init__(self, mlp_model=None, preprocessor=None):
        self.mlp_model = mlp_model
        self.preprocessor = preprocessor
        self._fallback = mlp_model is None

        if mlp_model is None:
            self._try_load_model()

    def _try_load_model(self):
        for run_dir in sorted(ARTIFACTS_DIR.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            model_path = run_dir / "mlp_model.pt"
            if model_path.exists():
                try:
                    from ml.models.suitability_mlp import SuitabilityMLP
                    from ml.features.feature_schema import get_feature_names

                    n_features = len(get_feature_names())
                    self.mlp_model = SuitabilityMLP(
                        input_dim=n_features,
                        hidden_dims=[128, 64, 32],
                        dropout=0.3,
                    )
                    self.mlp_model.load_state_dict(
                        torch.load(model_path, map_location="cpu", weights_only=True)
                    )
                    self.mlp_model.eval()
                    self._fallback = False
                    return
                except Exception:
                    continue
        self._fallback = True

    def score(
        self,
        profile: UserProfileIn,
        targets: NutrientTargets,
        candidates: dict[str, list[dict]],
    ) -> dict[str, list[dict]]:
        if self._fallback:
            return self._fallback_score(profile, targets, candidates)
        return self._mlp_score(profile, targets, candidates)

    def _fallback_score(
        self,
        profile: UserProfileIn,
        targets: NutrientTargets,
        candidates: dict[str, list[dict]],
    ) -> dict[str, list[dict]]:
        result = {}
        for meal_type, foods in candidates.items():
            for food in foods:
                food["suitability_score"] = food.get("retrieval_score", 0.5)
            result[meal_type] = foods
        return result

    def _mlp_score(
        self,
        profile: UserProfileIn,
        targets: NutrientTargets,
        candidates: dict[str, list[dict]],
    ) -> dict[str, list[dict]]:
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

        result = {}
        for meal_type, foods in candidates.items():
            batch_features = []
            for food in foods:
                ff = extract_food_features(food)

                cal_dev = abs(ff["calories_kcal"] - targets.calories_kcal / 3) / max(targets.calories_kcal / 3, 1)
                protein_dev = abs(ff["protein_g"] - targets.protein_g / 3) / max(targets.protein_g / 3, 1)
                sodium_ratio = ff["sodium_mg"] / max(targets.sodium_mg / 3, 1) if targets.sodium_mg > 0 else 0.0

                meal_bf = 1.0 if meal_type == "breakfast" else 0.0
                meal_l = 1.0 if meal_type == "lunch" else 0.0
                meal_d = 1.0 if meal_type == "dinner" else 0.0

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
                    meal_bf, meal_l, meal_d,
                ]
                batch_features.append(feature_vec)

            if batch_features:
                X = torch.tensor(batch_features, dtype=torch.float32)
                with torch.no_grad():
                    raw_scores = torch.sigmoid(self.mlp_model(X)).squeeze(-1).numpy()

                for i, food in enumerate(foods):
                    mlp_score = float(raw_scores[i])
                    retrieval_score = food.get("retrieval_score", 0.5)
                    food["suitability_score"] = round(0.6 * mlp_score + 0.4 * retrieval_score, 4)
                    food["mlp_score"] = round(mlp_score, 4)
            else:
                for food in foods:
                    food["suitability_score"] = food.get("retrieval_score", 0.5)

            result[meal_type] = foods

        return result
