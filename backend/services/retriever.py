"""
Hybrid retriever.
Uses GNN food embeddings (when available) + content-based scoring.
"""
import numpy as np
from backend.schemas.user import UserProfileIn, NutrientTargets
from backend.services.cold_start import ColdStartEncoder
from backend.ml_runtime.model_bundle import ModelBundle


class ContentRetriever:
    def __init__(self, user_encoder: ColdStartEncoder, bundle: ModelBundle | None = None):
        self.user_encoder = user_encoder
        self.bundle = bundle
        self._food_emb_matrix = None
        self._food_emb_norm = None
        self._food_id_to_idx = {}
        self._user_emb_norm = None

        if bundle and bundle.food_embeddings is not None:
            self._food_emb_matrix = bundle.food_embeddings
            self._food_id_to_idx = bundle.food_id_to_index
            self._food_emb_norm = self._normalize(self._food_emb_matrix)

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def set_user_embedding(self, user_embedding: np.ndarray):
        if user_embedding is not None and self._food_emb_matrix is not None:
            if len(user_embedding) == self._food_emb_matrix.shape[1]:
                norm = np.linalg.norm(user_embedding)
                if norm > 0:
                    self._user_emb_norm = user_embedding / norm
                else:
                    self._user_emb_norm = user_embedding
            else:
                self._user_emb_norm = None
        else:
            self._user_emb_norm = None

    def retrieve(
        self,
        profile: UserProfileIn,
        safe_foods: list[dict],
        targets: NutrientTargets,
        meal_targets: dict[str, NutrientTargets],
        top_k: int = 50,
        excluded_food_ids: set[str] | None = None,
    ) -> dict[str, list[dict]]:
        excluded = excluded_food_ids or set()
        result: dict[str, list[dict]] = {}

        for meal_type in ["breakfast", "lunch", "dinner"]:
            meal_target = meal_targets.get(meal_type, targets)
            candidates = [
                f for f in safe_foods
                if meal_type in f.get("meal_types", [])
                and f["food_id"] not in excluded
            ]

            scored = []
            for food in candidates:
                score = self._score_food(food, profile, meal_target)
                scored.append((score, food))

            scored.sort(key=lambda x: x[0], reverse=True)
            result[meal_type] = [
                {**food, "retrieval_score": score}
                for score, food in scored[:top_k]
            ]

        return result

    def _score_food(
        self,
        food: dict,
        profile: UserProfileIn,
        meal_target: NutrientTargets,
    ) -> float:
        tag_affinity = self.user_encoder.compute_tag_affinity(food, profile)

        cal_deviation = abs(food.get("calories_kcal", 0) - meal_target.calories_kcal)
        cal_score = max(0, 1.0 - cal_deviation / max(meal_target.calories_kcal, 1))

        protein_deviation = abs(food.get("protein_g", 0) - meal_target.protein_g)
        protein_score = max(0, 1.0 - protein_deviation / max(meal_target.protein_g, 1))

        sodium_penalty = 0.0
        if meal_target.sodium_mg > 0:
            sodium_ratio = food.get("sodium_mg", 0) / meal_target.sodium_mg
            if sodium_ratio > 1.0:
                sodium_penalty = min(0.3, (sodium_ratio - 1.0) * 0.15)

        budget_score = 1.0
        if profile.budget_per_day and profile.budget_per_day > 0:
            per_meal_budget = profile.budget_per_day / 3
            cost = food.get("cost_estimate", 0)
            if cost > per_meal_budget:
                budget_score = max(0, 1.0 - (cost - per_meal_budget) / per_meal_budget)

        content_score = (
            0.30 * tag_affinity
            + 0.30 * cal_score
            + 0.15 * protein_score
            + 0.15 * budget_score
            + 0.10 * (1.0 - sodium_penalty)
        )

        gnn_score = self._compute_gnn_similarity(food)

        if gnn_score is not None:
            score = 0.5 * content_score + 0.5 * gnn_score
        else:
            score = content_score

        return round(score, 4)

    def _compute_gnn_similarity(self, food: dict) -> float | None:
        if self._food_emb_norm is None or self._user_emb_norm is None:
            return None

        food_id = food.get("food_id")
        food_idx = self._food_id_to_idx.get(food_id)
        if food_idx is None:
            return None

        food_vec = self._food_emb_norm[food_idx]
        similarity = float(np.dot(self._user_emb_norm, food_vec))

        return max(0.0, (similarity + 1.0) / 2.0)
