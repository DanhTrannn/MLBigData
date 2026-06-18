"""
Cold-start user encoder.
Creates user vector from profile, preferred tags, and selected likes.
Uses GNN food embeddings and ProfileEncoder when available.
"""
import numpy as np
from backend.schemas.user import UserProfileIn
from backend.ml_runtime.model_bundle import ModelBundle


class ColdStartEncoder:
    def __init__(self, all_tags: list[str] | None = None, bundle: ModelBundle | None = None):
        self.all_tags = all_tags or []
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.all_tags)}
        self.bundle = bundle
        self._profile_encoder = None

        if bundle and bundle.profile_encoder is not None:
            self._profile_encoder = bundle.profile_encoder

    def encode(self, profile: UserProfileIn) -> np.ndarray:
        if self._profile_encoder is not None:
            return self._encode_with_profile_encoder(profile)
        return self._encode_heuristic(profile)

    def _encode_with_profile_encoder(self, profile: UserProfileIn) -> np.ndarray:
        import torch
        features = np.array([
            profile.age / 100.0,
            profile.height_cm / 200.0,
            profile.weight_kg / 150.0,
            profile.bmi / 40.0,
            {"sedentary": 0.0, "light": 0.33, "moderate": 0.66, "high": 1.0}.get(profile.activity_level.value, 0.5),
            {"lose": 0.0, "maintain": 0.5, "gain": 1.0}.get(profile.goal.value, 0.5),
            1.0 if any(d.value == "hypertension" for d in profile.diseases) else 0.0,
            1.0 if any(d.value == "diabetes" for d in profile.diseases) else 0.0,
            1.0 if any(d.value == "gout" for d in profile.diseases) else 0.0,
        ], dtype=np.float32)

        with torch.no_grad():
            tensor = torch.tensor(features).unsqueeze(0)
            embedding = self._profile_encoder(tensor).squeeze(0).numpy()

        return embedding

    def _encode_heuristic(self, profile: UserProfileIn) -> np.ndarray:
        parts = []

        parts.append(np.array([
            profile.age / 100.0,
            profile.height_cm / 200.0,
            profile.weight_kg / 150.0,
            profile.bmi / 40.0,
        ]))

        activity_map = {"sedentary": 0.0, "light": 0.33, "moderate": 0.66, "high": 1.0}
        goal_map = {"lose": 0.0, "maintain": 0.5, "gain": 1.0}
        parts.append(np.array([
            activity_map.get(profile.activity_level.value, 0.5),
            goal_map.get(profile.goal.value, 0.5),
        ]))

        disease_vec = np.zeros(3)
        disease_map = {"hypertension": 0, "diabetes": 1, "gout": 2}
        for d in profile.diseases:
            if d.value in disease_map:
                disease_vec[disease_map[d.value]] = 1.0
        parts.append(disease_vec)

        if self.tag_to_idx:
            tag_vec = np.zeros(len(self.tag_to_idx))
            for tag in profile.preferred_tags:
                tag_lower = tag.lower()
                if tag_lower in self.tag_to_idx:
                    tag_vec[self.tag_to_idx[tag_lower]] = 1.0
            parts.append(tag_vec)

        return np.concatenate(parts).astype(np.float32)

    def compute_tag_affinity(self, food: dict, profile: UserProfileIn) -> float:
        if not profile.preferred_tags:
            return 0.5
        food_tags = set(t.lower() for t in food.get("diet_tags", []))
        preferred = set(t.lower() for t in profile.preferred_tags)
        if not preferred:
            return 0.5
        overlap = len(food_tags & preferred)
        return min(1.0, 0.3 + 0.35 * overlap)

    def compute_like_affinity(self, food: dict, profile: UserProfileIn) -> float:
        if not profile.selected_likes:
            return 0.5

        if (self.bundle and self.bundle.food_embeddings is not None
                and self.bundle.food_id_to_index):
            food_id = food.get("food_id")
            food_idx = self.bundle.food_id_to_index.get(food_id)
            if food_idx is not None:
                food_vec = self.bundle.food_embeddings[food_idx]
                food_norm = food_vec / (np.linalg.norm(food_vec) + 1e-8)

                liked_vecs = []
                for liked_id in profile.selected_likes:
                    liked_idx = self.bundle.food_id_to_index.get(liked_id)
                    if liked_idx is not None:
                        lv = self.bundle.food_embeddings[liked_idx]
                        lv_norm = lv / (np.linalg.norm(lv) + 1e-8)
                        liked_vecs.append(lv_norm)

                if liked_vecs:
                    avg_liked = np.mean(liked_vecs, axis=0)
                    similarity = float(np.dot(food_norm, avg_liked))
                    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))

        food_ingredients = set(food.get("ingredients", []))
        best_score = 0.0
        return max(0.3, best_score + 0.5)
