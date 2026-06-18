"""
Recommendation Orchestrator.
Coordinates the full recommendation pipeline:
target calculation → rule filtering → retrieval → scoring → optimization → explanation.
"""
import uuid
from datetime import datetime
from pathlib import Path

from backend.schemas.user import UserProfileIn, NutrientTargets
from backend.schemas.recommendation import (
    DayPlanResponse, MealItem, MealNutrition, DaySummary,
)
from backend.services.target_calculator import TargetCalculator
from backend.services.rule_engine import DiseaseRuleEngine
from backend.services.cold_start import ColdStartEncoder
from backend.services.retriever import ContentRetriever
from backend.services.scorer import SuitabilityScorer
from backend.services.optimizer import MealOptimizer
from backend.services.explainer import Explainer
from backend.ml_runtime.model_bundle import ModelBundle
from backend.utils.logging import logger

PROCESSED_FOODS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "foods.json"


class RecommendationOrchestrator:
    def __init__(self, bundle: ModelBundle | None = None):
        self.target_calculator = TargetCalculator()
        self.rule_engine = DiseaseRuleEngine()
        self.foods = self._load_foods()
        all_tags = self._extract_all_tags(self.foods)

        self.bundle = bundle or ModelBundle()

        self.user_encoder = ColdStartEncoder(all_tags=all_tags, bundle=self.bundle)
        self.retriever = ContentRetriever(self.user_encoder, bundle=self.bundle)
        self.scorer = SuitabilityScorer(mlp_model=self.bundle.mlp_model)
        self.optimizer = MealOptimizer()
        self.explainer = Explainer(bundle=self.bundle)
        self._file_store = None

        logger.info(f"Orchestrator initialized with bundle version={self.bundle.version}")

    def set_file_store(self, file_store):
        self._file_store = file_store

    def recommend_day(
        self,
        profile: UserProfileIn,
        excluded_food_ids: set[str] | None = None,
    ) -> DayPlanResponse:
        logger.info(f"Generating recommendation for age={profile.age}, diseases={[d.value for d in profile.diseases]}")

        targets = self.target_calculator.calculate(profile)
        meal_targets = self.target_calculator.get_per_meal_targets(targets)

        diseases = [d.value for d in profile.diseases]
        allergies = [a.value for a in profile.allergies]
        dislikes = [i for i in profile.disliked_ingredients]

        safe_foods, rule_trace = self.rule_engine.filter_all(
            self.foods, diseases, allergies, dislikes, profile.lang,
        )
        logger.info(f"Rule engine: {len(safe_foods)}/{len(self.foods)} foods passed")

        if not safe_foods:
            from backend.utils.exceptions import NoSafeCandidateError
            raise NoSafeCandidateError(
                "Không có món nào an toàn với các ràng buộc đã cho. "
                "Hãy kiểm tra lại danh sách dị ứng và bệnh lý."
            )

        user_vector = self.user_encoder.encode(profile)
        self.retriever.set_user_embedding(user_vector)

        candidates = self.retriever.retrieve(
            profile, safe_foods, targets, meal_targets,
            top_k=100, excluded_food_ids=excluded_food_ids,
        )

        scored = self.scorer.score(profile, targets, candidates)

        plan = self.optimizer.optimize(profile, targets, scored)

        explained_plan = self.explainer.explain(
            profile, plan, rule_trace, targets, profile.lang,
        )

        summary_data = self.explainer.explain_plan_summary(
            plan, targets, profile.lang,
        )

        meals = []
        for item in explained_plan:
            meals.append(MealItem(
                meal_type=item.get("meal_type", "lunch"),
                food_id=item["food_id"],
                food_name=item["name"],
                serving=1.0,
                suitability_score=item.get("suitability_score", 0.5),
                mlp_score=item.get("mlp_score"),
                nutrition=MealNutrition(
                    calories_kcal=item.get("calories_kcal", 0),
                    protein_g=item.get("protein_g", 0),
                    carb_g=item.get("carb_g", 0),
                    fat_g=item.get("fat_g", 0),
                    sugar_g=item.get("sugar_g", 0),
                    fiber_g=item.get("fiber_g", 0),
                    sodium_mg=item.get("sodium_mg", 0),
                ),
                cost=item.get("cost_estimate", 0),
                explanations=item.get("explanations", []),
            ))

        response = DayPlanResponse(
            recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            rule_version=self.rule_engine.version,
            model_version=self.bundle.version,
            targets={
                "calories_kcal": targets.calories_kcal,
                "protein_g": targets.protein_g,
                "carb_g": targets.carb_g,
                "fat_g": targets.fat_g,
                "sugar_g": targets.sugar_g,
                "sodium_mg": targets.sodium_mg,
            },
            meals=meals,
            summary=DaySummary(**summary_data),
        )

        if self._file_store:
            self._file_store.append_recommendation(response)

        return response

    def _load_foods(self) -> list[dict]:
        import json
        if PROCESSED_FOODS_PATH.exists():
            with open(PROCESSED_FOODS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.warning("Processed foods not found, running preprocessing...")
        from ml.pipelines.preprocess_foods import preprocess
        records, _ = preprocess()
        return records

    def _extract_all_tags(self, foods: list[dict]) -> list[str]:
        tags = set()
        for food in foods:
            for tag in food.get("diet_tags", []):
                tags.add(tag.lower())
        return sorted(tags)
