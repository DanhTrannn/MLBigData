from typing import Optional
from pydantic import BaseModel, Field


class MealNutrition(BaseModel):
    calories_kcal: float = 0
    protein_g: float = 0
    carb_g: float = 0
    fat_g: float = 0
    sugar_g: float = 0
    fiber_g: float = 0
    sodium_mg: float = 0


class MealItem(BaseModel):
    meal_type: str
    food_id: str
    food_name: str
    serving: float = 1.0
    suitability_score: float = 0.0
    mlp_score: Optional[float] = None
    nutrition: MealNutrition
    cost: float = 0.0
    explanations: list[str] = Field(default_factory=list)


class DaySummary(BaseModel):
    total_calories_kcal: float = 0
    total_protein_g: float = 0
    total_carb_g: float = 0
    total_fat_g: float = 0
    total_sugar_g: float = 0
    total_fiber_g: float = 0
    total_sodium_mg: float = 0
    estimated_cost: float = 0
    constraint_status: str = "pass"
    target_calories_kcal: float = 0
    calorie_deviation: float = 0


class DayPlanResponse(BaseModel):
    recommendation_id: str
    rule_version: str
    model_version: str
    targets: dict
    meals: list[MealItem]
    summary: DaySummary
    alternatives: list[list[MealItem]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SwapRequest(BaseModel):
    recommendation_id: str
    food_id_to_remove: str
    meal_type: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[dict] = None
