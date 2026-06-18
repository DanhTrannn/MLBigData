from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ActivityLevel(str, Enum):
    sedentary = "sedentary"
    light = "light"
    moderate = "moderate"
    high = "high"


class Goal(str, Enum):
    maintain = "maintain"
    lose = "lose"
    gain = "gain"


class Disease(str, Enum):
    hypertension = "hypertension"
    diabetes = "diabetes"
    gout = "gout"


class Allergen(str, Enum):
    shellfish = "shellfish"
    peanut = "peanut"
    milk = "milk"
    egg = "egg"
    soy = "soy"
    wheat = "wheat"
    fish = "fish"


class UserProfileIn(BaseModel):
    age: int = Field(..., ge=1, le=120)
    height_cm: float = Field(..., gt=50, le=250)
    weight_kg: float = Field(..., gt=10, le=300)
    activity_level: ActivityLevel = ActivityLevel.moderate
    goal: Goal = Goal.maintain
    diseases: list[Disease] = Field(default_factory=list)
    allergies: list[Allergen] = Field(default_factory=list)
    disliked_ingredients: list[str] = Field(default_factory=list)
    preferred_tags: list[str] = Field(default_factory=list)
    budget_per_day: Optional[float] = Field(default=None, ge=0)
    selected_likes: list[str] = Field(default_factory=list)
    sex: Optional[str] = Field(default=None)
    lang: str = Field(default="vi", pattern="^(vi|en)$")

    @property
    def bmi(self) -> float:
        height_m = self.height_cm / 100
        return self.weight_kg / (height_m * height_m)


class NutrientTargets(BaseModel):
    calories_kcal: float
    protein_g: float = 0
    carb_g: float = 0
    fat_g: float = 0
    sugar_g: float = 0
    fiber_g: float = 0
    sodium_mg: float = 0


class UserProfileInternal(BaseModel):
    user_id: str
    profile: UserProfileIn
    bmi: float
    targets: NutrientTargets
