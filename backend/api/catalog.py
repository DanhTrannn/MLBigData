"""
Catalog API endpoints.
Provides lists of supported diseases and allergens.
"""
import yaml
from pathlib import Path
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/catalog", tags=["catalog"])

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


@router.get("/diseases")
async def get_diseases():
    return {
        "diseases": [
            {"id": "hypertension", "name_vi": "Tăng huyết áp", "name_en": "Hypertension"},
            {"id": "diabetes", "name_vi": "Tiểu đường", "name_en": "Diabetes"},
            {"id": "gout", "name_vi": "Gout", "name_en": "Gout"},
        ],
        "version": "2026-01",
    }


@router.get("/allergens")
async def get_allergens():
    return {
        "allergens": [
            {"id": "shellfish", "name_vi": "Hải sản", "name_en": "Shellfish"},
            {"id": "peanut", "name_vi": "Đậu phộng", "name_en": "Peanut"},
            {"id": "milk", "name_vi": "Sữa", "name_en": "Milk"},
            {"id": "egg", "name_vi": "Trứng", "name_en": "Egg"},
            {"id": "soy", "name_vi": "Đậu nành", "name_en": "Soy"},
            {"id": "wheat", "name_vi": "Lúa mì", "name_en": "Wheat"},
            {"id": "fish", "name_vi": "Cá", "name_en": "Fish"},
        ],
        "version": "2026-01",
    }


@router.get("/meal-types")
async def get_meal_types():
    return {
        "meal_types": [
            {"id": "breakfast", "name_vi": "Bữa sáng", "name_en": "Breakfast"},
            {"id": "lunch", "name_vi": "Bữa trưa", "name_en": "Lunch"},
            {"id": "dinner", "name_vi": "Bữa tối", "name_en": "Dinner"},
            {"id": "snack", "name_vi": "Bữa phụ", "name_en": "Snack"},
        ],
    }
