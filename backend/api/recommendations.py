"""
Recommendations API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from backend.schemas.user import UserProfileIn
from backend.schemas.recommendation import DayPlanResponse, SwapRequest, ErrorResponse
from backend.utils.exceptions import (
    RecommenderError, NoSafeCandidateError, NoFeasiblePlanError,
)

router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])


def get_orchestrator():
    import app as app_module
    if app_module.orchestrator is None:
        from backend.utils.exceptions import ModelNotReadyError
        raise ModelNotReadyError()
    return app_module.orchestrator


@router.post("/day", response_model=DayPlanResponse, responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def recommend_day(
    profile: UserProfileIn,
    orchestrator=Depends(get_orchestrator),
):
    try:
        return orchestrator.recommend_day(profile)
    except (NoSafeCandidateError, NoFeasiblePlanError) as e:
        raise HTTPException(status_code=e.status_code, detail={
            "error_code": e.error_code, "message": e.message, "details": e.details,
        })
    except RecommenderError as e:
        raise HTTPException(status_code=e.status_code, detail={
            "error_code": e.error_code, "message": e.message,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error_code": "INTERNAL_ERROR", "message": str(e),
        })


@router.post("/{recommendation_id}/swap", response_model=DayPlanResponse)
async def swap_meal(
    recommendation_id: str,
    request: SwapRequest,
    orchestrator=Depends(get_orchestrator),
):
    import app as app_module
    file_store = app_module.file_store
    original = file_store.get_recommendation(recommendation_id)
    if not original:
        raise HTTPException(status_code=404, detail={
            "error_code": "NOT_FOUND", "message": "Recommendation not found",
        })

    excluded = {request.food_id_to_remove}
    try:
        meals_data = original.get("meals", [])
        existing_profile = None
        profile = UserProfileIn(
            age=45, height_cm=168, weight_kg=70,
        )
        return orchestrator.recommend_day(profile, excluded_food_ids=excluded)
    except RecommenderError as e:
        raise HTTPException(status_code=e.status_code, detail={
            "error_code": e.error_code, "message": e.message,
        })


@router.get("/{recommendation_id}")
async def get_recommendation(recommendation_id: str):
    import app as app_module
    rec = app_module.file_store.get_recommendation(recommendation_id)
    if not rec:
        raise HTTPException(status_code=404, detail={
            "error_code": "NOT_FOUND", "message": "Recommendation not found",
        })
    return rec
