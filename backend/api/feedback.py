"""
Feedback API endpoints.
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException

from backend.schemas.feedback import FeedbackIn, FeedbackRecord

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


@router.post("", status_code=201)
async def submit_feedback(feedback: FeedbackIn):
    import app as app_module
    file_store = app_module.file_store

    record = FeedbackRecord(
        user_id=f"anon_{uuid.uuid4().hex[:8]}",
        recommendation_id=feedback.recommendation_id,
        food_id=feedback.food_id,
        event_type=feedback.event_type,
        event_value=feedback.event_value,
        meal_type=feedback.meal_type,
        timestamp=feedback.timestamp or datetime.now(),
    )

    file_store.append_feedback(record)
    return {"status": "ok", "feedback_id": record.user_id}
