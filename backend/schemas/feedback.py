from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class EventType(str, Enum):
    like = "like"
    dislike = "dislike"
    eaten = "eaten"
    swap = "swap"
    rating = "rating"


class FeedbackIn(BaseModel):
    recommendation_id: str
    food_id: str
    event_type: EventType
    event_value: Optional[float] = Field(default=None, ge=1, le=5)
    meal_type: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class FeedbackRecord(BaseModel):
    user_id: str
    recommendation_id: str
    food_id: str
    event_type: EventType
    event_value: Optional[float] = None
    meal_type: Optional[str] = None
    timestamp: datetime
