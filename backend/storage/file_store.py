"""
Local file storage.
Handles reading processed foods and appending recommendations/feedback to JSONL.
"""
import json
from pathlib import Path
from datetime import datetime

from backend.schemas.recommendation import DayPlanResponse
from backend.schemas.feedback import FeedbackRecord
from backend.storage.atomic_io import append_jsonl, read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
RECOMMENDATIONS_PATH = RUNTIME_DIR / "recommendations.jsonl"
FEEDBACK_PATH = RUNTIME_DIR / "feedback.jsonl"


class FileStore:
    def __init__(
        self,
        recommendations_path: Path = RECOMMENDATIONS_PATH,
        feedback_path: Path = FEEDBACK_PATH,
    ):
        self.recommendations_path = recommendations_path
        self.feedback_path = feedback_path
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    def append_recommendation(self, response: DayPlanResponse) -> None:
        record = response.model_dump()
        record["timestamp"] = datetime.now().isoformat()
        append_jsonl(self.recommendations_path, record)

    def append_feedback(self, feedback: FeedbackRecord) -> None:
        record = feedback.model_dump()
        record["timestamp"] = feedback.timestamp.isoformat()
        append_jsonl(self.feedback_path, record)

    def get_recommendation(self, recommendation_id: str) -> dict | None:
        records = read_jsonl(self.recommendations_path)
        for record in reversed(records):
            if record.get("recommendation_id") == recommendation_id:
                return record
        return None

    def get_recent_recommendations(self, limit: int = 10) -> list[dict]:
        records = read_jsonl(self.recommendations_path)
        return records[-limit:]
