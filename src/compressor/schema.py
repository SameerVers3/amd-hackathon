import logging
from typing import List
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

class SemanticScene(BaseModel):
    scene_id: int = Field(..., description="Sequential ID of the scene")
    time_start_sec: float = Field(..., description="Start time of the scene in seconds")
    time_end_sec: float = Field(..., description="End time of the scene in seconds")
    scene_summary: str = Field(..., description="Narrative summary of the visual action in the scene")
    key_objects: List[str] = Field(..., description="Most critical physical objects in the scene")
    key_actions: List[str] = Field(..., description="Most critical discrete actions")
    dialogue_summary: str = Field(..., description="Cohesive one-sentence summary of the spoken dialogue")

class SceneTimeline(BaseModel):
    scenes: List[SemanticScene] = Field(..., description="Array of chronologically ordered narrative scenes")

def get_fireworks_schema() -> dict:
    return {
        "type": "json_object",
        "schema": SceneTimeline.model_json_schema()
    }

def validate_payload(raw_json: dict) -> dict | None:
    try:
        validated = SceneTimeline.model_validate(raw_json)
        return validated.model_dump()
    except ValidationError as e:
        log.error("Pydantic validation failed: %s", e)
        return None
