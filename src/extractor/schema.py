import logging
from typing import List
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)


class AtomicFacts(BaseModel):
    visual_objects: List[str] = Field(
        ..., description="List of physical objects visible in the scene"
    )
    actions: List[str] = Field(
        ..., description="List of discrete actions happening in the scene"
    )
    setting: str = Field(
        ..., description="Description of the physical setting or background"
    )
    camera_or_lighting: str = Field(
        ..., description="Details about the camera angle, movement, or lighting"
    )
    audio_visual_correlation: str = Field(
        ..., description="How the audio dialogue correlates with the visuals"
    )


def get_fireworks_schema() -> dict:
    return {
        "type": "json_object",
        "schema": AtomicFacts.model_json_schema()
    }


def _deduplicate_list_case_insensitive(items: List[str]) -> List[str]:
    seen_lower = set()
    result = []
    for item in items:
        lower_item = item.strip().lower()
        if lower_item and lower_item not in seen_lower:
            seen_lower.add(lower_item)
            result.append(item.strip())
    return result


def validate_and_deduplicate(raw_json: dict) -> dict | None:
    try:
        validated = AtomicFacts.model_validate(raw_json)
        
        # Post-Processing De-duplication
        validated.visual_objects = _deduplicate_list_case_insensitive(validated.visual_objects)
        validated.actions = _deduplicate_list_case_insensitive(validated.actions)
        
        return validated.model_dump()
        
    except ValidationError as e:
        log.error("Pydantic validation failed: %s", e)
        return None
