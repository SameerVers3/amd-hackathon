import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ExtractorConfig:
    
    fireworks_api_key: str = field(
        default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", "")
    )
    fireworks_api_base: str = "https://api.fireworks.ai/inference/v1"
    
    chat_endpoint: str = "/chat/completions"
    
    gemma_model: str = field(
        default_factory=lambda: os.environ.get("GEMMA_MODEL", "accounts/fireworks/models/minimax-m3") # using this just to complete the pipeline, will change eventually to gemma
    )
   
    # this is also paid -> someone please give free way to use gemma :(

    google_api_key: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY", "")
    )
    google_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    
    google_model: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_MODEL", "gemini-1.5-flash")
    )
    
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 300
    
    max_concurrent_requests: int = 20
    
    tpm_limit: int = field(
        default_factory=lambda: int(os.environ.get("TPM_LIMIT", 200_000))
    )
    
    downscale_threshold: int = 5
    
    downscale_resolution: Tuple[int, int] = (768, 768)
    
    system_prompt: str = (
        "You are a multimodal atomic fact extractor. "
        "Analyze the provided visual frames and the accompanying audio transcript. "
        "Extract exactly the requested fields into valid JSON with the following schema:\n"
        "{\n"
        '  "visual_objects": ["list of physical objects"],\n'
        '  "actions": ["list of discrete actions"],\n'
        '  "setting": "description of setting/background",\n'
        '  "camera_or_lighting": "details about camera angle or lighting",\n'
        '  "audio_visual_correlation": "how audio correlates with visuals"\n'
        "}"
    )
