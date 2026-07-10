import os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class SynthesizerConfig:
    fireworks_api_key: str = field(
        default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", "")
    )
    fireworks_api_base: str = "https://api.fireworks.ai/inference/v1"
    chat_endpoint: str = "/chat/completions"
    
    model_id: str = field(
        default_factory=lambda: os.environ.get("GEMMA_31B_MODEL", "accounts/fireworks/models/minimax-m3")
    )
    
    max_retries: int = 3
    max_tokens: int = 4000
    
    # Dynamic Sampling Parameters per style
    style_sampling: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "formal": {"temperature": 0.2, "top_p": 0.9},
        "sarcastic": {"temperature": 0.6, "top_p": 0.95},
        "humorous_tech": {"temperature": 0.7, "top_p": 0.95},
        "humorous_non_tech": {"temperature": 0.7, "top_p": 0.95}
    })
