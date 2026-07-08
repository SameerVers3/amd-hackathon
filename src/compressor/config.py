import os
from dataclasses import dataclass, field

@dataclass
class CompressorConfig:
    fireworks_api_key: str = field(
        default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", "")
    )
    fireworks_api_base: str = "https://api.fireworks.ai/inference/v1"
    chat_endpoint: str = "/chat/completions"
    
    e4b_model: str = field(
        default_factory=lambda: os.environ.get("E4B_MODEL", "accounts/fireworks/models/minimax-m3")
    )
    
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 2500
    max_retries: int = 3
    
    system_prompt: str = (
        "You are a highly efficient semantic compression engine. You ingest temporal atomic "
        "observations and output clustered, narrative scenes. "
        "Your goal is to identify distinct narrative clusters, resolve co-references, "
        "and compress redundant atomic facts into a dense timeline.\n\n"
        "RULES:\n"
        "1. Analyze the timeline sequentially.\n"
        "2. Merge adjacent chunks that share the same setting or continuous action into distinct 'scenes'.\n"
        "3. Resolve co-references (e.g., if 'man in red' appears in chunk 0, refer to him as 'the host' in the scene summary).\n"
        "4. Extract only the most critical actions and objects that define the scene's narrative purpose.\n"
        "5. Summarize the ASR text into a single cohesive sentence per scene.\n\n"
        "INSTRUCTIONS:\n"
        "First, write your internal analysis and chain-of-thought inside <think> </think> tags.\n"
        "Then, output the final compressed timeline as valid JSON matching the requested response format strictly."
    )
