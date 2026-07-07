import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class IngestorConfig:
    """Configuration for the Adaptive Ingestion pipeline."""

    # Motion-Density Thresholds (0–255 grayscale MAD scale) 
    epsilon_low: float = 5.0
    epsilon_high: float = 30.0

    # Sampling Rates (target FPS for each motion class) 
    fps_static: float = 0.33   # 1 frame per 3 seconds
    fps_normal: float = 2.0    # 2 frames per second
    fps_high: float = 5.0      # 5 frames per second

    # Frame Budget 
    # Scales linearly: N_max = frames_per_minute * (duration_min)
    frames_per_minute: int = 100

    #  Temporal Chunking 
    window_sec: float = 4.0    # chunk window duration
    step_sec: float = 4.0      # step size (non-overlapping by default)

    #  Visual Token Budget 
    max_visual_tokens_per_chunk: int = 15_000

    # [placeholder right now]
    token_tiers: Dict[int, int] = field(
        default_factory=lambda: {768: 1024, 0: 256}
    )

    downscale_resolution: Tuple[int, int] = (512, 512)

    frames_output_dir: str = field(
        default_factory=lambda: os.environ.get("FRAMES_OUTPUT_DIR", "/tmp/frames")
    )

    fireworks_api_key: str = field(
        default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", "")
    )
    fireworks_api_base: str = "https://api.fireworks.ai/inference/v1"
    whisper_model: str = field(
        default_factory=lambda: os.environ.get("WHISPER_MODEL", "whisper-v3")
    )
    whisper_language: str = "en"
    whisper_temperature: float = 0.0

    audio_sample_rate: int = 16_000  # in kilo hetz

    analysis_fps: float = 5.0

    jpeg_quality: int = 85
