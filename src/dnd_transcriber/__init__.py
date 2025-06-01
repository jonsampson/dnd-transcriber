"""D&D Transcriber - LLM-Enhanced WhisperX Pipeline for D&D Session Transcription."""

__version__ = "0.1.0"

from .config import PipelineConfig
from .models import Segment, Speaker, TranscriptionOutput
from .pipeline import D_DTranscriptionPipeline
from .roster import CharacterRoster

__all__ = [
    "D_DTranscriptionPipeline",
    "PipelineConfig",
    "TranscriptionOutput",
    "Segment",
    "Speaker",
    "CharacterRoster",
]
