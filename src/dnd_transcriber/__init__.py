"""D&D Transcriber - LLM-Enhanced WhisperX Pipeline for D&D Session Transcription."""

__version__ = "0.1.0"

from .pipeline import D_DTranscriptionPipeline
from .config import PipelineConfig
from .models import TranscriptionOutput, Segment, Speaker
from .roster import CharacterRoster

__all__ = [
    "D_DTranscriptionPipeline",
    "PipelineConfig", 
    "TranscriptionOutput",
    "Segment",
    "Speaker",
    "CharacterRoster",
]