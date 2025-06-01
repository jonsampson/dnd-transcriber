from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Speaker(BaseModel):
    """Represents a speaker in the transcription."""
    id: str
    name: Optional[str] = None


class Segment(BaseModel):
    """Represents a transcribed segment of audio."""
    text: str
    speaker: str
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class TranscriptionOutput(BaseModel):
    """Complete transcription output with metadata."""
    segments: List[Segment]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    audio_duration: float = Field(..., description="Total audio duration in seconds")