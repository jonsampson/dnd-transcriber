import pytest
import tempfile
from pathlib import Path
from dnd_transcriber.config import PipelineConfig, WhisperXConfig, OllamaConfig, DemucsConfig
from dnd_transcriber.models import Segment


@pytest.fixture
def sample_config():
    """Sample pipeline configuration for testing."""
    return PipelineConfig(
        whisperx=WhisperXConfig(
            model="base",
            device="cpu",
            compute_type="float16",
            language="en"
        ),
        ollama=OllamaConfig(
            model_name="mistral-nemo:12b-instruct-2407-fp16",
            api_url="http://localhost:11434",
            temperature=0.1
        ),
        demucs=DemucsConfig(
            model_name="htdemucs",
            device="cpu",
            segment_length=None
        ),
        output_format="json"
    )


@pytest.fixture
def sample_whisperx_output():
    """Sample WhisperX output dictionary for testing."""
    return {
        "segments": [
            {
                "text": "Hello, welcome to our D&D session.",
                "speaker": "DM",
                "start": 0.0,
                "end": 3.5,
                "confidence": 0.95
            },
            {
                "text": "I cast magic missile at the goblin.",
                "speaker": "Player1",
                "start": 4.0,
                "end": 6.8,
                "confidence": 0.87
            }
        ],
        "language": "en",
        "model": "large-v2",
        "duration": 10.0
    }


@pytest.fixture
def sample_segments():
    """Sample list of Segment objects for testing."""
    return [
        Segment(
            text="Hello, welcome to our D&D session.",
            speaker="DM",
            start_time=0.0,
            end_time=3.5,
            confidence=0.95
        ),
        Segment(
            text="I cast magic missile at the goblin.",
            speaker="Player1",
            start_time=4.0,
            end_time=6.8,
            confidence=0.87
        )
    ]


@pytest.fixture
def temp_audio_file():
    """Temporary audio file path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)