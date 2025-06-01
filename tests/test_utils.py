from pathlib import Path
from unittest.mock import MagicMock, patch

from dnd_transcriber.models import Segment, TranscriptionOutput
from dnd_transcriber.utils.audio import get_audio_duration, validate_audio_format
from dnd_transcriber.utils.time import (
    parse_timestamp,
    seconds_to_readable,
    seconds_to_srt_time,
)


class TestTimeUtils:
    """Test time utility functions."""

    def test_seconds_to_srt_time(self):
        """Test SRT timestamp conversion."""
        assert seconds_to_srt_time(3661.5) == "01:01:01,500"
        assert seconds_to_srt_time(0.123) == "00:00:00,123"
        assert seconds_to_srt_time(90.0) == "00:01:30,000"

    def test_seconds_to_readable(self):
        """Test readable time format conversion."""
        assert seconds_to_readable(3661) == "1:01:01"
        assert seconds_to_readable(90) == "1:30"
        assert seconds_to_readable(45) == "0:45"

    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        assert parse_timestamp("1:30:45") == 5445.0
        assert parse_timestamp("5:30") == 330.0
        assert parse_timestamp("45") == 45.0


class TestAudioUtils:
    """Test audio utility functions."""

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    def test_validate_audio_format_valid(self, mock_exists, mock_run):
        """Test audio format validation with valid file."""
        mock_exists.return_value = True
        mock_run.return_value = MagicMock()
        test_path = Path("test.wav")

        assert validate_audio_format(test_path) is True

    @patch('subprocess.run')
    def test_get_audio_duration(self, mock_run):
        """Test audio duration retrieval."""
        mock_run.return_value = MagicMock(stdout="123.45\n")
        duration = get_audio_duration(Path("test.wav"))
        assert duration == 123.45


class TestModels:
    """Test Pydantic model functionality."""

    def test_segment_model(self):
        """Test Segment model creation and serialization."""
        segment = Segment(
            text="Test text",
            speaker="TestSpeaker",
            start_time=0.0,
            end_time=5.0,
            confidence=0.95
        )
        assert segment.text == "Test text"
        assert segment.end_time - segment.start_time == 5.0

        # Test JSON serialization
        data = segment.model_dump()
        assert data["text"] == "Test text"
        assert data["confidence"] == 0.95

    def test_transcription_output_model(self, sample_segments):
        """Test TranscriptionOutput model with segments."""
        output = TranscriptionOutput(
            segments=sample_segments,
            audio_duration=10.0,
            metadata={"test": "value"}
        )
        assert len(output.segments) == 2
        assert output.audio_duration == 10.0
        assert output.metadata["test"] == "value"
