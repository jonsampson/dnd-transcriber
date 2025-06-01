from pathlib import Path
from typing import Any

import whisperx

from .config import WhisperXConfig
from .utils.audio import get_audio_duration, validate_audio_format


class WhisperXTranscriber:
    """WhisperX transcription wrapper."""

    def __init__(self, config: WhisperXConfig):
        """Initialize transcriber with WhisperX configuration."""
        self.config = config
        self.model = None
        self.align_model = None
        self.diarize_model = None

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        """Transcribe audio file using WhisperX pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            Raw WhisperX output dictionary

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is not valid
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not validate_audio_format(audio_path):
            raise ValueError(f"Invalid audio format: {audio_path}")

        # Load models if not already loaded
        if self.model is None:
            self.model = whisperx.load_model(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
                language=self.config.language
            )

        # Load audio
        audio = whisperx.load_audio(str(audio_path))

        # Transcribe
        result = self.model.transcribe(audio, batch_size=16)

        # Add audio duration to result
        result["duration"] = get_audio_duration(audio_path)
        result["model"] = self.config.model

        # Align whisper output
        if self.align_model is None:
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=self.config.device
            )
            self.align_model = model_a

        result = whisperx.align(result["segments"], self.align_model, metadata, audio, self.config.device, return_char_alignments=False)

        return result
