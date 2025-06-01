from pathlib import Path
from typing import Any

import whisperx

from .config import WhisperXConfig
from .utils.audio import get_audio_duration, split_audio_file, validate_audio_format


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
            loaded_model = whisperx.load_model(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
                language=self.config.language,
            )
            if loaded_model is None:
                raise ValueError("Failed to load WhisperX model")
            self.model = loaded_model

        # Load audio
        print(f"🎤 Loading audio: {audio_path.name}")
        audio = whisperx.load_audio(str(audio_path))

        # Transcribe
        print("🎯 Transcribing audio with WhisperX...")
        assert self.model is not None  # mypy hint: model is guaranteed to be loaded
        result = self.model.transcribe(audio, batch_size=16)  # type: ignore

        # Add audio duration to result
        result["duration"] = get_audio_duration(audio_path)
        result["model"] = self.config.model

        # Align whisper output
        if self.align_model is None:
            print("📐 Loading alignment model...")
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self.config.device
            )
            self.align_model = model_a

        print("🔧 Aligning transcription segments...")
        result = whisperx.align(
            result["segments"],
            self.align_model,
            metadata,
            audio,
            self.config.device,
            return_char_alignments=False,
        )  # type: ignore

        return result  # type: ignore[no-any-return]

    def retranscribe_segment(
        self, audio_path: Path, start_time: float, end_time: float
    ) -> dict[str, Any]:
        """Retranscribe a specific time segment with enhanced parameters.

        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            WhisperX result for the time segment

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If time range is invalid
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if start_time >= end_time or start_time < 0:
            raise ValueError(f"Invalid time range: {start_time} to {end_time}")

        print(f"🔄 Retranscribing segment {start_time:.1f}s-{end_time:.1f}s")

        # Create audio chunk for the specific time range
        chunks = [(start_time, end_time)]
        chunk_paths = split_audio_file(audio_path, chunks)

        if not chunk_paths:
            raise ValueError("Failed to create audio chunk")

        try:
            # Load models if not already loaded
            if self.model is None:
                loaded_model = whisperx.load_model(
                    self.config.model,
                    device=self.config.device,
                    compute_type=self.config.compute_type,
                    language=self.config.language,
                )
                if loaded_model is None:
                    raise ValueError("Failed to load WhisperX model")
                self.model = loaded_model

            # Load the audio chunk
            audio = whisperx.load_audio(str(chunk_paths[0]))

            # Transcribe with enhanced parameters for better accuracy
            print("   🎯 Retranscribing with enhanced parameters...")
            result = self.model.transcribe(  # type: ignore[attr-defined]
                audio,
                batch_size=8,  # Smaller batch for better accuracy
                chunk_length_s=15,  # Shorter chunks for focus
                print_progress=False,
            )  # type: ignore

            # Align the results
            if self.align_model is None:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], device=self.config.device
                )
                self.align_model = model_a

            result = whisperx.align(
                result["segments"],
                self.align_model,
                metadata,
                audio,
                self.config.device,
                return_char_alignments=False,
            )  # type: ignore

            # Adjust timestamps to match original audio file
            if "segments" in result:
                for segment in result["segments"]:
                    if "start" in segment:
                        segment["start"] += start_time
                    if "end" in segment:
                        segment["end"] += start_time

            result["retranscribed"] = True
            result["original_start"] = start_time
            result["original_end"] = end_time

            return result  # type: ignore[no-any-return]

        finally:
            # Clean up temporary audio chunk
            for chunk_path in chunk_paths:
                chunk_path.unlink(missing_ok=True)
