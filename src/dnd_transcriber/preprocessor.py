import subprocess
from pathlib import Path

from .config import DemucsConfig
from .utils.audio import validate_audio_format


class AudioPreprocessor:
    """Minimal Demucs audio preprocessor for voice separation."""

    def __init__(self, config: DemucsConfig):
        """Initialize preprocessor with Demucs configuration."""
        self.config = config

    def separate_audio(self, input_path: Path) -> Path:
        """Separate audio and return path to vocals track.

        Args:
            input_path: Path to input audio file

        Returns:
            Path to separated vocals file

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input file is not valid audio
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        if not validate_audio_format(input_path):
            raise ValueError(f"Invalid audio format: {input_path}")

        output_dir = input_path.parent / "separated"
        vocals_path = output_dir / self.config.model_name / input_path.stem / "vocals.wav"

        # Build Demucs command
        cmd = ["python", "-m", "demucs", "--device", self.config.device]

        if self.config.segment_length:
            cmd.extend(["--segment", str(self.config.segment_length)])

        cmd.extend(["-n", self.config.model_name, "-o", str(output_dir), str(input_path)])

        # Run Demucs separation
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Demucs separation failed: {e}") from e

        if not vocals_path.exists():
            raise RuntimeError(f"Expected vocals file not found: {vocals_path}")

        return vocals_path
