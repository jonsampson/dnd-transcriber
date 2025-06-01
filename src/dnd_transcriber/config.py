import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

load_dotenv()


class WhisperXConfig(BaseModel):
    """Configuration for WhisperX transcription."""

    model: str = Field(default="large-v2", description="WhisperX model name")
    device: str = Field(default="cuda", description="Device to run on")
    compute_type: str = Field(default="float16", description="Compute precision")
    language: str | None = Field(default=None, description="Audio language")

    @validator("device")
    def validate_device(cls, v: str) -> str:
        if v not in ["cuda", "cpu", "auto"]:
            raise ValueError("Device must be 'cuda', 'cpu', or 'auto'")
        return v

    @validator("compute_type")
    def validate_compute_type(cls, v: str) -> str:
        if v not in ["float16", "int8", "int8_float16"]:
            raise ValueError(
                "Compute type must be 'float16', 'int8', or 'int8_float16'"
            )
        return v


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM."""

    model_name: str = Field(
        default="mistral-nemo:12b-instruct-2407-fp16", description="Ollama model name"
    )
    api_url: str = Field(default="http://localhost:11434", description="Ollama API URL")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Sampling temperature"
    )

    class Config:
        env_prefix = "OLLAMA_"


class DemucsConfig(BaseModel):
    """Configuration for Demucs audio separation."""

    model_name: str = Field(default="htdemucs", description="Demucs model name")
    device: str = Field(default="cuda", description="Device to run on")
    segment_length: int | None = Field(
        default=None, description="Segment length in seconds"
    )

    @validator("device")
    def validate_device(cls, v: str) -> str:
        if v not in ["cuda", "cpu", "auto"]:
            raise ValueError("Device must be 'cuda', 'cpu', or 'auto'")
        return v


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""

    whisperx: WhisperXConfig = Field(default_factory=WhisperXConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    demucs: DemucsConfig = Field(default_factory=DemucsConfig)
    output_format: str = Field(default="json", description="Output format")

    @validator("output_format")
    def validate_output_format(cls, v: str) -> str:
        if v not in ["json", "srt", "txt"]:
            raise ValueError("Output format must be 'json', 'srt', or 'txt'")
        return v

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load configuration from environment variables."""
        return cls(
            whisperx=WhisperXConfig(
                model=os.getenv("WHISPERX_MODEL", "large-v2"),
                device=os.getenv("WHISPERX_DEVICE", "cuda"),
                compute_type=os.getenv("WHISPERX_COMPUTE_TYPE", "float16"),
                language=os.getenv("WHISPERX_LANGUAGE"),
            ),
            ollama=OllamaConfig(
                model_name=os.getenv(
                    "OLLAMA_MODEL_NAME", "mistral-nemo:12b-instruct-2407-fp16"
                ),
                api_url=os.getenv("OLLAMA_API_URL", "http://localhost:11434"),
                temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            ),
            demucs=DemucsConfig(
                model_name=os.getenv("DEMUCS_MODEL", "htdemucs"),
                device=os.getenv("DEMUCS_DEVICE", "cuda"),
                segment_length=int(seg_len)
                if (seg_len := os.getenv("DEMUCS_SEGMENT_LENGTH")) is not None
                else None,
            ),
            output_format=os.getenv("OUTPUT_FORMAT", "json"),
        )
