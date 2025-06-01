# Development Prompts Log

This document contains all the prompts used to develop the D&D Transcriber project from start to finish.

## Initial Setup

### Project Structure Creation
```
I've added a REAMDE to give you slight more scope. Create a Python project structure for a D&D audio transcription pipeline using pyenv and poetry.

Requirements:
- Use pyenv to create a Python 3.11 virtual environment named "dnd-transcriber"
- Initialize a poetry project with appropriate metadata
- Project name: dnd-transcriber
- Create basic directory structure: src/, tests/, docs/, data/, src/dnd_transcriber/utils/
- Add .gitignore for Python projects
- Add README.md with project title only

Implement only the requested functionality. Do not add any features beyond what's specified. Keep the implementation minimal and focused.
```

### Dependencies Setup
```
Update the poetry pyproject.toml file to include the core dependencies for the transcription pipeline:

Add these dependencies:
- whisperx (from git: https://github.com/m-bain/whisperX)
- torch (with CUDA support)
- ollama
- demucs
- pydantic for configuration
- click for CLI
- python-dotenv for environment variables

Add dev dependencies:
- pytest
- black
- ruff
- mypy

Do not install the dependencies, only update pyproject.toml. Do not add any other dependencies or configuration beyond what's listed.
```

## Utility Functions

### Time Utilities
```
Create time formatting utilities in src/dnd_transcriber/utils/time.py:

Implement:
- seconds_to_srt_time(seconds: float) -> str
- seconds_to_readable(seconds: float) -> str  
- parse_timestamp(timestamp: str) -> float

Simple conversions only. Keep under 40 lines.
```

### Audio Utilities
```
Create audio utility functions in src/dnd_transcriber/utils/audio.py:

Implement only:
- get_audio_duration(path: Path) -> float
- validate_audio_format(path: Path) -> bool
- split_audio_file(path: Path, chunks: list) -> list[Path]

Use standard libraries only. No complex audio processing. Keep under 60 lines.
```

## Data Models

### Core Models
```
Create output data models in src/dnd_transcriber/models.py:

Define Pydantic models for:
- Speaker: id, name (optional)
- Segment: text, speaker, start_time, end_time, confidence
- TranscriptionOutput: segments, metadata, audio_duration

Basic models only, no methods beyond Pydantic defaults. Keep under 50 lines.
```

### Configuration Models
```
Create Pydantic configuration models for the pipeline in src/dnd_transcriber/config.py:

Define these models only:
- WhisperXConfig: model, device, compute_type, language
- OllamaConfig: model_name, api_url, temperature
- DemucsConfig: model_name, device, segment_length
- PipelineConfig: whisperx, ollama, demucs, output_format

Include basic validation and environment variable loading with python-dotenv. Keep it under 80 lines.
```

## Core Components

### Character Roster Management
```
Create character roster management in src/dnd_transcriber/roster.py:

Implement:
- CharacterRoster class
- load_from_file(path: Path) method for JSON roster
- find_closest_match(name: str) -> str using simple distance
- correct_names_in_text(text: str) -> str

Basic string matching only. No ML, no complex algorithms. Keep under 70 lines.
```

### Context Window Management
```
Add context windowing to src/dnd_transcriber/context.py:

Implement ContextWindowManager with:
- __init__(window_size: int, overlap: int)
- create_windows(segments: list) -> list of tuples
- Each tuple: (current_segment, context_before, context_after)

No complex logic, just sliding window creation. Keep under 60 lines.
```

### Format Conversion
```
Create a format converter in src/dnd_transcriber/formatter.py:

Implement:
- convert_whisperx_output(whisperx_data: dict) -> TranscriptionOutput
- export_to_text(output: TranscriptionOutput) -> str
- export_to_srt(output: TranscriptionOutput) -> str

Use the time utilities from utils/time.py. Simple conversions only. Keep under 80 lines.
```

### Confidence Analysis
```
Add confidence analysis in src/dnd_transcriber/confidence.py:

Implement:
- identify_low_confidence_segments(segments: list, threshold: float) -> list
- group_adjacent_segments(segments: list) -> list of groups
- Returns segment indices only

Simple threshold checking. No statistical analysis. Keep under 50 lines.
```

## Pipeline Components

### Audio Preprocessor
```
Create a minimal Demucs audio preprocessor in src/dnd_transcriber/preprocessor.py:

Implement:
- AudioPreprocessor class with __init__(config: DemucsConfig)
- separate_audio(input_path: Path) -> Path method that returns path to vocals
- Use audio utilities for validation
- Basic error handling for file not found only

Do not implement: progress bars, logging, multiple file support. Keep under 50 lines.
```

### Transcription Wrapper
```
Create a WhisperX transcription wrapper in src/dnd_transcriber/transcriber.py:

Implement:
- WhisperXTranscriber class with __init__(config: WhisperXConfig)
- transcribe(audio_path: Path) -> dict method
- Use audio utilities for duration and validation
- Returns raw WhisperX output format

No post-processing, no formatting, no extra features. Keep under 60 lines.
```

### LLM Validator
```
Create an Ollama-based validator in src/dnd_transcriber/validator.py:

Implement:
- TranscriptionValidator class with __init__(config: OllamaConfig, roster: Optional[CharacterRoster])
- validate_segment(text: str, context: str) -> str method
- Use character roster if provided
- Basic prompt template for D&D context validation

Do not add: streaming, retry logic, multiple models. Keep under 70 lines.
```

## Main Pipeline

### Core Pipeline
```
Create the main pipeline class in src/dnd_transcriber/pipeline.py:

Implement D&DTranscriptionPipeline with:
- __init__ accepting PipelineConfig and optional CharacterRoster
- process_audio(input_path: Path) -> TranscriptionOutput method
- Sequential execution: preprocess -> transcribe -> validate -> format
- Use all utility functions and components
- Return structured TranscriptionOutput

Do not add: parallel processing, checkpointing, progress tracking. Keep under 100 lines.
```

### Multi-pass Enhancement
```
Update src/dnd_transcriber/pipeline.py to add multi-pass support:

Add to D&DTranscriptionPipeline:
- reprocess_segments(indices: list, audio_path: Path) method
- merge_transcriptions(original: dict, reprocessed: dict) method  
- integrate confidence analyzer
- Update process_audio to optionally use multi-pass

Modify existing code minimally. Add only what's specified.
```

## CLI and Entry Points

### CLI Interface
```
Create a CLI interface in src/dnd_transcriber/cli.py using click:

Commands:
- transcribe: takes input audio path, output path, optional character roster
- Loads config from .env or defaults
- Calls pipeline.process_audio()
- Saves output using formatter (JSON/text/SRT based on extension)

Only implement the basic flow. No fancy output, no interactive mode. Keep under 80 lines.
```

### Main Entry Point
```
Create the main package entry point in src/dnd_transcriber/__main__.py:

Implement:
- Import and call CLI when run as module
- Handle KeyboardInterrupt gracefully
- Basic error message for missing dependencies

Keep it minimal, under 20 lines. No fancy error handling or logging setup.
```

## Configuration and Documentation

### Configuration Files
```
Create configuration files:

.env.example should include:
- WHISPERX_MODEL=large-v2
- OLLAMA_MODEL=mistral
- DEMUCS_MODEL=htdemucs
- OUTPUT_FORMAT=json
- OLLAMA_API_URL=http://localhost:11434

Create src/dnd_transcriber/__init__.py with version and basic imports.
```

### Testing Setup
```
Create test fixtures in tests/conftest.py:

Implement pytest fixtures for:
- sample_config: returns PipelineConfig
- sample_whisperx_output: returns mock WhisperX output dict
- sample_segments: returns list of Segment objects
- temp_audio_file: creates a temporary audio file path

Basic fixtures only, minimal mock data. Keep under 60 lines.
```

### Unit Tests - Utilities
```
Create unit tests in tests/test_utils.py:

Test only:
- Time conversion functions with known values
- Audio validation with mock paths
- Basic model serialization

5-6 simple tests total. Keep under 80 lines.
```

### Unit Tests - Components
```
Create unit tests in tests/test_components.py:

Test only:
- CharacterRoster.find_closest_match with known names
- ContextWindowManager.create_windows with simple data
- Confidence analyzer with mock segments

3-4 tests per component. No integration tests. Keep under 100 lines.
```

### Documentation Updates
```
Update README.md with:

Include only:
- Brief description (2-3 sentences)
- Installation steps using pyenv and poetry
- Basic usage example
- Configuration via .env
- Output format options

Do not add: badges, detailed API docs, contribution guidelines. Keep under 100 lines.
```

## Testing and Validation

### Test Execution
```
Run the tests and ensure they work as intented.
```

### Audio Format Support
```
does this program take mp3 files as input in addition to wav?
```

### Documentation Enhancement
```
update readme with that helpful detail
```

### Development Log Creation
```
Create a new markdown file that contains all the prompts used to get to this point in development
```

## Summary

This project was built incrementally using focused, specific prompts that:

1. **Established clear boundaries** - Each prompt specified exactly what to implement and what NOT to implement
2. **Maintained simplicity** - Consistently requested minimal implementations without over-engineering
3. **Built systematically** - Started with utilities, moved to models, then components, then integration
4. **Enforced constraints** - Line limits and feature restrictions kept the codebase focused
5. **Validated incrementally** - Tests were created and run to ensure functionality

The result is a complete, working D&D transcription pipeline with ~2000 lines of focused, tested code that does exactly what was requested without unnecessary complexity.