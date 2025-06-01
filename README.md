# D&D Transcriber

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-dependency%20management-blue.svg)](https://python-poetry.org/)
[![Tests](https://github.com/jonsampson/dnd-transcriber/workflows/Tests/badge.svg)](https://github.com/jonsampson/dnd-transcriber/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A specialized transcription pipeline for D&D sessions that combines WhisperX transcription with LLM-based validation. The system separates audio from background music, transcribes multi-speaker sessions, and corrects character names and fantasy terminology using local Ollama models.

## Installation

1. Install Python 3.11 using pyenv:
```bash
pyenv install 3.11.0
pyenv virtualenv 3.11.0 dnd-transcriber
pyenv local dnd-transcriber
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up Ollama (for LLM validation):
```bash
# Install Ollama and pull the model
ollama pull mistral-nemo:12b-instruct-2407-fp16
```

4. Install pre-commit hooks (optional but recommended):
```bash
poetry run pre-commit install
```

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
# WhisperX settings
WHISPERX_MODEL=large-v2
WHISPERX_DEVICE=cuda

# Ollama settings
OLLAMA_MODEL_NAME=mistral-nemo:12b-instruct-2407-fp16
OLLAMA_API_URL=http://localhost:11434

# Demucs settings
DEMUCS_MODEL=htdemucs
DEMUCS_DEVICE=cuda

# Output format
OUTPUT_FORMAT=json
```

## Usage

Basic transcription:
```bash
python -m dnd_transcriber transcribe input.wav output.json
```

With character roster:
```bash
python -m dnd_transcriber transcribe input.wav output.srt --roster characters.json
```

Disable multi-pass processing:
```bash
python -m dnd_transcriber transcribe input.wav output.txt --no-multipass
```

**Supported audio formats:** WAV, MP3, FLAC, M4A, OGG, AAC

## Character Roster Format

Create a JSON file with character and player names:
```json
{
  "characters": {
    "Gandalf": "wizard",
    "Frodo": "hobbit"
  },
  "players": ["Alice", "Bob", "Charlie"]
}
```

## Output Formats

- `.json`: Structured output with timestamps and metadata
- `.srt`: Standard subtitle format with speaker labels
- `.txt`: Plain text with timestamps and speaker names

The output format is determined automatically by the file extension.
