import json
import sys

import requests

from .config import OllamaConfig
from .roster import CharacterRoster


class TranscriptionValidator:
    """Ollama-based transcription validator for D&D sessions."""

    def __init__(self, config: OllamaConfig, roster: CharacterRoster | None = None):
        """Initialize validator with Ollama configuration and character roster."""
        self.config = config
        self.roster = roster

    def validate_segment(self, text: str, context: str = "") -> str:
        """Validate and correct transcription segment using LLM.

        Args:
            text: Transcribed text to validate
            context: Surrounding context for validation

        Returns:
            Corrected text
        """
        if not text.strip():
            return text

        # Apply character roster corrections first if available
        if self.roster:
            text = self.roster.correct_names_in_text(text)

        # Build validation prompt
        prompt = self._build_prompt(text, context)

        # Call Ollama API with streaming
        try:
            print(f"🤖 Validating: {text[:50]}...")
            sys.stdout.flush()

            response = requests.post(
                f"{self.config.api_url}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "temperature": self.config.temperature,
                    "stream": True,
                },
                timeout=60,
                stream=True,
            )
            response.raise_for_status()

            corrected_text = ""
            print("   ", end="")

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        if chunk.get("response"):
                            token = chunk["response"]
                            corrected_text += token
                            print(token, end="", flush=True)
                    except json.JSONDecodeError:
                        continue

            print()  # New line after streaming
            corrected_text = corrected_text.strip()

            # Return corrected text or original if correction is empty
            return corrected_text if corrected_text else text

        except (requests.RequestException, KeyError) as e:
            print(f"   ⚠️  Validation failed: {e}")
            # Return original text if validation fails
            return text

    def _build_prompt(self, text: str, context: str) -> str:
        """Build validation prompt for D&D transcription."""
        character_info = ""
        if self.roster and self.roster.characters:
            char_names = ", ".join(self.roster.characters.keys())
            player_names = ", ".join(self.roster.player_names)
            character_info = f"Characters: {char_names}\nPlayers: {player_names}\n"

        prompt = f"""You are correcting transcription errors in a D&D session transcript.

{character_info}
Context: {context}

Original text: "{text}"

Correct only obvious transcription errors, fantasy terms, and character names. Keep the same meaning and style. Return only the corrected text with no explanations.

Corrected text:"""

        return prompt
