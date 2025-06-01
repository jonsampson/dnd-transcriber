import json
import sys
from pathlib import Path

# Forward declaration for type annotation
from typing import TYPE_CHECKING, Any

import requests

from .config import OllamaConfig
from .roster import CharacterRoster

if TYPE_CHECKING:
    from .transcriber import WhisperXTranscriber


class TranscriptionValidator:
    """Ollama-based transcription validator for D&D sessions."""

    def __init__(self, config: OllamaConfig, roster: CharacterRoster | None = None):
        """Initialize validator with Ollama configuration and character roster."""
        self.config = config
        self.roster = roster
        self.transcriber: "WhisperXTranscriber | None" = None  # Will be set by pipeline

    def validate_segment(
        self,
        text: str,
        context: str = "",
        audio_path: Path | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        original_confidence: float | None = None,
    ) -> str:
        """Validate and correct transcription segment using LLM.

        Args:
            text: Transcribed text to validate
            context: Surrounding context for validation
            audio_path: Path to original audio file for retranscription
            start_time: Segment start time for retranscription
            end_time: Segment end time for retranscription
            original_confidence: Original segment confidence score

        Returns:
            Corrected text (possibly from retranscription)
        """
        if not text.strip():
            return text

        # Apply character roster corrections first if available
        if self.roster:
            text = self.roster.correct_names_in_text(text)

        # First, check if the text fits the context
        if context.strip():
            fits_context = self._check_context_fit(text, context)
            if fits_context:
                print("   ✅ Text fits context, skipping correction")
                return text
            else:
                # Context doesn't fit - try retranscription if possible
                if (
                    self.transcriber
                    and audio_path
                    and start_time is not None
                    and end_time is not None
                ):
                    retranscribed_data = self._try_retranscription(
                        text, audio_path, start_time, end_time, original_confidence
                    )
                    retranscribed_text: str = retranscribed_data["text"]

                    if retranscribed_text != text:
                        # Check if retranscribed text fits context better
                        retrans_fits = self._check_context_fit(
                            retranscribed_text, context
                        )
                        if retrans_fits:
                            print(
                                f"   ✅ Retranscription fits context: {retranscribed_text}"
                            )
                            return retranscribed_text
                        elif retranscribed_data["confidence_improved"]:
                            print(
                                f"   📊 Using retranscription (better confidence): {retranscribed_text}"
                            )
                            return retranscribed_text
                        else:
                            print(
                                "   🔄 Retranscription didn't improve context or confidence"
                            )
                            # Continue with LLM correction on original text

        # Build validation prompt
        prompt = self._build_prompt(text, context)

        # Call Ollama API with streaming
        try:
            print(f"🤖 Correcting: {text}")
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

IMPORTANT RULES:
- Only fix clear transcription errors (misheard words, garbled speech)
- Only replace character names if you're absolutely certain of a mismatch
- DO NOT replace pronouns (he, she, they, I, you, etc.) with names
- DO NOT add character names that aren't in the roster above
- DO NOT change the speaker's voice or perspective
- Keep the exact same meaning, tone, and grammatical structure
- If unsure, leave the text unchanged

Return only the corrected text with no explanations.

Corrected text:"""

        return prompt

    def _check_context_fit(self, text: str, context: str) -> bool:
        """Check if text fits naturally in the given context.

        Args:
            text: Text to validate
            context: Recent conversation context

        Returns:
            True if text fits context well, False if it needs correction
        """
        character_info = ""
        if self.roster and self.roster.characters:
            char_names = ", ".join(self.roster.characters.keys())
            player_names = ", ".join(self.roster.player_names)
            character_info = f"Characters: {char_names}\nPlayers: {player_names}\n"

        prompt = f"""You are analyzing a D&D session transcript for context consistency.

{character_info}
Recent conversation context:
{context}

Current text to evaluate: "{text}"

Does this text fit naturally and logically in the conversation context? Consider:
- Does it make sense given what was just said?
- Are any names/terms clearly misheard or incorrect?
- Does the grammar and flow work naturally?

IMPORTANT: If you answer "NO", the system will attempt to re-transcribe this audio segment with enhanced parameters before trying other corrections. Only say "NO" if you believe the transcription has genuine errors that could benefit from re-processing the original audio.

Answer ONLY with "YES" if the text fits well as-is, or "NO" if it likely contains transcription errors that would benefit from re-transcription.

Answer:"""

        try:
            print(f"🔍 Context check: {text}")
            response = requests.post(
                f"{self.config.api_url}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "temperature": 0.1,  # Low temperature for consistent yes/no answers
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            answer: str = result.get("response", "").strip().upper()
            print(f"   Context fit: {answer}")

            # Return True if answer starts with YES, False otherwise
            return bool(answer.startswith("YES"))

        except (requests.RequestException, KeyError) as e:
            print(f"   ⚠️  Context check failed: {e}")
            # If context check fails, proceed with correction anyway
            return False

    def _try_retranscription(
        self,
        original_text: str,
        audio_path: Path,
        start_time: float,
        end_time: float,
        original_confidence: float | None = None,
    ) -> dict[str, Any]:
        """Attempt retranscription of a segment with enhanced parameters.

        Args:
            original_text: Original transcribed text
            audio_path: Path to audio file
            start_time: Segment start time
            end_time: Segment end time
            original_confidence: Original confidence score

        Returns:
            Dict with 'text', 'confidence', and 'confidence_improved' keys
        """
        default_result = {
            "text": original_text,
            "confidence": original_confidence or 0.0,
            "confidence_improved": False,
        }

        try:
            if not self.transcriber:
                return default_result

            # Attempt retranscription
            result = self.transcriber.retranscribe_segment(
                audio_path, start_time, end_time
            )

            # Extract text and confidence from retranscription result
            if "segments" in result and result["segments"]:
                # Combine all segments into one text and calculate average confidence
                retranscribed_text = " ".join(
                    seg.get("text", "") for seg in result["segments"]
                )
                retranscribed_text = retranscribed_text.strip()

                # Calculate average confidence of retranscribed segments
                confidences = [
                    seg.get("confidence", 0.0)
                    for seg in result["segments"]
                    if "confidence" in seg
                ]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.0
                )

                if retranscribed_text:
                    confidence_improved = (
                        original_confidence is None
                        or avg_confidence
                        > original_confidence + 0.05  # 5% improvement threshold
                    )

                    conf_str = (
                        f"{original_confidence:.2f}"
                        if original_confidence is not None
                        else "N/A"
                    )
                    print(f"   🔄 Original: {original_text} (conf: {conf_str})")
                    print(
                        f"   🔄 Retranscribed: {retranscribed_text} (conf: {avg_confidence:.2f})"
                    )

                    return {
                        "text": retranscribed_text,
                        "confidence": avg_confidence,
                        "confidence_improved": confidence_improved,
                    }

        except Exception as e:
            print(f"   ⚠️  Retranscription failed: {e}")

        return default_result
