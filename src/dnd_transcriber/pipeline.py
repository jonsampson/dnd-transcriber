from pathlib import Path
from typing import Any

from .confidence import identify_low_confidence_segments
from .config import PipelineConfig
from .context import ContextWindowManager
from .formatter import convert_whisperx_output
from .models import TranscriptionOutput
from .preprocessor import AudioPreprocessor
from .roster import CharacterRoster
from .transcriber import WhisperXTranscriber
from .validator import TranscriptionValidator


class D_DTranscriptionPipeline:
    """Main D&D transcription pipeline combining preprocessing, transcription, and validation."""

    def __init__(self, config: PipelineConfig, roster: CharacterRoster | None = None):
        """Initialize pipeline with configuration and optional character roster."""
        self.config = config
        self.roster = roster

        # Initialize components
        self.preprocessor = AudioPreprocessor(config.demucs)
        self.transcriber = WhisperXTranscriber(config.whisperx)
        self.validator = TranscriptionValidator(config.ollama, roster)
        self.validator.transcriber = self.transcriber  # Pass transcriber reference
        self.context_manager = ContextWindowManager(window_size=5, overlap=1)

    def process_audio(
        self,
        input_path: Path,
        use_multipass: bool = True,
        skip_preprocessing: bool = False,
    ) -> TranscriptionOutput:
        """Process audio file through complete transcription pipeline.

        Args:
            input_path: Path to input audio file
            use_multipass: Enable retranscription for segments that don't fit context (recommended)
            skip_preprocessing: Whether to skip Demucs audio separation

        Returns:
            Structured transcription output
        """
        # Step 1: Preprocess audio with Demucs (if not skipped)
        if skip_preprocessing:
            print("⏭️  Skipping audio preprocessing")
            vocals_path = input_path
        else:
            print("🎵 Separating vocals from background music...")
            vocals_path = self.preprocessor.separate_audio(input_path)

        # Step 2: Transcribe with WhisperX
        whisperx_result = self.transcriber.transcribe(vocals_path)

        # Step 3: Convert to structured format
        transcription_output = convert_whisperx_output(whisperx_result)

        # Step 4: Multi-pass processing is now handled per-segment in validation
        if use_multipass:
            print("🔍 Multi-pass processing enabled (handled per-segment in validation)")
        else:
            print("⏭️  Multi-pass processing disabled")

        # Step 5: Validate segments with low confidence
        print("🔍 Identifying segments for LLM validation...")
        validated_segments = self._validate_segments(
            transcription_output.segments, vocals_path, use_multipass
        )

        # Step 6: Update transcription with validated segments
        transcription_output.segments = validated_segments

        # Step 7: Remove any duplicate or overlapping segments
        transcription_output.segments = self._deduplicate_segments(
            transcription_output.segments
        )

        return transcription_output

    def _validate_segments(
        self, segments: list[Any], audio_path: Path, use_retranscription: bool = True
    ) -> list[Any]:
        """Validate transcription segments using LLM."""
        # Identify low confidence segments
        low_confidence_indices = identify_low_confidence_segments(
            segments, threshold=0.7
        )

        # Create context windows for validation
        context_windows = self.context_manager.create_windows(segments)

        validated_segments = []
        validation_count = 0

        for i, (segment, context_before, context_after) in enumerate(context_windows):
            # Build context string with comprehensive context
            context_text = ""
            if context_before:
                # Include more context (up to 10 previous segments) for better analysis
                context_segments = [seg.text for seg in context_before[-10:]]
                context_text = " ".join(context_segments)
            if context_after:
                # Include fewer future segments to avoid spoilers
                future_segments = [seg.text for seg in context_after[:2]]
                if future_segments:
                    context_text += " " + " ".join(future_segments)

            # Validate if low confidence or contains potential character names
            should_validate = (
                i in low_confidence_indices
                or self._might_contain_character_names(segment.text)
            )

            if should_validate:
                validation_count += 1
                print(f"📝 Segment {validation_count}: {i+1}/{len(segments)}")
                corrected_text = self.validator.validate_segment(
                    segment.text,
                    context_text.strip(),
                    audio_path=audio_path if use_retranscription else None,
                    start_time=getattr(segment, "start_time", None)
                    if use_retranscription
                    else None,
                    end_time=getattr(segment, "end_time", None)
                    if use_retranscription
                    else None,
                    original_confidence=getattr(segment, "confidence", None),
                )
                segment.text = corrected_text

            validated_segments.append(segment)

        if validation_count == 0:
            print("✅ No segments needed LLM validation")
        else:
            print(f"✅ Validated {validation_count} segments with LLM")

        return validated_segments

    def _might_contain_character_names(self, text: str) -> bool:
        """Check if text might contain character names worth validating."""
        if not self.roster:
            return False

        # Simple check for capitalized words that might be character names
        words = text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                return True

        return False

    def _deduplicate_segments(self, segments: list[Any]) -> list[Any]:
        """Remove duplicate or overlapping segments.

        Args:
            segments: List of transcription segments

        Returns:
            Deduplicated segments sorted by start time
        """
        if not segments:
            return segments

        print("🧹 Checking for duplicate segments...")

        # Sort segments by start time
        sorted_segments = sorted(
            segments, key=lambda seg: getattr(seg, "start_time", 0)
        )

        deduplicated = []
        prev_segment = None
        duplicates_removed = 0

        for segment in sorted_segments:
            if prev_segment is None:
                deduplicated.append(segment)
                prev_segment = segment
                continue

            # Check for duplicates by comparing text content and timing
            current_text = getattr(segment, "text", "").strip()
            prev_text = getattr(prev_segment, "text", "").strip()
            current_start = getattr(segment, "start_time", 0)
            prev_start = getattr(prev_segment, "start_time", 0)
            current_end = getattr(segment, "end_time", 0)
            prev_end = getattr(prev_segment, "end_time", 0)

            # Check for exact text duplicates
            if current_text == prev_text:
                print(f"   🗑️  Removing exact duplicate: {current_text[:50]}")
                duplicates_removed += 1
                continue

            # Check for overlapping time ranges with similar content
            time_overlap = (
                current_start < prev_end
                and current_end > prev_start
                and abs(current_start - prev_start) < 5.0  # Within 5 seconds
            )

            if time_overlap and len(current_text) > 0 and len(prev_text) > 0:
                # Check if one text contains the other (partial overlap)
                if current_text in prev_text or prev_text in current_text:
                    # Keep the longer, more complete text
                    if len(current_text) > len(prev_text):
                        print("   🔄 Replacing shorter segment with longer version")
                        deduplicated[-1] = segment  # Replace previous with current
                        prev_segment = segment
                    else:
                        print("   🗑️  Removing shorter overlapping segment")
                        duplicates_removed += 1
                    continue

            # Check for segments that are obvious repetitions
            words_current = current_text.split()
            words_prev = prev_text.split()
            if len(words_current) > 3 and len(words_prev) > 3:
                # Check if first few words are the same (likely repetition)
                if words_current[:3] == words_prev[:3]:
                    print(f"   🗑️  Removing likely repetition: {current_text[:50]}")
                    duplicates_removed += 1
                    continue

            # Keep this segment
            deduplicated.append(segment)
            prev_segment = segment

        if duplicates_removed > 0:
            print(f"✅ Removed {duplicates_removed} duplicate/overlapping segments")
        else:
            print("✅ No duplicates found")

        return deduplicated
