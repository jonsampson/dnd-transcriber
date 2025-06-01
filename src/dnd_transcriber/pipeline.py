from pathlib import Path
from typing import Optional, List, Dict, Any
from .config import PipelineConfig
from .roster import CharacterRoster
from .preprocessor import AudioPreprocessor
from .transcriber import WhisperXTranscriber
from .validator import TranscriptionValidator
from .context import ContextWindowManager
from .confidence import identify_low_confidence_segments, group_adjacent_segments
from .formatter import convert_whisperx_output
from .models import TranscriptionOutput
from .utils.audio import split_audio_file


class D_DTranscriptionPipeline:
    """Main D&D transcription pipeline combining preprocessing, transcription, and validation."""
    
    def __init__(self, config: PipelineConfig, roster: Optional[CharacterRoster] = None):
        """Initialize pipeline with configuration and optional character roster."""
        self.config = config
        self.roster = roster
        
        # Initialize components
        self.preprocessor = AudioPreprocessor(config.demucs)
        self.transcriber = WhisperXTranscriber(config.whisperx)
        self.validator = TranscriptionValidator(config.ollama, roster)
        self.context_manager = ContextWindowManager(window_size=5, overlap=1)
    
    def process_audio(self, input_path: Path, use_multipass: bool = True, skip_preprocessing: bool = False) -> TranscriptionOutput:
        """Process audio file through complete transcription pipeline.
        
        Args:
            input_path: Path to input audio file
            use_multipass: Whether to use multi-pass processing for low confidence segments
            skip_preprocessing: Whether to skip Demucs audio separation
            
        Returns:
            Structured transcription output
        """
        # Step 1: Preprocess audio with Demucs (if not skipped)
        if skip_preprocessing:
            vocals_path = input_path
        else:
            vocals_path = self.preprocessor.separate_audio(input_path)
        
        # Step 2: Transcribe with WhisperX
        whisperx_result = self.transcriber.transcribe(vocals_path)
        
        # Step 3: Convert to structured format
        transcription_output = convert_whisperx_output(whisperx_result)
        
        # Step 4: Multi-pass processing if enabled
        if use_multipass:
            low_confidence_indices = identify_low_confidence_segments(
                transcription_output.segments, threshold=0.6
            )
            
            if low_confidence_indices:
                reprocessed_result = self.reprocess_segments(low_confidence_indices, vocals_path)
                whisperx_result = self.merge_transcriptions(whisperx_result, reprocessed_result)
                transcription_output = convert_whisperx_output(whisperx_result)
        
        # Step 5: Validate segments with low confidence
        validated_segments = self._validate_segments(transcription_output.segments)
        
        # Step 6: Update transcription with validated segments
        transcription_output.segments = validated_segments
        
        return transcription_output
    
    def _validate_segments(self, segments: list) -> list:
        """Validate transcription segments using LLM."""
        # Identify low confidence segments
        low_confidence_indices = identify_low_confidence_segments(
            segments, threshold=0.7
        )
        
        # Create context windows for validation
        context_windows = self.context_manager.create_windows(segments)
        
        validated_segments = []
        
        for i, (segment, context_before, context_after) in enumerate(context_windows):
            # Build context string
            context_text = ""
            if context_before:
                context_text += " ".join([seg.text for seg in context_before[-2:]])
            if context_after:
                context_text += " " + " ".join([seg.text for seg in context_after[:2]])
            
            # Validate if low confidence or contains potential character names
            should_validate = (
                i in low_confidence_indices or
                self._might_contain_character_names(segment.text)
            )
            
            if should_validate:
                corrected_text = self.validator.validate_segment(
                    segment.text, 
                    context_text.strip()
                )
                segment.text = corrected_text
            
            validated_segments.append(segment)
        
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
    
    def reprocess_segments(self, indices: List[int], audio_path: Path) -> Dict[str, Any]:
        """Reprocess specific segments with different parameters.
        
        Args:
            indices: List of segment indices to reprocess
            audio_path: Path to processed audio file
            
        Returns:
            Reprocessed WhisperX result
        """
        # Group adjacent segments for batch processing
        segment_groups = group_adjacent_segments(indices)
        
        reprocessed_segments = {}
        
        for group in segment_groups:
            start_idx, end_idx = group[0], group[-1] + 1
            
            # Create time chunks for audio splitting
            # Note: This is a simplified approach, would need actual timing data
            start_time = start_idx * 30  # Assume 30-second segments
            end_time = end_idx * 30
            
            chunks = [(start_time, end_time)]
            chunk_paths = split_audio_file(audio_path, chunks)
            
            if chunk_paths:
                chunk_result = self.transcriber.transcribe(chunk_paths[0])
                
                # Map results back to original indices
                for i, segment in enumerate(chunk_result.get("segments", [])):
                    if start_idx + i in indices:
                        reprocessed_segments[start_idx + i] = segment
                
                # Clean up temporary chunk
                chunk_paths[0].unlink(missing_ok=True)
        
        return {"segments": list(reprocessed_segments.values())}
    
    def merge_transcriptions(self, original: Dict[str, Any], reprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Merge original and reprocessed transcriptions.
        
        Args:
            original: Original WhisperX result
            reprocessed: Reprocessed segments result
            
        Returns:
            Merged transcription result
        """
        merged = original.copy()
        reprocessed_segments = {i: seg for i, seg in enumerate(reprocessed.get("segments", []))}
        
        # Replace segments that were reprocessed
        for i, segment in enumerate(merged.get("segments", [])):
            if i in reprocessed_segments:
                # Choose better confidence score
                reprocessed_seg = reprocessed_segments[i]
                if (reprocessed_seg.get("confidence", 0) > segment.get("confidence", 0)):
                    merged["segments"][i] = reprocessed_seg
        
        return merged