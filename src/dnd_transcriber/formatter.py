from typing import Dict, Any
from .models import TranscriptionOutput, Segment, Speaker
from .utils.time import seconds_to_srt_time, seconds_to_readable


def convert_whisperx_output(whisperx_data: dict) -> TranscriptionOutput:
    """Convert WhisperX output dictionary to TranscriptionOutput model."""
    segments = []
    
    for segment_data in whisperx_data.get('segments', []):
        segment = Segment(
            text=segment_data.get('text', '').strip(),
            speaker=segment_data.get('speaker', 'Unknown'),
            start_time=segment_data.get('start', 0.0),
            end_time=segment_data.get('end', 0.0),
            confidence=segment_data.get('confidence')
        )
        segments.append(segment)
    
    # Extract metadata
    metadata = {
        'language': whisperx_data.get('language'),
        'model': whisperx_data.get('model'),
        'segments_count': len(segments)
    }
    
    # Calculate total duration from segments or use provided duration
    audio_duration = whisperx_data.get('duration', 0.0)
    if not audio_duration and segments:
        audio_duration = max(seg.end_time for seg in segments)
    
    return TranscriptionOutput(
        segments=segments,
        metadata=metadata,
        audio_duration=audio_duration
    )


def export_to_text(output: TranscriptionOutput) -> str:
    """Export transcription to plain text format."""
    lines = []
    
    for segment in output.segments:
        timestamp = seconds_to_readable(segment.start_time)
        speaker = segment.speaker or "Unknown"
        text = segment.text.strip()
        
        if text:
            lines.append(f"[{timestamp}] {speaker}: {text}")
    
    return "\n".join(lines)


def export_to_srt(output: TranscriptionOutput) -> str:
    """Export transcription to SRT subtitle format."""
    srt_lines = []
    
    for i, segment in enumerate(output.segments, 1):
        if not segment.text.strip():
            continue
        
        start_time = seconds_to_srt_time(segment.start_time)
        end_time = seconds_to_srt_time(segment.end_time)
        
        speaker = segment.speaker or "Unknown"
        text = f"{speaker}: {segment.text.strip()}"
        
        srt_lines.extend([
            str(i),
            f"{start_time} --> {end_time}",
            text,
            ""  # Empty line between subtitles
        ])
    
    return "\n".join(srt_lines)