import json
from pathlib import Path
import click
from .config import PipelineConfig
from .roster import CharacterRoster
from .pipeline import D_DTranscriptionPipeline
from .formatter import export_to_text, export_to_srt


@click.group()
def cli():
    """D&D Transcription Pipeline CLI."""
    pass


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option('--roster', type=click.Path(exists=True, path_type=Path), help='Character roster JSON file')
@click.option('--no-multipass', is_flag=True, help='Disable multi-pass processing')
@click.option('--skip-preprocessing', is_flag=True, help='Skip Demucs audio separation')
def transcribe(input_path: Path, output_path: Path, roster: Path = None, no_multipass: bool = False, skip_preprocessing: bool = False):
    """Transcribe audio file using D&D-optimized pipeline."""
    
    # Load configuration from environment
    config = PipelineConfig.from_env()
    
    # Load character roster if provided
    character_roster = None
    if roster:
        character_roster = CharacterRoster()
        character_roster.load_from_file(roster)
    
    # Initialize pipeline
    pipeline = D_DTranscriptionPipeline(config, character_roster)
    
    # Process audio
    click.echo(f"Processing audio: {input_path}")
    result = pipeline.process_audio(input_path, use_multipass=not no_multipass, skip_preprocessing=skip_preprocessing)
    
    # Determine output format from file extension
    output_ext = output_path.suffix.lower()
    
    if output_ext == '.json':
        # Export as JSON
        output_data = {
            'segments': [
                {
                    'text': seg.text,
                    'speaker': seg.speaker,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'confidence': seg.confidence
                }
                for seg in result.segments
            ],
            'metadata': result.metadata,
            'audio_duration': result.audio_duration
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    elif output_ext == '.srt':
        # Export as SRT
        srt_content = export_to_srt(result)
        with open(output_path, 'w') as f:
            f.write(srt_content)
    
    else:
        # Export as plain text (default)
        text_content = export_to_text(result)
        with open(output_path, 'w') as f:
            f.write(text_content)
    
    click.echo(f"Transcription saved to: {output_path}")
    click.echo(f"Processed {len(result.segments)} segments in {result.audio_duration:.1f} seconds of audio")


if __name__ == '__main__':
    cli()