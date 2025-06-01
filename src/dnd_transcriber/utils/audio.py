import subprocess
from pathlib import Path


def get_audio_duration(path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def validate_audio_format(path: Path) -> bool:
    """Validate if file is a supported audio format."""
    if not path.exists():
        return False

    supported_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
    if path.suffix.lower() not in supported_extensions:
        return False

    # Verify file can be read by ffprobe
    cmd = ["ffprobe", "-v", "quiet", "-select_streams", "a:0", str(path)]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def split_audio_file(path: Path, chunks: list[tuple[float, float]]) -> list[Path]:
    """Split audio file into chunks. chunks = [(start_seconds, end_seconds), ...]"""
    output_paths = []

    for i, (start, end) in enumerate(chunks):
        output_path = path.parent / f"{path.stem}_chunk_{i:03d}{path.suffix}"

        cmd = [
            "ffmpeg",
            "-i",
            str(path),
            "-ss",
            str(start),
            "-t",
            str(end - start),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            output_paths.append(output_path)
        except subprocess.CalledProcessError:
            # Clean up partial outputs on failure
            for p in output_paths:
                p.unlink(missing_ok=True)
            return []

    return output_paths
