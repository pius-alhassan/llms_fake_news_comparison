import subprocess
import uuid
from pathlib import Path
import os
import cv2

TEMP_DIR = Path("server/temp/frames")

# ensure temp directory exists
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def create_temp_session_dir():
    """Creates a unique temp directory for each video processing request."""
    session_id = str(uuid.uuid4())
    session_path = TEMP_DIR / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path


def get_video_duration(video_path):
    """Get duration of video in seconds using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps and frame_count:
        duration = frame_count / fps
        return duration
    return None


def extract_frames(video_path, frame_count=3):
    """
    Extracts N frames from the video at evenly spaced intervals.
    Returns: { "session_dir": "...", "frames": ["path1.jpg", "path2.jpg"] }
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Create unique session folder
    session_dir = create_temp_session_dir()

    # Determine video duration
    duration = get_video_duration(video_path)
    if not duration:
        raise ValueError("Cannot read video duration.")

    # Compute timestamps
    timestamps = []
    for i in range(frame_count):
        t = (i + 1) / (frame_count + 1) * duration
        timestamps.append(t)

    frame_paths = []

    # Extract each frame using ffmpeg
    for idx, ts in enumerate(timestamps, start=1):
        output_frame = session_dir / f"frame_{idx}.jpg"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(output_frame)
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if output_frame.exists():
            frame_paths.append(str(output_frame))

    return {
        "session_dir": str(session_dir),
        "frames": frame_paths,
        "frame_count": len(frame_paths)
    }


def cleanup_session(session_dir):
    """Deletes a session directory and its frames."""
    session_dir = Path(session_dir)
    if session_dir.exists():
        for f in session_dir.glob("*"):
            f.unlink()
        session_dir.rmdir()
