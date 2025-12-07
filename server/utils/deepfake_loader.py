import os
import pandas as pd
from pathlib import Path

def load_deepfake_metadata():
    """
    Loads metadata CSV containing:
    video_path | label | manipulation_type | notes
    """
    metadata_path = Path("data/samples/deepfake_metadata.csv")

    if not metadata_path.exists():
        raise FileNotFoundError("Deepfake metadata CSV missing.")

    return pd.read_csv(metadata_path)


def get_video_files():
    """
    Lists all video files in the raw directory.
    """
    raw_dir = Path("data/raw/faceforensics")
    video_files = list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.avi")) + list(raw_dir.glob("*.mov"))
    return video_files
