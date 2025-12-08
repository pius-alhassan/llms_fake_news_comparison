import pandas as pd
from pathlib import Path
from helpers import load_metadata
import random

def get_random_videos(batch_size=10):
    """
    Returns a list of random video metadata entries.
    """
    df = load_metadata()

    # randomly sample rows (without replacement)
    sample_df = df.sample(n=batch_size, replace=False).reset_index(drop=True)

    # convert to dict format for API
    results = []
    for _, row in sample_df.iterrows():
        results.append({
            "id": int(row["id"]),
            "file_name": row["file_name"],
            "label": row["label"],    # either real or fake
            "video_type": row["video_type"],  # category like FaceSwap, etc.
            "frame_counts": int(row.get("frame_counts", 0)),
            "width": int(row.get("width", 0)),
            "height": int(row.get("height", 0)),
            "codec": row.get("codec", None),
            "file_size": float(row.get("file_size", 0)),
        })
    return results
