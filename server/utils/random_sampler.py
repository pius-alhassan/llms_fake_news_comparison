import pandas as pd
from pathlib import Path
import random

METADATA_PATH = Path("data/metadata/llm_evaluation_metadata.xlsx")

# cache dataframe
_cached_df = None

def load_metadata():
    global _cached_df
    if _cached_df is None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
        _cached_df = pd.read_excel(METADATA_PATH, engine="openpyxl")
    return _cached_df


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
