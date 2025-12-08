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


def get_video_entry(video_id: int):
    df = load_metadata()
    row = df[df["id"] == video_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()