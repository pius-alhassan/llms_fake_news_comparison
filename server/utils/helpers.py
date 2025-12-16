import pandas as pd
from pathlib import Path
import re
import json


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

def strip_markdown_json(text: str) -> str:
    """
    Removes ```json ... ``` or ``` ... ``` wrappers from LLM output
    """
    if not text:
        return text

    text = text.strip()

    # Remove ```json or ``` wrappers
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)

    return text.strip()

def extract_json_from_llm(text: str) -> dict:
    """
    Extract JSON object from LLM output wrapped in markdown or text.
    """
    if not text:
        raise ValueError("Empty LLM output")

    # Remove markdown fences
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")