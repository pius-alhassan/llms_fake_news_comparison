import pandas as pd
from pathlib import Path


# ------------------ PATHS ------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"

VIDEO_DIR = RAW_DIR / "video" / "faceforensics"
TEXT_DIR = RAW_DIR / "text" / "liar_dataset"

VIDEO_METADATA_FILE = METADATA_DIR / "llm_evaluation_metadata.xlsx"


# ------------------ VALIDATION ------------------

REQUIRED_VIDEO_COLUMNS = {
    "id",
    "file_name",
    "Label",
    "video_type",
    "frame_count",
    "width",
    "height",
    "codec",
    "file_size"
}

REQUIRED_TEXT_COLUMNS = {
    "id",
    "label",
    "statement"
}


# ------------------ LOADERS ------------------

def load_video_metadata() -> pd.DataFrame:
    """
    Load FaceForensics video metadata for LLM evaluation.
    Returns a validated pandas DataFrame.
    """
    if not VIDEO_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Video metadata file not found: {VIDEO_METADATA_FILE}"
        )

    df = pd.read_excel(VIDEO_METADATA_FILE)

    missing = REQUIRED_VIDEO_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required video metadata columns: {missing}"
        )

    # Normalize labels
    df["Label"] = df["Label"].str.lower().str.strip()

    return df.sort_values("id").reset_index(drop=True)


def load_text_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load LIAR dataset split.
    split: train | test | valid
    """
    split = split.lower()
    file_map = {
        "train": "train.tsv",
        "test": "test.tsv",
        "valid": "valid.tsv"
    }

    if split not in file_map:
        raise ValueError("split must be one of: train, test, valid")

    file_path = TEXT_DIR / file_map[split]
    if not file_path.exists():
        raise FileNotFoundError(f"LIAR dataset not found: {file_path}")

    df = pd.read_csv(file_path, sep="\t", header=None)

    df.columns = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "job",
        "state",
        "party",
        "barely_true",
        "false",
        "half_true",
        "mostly_true",
        "pants_on_fire",
        "context"
    ]

    df["label"] = df["label"].str.lower().str.strip()

    return df


# ------------------ UTILITY ------------------

def get_video_path(file_name: str) -> Path:
    """
    Return absolute path to a FaceForensics video file.
    """
    path = VIDEO_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    return path
