import pandas as pd
from pathlib import Path

def load_liar_split(split="train"):
    """
    Loads the LIAR dataset split (train/test/valid)
    Returns a pandas DataFrame with clean column names.
    """
    liar_dir = Path("data/raw/text/liar_dataset")
    filepath = liar_dir / f"{split}.tsv"

    if not filepath.exists():
        raise FileNotFoundError(f"LIAR file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=[
            "id", "label", "statement", "subjects", "speaker", "speaker_occupation",
            "state", "party", "barely_true", "false", "half_true",
            "mostly_true", "pants_on_fire", "context"
        ]
    )
    return df


def load_liar_all():
    """
    Load train, test, valid into a dictionary of DataFrames.
    """
    return {
        "train": load_liar_split("train"),
        "test": load_liar_split("test"),
        "valid": load_liar_split("valid")
    }

def normalize_label(label):
    return "FAKE" if label in ["false", "pants-fire", "barely-true"] else "REAL"

def sample_liar_statements(batch=10, split="test"):
    df = load_liar_split(split)
    rows = df.sample(batch)

    samples = []
    for _, row in rows.iterrows():
        samples.append({
            "id": row["id"],
            "text": row["statement"],
            "label": normalize_label(row["label"]),
            "speaker": row["speaker"],
            "context": row["context"]
        })

    return samples
