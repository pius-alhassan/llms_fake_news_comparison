import pandas as pd
from pathlib import Path

def load_liar_split(split="train"):
    """
    Loads the LIAR dataset split (train/test/valid)
    Returns a pandas DataFrame with clean column names.
    """
    liar_dir = Path("data/raw/liar")
    filepath = liar_dir / f"{split}.tsv"

    if not filepath.exists():
        raise FileNotFoundError(f"LIAR file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=[
            "id", "label", "statement", "subjects", "speaker", "speaker_job",
            "state", "party", "barely_true_ct", "false_ct", "half_true_ct",
            "mostly_true_ct", "pants_on_fire_ct", "context"
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
