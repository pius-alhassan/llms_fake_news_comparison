import time
import json
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime

from server.utils.datasets import load_video_metadata
from server.utils.random_sampler import get_random_videos
from server.preprocessing.preprocess_video import extract_frames, cleanup_session
from server.llms.gemini_wrapper import GeminiLLM
from server.llms.openai_wrapper import OpenAIWrapper
from server.llms.mistral_wrapper import MistralLLM


# ------------------ CONFIG ------------------

FRAME_COUNT = 3
SUBSET_SIZE = None  # None = full dataset
MODELS = {
    "gemini": GeminiLLM(),
    "gpt4o": OpenAIWrapper(),
    "mistral": MistralLLM()
}

VIDEO_DIR = Path("data/raw/video/faceforensics")
OUTPUT_ROOT = Path("data/outputs/video_evaluations")


# ------------------ HELPERS ------------------

def create_eval_session():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = OUTPUT_ROOT / f"eval_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, timestamp


def save_config(session_dir, timestamp):
    config = {
        "mode": "video",
        "subset_size": SUBSET_SIZE,
        "frame_count": FRAME_COUNT,
        "models": list(MODELS.keys()),
        "dataset": "FaceForensics",
        "timestamp": timestamp
    }
    with open(session_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


# ------------------ MAIN EVALUATION ------------------

def run_evaluation():
    session_dir, timestamp = create_eval_session()
    save_config(session_dir, timestamp)

    metadata = load_video_metadata()

    if SUBSET_SIZE:
        metadata = metadata.sample(SUBSET_SIZE, random_state=42)

    results = []

    for _, row in metadata.iterrows():
        video_path = VIDEO_DIR / row["file_name"]
        true_label = row["Label"]

        extraction = extract_frames(video_path, frame_count=FRAME_COUNT)

        for model_name, model in MODELS.items():
            start = time.time()
            prediction = model.predict_video(extraction["frames"])
            latency = round((time.time() - start) * 1000, 2)

            pred_label = prediction.get("prediction", "unknown")

            results.append({
                "video_id": row["id"],
                "file_name": row["file_name"],
                "video_type": row["video_type"],
                "true_label": true_label,
                "model": model_name,
                "predicted_label": pred_label,
                "confidence": prediction.get("confidence"),
                "correct": pred_label == true_label,
                "latency_ms": latency
            })

        cleanup_session(extraction["session_dir"])

    # Save outputs
    df = pd.DataFrame(results)
    df.to_csv(session_dir / "results.csv", index=False)
    df.to_json(session_dir / "results.json", indent=2)

    return df, session_dir


if __name__ == "__main__":
    run_evaluation()
