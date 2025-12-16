import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from server.utils.datasets import load_text_dataset
from server.llms.gemini_wrapper import GeminiLLM
from server.llms.openai_wrapper import OpenAIWrapper
from server.llms.mistral_wrapper import MistralLLM


# ------------------ CONFIG ------------------

SUBSET_SIZE = 100
MODELS = {
    "gemini": GeminiLLM(),
    "gpt4o": OpenAIWrapper(),
    "mistral": MistralLLM()
}

OUTPUT_ROOT = Path("data/outputs/text_evaluations")


# ------------------ HELPERS ------------------

def map_liar_label(label):
    fake = {"pants-fire", "false", "barely-true"}
    return "fake" if label in fake else "real"


def create_eval_session():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = OUTPUT_ROOT / f"eval_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d, ts


# ------------------ MAIN ------------------

def run_evaluation():
    session_dir, timestamp = create_eval_session()

    df = load_text_dataset("test")
    if SUBSET_SIZE:
        df = df.sample(SUBSET_SIZE, random_state=42)

    results = []

    for _, row in df.iterrows():
        text = row["statement"]
        true_label = map_liar_label(row["label"])

        for model_name, model in MODELS.items():
            start = time.time()
            prediction = model.predict_text(text)
            latency = round((time.time() - start) * 1000, 2)

            pred_label = prediction.get("prediction", "unknown")

            results.append({
                "id": row["id"],
                "model": model_name,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": prediction.get("confidence"),
                "correct": pred_label == true_label,
                "latency_ms": latency
            })

    # Save
    df_out = pd.DataFrame(results)
    df_out.to_csv(session_dir / "results.csv", index=False)
    df_out.to_json(session_dir / "results.json", indent=2)

    with open(session_dir / "config.json", "w") as f:
        json.dump({
            "dataset": "LIAR",
            "subset_size": SUBSET_SIZE,
            "models": list(MODELS.keys()),
            "timestamp": timestamp
        }, f, indent=2)

    return df_out, session_dir


if __name__ == "__main__":
    run_evaluation()
