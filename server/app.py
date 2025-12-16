from pathlib import Path
import sys
from flask import Flask, jsonify, request
import json


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from .utils.random_sampler import get_random_videos
from .utils.helpers import get_video_entry, strip_markdown_json
from .utils.liar_loader import sample_liar_statements

from .preprocessing.preprocess_video import extract_frames, cleanup_session
from .llms.gemini_wrapper import GeminiLLM
from .llms.mistral_wrapper import MistralLLM
from .llms.openai_wrapper import OpenAIWrapper


app = Flask(__name__)

MODEL_REGISTRY = {
    "gemini": GeminiLLM(),
    "mistral": MistralLLM(),
    "gpt4o": OpenAIWrapper()
}


@app.route("/api/videos/random", methods=["GET"])
def api_random_videos():
    batch = request.args.get("batch", default=10, type=int)
    try:
        videos = get_random_videos(batch)
        return jsonify({"status": "ok", "count": len(videos), "videos": videos})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/videos/predict/<int:video_id>", methods=["GET"])
def predict_video(video_id):
    # Choose model
    model_name = request.args.get("model", "gemini").lower()
    if model_name not in MODEL_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown model '{model_name}'"}), 400

    model = MODEL_REGISTRY[model_name]

    # Get metadata row
    entry = get_video_entry(video_id)
    if entry is None:
        return jsonify({"status": "error", "message": "Video ID not found"}), 404

    video_path = Path("data/raw/video/faceforensics") / entry["file_name"]
    if not video_path.exists():
        return jsonify({"status": "error", "message": f"Video file not found: {video_path}"}), 404

    try:
        # 1. Extract frames dynamically
        extraction = extract_frames(video_path, frame_count=3)

        # 2. Send frames to LLM
        prediction = model.predict_video(extraction["frames"])
        print(">>> Prediction:", prediction)
        raw_output = prediction.get("raw_output", "")
        clean_json = strip_markdown_json(raw_output)

        try:
            parsed = json.loads(clean_json)
        except json.JSONDecodeError:
            parsed = {
                "prediction": "unknown",
                "confidence": 0,
                "explanation": raw_output
            }


        # 3. Clean up temp frames
        cleanup_session(extraction["session_dir"])

        # 4. Build API response
        return jsonify({
            "status": "ok",
            "video_id": video_id,
            "file_name": entry["file_name"],
            "ground_truth": entry["label"],
            "video_type": entry["video_type"],
            "llm_used": model_name,

            # FLATTENING THE OUTPUT
            "prediction": parsed["prediction"],
            "confidence": parsed["confidence"],
            "explanation": parsed["explanation"]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/text/random", methods=["GET"])
def api_random_texts():
    batch = request.args.get("batch", default=10, type=int)
    split = request.args.get("split", default="test", type=str)

    try:
        samples = sample_liar_statements(batch=batch, split=split)
        return jsonify({
            "status": "ok",
            "count": len(samples),
            "texts": samples
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/text/predict", methods=["POST"])
def predict_text():
    data = request.get_json()

    text = data.get("text", "").strip()
    ground_truth = data.get("ground_truth", "UNKNOWN")
    model_name = data.get("model", "gemini").lower()

    print("text:", text, "ground_truth:", ground_truth, "model:", model_name)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if model_name not in MODEL_REGISTRY:
        return jsonify({"error": "Unknown model"}), 400

    try:
        llm = MODEL_REGISTRY[model_name]
        llm_result = llm.predict_text(text)

        print("LLM raw result:", llm_result)

        # Defensive validation
        if not isinstance(llm_result, dict):
            raise ValueError("LLM did not return a dictionary")

        required_keys = {"prediction", "confidence", "explanation"}
        if not required_keys.issubset(llm_result):
            raise ValueError(f"Missing keys in LLM result: {llm_result}")

        return jsonify({
            "ground_truth": ground_truth,
            "prediction": llm_result["prediction"].upper(),
            "confidence": float(llm_result["confidence"]),
            "explanation": llm_result["explanation"],
            "llm_used": model_name,
            "modality": "text"
        })

    except Exception as e:
        print("TEXT PREDICTION ERROR:", e)
        return jsonify({"error": str(e)}), 500
