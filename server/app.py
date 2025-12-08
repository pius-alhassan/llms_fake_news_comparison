from flask import Flask, jsonify, request
from utils.random_sampler import *
from utils.helpers import get_video_entry
from preprocessing.preprocess_video import extract_frames, cleanup_session
from llms.gemini_wrapper import GeminiLLM
# from llms.mistral_wrapper import MistralLLM
# from llms.openai_wrapper import OpenAIWrapper


app = Flask(__name__)

MODEL_REGISTRY = {
    "gemini": GeminiLLM(),
    # "mistral": MistralLLM(),
    # "gpt4o": OpenAIWrapper()
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

        # 3. Clean up temp frames
        cleanup_session(extraction["session_dir"])

        # 4. Build API response
        return jsonify({
            "status": "ok",
            "video_id": video_id,
            "file_name": entry["file_name"],
            "label": entry["Label"],
            "video_type": entry["video_type"],
            "llm_used": model_name,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

