from flask import Flask, jsonify, request
from utils.random_sampler import get_random_videos

app = Flask(__name__)

@app.route("/api/videos/random", methods=["GET"])
def api_random_videos():
    batch = request.args.get("batch", default=10, type=int)
    try:
        videos = get_random_videos(batch)
        return jsonify({"status": "ok", "count": len(videos), "videos": videos})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
