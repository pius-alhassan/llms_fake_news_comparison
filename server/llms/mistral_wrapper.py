# server/llms/mistral_wrapper.py
import os
import base64
import json
import requests
from dotenv import load_dotenv
from server.llms.base import BaseLLM

load_dotenv()

class MistralLLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env")
        # Default endpoint - adjust if your account uses another host
        self.endpoint = "https://api.mistral.ai/v1/generate"
        self.model = "pixtral-12b-2409"  # change if you use a different Pixtral model

    def _frames_to_base64(self, frame_paths):
        parts = []
        for p in frame_paths:
            with open(p, "rb") as f:
                parts.append(base64.b64encode(f.read()).decode("utf-8"))
        return parts

    def predict_video(self, frame_paths: list[str]) -> dict:
        try:
            b64_images = self._frames_to_base64(frame_paths)
            # Place images inline in prompt (simple portable approach)
            prompt_images = ""
            for i, b64 in enumerate(b64_images, 1):
                prompt_images += f"[Image_{i}_base64]\n{b64}\n[/Image_{i}_base64]\n"

            prompt = (
                "You are a video forensic analyst. Use the base64-encoded images below. "
                "Decide if the source video is 'real' or 'fake'. Return valid JSON with "
                "keys: prediction, confidence(0-1), explanation.\n\n"
                f"{prompt_images}\nRespond with JSON only."
            )

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Minimal payload - Mistral docs may offer richer parameters
            data = {
                "model": self.model,
                "input": prompt,
                "max_tokens": 512,
            }

            resp = requests.post(self.endpoint, headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            text = resp.json().get("output", "") or resp.text

            # Try to extract JSON
            try:
                parsed = json.loads(text.strip())
            except Exception:
                # sometimes Mistral returns structured JSON in a field; attempt best-effort
                # if response is dict with output_text
                if isinstance(resp.json(), dict):
                    # try common fields
                    for k in ("output", "text", "result"):
                        if isinstance(resp.json().get(k), str):
                            try:
                                parsed = json.loads(resp.json().get(k))
                                break
                            except Exception:
                                parsed = {"raw_output": resp.json().get(k)}
                                break
                else:
                    parsed = {"raw_output": text}
            return parsed
        except Exception as e:
            return {"error": str(e)}
