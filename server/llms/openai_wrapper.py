import os
import base64
import json
from dotenv import load_dotenv
from server.llms.base import BaseLLM
from openai import OpenAI

load_dotenv()

class OpenAIWrapper(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")

        self.client = OpenAI(api_key=self.api_key)

        # Modern OpenAI multimodal model
        self.model = "gpt-4o-mini"

    def _frames_to_base64(self, frame_paths):
        encoded = []
        for p in frame_paths:
            with open(p, "rb") as f:
                encoded.append(base64.b64encode(f.read()).decode("utf-8"))
        return encoded

    def predict_video(self, frame_paths: list[str]) -> dict:
        try:
            b64_images = self._frames_to_base64(frame_paths)

            # Build prompt
            image_blocks = []
            for i, b64 in enumerate(b64_images, start=1):
                image_blocks.append(
                    f"[Image {i} Base64]\n{b64}\n[/Image {i} Base64]\n"
                )

            prompt = (
                "You are a forensic video analyst.\n"
                "Analyze the base64-encoded frames below and determine whether the "
                "source video is REAL or FAKE.\n\n"
                "Return ONLY a JSON object with keys:\n"
                "prediction (real|fake), confidence (0-1), explanation.\n\n"
                + "\n".join(image_blocks)
                + "\nRespond with valid JSON."
            )

            # Modern call using new OpenAI Responses API
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=300
            )

            text = response.output_text

            # Try parse JSON
            try:
                return json.loads(text)
            except Exception:
                return {"raw_output": text}

        except Exception as e:
            return {"error": str(e)}
