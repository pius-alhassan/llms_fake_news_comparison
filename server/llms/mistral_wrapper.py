import os
import base64
import json, re
import requests
from dotenv import load_dotenv
from .base import BaseLLM

load_dotenv()

class MistralLLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env")

        self.endpoint = "https://api.mistral.ai/v1/generate"
        self.model = "pixtral-12b-2409"

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
                "keys: prediction, confidence(0%-100%), explanation.\n\n"
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
        
    def predict_text(self, text: str) -> dict:
        """
        Predict fake/real for a text statement using Mistral Pixtral
        """
        try:
            prompt = (
                "You are a forensic news analyst.\n"
                "Analyze the following statement and determine if it is 'fake' or 'real'.\n"
                "Return JSON only with keys: prediction (fake|real), confidence (0-100), explanation.\n\n"
                f"Text: {text}\n"
            )

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "input": prompt,
                "max_tokens": 512
            }

            resp = requests.post(self.endpoint, headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            
            # Try to get output from different possible response formats
            response_data = resp.json()
            text_output = response_data.get("output", "") or response_data.get("choices", [{}])[0].get("message", {}).get("content", "") or resp.text
            
            if not text_output:
                text_output = str(response_data)

            # --- Clean markdown code blocks ---
            cleaned = re.sub(r"^```json\s*", "", text_output, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                # Try to find JSON in the text if it's not the whole response
                try:
                    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                    else:
                        return {
                            "prediction": "UNKNOWN",
                            "confidence": 0.0,
                            "explanation": f"Failed to parse JSON from response: {cleaned[:200]}"
                        }
                except Exception:
                    return {
                        "prediction": "UNKNOWN",
                        "confidence": 0.0,
                        "explanation": f"Failed to parse JSON from response: {cleaned[:200]}"
                    }

            # --- Enforce schema ---
            prediction = str(parsed.get("prediction", "UNKNOWN")).upper()
            if prediction not in ["REAL", "FAKE"]:
                prediction = "UNKNOWN"
                
            confidence = parsed.get("confidence", 50)
            try:
                confidence = float(confidence)
                if confidence < 0:
                    confidence = 0.0
                elif confidence > 100:
                    confidence = 100.0
            except (ValueError, TypeError):
                confidence = 50.0
                
            return {
                "prediction": prediction,
                "confidence": confidence,
                "explanation": str(parsed.get("explanation", ""))
            }

        except requests.exceptions.RequestException as e:
            return {
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "explanation": f"Mistral API request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "explanation": f"Mistral prediction failed: {str(e)}"
            }
