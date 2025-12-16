import os
import json, re
from dotenv import load_dotenv
from google import generativeai as genai
from .base import BaseLLM

load_dotenv()  # loading GEMINI_API_KEY from environment variable file

class GeminiLLM(BaseLLM):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")  
    
    def predict_video(self, frame_paths: list[str]) -> dict:
        """
        Sends frames to Gemini and returns prediction.
        """
        try:
            parts = []
            
            for fpath in frame_paths:
                with open(fpath, "rb") as img_file:
                    img_bytes = img_file.read()
                    
                parts.append({
                    "inline_data": {
                        "data": img_bytes,
                        "mime_type": "image/jpeg"
                    }
                })
            parts.insert(0,{
                "text":"""
                        You are a fake-video detection expert.
                        Analyze these video frames and determine whether the video is REAL or FAKE.
                        Provide:
                        1. Final decision: "real" or "fake"
                        2. Confidence level (0%–100%)
                        3. Short explanation describing visual clues.

                        Return result as valid JSON:
                        {
                        "prediction": "...",
                        "confidence": ...,
                        "explanation": "..."
                        }
                    """
            })

            response = self.model.generate_content(
                contents=[{"role": "user", "parts": parts}]
            )
            
            output_text = response.text.strip()
            import json
            try:
                parsed = json.loads(output_text)
            except:
                parsed = {"raw_output": output_text}
            return parsed

        except Exception as e:
            return {"error": str(e)}
        
    def predict_text(self, text: str) -> dict:
        """
        Predict if a text statement is fake or real using Gemini.
        Returns a dict with keys: prediction, confidence, explanation
        """
        try:
            prompt = f"""
                    You are a fake news detection expert.

                    Analyze the following news statement and determine whether it is REAL or FAKE.

                    Return your answer strictly as JSON with keys:
                    - prediction (REAL or FAKE)
                    - confidence (0–100)
                    - explanation (short reasoning)

                    Statement:
                    {text}
                    """

            response = self.model.generate_content(prompt)
            text_output = response.text.strip()

            # --- Clean markdown code blocks ---
            # Remove ```json and ``` markers
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
            # Ensure the response has the expected keys with proper formatting
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

        except Exception as e:
            return {
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "explanation": f"Gemini prediction failed: {str(e)}"
            }