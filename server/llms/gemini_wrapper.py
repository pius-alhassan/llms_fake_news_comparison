import os
from dotenv import load_dotenv
from google import generativeai as genai
from base import BaseLLM

load_dotenv()  # loading GEMINI_API_KEY

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
