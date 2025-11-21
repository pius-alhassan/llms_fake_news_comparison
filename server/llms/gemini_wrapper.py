# server/llms/gemini_wrapper.py
from .base import LLMBase
import google.generativeai as genai

class GeminiLLM(LLMBase):
    def __init__(self, api_key):
        super().__init__(api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")

    def generate(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text

    def classify_fake_news(self, input_data, mode="text"):
        if mode == "text":
            prompt = f"Determine if this text is fake news or real:\n{input_data}\nAnswer with 'Fake' or 'Real'."
            return self.generate(prompt)
        # TODO: extend image/audio/video support
        return "Mode not implemented yet"
