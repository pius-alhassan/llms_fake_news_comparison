# server/llms/base.py
from abc import ABC, abstractmethod

class LLMBase(ABC):
    def __init__(self, api_key=None):
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt):
        pass

    @abstractmethod
    def classify_fake_news(self, input_data, mode="text"):
        """
        mode can be: text, image, audio, video
        """
        pass
