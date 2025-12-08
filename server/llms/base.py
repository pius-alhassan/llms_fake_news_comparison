from abc import ABC, abstractmethod

class BaseLLM(ABC):
    
    @abstractmethod
    def predict_video(self, frame_paths: list[str]) -> dict:
        """Takes extracted frame paths and returns prediction dict."""
        pass
