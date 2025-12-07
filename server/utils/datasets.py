from .liar_loader import load_liar_all
from .deepfake_loader import load_deepfake_metadata
from ..models.multimodal import extract_key_frame, extract_audio_text

class TextDataset:
    def __init__(self):
        self.splits = load_liar_all()

class DeepfakeDataset:
    def __init__(self):
        self.metadata = load_deepfake_metadata()

    def get_frame_for_video(self, video_path):
        return extract_key_frame(video_path, "data/samples/deepfake_frames")

    def get_audio_text(self, video_path):
        return extract_audio_text(video_path)
