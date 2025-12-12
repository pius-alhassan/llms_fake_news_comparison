from pathlib import Path
import sys

# Add project root to PYTHONPATH
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from server.preprocessing.preprocess_video import extract_frames

print(extract_frames("data/raw/video/faceforensics/01_03__deepfake.mp4"))
