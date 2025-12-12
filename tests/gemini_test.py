import sys
import pathlib
import json

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from server.llms.gemini_wrapper import GeminiLLM

# module path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from server.preprocessing.preprocess_video import extract_frames, cleanup_session
from server.llms.gemini_wrapper import GeminiLLM
from pathlib import Path

# Choose a real video from your dataset
VIDEO_FILE_01 = "000_003_DF.mp4"   
VIDEO_FILE_02 = "01_03__deepfake.mp4"
VIDEO_FILE_03 = "437_original.mp4"
VIDEO_PATH = Path("data/raw/video/faceforensics") / VIDEO_FILE_03

print("\n========================")
print("🔍 GEMINI VIDEO TEST")
print("========================\n")

# 1. Extract frames dynamically
# 1. Extract frames
print(f"📥 Extracting frames from: {VIDEO_FILE_03}")
extraction = extract_frames(VIDEO_PATH, frame_count=3)

print("\n🖼️ Extracted Frame Paths:")
for f in extraction["frames"]:
    print("   -", f)

# 2. Test Gemini
print("\n🤖 Sending frames to Gemini...")
model = GeminiLLM()
result = model.predict_video(extraction["frames"])

# 3. Pretty Print Output
print("\n========================")
print("📌 Formatted GEMINI OUTPUT ")
print("========================\n")
try:
    print(json.dumps(result, indent=2))
except:
    print(result)

# 4. Clean up
cleanup_session(extraction["session_dir"])
print("\n🧹 Temporary frames cleaned up.")
print("\n✅ Test Completed Successfully.\n")
