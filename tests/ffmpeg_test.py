import subprocess

video_path = r"C:\Users\value\Documents\projects\llms_fake_news_comparison\data\raw\video\faceforensics\000_003_DF.mp4"

# Test 1: Check if ffmpeg is available
try:
    result = subprocess.run(["ffmpeg", "-version"], 
                          capture_output=True, text=True)
    print("ffmpeg available:", "ffmpeg" in result.stdout.lower())
except FileNotFoundError:
    print("❌ ffmpeg not found in PATH!")

# Test 2: Try ffmpeg with your video
cmd = ["ffmpeg", "-i", video_path]
result = subprocess.run(cmd, capture_output=True, text=True)
print("ffmpeg output:", result.stderr[:500])  # First 500 chars