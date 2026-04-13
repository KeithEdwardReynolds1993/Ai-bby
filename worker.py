import os
import time
import subprocess
from pathlib import Path

TMP_DIR = Path("/tmp/ai_bby")
TMP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = TMP_DIR / "test.mp4"

def run():
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "color=c=black:s=1080x1920:d=5",
        "-vf", "drawtext=text='AI VIDEO TEST':fontcolor=white:fontsize=60:x=(w-text_w)/2:y=(h-text_h)/2",
        str(OUTPUT)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("RETURN:", result.returncode)
    print(result.stdout)
    print(result.stderr)

    if OUTPUT.exists():
        print("✅ VIDEO CREATED:", OUTPUT)

while True:
    print("Running job...")
    run()
    time.sleep(60)
