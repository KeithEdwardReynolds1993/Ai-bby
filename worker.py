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
        "-i", "color=c=black:s=540x960:d=3",
        "-vf", "drawtext=text='AI VIDEO TEST':fontcolor=white:fontsize=32:x=(w-text_w)/2:y=(h-text_h)/2",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "32",
        "-pix_fmt", "yuv420p",
        "-threads", "1",
        str(OUTPUT)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("RETURN:", result.returncode, flush=True)
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)

    if OUTPUT.exists():
        print("VIDEO CREATED:", OUTPUT, flush=True)

while True:
    print("Running job...", flush=True)
    run()
    time.sleep(60)
