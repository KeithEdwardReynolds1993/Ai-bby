import os
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account


# =========================
# PATHS
# =========================

TMP_DIR = Path("/tmp/ai_bby")
INPUT_DIR = TMP_DIR / "input"
OUTPUT_DIR = TMP_DIR / "output"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = [
    INPUT_DIR / "clip1.mp4",
    INPUT_DIR / "clip2.mp4",
    INPUT_DIR / "clip3.mp4",
]

NORMALIZED = [
    TMP_DIR / "clip1_norm.mp4",
    TMP_DIR / "clip2_norm.mp4",
    TMP_DIR / "clip3_norm.mp4",
]

CONCAT_LIST = TMP_DIR / "clips.txt"
MERGED = TMP_DIR / "merged.mp4"
FINAL_TMP = TMP_DIR / "final.mp4"
FINAL_OUT = OUTPUT_DIR / "final.mp4"


# =========================
# ENV
# =========================

CAPTION_TEXT = os.getenv("CAPTION_TEXT", "Fun times in AI village")

GOOGLE_DRIVE_OUTPUT_FOLDER_ID = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")

SERVICE_ACCOUNT_INFO = json.loads(
    os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
)


# =========================
# HELPERS
# =========================

def log(*args):
    print(*args, flush=True)


def run_cmd(cmd):
    log("\n>>>", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.stdout:
        log(p.stdout[-3000:])
    if p.stderr:
        log(p.stderr[-3000:])

    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

    return p


# =========================
# VALIDATION
# =========================

def validate_inputs():
    missing = [str(p) for p in INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input clips:\n" + "\n".join(missing)
        )


# =========================
# FFMEG PIPELINE
# =========================

def normalize(src, dst):
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,fps=30,format=yuv420p",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        str(dst)
    ])


def write_concat():
    CONCAT_LIST.write_text(
        "\n".join([f"file '{p}'" for p in NORMALIZED])
    )


def concat():
    run_cmd([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(CONCAT_LIST),
        "-c", "copy",
        str(MERGED)
    ])


def add_caption():
    text = CAPTION_TEXT.replace(":", r"\:")

    vf = (
        "drawbox=x=120:y=1450:w=840:h=180:"
        "color=black@0.85:t=fill,"
        f"drawtext=text='{text}':"
        "fontcolor=white:fontsize=52:"
        "x=(w-text_w)/2:y=1507"
    )

    run_cmd([
        "ffmpeg", "-y",
        "-i", str(MERGED),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "20",
        "-preset", "veryfast",
        "-c:a", "copy",
        str(FINAL_TMP)
    ])


# =========================
# GOOGLE DRIVE UPLOAD
# =========================

def upload_to_drive(file_path: Path):
    log("Uploading to Google Drive...")

    creds = service_account.Credentials.from_service_account_info(
        SERVICE_ACCOUNT_INFO,
        scopes=["https://www.googleapis.com/auth/drive"]
    )

    service = build("drive", "v3", credentials=creds)

    file_metadata = {
        "name": file_path.name,
        "parents": [GOOGLE_DRIVE_OUTPUT_FOLDER_ID]
    }

    media = MediaFileUpload(str(file_path), mimetype="video/mp4")

    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    log("Upload complete:", uploaded.get("id"))


# =========================
# MAIN
# =========================

def main():
    log("AI Bby Worker Starting...")

    validate_inputs()

    # Normalize clips
    for s, d in zip(INPUTS, NORMALIZED):
        normalize(s, d)

    write_concat()
    concat()
    add_caption()

    # Save local copy
    shutil.copy2(FINAL_TMP, FINAL_OUT)

    log("Final created:", FINAL_OUT)

    # Upload to Drive
    if GOOGLE_DRIVE_OUTPUT_FOLDER_ID:
        upload_to_drive(FINAL_OUT)
    else:
        log("No output folder set — skipping upload")


if __name__ == "__main__":
    main()
