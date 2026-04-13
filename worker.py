import os
import json
import shutil
import subprocess
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

# =========================
# PATHS (Render safe)
# =========================
TMP_DIR = Path("/tmp/ai_bby")
INPUT_DIR = TMP_DIR / "input"
OUTPUT_DIR = TMP_DIR / "output"

TMP_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# ENV VARS
# =========================
CAPTION_TEXT = os.getenv("CAPTION_TEXT", "AI Video")

INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
ARCHIVE_FOLDER = os.getenv("GOOGLE_DRIVE_ARCHIVE_FOLDER_ID")
FAILED_FOLDER = os.getenv("GOOGLE_DRIVE_FAILED_FOLDER_ID")

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))

SCOPES = ["https://www.googleapis.com/auth/drive"]

creds = service_account.Credentials.from_service_account_info(
    SERVICE_ACCOUNT_INFO, scopes=SCOPES
)

drive = build("drive", "v3", credentials=creds)

# =========================
# LOGGING
# =========================
def log(*args):
    print(*args, flush=True)

# =========================
# SHELL RUNNER
# =========================
def run(cmd):
    log(">>>", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        log(result.stdout[-2000:])
    if result.stderr:
        log(result.stderr[-2000:])
    if result.returncode != 0:
        raise RuntimeError("Command failed")
    return result

# =========================
# GET CLIPS FROM DRIVE
# =========================
def get_clips():
    query = f"'{INCOMING_FOLDER}' in parents and mimeType contains 'video/'"
    res = drive.files().list(
        q=query,
        fields="files(id, name)",
        pageSize=3
    ).execute()

    return res.get("files", [])

# =========================
# DOWNLOAD FILE
# =========================
def download(file_id, path):
    request = drive.files().get_media(fileId=file_id)
    fh = io.FileIO(path, "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

# =========================
# MOVE FILE
# =========================
def move(file_id, folder_id):
    file = drive.files().get(fileId=file_id, fields="parents").execute()
    prev = ",".join(file.get("parents"))

    drive.files().update(
        fileId=file_id,
        addParents=folder_id,
        removeParents=prev,
        fields="id, parents"
    ).execute()

# =========================
# UPLOAD OUTPUT
# =========================
def upload(file_path):
    media = MediaFileUpload(file_path, mimetype="video/mp4")

    file = drive.files().create(
        body={
            "name": "final.mp4",
            "parents": [OUTPUT_FOLDER]
        },
        media_body=media,
        fields="id"
    ).execute()

    return file.get("id")

# =========================
# PROCESS PIPELINE
# =========================
def process():
    clips = get_clips()

    if len(clips) < 3:
        log("Not enough clips in Drive")
        return

    local = []

    # DOWNLOAD 3 CLIPS
    for i, clip in enumerate(clips[:3]):
        path = INPUT_DIR / f"clip{i+1}.mp4"
        download(clip["id"], path)
        local.append((clip["id"], path))

    # NORMALIZE
    norm = []
    for i, (_, path) in enumerate(local):
        out = TMP_DIR / f"norm{i}.mp4"

        run([
            "ffmpeg", "-y",
            "-i", str(path),
            "-vf",
            "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            str(out)
        ])

        norm.append(out)

    # CONCAT
    list_file = TMP_DIR / "list.txt"
    list_file.write_text("\n".join([f"file '{p}'" for p in norm]))

    merged = TMP_DIR / "merged.mp4"

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(merged)
    ])

    # CAPTION
    final = TMP_DIR / "final.mp4"

    run([
        "ffmpeg", "-y",
        "-i", str(merged),
        "-vf",
        f"drawtext=text='{CAPTION_TEXT}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=h-200",
        "-c:v", "libx264",
        "-crf", "20",
        str(final)
    ])

    # UPLOAD RESULT
    upload(str(final))

    # ARCHIVE ORIGINALS
    for clip in clips[:3]:
        move(clip["id"], ARCHIVE_FOLDER)

    log("DONE")

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    process()
