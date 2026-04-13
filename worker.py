import os
import json
import shutil
import subprocess
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

TMP_DIR = Path("/tmp/ai_bby")
INPUT_DIR = TMP_DIR / "input"
OUTPUT_DIR = TMP_DIR / "output"

TMP_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAPTION_TEXT = os.getenv("CAPTION_TEXT", "AI video")

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


def log(*args):
    print(*args, flush=True)


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


def list_clips():
    query = f"'{INCOMING_FOLDER}' in parents and mimeType contains 'video/'"
    results = drive.files().list(q=query, pageSize=3, fields="files(id, name)").execute()
    return results.get("files", [])


def download_file(file_id, name, dest):
    request = drive.files().get_media(fileId=file_id)
    fh = io.FileIO(dest, "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()


def move_file(file_id, new_folder):
    file = drive.files().get(fileId=file_id, fields="parents").execute()
    previous_parents = ",".join(file.get("parents"))
    drive.files().update(
        fileId=file_id,
        addParents=new_folder,
        removeParents=previous_parents,
        fields="id, parents",
    ).execute()


def upload_output(path):
    media = MediaFileUpload(path, mimetype="video/mp4")
    file = drive.files().create(
        body={"name": "final.mp4", "parents": [OUTPUT_FOLDER]},
        media_body=media,
        fields="id",
    ).execute()
    return file.get("id")


def process():
    clips = list_clips()

    if len(clips) < 3:
        log("Not enough clips found")
        return

    local_paths = []

    for i, clip in enumerate(clips[:3]):
        local_path = INPUT_DIR / f"clip{i+1}.mp4"
        download_file(clip["id"], clip["name"], str(local_path))
        local_paths.append(local_path)

    norm_paths = []
    for i, path in enumerate(local_paths):
        out = TMP_DIR / f"norm{i}.mp4"
        run([
            "ffmpeg", "-y",
            "-i", str(path),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            str(out)
        ])
        norm_paths.append(out)

    concat_file = TMP_DIR / "list.txt"
    concat_file.write_text("\n".join([f"file '{p}'" for p in norm_paths]))

    merged = TMP_DIR / "merged.mp4"

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(merged)
    ])

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

    upload_output(str(final))

    for clip in clips[:3]:
        move_file(clip["id"], ARCHIVE_FOLDER)

    log("DONE")


if __name__ == "__main__":
    process()
