import os
import json
import subprocess
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account


# =========================
# PATHS
# =========================

TMP = Path("/tmp/ai_bby")
INPUT = TMP / "input"
OUTPUT = TMP / "output"

INPUT.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

MERGED = TMP / "merged.mp4"
FINAL = TMP / "final.mp4"


# =========================
# ENV
# =========================

SERVICE_ACCOUNT_INFO = json.loads(
    os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
)

INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
ARCHIVE_FOLDER = os.getenv("GOOGLE_DRIVE_ARCHIVE_FOLDER_ID")

CAPTION = os.getenv("CAPTION_TEXT", "Made with AI")


# =========================
# DRIVE CLIENT
# =========================

def drive():
    creds = service_account.Credentials.from_service_account_info(
        SERVICE_ACCOUNT_INFO,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


# =========================
# DOWNLOAD INPUTS
# =========================

def download_inputs():
    service = drive()

    results = service.files().list(
        q=f"'{INCOMING_FOLDER}' in parents and trashed=false",
        fields="files(id, name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives"
    ).execute()

    files = results.get("files", [])

    if not files:
        raise Exception("No input videos in Drive incoming folder")

    print(f"Found {len(files)} files")

    for i, f in enumerate(files):
        file_id = f["id"]
        target = INPUT / f"clip{i+1}.mp4"

        request = service.files().get_media(fileId=file_id)
        fh = open(target, "wb")

        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        fh.close()

        print("Downloaded:", f["name"])


# =========================
# FFmpeg PIPELINE
# =========================

def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_video():
    clips = sorted(INPUT.glob("*.mp4"))

    if not clips:
        raise Exception("No downloaded clips found")

    list_file = TMP / "list.txt"
    list_file.write_text(
        "\n".join([f"file '{c}'" for c in clips])
    )

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(MERGED)
    ])

    run([
        "ffmpeg", "-y",
        "-i", str(MERGED),
        "-vf",
        f"drawtext=text='{CAPTION}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=h-200",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "copy",
        str(FINAL)
    ])


# =========================
# UPLOAD OUTPUT
# =========================

def upload_output():
    service = drive()

    file_metadata = {
        "name": "final.mp4",
        "parents": [OUTPUT_FOLDER]
    }

    media = MediaFileUpload(str(FINAL), mimetype="video/mp4")

    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True
    ).execute()

    print("Uploaded:", uploaded.get("id"))


# =========================
# ARCHIVE INPUTS
# =========================

def archive_inputs():
    service = drive()

    files = service.files().list(
        q=f"'{INCOMING_FOLDER}' in parents and trashed=false",
        fields="files(id, name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives"
    ).execute().get("files", [])

    for f in files:
        service.files().update(
            fileId=f["id"],
            addParents=ARCHIVE_FOLDER,
            removeParents=INCOMING_FOLDER,
            supportsAllDrives=True
        ).execute()

    print("Archived inputs")


# =========================
# MAIN
# =========================

def main():
    print("AI VIDEO PIPELINE START")

    download_inputs()
    build_video()
    upload_output()
    archive_inputs()

    print("DONE")


if __name__ == "__main__":
    main()
