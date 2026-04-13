import os
import io
import json
import hashlib
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account


# =========================
# PATHS
# =========================

TMP = Path("/tmp/ai_bby")
INPUT = TMP / "input"
OUTPUT = TMP / "output"
MERGED = TMP / "merged.mp4"
STATE_LOCAL = TMP / "processed_batches.json"


# =========================
# ENV
# =========================

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))

INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
ARCHIVE_FOLDER = os.getenv("GOOGLE_DRIVE_ARCHIVE_FOLDER_ID")

CAPTION = os.getenv("CAPTION_TEXT", "Made with AI")
STATE_FILENAME = os.getenv("STATE_FILENAME", "processed_batches.json")
MIN_FILE_AGE_SECONDS = int(os.getenv("MIN_FILE_AGE_SECONDS", "30"))
MAX_CAPTION_CHARS = int(os.getenv("MAX_CAPTION_CHARS", "60"))


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
# HELPERS
# =========================

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def ensure_dirs():
    """Create tmp dirs if they don't exist — never wipes them."""
    INPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

def clean_run_artifacts():
    """Only clears per-run files (merged + output), not downloaded inputs."""
    if MERGED.exists():
        MERGED.unlink()
    for f in OUTPUT.glob("*.mp4"):
        f.unlink()

def run(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)

def ffmpeg_escape(text: str) -> str:
    return (
        text.replace("\\", r"\\\\")
            .replace("'", r"'\''")
            .replace(":", r"\:")
            .replace("%", r"\%")
            .replace(",", r"\,")
            .replace("[", r"\[")
            .replace("]", r"\]")
            .replace("\n", " ")
            .replace("\r", "")
    )


# =========================
# DRIVE FILE LISTING
# =========================

def list_incoming_files():
    service = drive()

    results = service.files().list(
        q=f"'{INCOMING_FOLDER}' in parents and trashed=false",
        fields="files(id, name, parents, md5Checksum, modifiedTime, mimeType)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives"
    ).execute()

    files = results.get("files", [])
    video_files = [f for f in files if f.get("mimeType", "").startswith("video/")]

    print(f"Found {len(video_files)} incoming video file(s)")
    return video_files


def file_is_old_enough(file_obj):
    modified = file_obj.get("modifiedTime")
    if not modified:
        return True
    dt = datetime.fromisoformat(modified.replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - dt).total_seconds()
    return age >= MIN_FILE_AGE_SECONDS


def all_files_stable(files):
    return all(file_is_old_enough(f) for f in files)


# =========================
# STATE FILE IN DRIVE
# =========================

def find_state_file(service):
    results = service.files().list(
        q=(
            f"name='{STATE_FILENAME}' and "
            f"'{OUTPUT_FOLDER}' in parents and trashed=false"
        ),
        fields="files(id, name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives"
    ).execute()
    files = results.get("files", [])
    return files[0] if files else None


def load_state():
    service = drive()
    state_file = find_state_file(service)

    if not state_file:
        print("No remote state file found. Starting fresh.")
        return {"processed_batches": {}, "processing_batches": {}}, None

    request = service.files().get_media(fileId=state_file["id"])
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    fh.seek(0)
    raw = fh.read().decode("utf-8").strip()

    if not raw:
        return {"processed_batches": {}, "processing_batches": {}}, state_file["id"]

    state = json.loads(raw)
    state.setdefault("processed_batches", {})
    state.setdefault("processing_batches", {})

    print("Loaded remote state file")
    return state, state_file["id"]


def save_state(state, state_file_id=None):
    service = drive()
    STATE_LOCAL.write_text(json.dumps(state, indent=2), encoding="utf-8")
    media = MediaFileUpload(str(STATE_LOCAL), mimetype="application/json")

    if state_file_id:
        service.files().update(
            fileId=state_file_id,
            media_body=media,
            supportsAllDrives=True
        ).execute()
        print("Updated remote state file")
        return state_file_id

    metadata = {
        "name": STATE_FILENAME,
        "parents": [OUTPUT_FOLDER],
        "mimeType": "application/json"
    }
    created = service.files().create(
        body=metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True
    ).execute()
    print("Created remote state file")
    return created["id"]


# =========================
# BATCH KEY
# =========================

def generate_batch_key(files):
    files_sorted = sorted(files, key=lambda f: (f["name"], f["id"]))
    key_string = "|".join(
        f"{f['id']}:{f.get('md5Checksum', 'no_md5')}"
        for f in files_sorted
    )
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()


# =========================
# OUTPUT CHECK
# =========================

def output_name_for_batch(batch_key):
    return f"final_{batch_key[:12]}.mp4"


def output_exists(batch_key):
    service = drive()
    filename = output_name_for_batch(batch_key)

    results = service.files().list(
        q=(
            f"name='{filename}' and "
            f"'{OUTPUT_FOLDER}' in parents and trashed=false"
        ),
        fields="files(id, name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives"
    ).execute()

    files = results.get("files", [])
    return files[0] if files else None


# =========================
# DOWNLOAD INPUTS
# =========================

def download_inputs(files):
    service = drive()
    local_clips = []
    files_sorted = sorted(files, key=lambda f: (f["name"], f["id"]))

    for i, f in enumerate(files_sorted):
        target = INPUT / f"clip{i+1}.mp4"

        # Skip re-downloading if already present and same checksum
        if target.exists():
            print(f"Already have {target.name}, skipping download")
            local_clips.append(target)
            continue

        request = service.files().get_media(fileId=f["id"])
        with open(target, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        local_clips.append(target)
        print("Downloaded:", f["name"], "->", target.name)

    return local_clips


# =========================
# FFMPEG PIPELINE
# =========================

def build_video(batch_key):
    clips = sorted(INPUT.glob("*.mp4"))

    if not clips:
        raise Exception("No downloaded clips found")

    if len(CAPTION) > MAX_CAPTION_CHARS:
        raise ValueError(
            f"Caption too long ({len(CAPTION)} chars). "
            f"Keep it under {MAX_CAPTION_CHARS} or set MAX_CAPTION_CHARS env var."
        )

    list_file = TMP / "list.txt"
    list_file.write_text(
        "\n".join([f"file '{c}'" for c in clips]),
        encoding="utf-8"
    )

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(MERGED)
    ])

    final_path = OUTPUT / output_name_for_batch(batch_key)
    safe_caption = ffmpeg_escape(CAPTION)

    pad = 40  # padding around text inside box

    drawtext = (
        f"drawtext=text='{safe_caption}':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
        f"fontcolor=white:"
        f"fontsize=54:"
        f"x=(w-text_w)/2:"
        f"y=(h*0.82)-(text_h/2):"
        f"box=1:"
        f"boxcolor=black@0.72:"
        f"boxborderw={pad}"
    )

    vf = (
        f"scale=1080:1920:force_original_aspect_ratio=decrease,"
        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
        f"{drawtext}"
    )

    run([
        "ffmpeg", "-y",
        "-i", str(MERGED),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-an",
        str(final_path)
    ])

    return final_path


# =========================
# UPLOAD OUTPUT
# =========================

def upload_output(final_path, batch_key):
    service = drive()

    file_metadata = {
        "name": final_path.name,
        "parents": [OUTPUT_FOLDER]
    }

    media = MediaFileUpload(str(final_path), mimetype="video/mp4")

    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name",
        supportsAllDrives=True
    ).execute()

    print("Uploaded output:", uploaded.get("name"), uploaded.get("id"))
    return uploaded


# =========================
# ARCHIVE INPUTS
# =========================

def archive_inputs(files):
    service = drive()

    for f in files:
        service.files().update(
            fileId=f["id"],
            addParents=ARCHIVE_FOLDER,
            removeParents=INCOMING_FOLDER,
            supportsAllDrives=True,
            fields="id, parents"
        ).execute()
        print("Archived:", f["name"])

    print("Archived all rendered inputs")


# =========================
# MAIN
# =========================

def main():
    print("AI VIDEO PIPELINE START")
    ensure_dirs()          # create dirs if missing, never wipes
    clean_run_artifacts()  # only clears merged + output mp4s from last run

    files = list_incoming_files()
    if not files:
        print("No input videos found. Exiting.")
        return

    if not all_files_stable(files):
        print(f"Files are too new. Waiting until they are at least {MIN_FILE_AGE_SECONDS}s old.")
        return

    batch_key = generate_batch_key(files)
    print("BATCH KEY:", batch_key)

    state, state_file_id = load_state()
    processed = state["processed_batches"]
    processing = state["processing_batches"]

    if batch_key in processed:
        print("Batch already processed. Skipping.")
        return

    existing_output = output_exists(batch_key)
    if existing_output:
        print("Output already exists in Drive. Marking batch processed and skipping.")
        processed[batch_key] = {
            "processed_at": utc_now_iso(),
            "output_file_id": existing_output["id"],
            "output_name": existing_output["name"]
        }
        state_file_id = save_state(state, state_file_id)
        return

    if batch_key in processing:
        print("Batch already in progress. Skipping to avoid duplicate render.")
        return

    processing[batch_key] = {
        "started_at": utc_now_iso(),
        "file_ids": [f["id"] for f in files],
        "file_names": [f["name"] for f in files]
    }
    state_file_id = save_state(state, state_file_id)

    try:
        download_inputs(files)       # skips files already on disk
        final_path = build_video(batch_key)
        uploaded = upload_output(final_path, batch_key)
        archive_inputs(files)

        processed[batch_key] = {
            "processed_at": utc_now_iso(),
            "output_file_id": uploaded["id"],
            "output_name": uploaded["name"]
        }
        print("Marked batch processed")

    finally:
        processing.pop(batch_key, None)
        state_file_id = save_state(state, state_file_id)

    print("DONE")


if __name__ == "__main__":
    main()
