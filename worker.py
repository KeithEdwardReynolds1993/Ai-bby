import os
import io
import json
import hashlib
import sqlite3
import subprocess
import threading
import traceback
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account
import requests

app = Flask('worker')

TMP = Path("/tmp/ai_bby")
INPUT = TMP / "input"
OUTPUT = TMP / "output"
MUSIC_DIR = TMP / "music"
MERGED = TMP / "merged.mp4"
MERGED_CAPPED = TMP / "merged_capped.mp4"
DB_PATH = TMP / "generations.db"

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
ARCHIVE_FOLDER = os.getenv("GOOGLE_DRIVE_ARCHIVE_FOLDER_ID")
MUSIC_FOLDER = os.getenv("GOOGLE_DRIVE_MUSIC_FOLDER_ID")
MAX_CAPTION_CHARS = int(os.getenv("MAX_CAPTION_CHARS") or "60")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TIKTOK_ACCESS_TOKEN = os.getenv("TIKTOK_ACCESS_TOKEN", "")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID", "")
MAX_OUTPUT_DURATION = 30

pipeline_status = {"running": False, "log": [], "done": False, "error": None}
pipeline_lock = threading.Lock()


# ─── Database ─────────────────────────────────────────────────────────────────

def init_db():
    TMP.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            caption TEXT,
            vibe TEXT,
            music TEXT,
            cut_style TEXT,
            transition TEXT,
            speed REAL,
            drive_file_id TEXT,
            drive_file_name TEXT,
            drive_thumb_url TEXT,
            status TEXT DEFAULT 'awaiting_approval',
            posted_at TEXT,
            post_error TEXT,
            file_hash TEXT
        )
    """)
    try:
        conn.execute("ALTER TABLE generations ADD COLUMN file_hash TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()

def db_insert_generation(caption, vibe, music, cut_style, transition, speed,
                         drive_file_id, drive_file_name, file_hash):
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute("""
        INSERT INTO generations
        (created_at, caption, vibe, music, cut_style, transition, speed,
         drive_file_id, drive_file_name, file_hash, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'awaiting_approval')
    """, (
        datetime.utcnow().isoformat(),
        caption,
        vibe,
        music,
        cut_style,
        transition,
        speed,
        drive_file_id,
        drive_file_name,
        file_hash
    ))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id

def db_update_thumb(gen_id, thumb_url):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("UPDATE generations SET drive_thumb_url=? WHERE id=?", (thumb_url, gen_id))
    conn.commit()
    conn.close()

def db_update_status(gen_id, status, post_error=None):
    conn = sqlite3.connect(str(DB_PATH))
    posted_at = datetime.utcnow().isoformat() if status == "posted" else None
    conn.execute(
        "UPDATE generations SET status=?, posted_at=?, post_error=? WHERE id=?",
        (status, posted_at, post_error, gen_id)
    )
    conn.commit()
    conn.close()

def db_get_all_generations():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM generations ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_generation(gen_id):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM generations WHERE id=?", (gen_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def db_find_by_hash(file_hash):
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT id FROM generations WHERE file_hash=? ORDER BY id DESC LIMIT 1",
        (file_hash,)
    ).fetchone()
    conn.close()
    return row


# ─── Deduplication ────────────────────────────────────────────────────────────

def hash_files(files):
    ids = sorted([str(f["id"]) for f in files if f.get("id")])
    if not ids:
        raise ValueError("No valid file IDs found for hashing")
    return hashlib.sha256(",".join(ids).encode()).hexdigest()


# ─── Drive ────────────────────────────────────────────────────────────────────

def drive():
    creds = service_account.Credentials.from_service_account_info(
        SERVICE_ACCOUNT_INFO,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def ensure_dirs():
    INPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)

def clean_run_artifacts():
    if MERGED.exists():
        MERGED.unlink()
    if MERGED_CAPPED.exists():
        MERGED_CAPPED.unlink()
    for f in OUTPUT.glob("*.mp4"):
        f.unlink()

def list_incoming_files():
    service = drive()
    results = service.files().list(
        q=f"'{INCOMING_FOLDER}' in parents and trashed=false",
        fields="files(id, name, modifiedTime, mimeType, size, thumbnailLink)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives"
    ).execute()
    files = results.get("files", [])
    return [f for f in files if f.get("mimeType", "").startswith("video/")]

def list_music_files():
    if not MUSIC_FOLDER:
        return []
    service = drive()
    results = service.files().list(
        q=f"'{MUSIC_FOLDER}' in parents and trashed=false",
        fields="files(id, name, mimeType, size)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives"
    ).execute()
    files = results.get("files", [])
    audio_mimes = ("audio/", "video/mp4")
    return [f for f in files if any(f.get("mimeType", "").startswith(m) for m in audio_mimes)]

def download_file(service, file_obj, target):
    if target.exists():
        log(f"Already have {target.name}, skipping download")
        return
    req = service.files().get_media(fileId=file_obj["id"])
    with open(target, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    log(f"Downloaded: {file_obj['name']} -> {target.name}")

def upload_output(final_path):
    service = drive()
    file_metadata = {"name": final_path.name, "parents": [OUTPUT_FOLDER]}
    media = MediaFileUpload(str(final_path), mimetype="video/mp4")
    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name, thumbnailLink",
        supportsAllDrives=True
    ).execute()
    log(f"Uploaded: {uploaded.get('name')} ({uploaded.get('id')})")
    return uploaded

def get_drive_thumb(file_id):
    try:
        service = drive()
        f = service.files().get(
            fileId=file_id,
            fields="thumbnailLink",
            supportsAllDrives=True
        ).execute()
        return f.get("thumbnailLink", "")
    except Exception:
        return ""

def archive_clips(selected_files):
    if not ARCHIVE_FOLDER:
        return
    service = drive()
    for f in selected_files:
        try:
            service.files().update(
                fileId=f["id"],
                addParents=ARCHIVE_FOLDER,
                removeParents=INCOMING_FOLDER,
                supportsAllDrives=True
            ).execute()
            log(f"Archived: {f['name']}")
        except Exception as e:
            log(f"Archive failed for {f['name']}: {e}")


# ─── Logging / shell ──────────────────────────────────────────────────────────

def log(msg):
    print(msg)
    pipeline_status["log"].append(msg)

def run(cmd):
    log(">>> " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


# ─── FFmpeg helpers ───────────────────────────────────────────────────────────

def ffmpeg_escape(text):
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

def get_video_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True,
        text=True
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])

def get_vibe_filters(vibe):
    filters = {
        "normal": "",
        "hype": ",eq=contrast=1.3:brightness=0.05:saturation=1.5",
        "cinematic": ",eq=contrast=1.1:brightness=-0.05:saturation=0.7,vignette",
        "dreamy": ",gblur=sigma=1.5,eq=brightness=0.08:saturation=0.8",
        "gritty": ",eq=contrast=1.4:brightness=-0.1:saturation=0.6,noise=alls=20:allf=t",
        "retro": ",curves=vintage,eq=saturation=0.8,vignette",
    }
    return filters.get(vibe, "")

XFADE_TRANSITIONS = {
    "cut": None,
    "fade": "fade",
    "dissolve": "dissolve",
    "wipeleft": "wipeleft",
    "wiperight": "wiperight",
    "slideleft": "slideleft",
    "slideright": "slideright",
    "zoom": "zoom",
    "pixelize": "pixelize",
    "radial": "radial",
}


# ─── Beat analysis ────────────────────────────────────────────────────────────

def analyze_music(music_path):
    log(f"Analyzing music: {music_path.name}")
    y, sr = librosa.load(str(music_path), sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).flat[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    downbeat_times = beat_times[::4].tolist()
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_max = float(np.max(rms))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.mean(centroid))
    duration = float(len(y) / sr)
    log(f"BPM: {tempo:.1f} | Downbeats: {len(downbeat_times)} | Duration: {duration:.1f}s")
    return {
        "bpm": tempo,
        "beat_times": beat_times.tolist(),
        "downbeat_times": downbeat_times,
        "duration": duration,
        "energy_mean": energy_mean,
        "energy_max": energy_max,
        "brightness": brightness,
    }

def get_cut_times(music_analysis, cut_style, num_clips, total_video_duration):
    downbeats = music_analysis["downbeat_times"]
    beats = music_analysis["beat_times"]
    if cut_style == "every_beat":
        times = beats
    elif cut_style == "every_downbeat":
        times = downbeats
    elif cut_style == "every_2_downbeats":
        times = downbeats[::2]
    elif cut_style == "every_4_downbeats" or cut_style == "phrase":
        times = downbeats[::4]
    else:
        times = downbeats
    return [t for t in times if 0 < t < total_video_duration]


# ─── AI Director ──────────────────────────────────────────────────────────────

def ask_openai(prompt, clip_names, music_files=None):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    music_list = "\n".join([f["name"] for f in music_files]) if music_files else "No music available"
    system = (
        "You are a professional AI video director. Return ONLY a JSON object with:\n"
        "caption, speed, vibe, music_file, cut_style, transition, caption_fade_in, "
        "caption_fade_out, ken_burns, explanation. No markdown, no extra text."
    )
    user = f"Clips: {', '.join(clip_names)}\n\nMusic:\n{music_list}\n\nPrompt: {prompt}"
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        },
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.7
        },
        timeout=30
    )
    if not resp.ok:
        raise ValueError(f"OpenAI error {resp.status_code}: {resp.text}")
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


# ─── Posting ──────────────────────────────────────────────────────────────────

def post_to_tiktok(video_path, caption):
    if not TIKTOK_ACCESS_TOKEN:
        raise ValueError("TIKTOK_ACCESS_TOKEN not set")
    video_size = os.path.getsize(str(video_path))
    init_resp = requests.post(
        "https://open.tiktokapis.com/v2/post/publish/video/init/",
        headers={
            "Authorization": f"Bearer {TIKTOK_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "post_info": {
                "title": caption,
                "privacy_level": "PUBLIC_TO_EVERYONE",
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": video_size,
                "total_chunk_count": 1
            }
        }
    )
    if not init_resp.ok:
        raise ValueError(f"TikTok init failed: {init_resp.text}")
    data = init_resp.json()["data"]
    publish_id = data["publish_id"]
    upload_url = data["upload_url"]
    with open(str(video_path), "rb") as f:
        video_bytes = f.read()
    upload_resp = requests.put(
        upload_url,
        headers={
            "Content-Type": "video/mp4",
            "Content-Range": f"bytes 0-{len(video_bytes)-1}/{len(video_bytes)}"
        },
        data=video_bytes
    )
    if not upload_resp.ok:
        raise ValueError(f"TikTok upload failed: {upload_resp.text}")
    log(f"Posted to TikTok: {publish_id}")
    return publish_id

def post_to_instagram(video_path, caption):
    if not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_ACCOUNT_ID:
        raise ValueError("INSTAGRAM_ACCESS_TOKEN or INSTAGRAM_ACCOUNT_ID not set")
    container_resp = requests.post(
        f"https://graph.facebook.com/v19.0/{INSTAGRAM_ACCOUNT_ID}/media",
        params={
            "media_type": "REELS",
            "caption": caption,
            "access_token": INSTAGRAM_ACCESS_TOKEN
        }
    )
    if not container_resp.ok:
        raise ValueError(f"Instagram container failed: {container_resp.text}")
    creation_id = container_resp.json()["id"]
    publish_resp = requests.post(
        f"https://graph.facebook.com/v19.0/{INSTAGRAM_ACCOUNT_ID}/media_publish",
        params={
            "creation_id": creation_id,
            "access_token": INSTAGRAM_ACCESS_TOKEN
        }
    )
    if not publish_resp.ok:
        raise ValueError(f"Instagram publish failed: {publish_resp.text}")
    log(f"Posted to Instagram: {publish_resp.json()}")
    return publish_resp.json().get("id")

def post_generation(gen_id):
    gen = db_get_generation(gen_id)
    if not gen:
        raise ValueError(f"Generation {gen_id} not found")
    service = drive()
    video_tmp = TMP / f"post_{gen_id}.mp4"
    req = service.files().get_media(fileId=gen["drive_file_id"])
    with open(str(video_tmp), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    errors = []
    try:
        post_to_tiktok(video_tmp, gen["caption"])
    except Exception as e:
        errors.append(f"TikTok: {e}")
        log(f"TikTok error: {e}")
    try:
        post_to_instagram(video_tmp, gen["caption"])
    except Exception as e:
        errors.append(f"Instagram: {e}")
        log(f"Instagram error: {e}")
    if video_tmp.exists():
        video_tmp.unlink()
    if errors:
        db_update_status(gen_id, "post_error", post_error="; ".join(errors))
        raise ValueError("; ".join(errors))
    db_update_status(gen_id, "posted")


# ─── Core pipeline ────────────────────────────────────────────────────────────

def build_video(selected_files, caption, speed=1.0, vibe="normal",
                music_file_obj=None, cut_style="every_downbeat",
                transition="cut", caption_fade_in=0.5, caption_fade_out=1.0,
                ken_burns=False):
    ensure_dirs()
    clean_run_artifacts()

    if len(caption) > MAX_CAPTION_CHARS:
        raise ValueError(f"Caption too long ({len(caption)} chars). Max {MAX_CAPTION_CHARS}.")

    service = drive()

    local_clips = []
    for i, f in enumerate(selected_files):
        target = INPUT / f"clip{i+1:02d}.mp4"
        raw = INPUT / f"raw{i+1:02d}.mp4"
        download_file(service, f, raw)
        if not target.exists():
            log(f"Transcoding clip{i+1} to H264...")
            run([
                "ffmpeg", "-y", "-i", str(raw),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k", str(target)
            ])
        local_clips.append(target)

    music_path = None
    music_analysis = None
    if music_file_obj:
        music_path = MUSIC_DIR / music_file_obj["name"]
        download_file(service, music_file_obj, music_path)
        music_analysis = analyze_music(music_path)

    capped_clips = []
    for i, clip in enumerate(local_clips):
        dur = get_video_duration(clip)
        out = INPUT / f"capped{i+1:02d}.mp4"
        run(["ffmpeg", "-y", "-i", str(clip), "-t", str(MAX_OUTPUT_DURATION), "-c", "copy", str(out)])
        capped_clips.append(out)
        log(f"Clip{i+1}: {dur:.1f}s -> capped at {MAX_OUTPUT_DURATION}s")
    local_clips = capped_clips

    if abs(speed - 1.0) > 0.01:
        sped_clips = []
        for i, clip in enumerate(local_clips):
            out = INPUT / f"sped{i+1:02d}.mp4"
            pts = round(1.0 / speed, 4)
            run(["ffmpeg", "-y", "-i", str(clip), "-vf", f"setpts={pts}*PTS", "-an", str(out)])
            sped_clips.append(out)
        local_clips = sped_clips

    durations = [get_video_duration(c) for c in local_clips]
    total_duration = sum(durations)
    log(f"Total duration before cap: {total_duration:.1f}s")

    if music_analysis and cut_style != "cut":
        cut_times = get_cut_times(music_analysis, cut_style, len(local_clips), total_duration)
        cut_times = [t for t in cut_times if t < MAX_OUTPUT_DURATION]
        boundaries = [0.0] + cut_times + [min(total_duration, MAX_OUTPUT_DURATION)]
        segment_durations = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        trimmed_clips = []

        clip_sequence = (local_clips * ((len(segment_durations) // len(local_clips)) + 1))[:len(segment_durations)]
        for i, (clip, seg_dur) in enumerate(zip(clip_sequence, segment_durations)):
            out = INPUT / f"trim{i+1:02d}.mp4"
            clip_dur = get_video_duration(clip)
            if clip_dur < seg_dur:
                loop_count = int(seg_dur / clip_dur) + 1
                run([
                    "ffmpeg", "-y", "-stream_loop", str(loop_count), "-i", str(clip),
                    "-t", str(seg_dur), "-c", "copy", str(out)
                ])
            else:
                run(["ffmpeg", "-y", "-i", str(clip), "-t", str(seg_dur), "-c", "copy", str(out)])
            trimmed_clips.append(out)

        local_clips = trimmed_clips
        durations = segment_durations

    list_file = TMP / "list.txt"
    list_file.write_text("\n".join([f"file '{c}'" for c in local_clips]), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(MERGED)])

    raw_merged_duration = get_video_duration(MERGED)
    log(f"Merged: {raw_merged_duration:.1f}s | Cap: {MAX_OUTPUT_DURATION}s")

    if raw_merged_duration > MAX_OUTPUT_DURATION:
        run(["ffmpeg", "-y", "-i", str(MERGED), "-t", str(MAX_OUTPUT_DURATION), "-c", "copy", str(MERGED_CAPPED)])
        merged_input = MERGED_CAPPED
    else:
        merged_input = MERGED

    merged_duration = min(raw_merged_duration, MAX_OUTPUT_DURATION)
    log(f"Final duration: {merged_duration:.1f}s")

    cap_hash = hashlib.sha256((caption + vibe + str(speed) + transition).encode()).hexdigest()[:8]
    final_path = OUTPUT / f"final_{cap_hash}.mp4"
    safe_caption = ffmpeg_escape(caption)
    vibe_filter = get_vibe_filters(vibe)

    fade_in_end = caption_fade_in + 0.5
    fade_out_start = max(fade_in_end + 0.5, merged_duration - caption_fade_out - 0.5)
    alpha_expr = (
        f"if(lt(t,{caption_fade_in}),0,"
        f"if(lt(t,{fade_in_end}),(t-{caption_fade_in})/0.5,"
        f"if(lt(t,{fade_out_start}),1,"
        f"if(lt(t,{fade_out_start+0.5}),({fade_out_start+0.5}-t)/0.5,0))))"
    )

    drawtext = (
        f"drawtext=text='{safe_caption}':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontcolor=white:fontsize=54:x=(w-text_w)/2:y=(h-text_h)/2:"
        f"box=1:boxcolor=black@0.72:boxborderw=40:alpha='{alpha_expr}'"
    )

    kb_filter = ",scale=iw*1.05:ih*1.05,crop=iw/1.05:ih/1.05" if ken_burns else ""
    vf = (
        f"scale=1080:1920:force_original_aspect_ratio=decrease,"
        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black{kb_filter}{vibe_filter},{drawtext}"
    )

    if music_path and music_path.exists():
        log(f"Mixing music: {music_path.name}")
        music_trimmed = TMP / "music_trimmed.wav"
        music_dur = music_analysis["duration"] if music_analysis else 999

        if music_dur < merged_duration:
            loop_count = int(merged_duration / music_dur) + 1
            run([
                "ffmpeg", "-y", "-stream_loop", str(loop_count), "-i", str(music_path),
                "-t", str(merged_duration),
                "-af", f"afade=t=out:st={merged_duration - 2}:d=2",
                str(music_trimmed)
            ])
        else:
            run([
                "ffmpeg", "-y", "-i", str(music_path), "-t", str(merged_duration),
                "-af", f"afade=t=out:st={merged_duration - 2}:d=2",
                str(music_trimmed)
            ])

        run([
            "ffmpeg", "-y",
            "-i", str(merged_input),
            "-i", str(music_trimmed),
            "-t", str(merged_duration),
            "-filter_complex", f"[0:v]{vf}[vout];[1:a]volume=0.85,afade=t=in:st=0:d=1[aout]",
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            str(final_path)
        ])
    else:
        run([
            "ffmpeg", "-y",
            "-i", str(merged_input),
            "-t", str(merged_duration),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-an",
            str(final_path)
        ])

    actual = get_video_duration(final_path)
    log(f"Verified duration: {actual:.1f}s")
    if actual > MAX_OUTPUT_DURATION + 1:
        raise RuntimeError(f"Output {actual:.1f}s exceeds {MAX_OUTPUT_DURATION}s cap!")

    return final_path


def run_pipeline(selected_files, caption, speed=1.0, vibe="normal",
                 music_file_obj=None, cut_style="every_downbeat",
                 transition="cut", caption_fade_in=0.5, caption_fade_out=1.0,
                 ken_burns=False):
    global pipeline_status

    with pipeline_lock:
        pipeline_status = {"running": True, "log": [], "done": False, "error": None}
        try:
            log("Starting pipeline...")

            file_hash = hash_files(selected_files)
            existing = db_find_by_hash(file_hash)
            if existing:
                log(f"Already processed this clip set (Gen #{existing[0]}). Skipping.")
                pipeline_status["done"] = True
                return

            log(f"Caption: {caption} | Speed: {speed}x | Vibe: {vibe} | Cut: {cut_style} | Transition: {transition}")
            if music_file_obj:
                log(f"Music: {music_file_obj['name']}")

            final_path = build_video(
                selected_files, caption, speed, vibe,
                music_file_obj, cut_style, transition,
                caption_fade_in, caption_fade_out, ken_burns
            )

            uploaded = upload_output(final_path)
            drive_file_id = uploaded.get("id")
            drive_file_name = uploaded.get("name")
            music_name = music_file_obj["name"] if music_file_obj else ""

            gen_id = db_insert_generation(
                caption, vibe, music_name, cut_style, transition, speed,
                drive_file_id, drive_file_name, file_hash
            )

            time.sleep(3)
            thumb = get_drive_thumb(drive_file_id)
            if thumb:
                db_update_thumb(gen_id, thumb)

            archive_clips(selected_files)
            log(f"Done! Generation #{gen_id} awaiting approval.")
            pipeline_status["done"] = True
            pipeline_status["gen_id"] = gen_id

        except Exception as e:
            log(f"Error: {e}")
            log(traceback.format_exc())
            pipeline_status["error"] = str(e)
        finally:
            pipeline_status["running"] = False


# ─── HTML ─────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clip Studio</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0a0a0f; --surface: #13131a; --surface2: #1c1c27;
  --border: #2a2a3d; --accent: #c8ff00; --accent2: #7c4dff;
  --text: #e8e8f0; --muted: #5a5a7a; --danger: #ff4d6d;
  --warn: #ffb800; --success: #00e676; --radius: 12px;
}
body { font-family: 'Syne', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 40px 24px; }
.wrap { max-width: 1200px; margin: 0 auto; }
header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 48px; }
h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; color: var(--accent); }
.subtitle { font-family: 'DM Mono', monospace; font-size: .75rem; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; }
.layout { display: grid; grid-template-columns: 480px 1fr; gap: 48px; align-items: start; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
section { margin-bottom: 32px; }
.section-label { font-family: 'DM Mono', monospace; font-size: .7rem; letter-spacing: .15em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; display: flex; align-items: center; gap: 10px; }
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.section-header .section-label { margin-bottom: 0; flex: 1; }
#clip-list, #music-list { display: flex; flex-direction: column; gap: 8px; min-height: 60px; }
.clip-item { display: flex; align-items: center; gap: 14px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 18px; cursor: grab; transition: border-color .15s, background .15s; user-select: none; }
.clip-item:hover { border-color: var(--accent2); background: var(--surface2); }
.clip-item.dragging { opacity: .4; }
.clip-item.drag-over { border-color: var(--accent); }
.clip-item.unselected .clip-name { color: var(--muted); }
.music-item { display: flex; align-items: center; gap: 14px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 18px; cursor: pointer; transition: border-color .15s, background .15s; }
.music-item:hover { border-color: var(--accent2); background: var(--surface2); }
.music-item.selected { border-color: var(--accent); background: var(--surface2); }
.music-item.selected .music-name { color: var(--accent); }
.thumb { width: 60px; height: 60px; object-fit: cover; border-radius: 6px; flex-shrink: 0; background: var(--surface2); }
.music-icon { width: 40px; height: 40px; border-radius: 8px; background: var(--surface2); display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
.clip-check { width: 20px; height: 20px; accent-color: var(--accent); cursor: pointer; flex-shrink: 0; }
.drag-handle { color: var(--muted); flex-shrink: 0; }
.clip-name, .music-name { flex: 1; font-size: .9rem; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.clip-meta { font-family: 'DM Mono', monospace; font-size: .68rem; color: var(--muted); flex-shrink: 0; }
.caption-wrap { position: relative; }
#caption { width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 80px 16px 20px; font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: var(--text); outline: none; transition: border-color .15s; }
#caption:focus { border-color: var(--accent); }
#caption.too-long { border-color: var(--danger); }
.char-count { position: absolute; right: 14px; top: 50%; transform: translateY(-50%); font-family: 'DM Mono', monospace; font-size: .7rem; color: var(--muted); pointer-events: none; }
.char-count.warn { color: var(--danger); }
#ai-prompt { width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 20px; font-family: 'Syne', sans-serif; font-size: 1rem; color: var(--text); outline: none; transition: border-color .15s; margin-bottom: 12px; }
#ai-prompt:focus { border-color: var(--accent2); }
#ai-plan { background: var(--surface2); border: 1px solid var(--accent2); border-radius: var(--radius); padding: 16px 20px; font-family: 'DM Mono', monospace; font-size: .8rem; line-height: 1.8; color: var(--text); margin-bottom: 16px; display: none; }
#ai-plan .plan-label { color: var(--accent2); font-size: .65rem; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 8px; }
.btn-row { display: flex; gap: 12px; flex-wrap: wrap; }
#ask-ai-btn { background: var(--accent2); color: white; border: none; border-radius: var(--radius); padding: 14px 28px; font-family: 'Syne', sans-serif; font-size: .95rem; font-weight: 800; cursor: pointer; transition: opacity .15s; }
#ask-ai-btn:hover:not(:disabled) { opacity: .8; }
#ask-ai-btn:disabled { opacity: .35; cursor: not-allowed; }
#run-btn { display: flex; align-items: center; gap: 12px; background: var(--accent); color: #0a0a0f; border: none; border-radius: var(--radius); padding: 14px 28px; font-family: 'Syne', sans-serif; font-size: .95rem; font-weight: 800; cursor: pointer; transition: opacity .15s, transform .1s; }
#run-btn:hover:not(:disabled) { opacity: .88; transform: translateY(-1px); }
#run-btn:disabled { opacity: .35; cursor: not-allowed; }
.spinner { width: 16px; height: 16px; border: 2px solid #0a0a0f; border-top-color: transparent; border-radius: 50%; animation: spin .7s linear infinite; display: none; }
@keyframes spin { to { transform: rotate(360deg); } }
#log-wrap { display: none; margin-top: 24px; }
#log { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; font-family: 'DM Mono', monospace; font-size: .75rem; line-height: 1.8; color: var(--text); max-height: 240px; overflow-y: auto; white-space: pre-wrap; }
.empty { padding: 32px; text-align: center; font-family: 'DM Mono', monospace; font-size: .8rem; color: var(--muted); border: 1px dashed var(--border); border-radius: var(--radius); }
.refresh-btn { background: none; border: 1px solid var(--border); border-radius: 8px; color: var(--muted); font-family: 'DM Mono', monospace; font-size: .7rem; padding: 6px 14px; cursor: pointer; transition: border-color .15s, color .15s; }
.refresh-btn:hover { border-color: var(--accent); color: var(--accent); }
.no-music-btn { background: none; border: 1px dashed var(--border); border-radius: var(--radius); color: var(--muted); font-family: 'DM Mono', monospace; font-size: .75rem; padding: 10px 18px; cursor: pointer; width: 100%; text-align: left; transition: border-color .15s; }
.no-music-btn:hover, .no-music-btn.selected { border-color: var(--accent); color: var(--accent); }
.gen-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }
.gen-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; display: flex; flex-direction: column; transition: border-color .15s; }
.gen-card:hover { border-color: var(--border); }
.gen-thumb { width: 100%; height: 160px; object-fit: cover; background: var(--surface2); display: block; }
.gen-thumb-placeholder { width: 100%; height: 160px; background: var(--surface2); display: flex; align-items: center; justify-content: center; font-size: 2.5rem; }
.gen-body { padding: 14px 16px; flex: 1; display: flex; flex-direction: column; gap: 6px; }
.gen-caption { font-size: .9rem; font-weight: 700; line-height: 1.3; }
.gen-status { display: inline-flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: .65rem; font-weight: 700; padding: 3px 8px; border-radius: 20px; width: fit-content; }
.status-awaiting_approval { background: rgba(255,184,0,.15); color: var(--warn); border: 1px solid rgba(255,184,0,.3); }
.status-approved { background: rgba(124,77,255,.15); color: var(--accent2); border: 1px solid rgba(124,77,255,.3); }
.status-posting { background: rgba(0,230,118,.1); color: var(--success); border: 1px solid rgba(0,230,118,.2); }
.status-posted { background: rgba(0,230,118,.15); color: var(--success); border: 1px solid rgba(0,230,118,.3); }
.status-post_error { background: rgba(255,77,109,.15); color: var(--danger); border: 1px solid rgba(255,77,109,.3); }
.gen-meta { font-family: 'DM Mono', monospace; font-size: .65rem; color: var(--muted); line-height: 1.6; }
.gen-date { font-family: 'DM Mono', monospace; font-size: .62rem; color: var(--muted); }
.gen-error { font-family: 'DM Mono', monospace; font-size: .65rem; color: var(--danger); word-break: break-all; }
.gen-actions { display: flex; gap: 8px; margin-top: 8px; }
.approve-btn { flex: 1; background: var(--accent); color: #0a0a0f; border: none; border-radius: 8px; padding: 9px; font-family: 'Syne', sans-serif; font-size: .82rem; font-weight: 800; cursor: pointer; transition: opacity .15s; }
.approve-btn:hover { opacity: .85; }
.post-btn { flex: 1; background: var(--accent2); color: white; border: none; border-radius: 8px; padding: 9px; font-family: 'Syne', sans-serif; font-size: .82rem; font-weight: 800; cursor: pointer; transition: opacity .15s; }
.post-btn:hover { opacity: .85; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Clip Studio</h1>
    <span class="subtitle">AI Video Director</span>
  </header>

  <div class="layout">
    <div>
      <section>
        <div class="section-header">
          <div class="section-label">Clips from Drive</div>
          <button class="refresh-btn" id="refresh-btn">&#8635; Refresh</button>
        </div>
        <div id="clip-list"><div class="empty">Loading clips...</div></div>
      </section>

      <section>
        <div class="section-header">
          <div class="section-label">Music</div>
          <button class="refresh-btn" id="refresh-music-btn">&#8635; Refresh</button>
        </div>
        <div id="music-list"><div class="empty">Loading music...</div></div>
      </section>

      <section>
        <div class="section-label">Title / Caption</div>
        <div class="caption-wrap">
          <input id="caption" type="text" placeholder="Enter title..." maxlength="100">
          <span class="char-count" id="char-count">0 / 60</span>
        </div>
      </section>

      <section>
        <div class="section-label">AI Director</div>
        <input id="ai-prompt" type="text" placeholder='e.g. "hype reel, fast cuts"'>
        <div id="ai-plan"><div class="plan-label">AI Plan</div><div id="ai-plan-text"></div></div>
        <div class="btn-row">
          <button id="ask-ai-btn">Ask AI</button>
          <button id="run-btn" disabled>
            <span class="spinner" id="spinner"></span>
            <span id="btn-label">Run Pipeline</span>
          </button>
        </div>
      </section>

      <div id="log-wrap">
        <div class="section-label" style="margin-bottom:14px;">Pipeline Log</div>
        <div id="log"></div>
      </div>
    </div>

    <div>
      <section>
        <div class="section-header">
          <div class="section-label">Generations</div>
          <button class="refresh-btn" id="refresh-gens-btn">&#8635; Refresh</button>
        </div>
        <div id="gen-grid" class="gen-grid"><div class="empty">No generations yet.</div></div>
      </section>
    </div>
  </div>
</div>

<script>
var MAX_CHARS = 60;
var allFiles = [];
var allMusic = [];
var dragSrc = null;
var selectedMusicId = null;
var aiSettings = {
  caption: "",
  speed: 1.0,
  vibe: "normal",
  cut_style: "every_downbeat",
  transition: "cut",
  caption_fade_in: 0.5,
  caption_fade_out: 1.0,
  ken_burns: false
};

function loadClips() {
  var list = document.getElementById("clip-list");
  list.innerHTML = "<div class='empty'>Fetching from Drive...</div>";
  fetch("/api/clips")
    .then(function(r) { return r.json(); })
    .then(function(data) {
      allFiles = data.files || [];
      renderClips();
      updateRunBtn();
    })
    .catch(function() {
      list.innerHTML = "<div class='empty'>Failed to load clips.</div>";
    });
}

function renderClips() {
  var list = document.getElementById("clip-list");
  if (!allFiles.length) {
    list.innerHTML = "<div class='empty'>No video files found.</div>";
    return;
  }

  list.innerHTML = "";
  allFiles.forEach(function(f) {
    var el = document.createElement("div");
    el.className = "clip-item";
    el.draggable = true;
    el.dataset.id = f.id;

    var mb = f.size ? (parseInt(f.size, 10) / 1048576).toFixed(1) + " MB" : "";
    var thumb = f.thumbnailLink
      ? "<img class='thumb' src='" + f.thumbnailLink + "'>"
      : "<div class='thumb'></div>";

    el.innerHTML =
      "<span class='drag-handle'>::</span>" +
      thumb +
      "<input type='checkbox' class='clip-check' checked>" +
      "<span class='clip-name'>" + f.name + "</span>" +
      "<span class='clip-meta'>" + mb + "</span>";

    el.querySelector(".clip-check").addEventListener("change", function() {
      el.classList.toggle("unselected", !this.checked);
      updateRunBtn();
    });

    el.addEventListener("dragstart", function(e) {
      dragSrc = el;
      el.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
    });

    el.addEventListener("dragover", function(e) {
      e.preventDefault();
      document.querySelectorAll(".clip-item").forEach(function(x) {
        x.classList.remove("drag-over");
      });
      el.classList.add("drag-over");
    });

    el.addEventListener("drop", function(e) {
      e.preventDefault();
      if (dragSrc === el) return;
      var items = Array.from(list.querySelectorAll(".clip-item"));
      var si = items.indexOf(dragSrc);
      var ti = items.indexOf(el);
      if (si < ti) list.insertBefore(dragSrc, el.nextSibling);
      else list.insertBefore(dragSrc, el);
      syncOrder();
    });

    el.addEventListener("dragend", function() {
      document.querySelectorAll(".clip-item").forEach(function(x) {
        x.classList.remove("dragging", "drag-over");
      });
    });

    list.appendChild(el);
  });
}

function loadMusic() {
  fetch("/api/music")
    .then(function(r) { return r.json(); })
    .then(function(data) {
      allMusic = data.files || [];
      renderMusic();
    })
    .catch(function() {
      var list = document.getElementById("music-list");
      list.innerHTML = "<div class='empty'>Failed to load music.</div>";
    });
}

function renderMusic() {
  var list = document.getElementById("music-list");
  list.innerHTML = "";

  var noMusic = document.createElement("button");
  noMusic.className = "no-music-btn" + (selectedMusicId === null ? " selected" : "");
  noMusic.textContent = "No music (video only)";
  noMusic.addEventListener("click", function() {
    selectedMusicId = null;
    renderMusic();
  });
  list.appendChild(noMusic);

  if (!allMusic.length) {
    var em = document.createElement("div");
    em.className = "empty";
    em.textContent = "No music files found.";
    list.appendChild(em);
    return;
  }

  allMusic.forEach(function(f) {
    var el = document.createElement("div");
    el.className = "music-item" + (selectedMusicId === f.id ? " selected" : "");
    el.dataset.id = f.id;

    var mb = f.size ? (parseInt(f.size, 10) / 1048576).toFixed(1) + " MB" : "";
    el.innerHTML =
      "<div class='music-icon'>~</div>" +
      "<span class='music-name'>" + f.name + "</span>" +
      "<span class='clip-meta'>" + mb + "</span>";

    el.addEventListener("click", function() {
      selectedMusicId = f.id;
      renderMusic();
    });

    list.appendChild(el);
  });
}

function loadGenerations() {
  fetch("/api/generations")
    .then(function(r) { return r.json(); })
    .then(function(data) {
      renderGenerations(data.generations || []);
    })
    .catch(function() {
      var grid = document.getElementById("gen-grid");
      grid.innerHTML = "<div class='empty'>Failed to load generations.</div>";
    });
}

function statusLabel(s) {
  return {
    awaiting_approval: "⏳ Awaiting Approval",
    approved: "✅ Approved",
    posting: "📤 Posting...",
    posted: "🎉 Posted",
    post_error: "❌ Post Error"
  }[s] || s;
}

function renderGenerations(gens) {
  var grid = document.getElementById("gen-grid");
  if (!gens.length) {
    grid.innerHTML = "<div class='empty'>No generations yet.</div>";
    return;
  }

  grid.innerHTML = "";

  gens.forEach(function(g) {
    var card = document.createElement("div");
    card.className = "gen-card";
    card.dataset.id = g.id;

    var thumbHtml;
    if (g.drive_thumb_url) {
      thumbHtml = '<img class="gen-thumb" src="' + g.drive_thumb_url + '">';
    } else {
      thumbHtml = '<div class="gen-thumb-placeholder">🎬</div>';
    }

    var date = new Date(g.created_at + "Z").toLocaleString();
    var metaParts = [];
    if (g.vibe) metaParts.push("Vibe: " + g.vibe);
    if (g.music) metaParts.push(g.music);
    if (g.cut_style) metaParts.push(g.cut_style);
    if (g.speed) metaParts.push(g.speed + "x");

    var actionsHtml = "";
    if (g.status === "awaiting_approval") {
      actionsHtml = "<div class='gen-actions'><button class='approve-btn' onclick='approveGen(" + g.id + ")'>✓ Approve</button></div>";
    } else if (g.status === "approved") {
      actionsHtml = "<div class='gen-actions'><button class='post-btn' onclick='postGen(" + g.id + ")'>🚀 Post Now</button></div>";
    } else if (g.status === "post_error") {
      actionsHtml = "<div class='gen-actions'><button class='post-btn' onclick='postGen(" + g.id + ")'>🔁 Retry</button></div>";
    }

    card.innerHTML =
      thumbHtml +
      "<div class='gen-body'>" +
        "<div class='gen-caption'>" + (g.caption || "") + "</div>" +
        "<span class='gen-status status-" + g.status + "'>" + statusLabel(g.status) + "</span>" +
        "<div class='gen-meta'>" + metaParts.join(" &bull; ") + "</div>" +
        (g.post_error ? "<div class='gen-error'>" + g.post_error + "</div>" : "") +
        "<div class='gen-date'>" + date + "</div>" +
        actionsHtml +
      "</div>";

    grid.appendChild(card);

    var img = card.querySelector(".gen-thumb");
    if (img) {
      img.onerror = function() {
        this.outerHTML = '<div class="gen-thumb-placeholder">🎬</div>';
      };
    }
  });
}

function approveGen(id) {
  fetch("/api/generations/" + id + "/approve", { method: "POST" })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.ok) loadGenerations();
      else alert("Error: " + data.error);
    });
}

function postGen(id) {
  if (!confirm("Post to TikTok + Instagram now?")) return;

  var card = document.querySelector(".gen-card[data-id='" + id + "']");
  if (card) {
    var a = card.querySelector(".gen-actions");
    if (a) {
      a.innerHTML = "<span style='color:var(--success);font-family:monospace;font-size:.8rem'>Posting...</span>";
    }
  }

  fetch("/api/generations/" + id + "/post", { method: "POST" })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      setTimeout(loadGenerations, 2000);
      if (data.error) alert("Error: " + data.error);
    });
}

function syncOrder() {
  var ids = Array.from(document.querySelectorAll(".clip-item")).map(function(el) {
    return el.dataset.id;
  });
  allFiles = ids.map(function(id) {
    return allFiles.find(function(f) { return f.id === id; });
  });
}

function updateRunBtn() {
  var caption = document.getElementById("caption").value.trim();
  var selected = getSelectedFiles();
  document.getElementById("run-btn").disabled = !caption || caption.length > MAX_CHARS || selected.length === 0;
}

function getSelectedFiles() {
  return Array.from(document.querySelectorAll(".clip-item"))
    .filter(function(el) { return el.querySelector(".clip-check").checked; })
    .map(function(el) { return allFiles.find(function(f) { return f.id === el.dataset.id; }); })
    .filter(Boolean);
}

function getSelectedMusic() {
  if (!selectedMusicId) return null;
  return allMusic.find(function(f) { return f.id === selectedMusicId; }) || null;
}

function askAI() {
  var prompt = document.getElementById("ai-prompt").value.trim();
  if (!prompt) return;

  var btn = document.getElementById("ask-ai-btn");
  btn.disabled = true;
  btn.textContent = "Thinking...";

  fetch("/api/ai-plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: prompt,
      clip_names: getSelectedFiles().map(function(f) { return f.name; })
    })
  })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      btn.disabled = false;
      btn.textContent = "Ask AI";

      if (data.error) {
        alert("AI error: " + data.error);
        return;
      }

      aiSettings = {
        caption: data.caption,
        speed: data.speed,
        vibe: data.vibe,
        cut_style: data.cut_style || "every_downbeat",
        transition: data.transition || "cut",
        caption_fade_in: data.caption_fade_in || 0.5,
        caption_fade_out: data.caption_fade_out || 1.0,
        ken_burns: data.ken_burns || false
      };

      document.getElementById("caption").value = data.caption;
      updateRunBtn();

      if (data.music_file) {
        var match = allMusic.find(function(f) { return f.name === data.music_file; });
        if (match) {
          selectedMusicId = match.id;
          renderMusic();
        }
      }

      document.getElementById("ai-plan-text").innerHTML =
        "<b>Caption:</b> " + data.caption +
        "<br><b>Music:</b> " + (data.music_file || "none") +
        "<br><b>Cut:</b> " + aiSettings.cut_style + " &bull; <b>Transition:</b> " + aiSettings.transition +
        "<br><b>Vibe:</b> " + data.vibe + " &bull; <b>Speed:</b> " + data.speed + "x &bull; <b>Ken Burns:</b> " + (data.ken_burns ? "yes" : "no") +
        "<br><b>Reasoning:</b> " + data.explanation;

      document.getElementById("ai-plan").style.display = "block";
    })
    .catch(function() {
      btn.disabled = false;
      btn.textContent = "Ask AI";
      alert("Failed to reach AI.");
    });
}

function runPipeline() {
  var caption = document.getElementById("caption").value.trim();
  var selected = getSelectedFiles();
  var music = getSelectedMusic();

  document.getElementById("run-btn").disabled = true;
  document.getElementById("spinner").style.display = "block";
  document.getElementById("btn-label").textContent = "Running...";
  document.getElementById("log-wrap").style.display = "block";
  document.getElementById("log").textContent = "";

  fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      files: selected,
      caption: caption,
      music_file: music,
      speed: aiSettings.speed,
      vibe: aiSettings.vibe,
      cut_style: aiSettings.cut_style,
      transition: aiSettings.transition,
      caption_fade_in: aiSettings.caption_fade_in,
      caption_fade_out: aiSettings.caption_fade_out,
      ken_burns: aiSettings.ken_burns
    })
  }).then(function() {
    pollLog();
  });
}

function pollLog() {
  var logEl = document.getElementById("log");
  var interval = setInterval(function() {
    fetch("/api/status")
      .then(function(r) { return r.json(); })
      .then(function(data) {
        logEl.textContent = data.log.join(String.fromCharCode(10));
        logEl.scrollTop = logEl.scrollHeight;

        if (!data.running) {
          clearInterval(interval);
          document.getElementById("spinner").style.display = "none";
          document.getElementById("run-btn").disabled = false;
          document.getElementById("btn-label").textContent = data.done ? "Done - Run Again" : "Run Pipeline";
          if (data.done) setTimeout(loadGenerations, 1500);
        }
      });
  }, 1000);
}

document.getElementById("caption").addEventListener("input", function() {
  var val = this.value;
  var count = document.getElementById("char-count");
  count.textContent = val.length + " / 60";
  count.classList.toggle("warn", val.length > MAX_CHARS);
  this.classList.toggle("too-long", val.length > MAX_CHARS);
  updateRunBtn();
});

document.getElementById("refresh-btn").addEventListener("click", loadClips);
document.getElementById("refresh-music-btn").addEventListener("click", loadMusic);
document.getElementById("refresh-gens-btn").addEventListener("click", loadGenerations);
document.getElementById("run-btn").addEventListener("click", runPipeline);
document.getElementById("ask-ai-btn").addEventListener("click", askAI);
document.getElementById("ai-prompt").addEventListener("keydown", function(e) {
  if (e.key === "Enter") askAI();
});

loadClips();
loadMusic();
loadGenerations();
setInterval(loadGenerations, 30000);
</script>
</body>
</html>"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return app.response_class(HTML.encode("utf-8"), mimetype="text/html")

@app.route("/api/clips")
def api_clips():
    try:
        return jsonify({"files": list_incoming_files()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/music")
def api_music():
    try:
        return jsonify({"files": list_music_files()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai-plan", methods=["POST"])
def api_ai_plan():
    try:
        data = request.json
        music_files = list_music_files()
        plan = ask_openai(data.get("prompt", ""), data.get("clip_names", []), music_files)
        return jsonify(plan)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/run", methods=["POST"])
def api_run():
    global pipeline_status
    if pipeline_status.get("running"):
        return jsonify({"error": "Already running"}), 409

    data = request.json or {}
    selected_files = data.get("files", [])
    caption = data.get("caption", "").strip()

    if not selected_files or not caption:
        return jsonify({"error": "Missing files or caption"}), 400

    try:
        file_hash = hash_files(selected_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    existing = db_find_by_hash(file_hash)
    if existing:
        return jsonify({"ok": True, "skipped": True, "existing_gen_id": existing[0]})

    thread = threading.Thread(
        target=run_pipeline,
        args=(
            selected_files,
            caption,
            float(data.get("speed") or 1.0),
            data.get("vibe") or "normal",
            data.get("music_file"),
            data.get("cut_style") or "every_downbeat",
            data.get("transition") or "cut",
            float(data.get("caption_fade_in") or 0.5),
            float(data.get("caption_fade_out") or 1.0),
            bool(data.get("ken_burns") or False),
        ),
        daemon=True
    )
    thread.start()
    return jsonify({"ok": True})

@app.route("/api/status")
def api_status():
    return jsonify(pipeline_status)

@app.route("/api/generations")
def api_generations():
    try:
        return jsonify({"generations": db_get_all_generations()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/generations/<int:gen_id>/approve", methods=["POST"])
def api_approve(gen_id):
    try:
        if not db_get_generation(gen_id):
            return jsonify({"error": "Not found"}), 404
        db_update_status(gen_id, "approved")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/generations/<int:gen_id>/post", methods=["POST"])
def api_post(gen_id):
    try:
        if not db_get_generation(gen_id):
            return jsonify({"error": "Not found"}), 404
        db_update_status(gen_id, "posting")
        thread = threading.Thread(target=_post_thread, args=(gen_id,), daemon=True)
        thread.start()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _post_thread(gen_id):
    try:
        post_generation(gen_id)
    except Exception as e:
        print(f"Post thread error: {e}")


if __name__ == "__main__":
    ensure_dirs()
    init_db()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 5000), debug=False)
