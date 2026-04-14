import os
import io
import json
import subprocess
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account

app = Flask("worker_horizontal")

TMP = Path("/tmp/ai_horiz")
INPUT = TMP / "input"
OUTPUT = TMP / "output"
MUSIC_DIR = TMP / "music"

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
INTERVIEW_FOLDER = os.getenv("GOOGLE_DRIVE_INTERVIEW_FOLDER_ID", "")
BROLL_FOLDER = os.getenv("GOOGLE_DRIVE_BROLL_FOLDER_ID", "")
TRANSCRIPTIONS_FOLDER = os.getenv("GOOGLE_DRIVE_TRANSCRIPTIONS_FOLDER_ID", "")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
MUSIC_FOLDER = os.getenv("GOOGLE_DRIVE_MUSIC_FOLDER_ID", "")
GUIDE_FOLDER = os.getenv("GOOGLE_DRIVE_GUIDE_FOLDER_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_CAPTION_CHARS = int(os.getenv("MAX_CAPTION_CHARS") or "80")
SEGMENT_DURATION = 30

pipeline_status = {"running": False, "log": [], "done": False, "error": None}
pipeline_lock = threading.Lock()
stop_event = threading.Event()


def plog(msg):
    print(msg)
    pipeline_status["log"].append(str(msg))


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
    for f in list(INPUT.glob("*")) + list(OUTPUT.glob("*")) + list(MUSIC_DIR.glob("*")):
        if f.is_file():
            f.unlink()


def run_cmd(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


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


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, check=True
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def wrap_caption(text, max_chars_per_line=32):
    words = text.split()
    lines, current = [], []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > max_chars_per_line and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


# =============================================================================
# DRIVE
# =============================================================================

def get_latest_video():
    service = drive()
    results = service.files().list(
        q=f"'{INTERVIEW_FOLDER}' in parents and trashed=false",
        fields="files(id,name,mimeType,size,thumbnailLink,modifiedTime)",
        orderBy="modifiedTime desc",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives"
    ).execute()
    files = [f for f in results.get("files", []) if f.get("mimeType", "").startswith("video/")]
    return files[0] if files else None


def get_transcript(video_name):
    if not TRANSCRIPTIONS_FOLDER:
        return None
    base = video_name.rsplit(".", 1)[0]
    service = drive()
    for ext in [".txt", ".srt", ".vtt"]:
        results = service.files().list(
            q=f"name='{base}{ext}' and '{TRANSCRIPTIONS_FOLDER}' in parents and trashed=false",
            fields="files(id,name)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives"
        ).execute()
        files = results.get("files", [])
        if files:
            req = service.files().get_media(fileId=files[0]["id"])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.seek(0)
            plog(f"Found transcript: {base}{ext}")
            return fh.read().decode("utf-8", errors="ignore")
    plog("No transcript found — will use vision analysis only")
    return None


def download_file(service, file_obj, target):
    plog(f"Downloading {file_obj['name']}...")
    req = service.files().get_media(fileId=file_obj["id"])
    with open(target, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                plog(f"  {int(status.progress() * 100)}%")


def upload_output(final_path):
    service = drive()
    uploaded = service.files().create(
        body={"name": final_path.name, "parents": [OUTPUT_FOLDER]},
        media_body=MediaFileUpload(str(final_path), mimetype="video/mp4"),
        fields="id,name",
        supportsAllDrives=True
    ).execute()
    plog(f"Uploaded: {uploaded.get('name')} ({uploaded.get('id')})")
    return uploaded


def list_music_files():
    if not MUSIC_FOLDER:
        return []
    try:
        service = drive()
        results = service.files().list(
            q=f"'{MUSIC_FOLDER}' in parents and trashed=false",
            fields="files(id,name,mimeType)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives"
        ).execute()
        return [f for f in results.get("files", [])
                if f.get("mimeType", "").startswith(("audio/", "video/mp4"))]
    except Exception as e:
        plog(f"Music fetch failed: {e}")
        return []


def load_style_guide():
    if not GUIDE_FOLDER:
        return ""
    try:
        service = drive()
        results = service.files().list(
            q=f"'{GUIDE_FOLDER}' in parents and trashed=false",
            fields="files(id,mimeType)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives"
        ).execute()
        files = results.get("files", [])
        if not files:
            return ""
        f = files[0]
        if "google-apps.document" in f.get("mimeType", ""):
            req = service.files().export_media(fileId=f["id"], mimeType="text/plain")
        else:
            req = service.files().get_media(fileId=f["id"])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return fh.read().decode("utf-8", errors="ignore")[:3000]
    except Exception as e:
        plog(f"Style guide error: {e}")
        return ""


# =============================================================================
# AUDIO WAVEFORM — find clean cut points (silence/pauses)
# =============================================================================

def find_clean_cut(video_path, target_time, search_window=2.0):
    """Find the nearest silence/pause within search_window seconds of target_time."""
    try:
        audio_tmp = INPUT / "audio_analysis.wav"
        run_cmd([
            "ffmpeg", "-y", "-i", str(video_path),
            "-ss", str(max(0, target_time - search_window)),
            "-t", str(search_window * 2),
            "-vn", "-ar", "16000", "-ac", "1",
            str(audio_tmp)
        ])
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_frames", "-select_streams", "a", str(audio_tmp)],
            capture_output=True, text=True
        )
        # Use ffmpeg silencedetect to find pauses
        silence = subprocess.run(
            ["ffmpeg", "-i", str(audio_tmp),
             "-af", "silencedetect=noise=-40dB:d=0.1",
             "-f", "null", "-"],
            capture_output=True, text=True
        )
        output = silence.stderr
        pauses = []
        for line in output.split("\n"):
            if "silence_start" in line:
                try:
                    t = float(line.split("silence_start:")[1].strip().split()[0])
                    # Convert back to original video time
                    original_t = max(0, target_time - search_window) + t
                    pauses.append(original_t)
                except Exception:
                    pass
        if pauses:
            # Find the pause closest to target_time
            best = min(pauses, key=lambda t: abs(t - target_time))
            plog(f"Clean cut found at {best:.2f}s (target was {target_time:.2f}s)")
            return best
    except Exception as e:
        plog(f"Clean cut detection failed: {e}")
    return target_time


# =============================================================================
# AI — pick best quote + music
# =============================================================================

def analyze_with_ai(transcript, music_files, style_guide="", prompt=""):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    music_list = "\n".join([f"- {f['name']}" for f in music_files]) or "No music available."

    system = (
        "You are a video editor. Given a transcript, find the most compelling 30-second quote or moment.\n"
        "Return ONLY JSON with:\n"
        "- quote: string (the exact words from the transcript to use as caption, max 80 chars)\n"
        "- start_time: float (approximate start time in seconds based on transcript position)\n"
        "- music_name: string (best matching music track name from the list)\n"
        "- explanation: string (1 sentence why this moment)\n"
        f"{('Style Guide: ' + style_guide) if style_guide else ''}"
    )

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Transcript:\n{transcript[:6000]}\n\nMusic:\n{music_list}" + (f"\n\nDirector note: {prompt}" if prompt else "")}
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
    result = json.loads(content.strip())

    music_name = result.get("music_name", "")
    music_file = next((f for f in music_files if f["name"] == music_name), None)
    if not music_file and music_files:
        import random
        music_file = random.choice(music_files)

    return result, music_file


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(prompt=""):
    global pipeline_status
    with pipeline_lock:
        pipeline_status = {"running": True, "log": [], "done": False, "error": None}
        try:
            ensure_dirs()
            clean_run_artifacts()

            stop_event.clear()
            plog("Getting latest video from Drive...")
            video = get_latest_video()
            if not video:
                raise ValueError("No videos found in incoming folder.")
            plog(f"Found: {video['name']}")

            if stop_event.is_set(): raise InterruptedError("Stopped.")
            transcript = get_transcript(video["name"])
            if not transcript:
                raise ValueError("No transcript found. Upload a matching .txt/.srt file to the transcriptions folder.")

            plog("Loading style guide...")
            style_guide = load_style_guide()

            plog("Fetching music tracks...")
            music_files = list_music_files()
            plog(f"Found {len(music_files)} music track(s)")

            if stop_event.is_set(): raise InterruptedError("Stopped.")
            plog("AI analyzing transcript...")
            result, selected_music = analyze_with_ai(transcript, music_files, style_guide, prompt)
            plog(f"Best moment: {result.get('explanation', '')}")
            plog(f"Quote: {result.get('quote', '')}")
            plog(f"Approx start: {result.get('start_time', 0):.1f}s")
            if selected_music:
                plog(f"Music: {selected_music['name']}")

            if stop_event.is_set(): raise InterruptedError("Stopped.")
            service = drive()
            raw = INPUT / video["name"]
            download_file(service, video, raw)

            total_duration = get_duration(raw)
            plog(f"Video duration: {total_duration:.1f}s")

            target_start = float(result.get("start_time", 0))
            target_start = max(0, min(target_start, total_duration - SEGMENT_DURATION))

            plog(f"Finding clean cut points near {target_start:.1f}s...")
            clean_start = find_clean_cut(raw, target_start)
            clean_end_target = clean_start + SEGMENT_DURATION
            clean_end = find_clean_cut(raw, min(clean_end_target, total_duration))
            actual_dur = clean_end - clean_start
            actual_dur = max(5, min(actual_dur, SEGMENT_DURATION))

            plog(f"Clean segment: {clean_start:.2f}s — {clean_start + actual_dur:.2f}s")

            trimmed = INPUT / "trimmed.mp4"
            run_cmd([
                "ffmpeg", "-y",
                "-ss", str(clean_start),
                "-i", str(raw),
                "-t", str(actual_dur),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-c:a", "aac", "-b:a", "128k",
                str(trimmed)
            ])

            # Caption from the quote
            caption = result.get("quote", "")[:MAX_CAPTION_CHARS]
            lines = wrap_caption(caption, max_chars_per_line=32)
            fontsize = 42
            line_h = fontsize + 10
            pad_x = 40
            pad_y = 20
            total_text_h = len(lines) * line_h
            block_h = total_text_h + pad_y * 2
            block_y = (1080 - block_h) // 2
            max_chars = max(len(l) for l in lines)
            est_w = int(max_chars * fontsize * 0.6)
            box_w = min(est_w + pad_x * 2, 1800)
            box_x = (1920 - box_w) // 2
            font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

            drawbox = f"drawbox=x={box_x}:y={block_y}:w={box_w}:h={block_h}:color=black:t=fill"
            drawtext_filters = [drawbox]
            for i, line in enumerate(lines):
                safe_line = ffmpeg_escape(line)
                y = block_y + pad_y + i * line_h
                drawtext_filters.append(
                    f"drawtext=text='{safe_line}':"
                    f"fontfile={font}:"
                    f"fontcolor=white:fontsize={fontsize}:"
                    f"x=(w-text_w)/2:y={y}"
                )

            vf = (
                "scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,"
            ) + ",".join(drawtext_filters)

            central = datetime.now(timezone(timedelta(hours=-5)))
            date_str = central.strftime("%m-%d-%Y %I:%M %p")
            safe_caption = " ".join(caption.split()[:6])
            orig_name = video["name"].rsplit(".", 1)[0]
            final_path = OUTPUT / f"{safe_caption}_{orig_name}_{date_str}.mp4"

            if selected_music:
                music_path = MUSIC_DIR / selected_music["name"]
                download_file(service, selected_music, music_path)
                music_trimmed = MUSIC_DIR / "music_trim.aac"
                run_cmd([
                    "ffmpeg", "-y", "-i", str(music_path),
                    "-t", str(actual_dur),
                    "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={max(0, actual_dur-2)}:d=2,volume=0.25",
                    str(music_trimmed)
                ])
                run_cmd([
                    "ffmpeg", "-y",
                    "-i", str(trimmed),
                    "-i", str(music_trimmed),
                    "-filter_complex", f"[0:v]{vf}[vout];[0:a]volume=1.0[va];[va][1:a]amix=inputs=2:duration=first[aout]",
                    "-map", "[vout]", "-map", "[aout]",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k",
                    str(final_path)
                ])
            else:
                run_cmd([
                    "ffmpeg", "-y",
                    "-i", str(trimmed),
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k",
                    str(final_path)
                ])

            upload_output(final_path)
            pipeline_status["done"] = True
            plog("Done!")

        except Exception as e:
            plog(f"Error: {e}")
            plog(traceback.format_exc())
            pipeline_status["error"] = str(e)
        finally:
            pipeline_status["running"] = False


# =============================================================================
# HTML
# =============================================================================

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Interview Editor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,sans-serif;background:#000;color:#fff;min-height:100vh;display:flex;align-items:center;justify-content:center}
.wrap{width:100%;max-width:480px;padding:40px 24px}
h1{font-size:1.4rem;font-weight:900;color:#00a6ff;margin-bottom:2px}
.sub{font-size:.65rem;color:#222;font-family:monospace;margin-bottom:28px}
.clip-card{background:#0d0d0d;border:1px solid #1c1c1c;border-radius:10px;padding:12px;display:flex;align-items:center;gap:12px;margin-bottom:20px;min-height:72px}
.clip-thumb{width:64px;height:64px;object-fit:cover;border-radius:6px;background:#111;flex-shrink:0}
.clip-info{flex:1;overflow:hidden}
.clip-name{font-size:.8rem;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#fff}
.clip-meta{font-size:.65rem;color:#333;font-family:monospace;margin-top:3px}

#refresh-btn{width:100%;background:none;border:1px solid #1a1a1a;border-radius:8px;color:#222;font-size:.65rem;font-family:monospace;padding:8px;cursor:pointer;transition:border-color .15s,color .15s;margin-bottom:24px}
#refresh-btn:hover{border-color:#00a6ff;color:#00a6ff}
#log{font-family:monospace;font-size:.68rem;line-height:1.8;color:#333;max-height:320px;overflow-y:auto;white-space:pre-wrap}
#log.active{color:#00a6ff}
#log.done{color:#00e676}
#log.error{color:#ff4d6d}
</style>
</head>
<body>
<div class="wrap">
  <h1>Interview Editor</h1>
  <div class="sub">TRANSCRIPT-DRIVEN 1920x1080</div>
  <div id="clip-card" class="clip-card">
    <div class="clip-thumb"></div>
    <div class="clip-info">
      <div class="clip-name">Loading latest clip...</div>
      <div class="clip-meta"></div>
    </div>
  </div>
  <div style="position:relative;margin-bottom:10px">
    <input id="prompt" style="width:100%;background:#0d0d0d;border:1px solid #1c1c1c;border-radius:10px;padding:13px 48px 13px 14px;font-size:.9rem;color:#fff;outline:none" placeholder="Director note e.g. focus on ROI moments...">
    <button id="go-btn" style="position:absolute;right:8px;top:50%;transform:translateY(-50%);width:32px;height:32px;background:#00a6ff;color:#000;border:none;border-radius:7px;font-weight:900;cursor:pointer;font-size:.9rem">&#9654;</button>
  </div>
  <button id="stop-btn" style="width:100%;background:none;border:1px solid #ff4d6d;border-radius:8px;color:#ff4d6d;font-size:.65rem;font-family:monospace;padding:8px;cursor:pointer;margin-bottom:8px;display:none">&#9632; Stop</button>
  <button id="refresh-btn">&#8635; Refresh Clip</button>
  <div id="log"></div>
</div>
<script>
var busy = false;
function setLog(text, cls) {
  var el = document.getElementById("log");
  el.textContent = text;
  el.className = cls || "";
}
function loadLatestClip() {
  fetch("/api/latest-clip")
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (!data.clip) { document.querySelector(".clip-name").textContent = "No clips found."; return; }
      var card = document.getElementById("clip-card");
      var mb = data.clip.size ? (parseInt(data.clip.size) / 1048576).toFixed(1) + " MB" : "";
      card.innerHTML = (data.clip.thumbnailLink
        ? "<img class='clip-thumb' src='" + data.clip.thumbnailLink + "'>"
        : "<div class='clip-thumb'></div>") +
        "<div class='clip-info'><div class='clip-name'>" + data.clip.name + "</div>" +
        "<div class='clip-meta'>" + mb + " &bull; latest</div></div>";
    });
}
function go() {
  if (busy) return;
  busy = true;
  document.getElementById("go-btn").disabled = true;
  setLog("Starting...", "active");
  var prompt = document.getElementById("prompt").value.trim();
  fetch("/api/run", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({prompt: prompt})})
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.error) { setLog("Error: " + data.error, "error"); busy = false; document.getElementById("go-btn").disabled = false; return; }
      pollLog();
    });
}
function pollLog() {
  var timer = setInterval(function() {
    fetch("/api/status").then(function(r) { return r.json(); }).then(function(data) {
      setLog((data.log || []).join(String.fromCharCode(10)), data.running ? "active" : (data.done ? "done" : "error"));
      if (!data.running) {
        clearInterval(timer);
        busy = false;
        document.getElementById("go-btn").disabled = false;
      }
    });
  }, 1000);
}
document.getElementById("go-btn").addEventListener("click", go);
document.getElementById("refresh-btn").addEventListener("click", loadLatestClip);
loadLatestClip();
</script>
</body>
</html>"""


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    return app.response_class(HTML.encode("utf-8"), mimetype="text/html")

@app.route("/api/latest-clip")
def api_latest_clip():
    try:
        clip = get_latest_video()
        return jsonify({"clip": clip})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/run", methods=["POST"])
def api_run():
    if pipeline_status.get("running"):
        return jsonify({"error": "Already running"}), 409
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    threading.Thread(target=run_pipeline, args=(prompt,), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    global pipeline_status
    stop_event.set()
    clean_run_artifacts()
    pipeline_status = {"running": False, "log": ["Stopped."], "done": False, "error": "Stopped by user"}
    return jsonify({"ok": True})

@app.route("/api/status")
def api_status():
    return jsonify(pipeline_status)

if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 5000), debug=False)
