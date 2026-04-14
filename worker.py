import os
import io
import json
import subprocess
import threading
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account

app = Flask("worker")

TMP = Path("/tmp/ai_bby")
INPUT = TMP / "input"
OUTPUT = TMP / "output"
MUSIC_DIR = TMP / "music"

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_CAPTION_CHARS = int(os.getenv("MAX_CAPTION_CHARS") or "60")
SEGMENT_DURATION = 15

pipeline_status = {"running": False, "log": [], "done": False, "error": None}
pipeline_lock = threading.Lock()


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



def wrap_caption(text, max_chars_per_line=20):
    """Wrap caption into multiple drawtext calls stacked vertically."""
    words = text.split()
    lines = []
    current = []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > max_chars_per_line and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines

def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, check=True
    )
    return float(json.loads(r.stdout)["format"]["duration"])


# =============================================================================
# DRIVE
# =============================================================================

def get_latest_video():
    service = drive()
    results = service.files().list(
        q=f"'{INCOMING_FOLDER}' in parents and trashed=false",
        fields="files(id,name,mimeType,size,thumbnailLink,modifiedTime)",
        orderBy="modifiedTime desc",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives"
    ).execute()
    files = [f for f in results.get("files", []) if f.get("mimeType", "").startswith("video/")]
    return files[0] if files else None


def download_file(service, file_obj, target):
    req = service.files().get_media(fileId=file_obj["id"])
    with open(target, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()


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


# =============================================================================
# VISION ANALYSIS
# =============================================================================

def analyze_thumbnail(thumbnail_url):
    if not thumbnail_url or not OPENAI_API_KEY:
        return {}
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this video thumbnail. Return ONLY JSON:\n"
                                "- description: string (1-2 sentences)\n"
                                "- mood: string (energetic/calm/funny/dramatic/inspirational)\n"
                                "- subjects: list of strings\n"
                                "- energy: string (low/medium/high)\n"
                                "- setting: string (indoor/outdoor/studio/unknown)"
                            )
                        },
                        {"type": "image_url", "image_url": {"url": thumbnail_url}}
                    ]
                }],
                "max_tokens": 300
            },
            timeout=20
        )
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception as e:
        plog(f"Vision analysis failed: {e}")
        return {}


# =============================================================================
# OPTICAL FLOW + FACE DETECTION
# =============================================================================

def find_best_segment(video_path, total_duration):
    if total_duration <= SEGMENT_DURATION:
        return 0.0

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_interval = max(1, int(fps * 0.5))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frame_scores = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (320, 180))
            motion = 0.0
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                motion = float(np.mean(np.abs(flow)))
            faces = face_cascade.detectMultiScale(small, 1.1, 4)
            face_score = min(len(faces) * 0.4, 1.0)
            frame_scores.append((frame_idx / fps, motion + face_score))
            prev_gray = small
        frame_idx += 1

    cap.release()

    if not frame_scores:
        return 0.0

    window = int(SEGMENT_DURATION / 0.5)
    best_score = -1
    best_start = 0.0

    for i in range(max(1, len(frame_scores) - window + 1)):
        score = sum(s for _, s in frame_scores[i:i + window])
        if score > best_score:
            best_score = score
            best_start = frame_scores[i][0]

    return min(best_start, total_duration - SEGMENT_DURATION)



GUIDE_FOLDER = os.getenv("GOOGLE_DRIVE_GUIDE_FOLDER_ID", "")
MUSIC_FOLDER = os.getenv("GOOGLE_DRIVE_MUSIC_FOLDER_ID", "")

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
        print(f"Style guide error: {e}")
        return ""

# =============================================================================
# AI CAPTION
# =============================================================================

def generate_caption(vision):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    vision_summary = (
        f"Description: {vision.get('description', 'unknown')}\n"
        f"Mood: {vision.get('mood', 'unknown')}\n"
        f"Energy: {vision.get('energy', 'unknown')}\n"
        f"Setting: {vision.get('setting', 'unknown')}\n"
        f"Subjects: {', '.join(vision.get('subjects', []))}"
    )

    style_guide = load_style_guide()
    vision_summary += f"\n\nStyle Guide:\n{style_guide}" if style_guide else ""

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You write short punchy video captions. Max {MAX_CAPTION_CHARS} chars. "
                        "Captions will be word-wrapped at ~20 characters per line and displayed "
                        "centered on a vertical video. Write captions that read naturally across "
                        "Max 8 words. Bold, punchy, impactful. Can be a short sentence or phrase. No filler words. Write like a motivational brand — confident, direct, memorable. Examples: 'This is how winners think' or 'Every rep builds the future' or 'Show up. Do the work.'. "
                        "Return ONLY JSON with key: caption"
                    )
                },
                {
                    "role": "user",
                    "content": f"Video analysis:\n{vision_summary}\n\nWrite the best caption for this clip."
                }
            ],
            "temperature": 0.8
        },
        timeout=20
    )
    if not resp.ok:
        raise ValueError(f"OpenAI error {resp.status_code}: {resp.text}")

    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())["caption"]



def get_random_music():
    if not MUSIC_FOLDER:
        return None
    try:
        service = drive()
        results = service.files().list(
            q=f"'{MUSIC_FOLDER}' in parents and trashed=false",
            fields="files(id,name,mimeType)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives"
        ).execute()
        files = [f for f in results.get("files", [])
                 if f.get("mimeType", "").startswith(("audio/", "video/mp4"))]
        if not files:
            return None
        import random
        return random.choice(files)
    except Exception as e:
        plog(f"Music fetch failed: {e}")
        return None

# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline():
    global pipeline_status
    with pipeline_lock:
        pipeline_status = {"running": True, "log": [], "done": False, "error": None}
        try:
            ensure_dirs()
            clean_run_artifacts()

            plog("Getting latest video from Drive...")
            video = get_latest_video()
            if not video:
                raise ValueError("No videos found in incoming folder.")
            plog(f"Found: {video['name']}")

            plog("Analyzing thumbnail...")
            vision = analyze_thumbnail(video.get("thumbnailLink", ""))
            if vision:
                plog(f"Vision: {vision.get('description', '')}")
                plog(f"Mood: {vision.get('mood', '')} | Energy: {vision.get('energy', '')}")

            plog("Generating caption...")
            caption = generate_caption(vision)
            # Hard truncate to 5 words max
            caption = " ".join(caption.split()[:8])
            plog(f"Caption: {caption}")

            plog("Downloading video...")
            service = drive()
            raw = INPUT / "raw_clip.mp4"
            download_file(service, video, raw)

            total_duration = get_duration(raw)
            plog(f"Duration: {total_duration:.1f}s")

            plog(f"Finding best {SEGMENT_DURATION}s segment...")
            best_start = find_best_segment(raw, total_duration)
            actual_seg = min(SEGMENT_DURATION, total_duration - best_start)
            plog(f"Best segment: {best_start:.1f}s — {best_start + actual_seg:.1f}s")

            trimmed = INPUT / "trimmed.mp4"
            run_cmd([
                "ffmpeg", "-y",
                "-ss", str(best_start),
                "-i", str(raw),
                "-t", str(actual_seg),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-an", str(trimmed)
            ])

            from datetime import timezone, timedelta
            central = datetime.now(timezone(timedelta(hours=-5)))
            date_str = central.strftime("%m-%d-%Y %I:%M %p")
            safe_caption = " ".join(caption.split()[:8]).rstrip(".")
            orig_name = video["name"].rsplit(".", 1)[0]
            final_path = OUTPUT / f"{safe_caption}_{orig_name}_{date_str}.mp4"
            lines = wrap_caption(caption, max_chars_per_line=14)
            fontsize = 52
            line_height = fontsize + 16
            pad = 24
            total_height = len(lines) * line_height
            # Center block vertically, each line centered horizontally
            base = (
                "scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
            )
            # Join lines with FFmpeg newline, single drawtext = no overlap
            joined = r"\n".join(ffmpeg_escape(l) for l in lines)
            drawtext = (
                f"drawtext=text='{joined}':"
                f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                f"fontcolor=white:fontsize={fontsize}:"
                f"x=(w-text_w)/2:y=(h-text_h)/2:"
                f"box=1:boxcolor=black:boxborderw={pad}:line_spacing=8"
            )
            vf = base + "," + drawtext

            plog("Picking music track...")
            music = get_random_music()
            if music:
                plog(f"Music: {music['name']}")
                music_path = MUSIC_DIR / music["name"]
                download_file(service, music, music_path)
                music_trimmed = MUSIC_DIR / "music_trim.aac"
                run_cmd([
                    "ffmpeg", "-y", "-i", str(music_path),
                    "-t", str(actual_seg),
                    "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={max(0, actual_seg-2)}:d=2,volume=0.8",
                    str(music_trimmed)
                ])
                run_cmd([
                    "ffmpeg", "-y",
                    "-i", str(trimmed),
                    "-i", str(music_trimmed),
                    "-filter_complex", f"[0:v]{vf}[vout];[1:a]anull[aout]",
                    "-map", "[vout]", "-map", "[aout]",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k",
                    str(final_path)
                ])
            else:
                plog("No music found, rendering without.")
                run_cmd([
                    "ffmpeg", "-y",
                    "-i", str(trimmed),
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-an", str(final_path)
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
<title>Clip Studio</title>
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



#go-btn{width:100%;background:#00a6ff;color:#000;border:none;border-radius:10px;padding:14px;font-size:1rem;font-weight:900;cursor:pointer;transition:opacity .15s;margin-bottom:10px;letter-spacing:.05em}
#go-btn:hover{opacity:.8}
#go-btn:disabled{opacity:.2;cursor:not-allowed}
#refresh-btn{width:100%;background:none;border:1px solid #1a1a1a;border-radius:8px;color:#222;font-size:.65rem;font-family:monospace;padding:8px;cursor:pointer;transition:border-color .15s,color .15s;margin-bottom:24px}
#refresh-btn:hover{border-color:#00a6ff;color:#00a6ff}
#log{font-family:monospace;font-size:.68rem;line-height:1.8;color:#333;max-height:280px;overflow-y:auto;white-space:pre-wrap}
#log.active{color:#00a6ff}
#log.done{color:#00e676}
#log.error{color:#ff4d6d}
</style>
</head>
<body>
<div class="wrap">
  <h1>Clip Studio</h1>
  <div class="sub">AI VIDEO DIRECTOR</div>

  <div id="clip-card" class="clip-card">
    <div class="clip-thumb"></div>
    <div class="clip-info">
      <div class="clip-name">Loading latest clip...</div>
      <div class="clip-meta"></div>
    </div>
  </div>

  <button id="go-btn">GO</button>
  <button id="refresh-btn">&#8635; Refresh Clip</button>
  <div id="log"></div>
</div>

<script>
var busy = false;
var currentClip = null;

function setLog(text, cls) {
  var el = document.getElementById("log");
  el.textContent = text;
  el.className = cls || "";
}

function loadLatestClip() {
  fetch("/api/latest-clip")
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.error || !data.clip) {
        document.querySelector(".clip-name").textContent = "No clips found in incoming folder.";
        return;
      }
      currentClip = data.clip;
      var card = document.getElementById("clip-card");
      var mb = data.clip.size ? (parseInt(data.clip.size) / 1048576).toFixed(1) + " MB" : "";
      card.innerHTML = (
        (data.clip.thumbnailLink
          ? "<img class='clip-thumb' src='" + data.clip.thumbnailLink + "'>"
          : "<div class='clip-thumb'></div>") +
        "<div class='clip-info'>" +
          "<div class='clip-name'>" + data.clip.name + "</div>" +
          "<div class='clip-meta'>" + mb + " &bull; latest</div>" +
        "</div>"
      );
    })
    .catch(function() {
      document.querySelector(".clip-name").textContent = "Failed to load clip.";
    });
}

function go() {
  if (busy) return;
  busy = true;
  document.getElementById("go-btn").disabled = true;
  setLog("Starting...", "active");

  fetch("/api/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({})
  }).then(function(r) { return r.json(); }).then(function(data) {
    if (data.error) {
      setLog("Error: " + data.error, "error");
      busy = false;
      document.getElementById("go-btn").disabled = false;
      return;
    }
    pollLog();
  }).catch(function(e) {
    setLog("Failed: " + e, "error");
    busy = false;
    document.getElementById("go-btn").disabled = false;
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
        if (data.done) loadLatestClip();
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
    threading.Thread(target=run_pipeline, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(pipeline_status)


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 5000), debug=False)
