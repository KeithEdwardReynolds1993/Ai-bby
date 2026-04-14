import os
import io
import json
import subprocess
import threading
import traceback
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account

app = Flask("worker_xml")

TMP = Path("/tmp/ai_xml")
INPUT = TMP / "input"
OUTPUT = TMP / "output"

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SEGMENT_DURATION = 15    # seconds per segment
MAX_SEGMENTS = 20        # max segments to find per clip
SAMPLE_INTERVAL = 1.0    # analyze every N seconds

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


def clean_run_artifacts():
    for f in list(INPUT.glob("*")) + list(OUTPUT.glob("*")):
        if f.is_file():
            f.unlink()


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True
    )
    data = json.loads(r.stdout)
    duration = float(data["format"]["duration"])
    # Get actual video dimensions
    width, height = 1920, 1080
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = stream.get("width", 1920)
            height = stream.get("height", 1080)
            break
    return duration, width, height


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
    plog(f"Downloading {file_obj['name']}...")
    req = service.files().get_media(fileId=file_obj["id"])
    with open(target, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                plog(f"  {int(status.progress() * 100)}%")


def upload_xml(xml_path):
    service = drive()
    uploaded = service.files().create(
        body={"name": xml_path.name, "parents": [OUTPUT_FOLDER]},
        media_body=MediaFileUpload(str(xml_path), mimetype="application/xml"),
        fields="id,name",
        supportsAllDrives=True
    ).execute()
    plog(f"Uploaded XML: {uploaded.get('name')} ({uploaded.get('id')})")
    return uploaded


# =============================================================================
# OPTICAL FLOW + FACE DETECTION — find best segments
# =============================================================================

def find_best_segments(video_path, total_duration):
    plog("Analyzing video with optical flow + face detection...")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_interval_frames = max(1, int(fps * SAMPLE_INTERVAL))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frame_scores = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval_frames == 0:
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
    plog(f"Analyzed {len(frame_scores)} sample frames")

    if not frame_scores:
        return [(0.0, SEGMENT_DURATION)]

    # Sliding window to find top segments
    window = max(1, int(SEGMENT_DURATION / SAMPLE_INTERVAL))
    scored_windows = []

    for i in range(len(frame_scores) - window + 1):
        score = sum(s for _, s in frame_scores[i:i + window])
        start = frame_scores[i][0]
        scored_windows.append((score, start))

    scored_windows.sort(reverse=True)

    # Pick top non-overlapping segments
    segments = []
    for score, start in scored_windows:
        end = start + SEGMENT_DURATION
        # Check no overlap with already selected
        overlap = False
        for s_start, s_end in segments:
            if not (end <= s_start or start >= s_end):
                overlap = True
                break
        if not overlap:
            actual_dur = min(SEGMENT_DURATION, total_duration - start)
            if actual_dur >= 3:  # skip tiny segments
                segments.append((start, end))
            if len(segments) >= MAX_SEGMENTS:
                break

    # Sort chronologically
    segments.sort()
    plog(f"Found {len(segments)} best segments")
    return [(s, min(SEGMENT_DURATION, total_duration - s)) for s, e in segments]


# =============================================================================
# FCPXML GENERATOR
# =============================================================================

def seconds_to_rational(seconds, fps=30):
    """Convert seconds to FCP rational time format."""
    frames = round(seconds * fps)
    return f"{frames}/30s" if fps == 30 else f"{round(seconds * 30)}/30s"


def generate_fcpxml(video_file, segments, duration, width, height, orig_filename):
    fps = 30
    uid = str(uuid.uuid4()).upper()
    asset_id = f"r1"
    format_id = f"r2"
    seq_id = f"r3"

    # Timeline duration = sum of all segments
    total_timeline = sum(d for _, d in segments)

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!DOCTYPE fcpxml>')
    lines.append('<fcpxml version="1.10">')
    lines.append('  <resources>')
    lines.append(f'    <format id="{format_id}" name="FFVideoFormat{height}p{fps}" '
                 f'frameDuration="1/{fps}s" width="{width}" height="{height}"/>')
    lines.append(f'    <asset id="{asset_id}" name="{orig_filename}" uid="{uid}" '
                 f'start="0s" duration="{seconds_to_rational(duration, fps)}" '
                 f'hasVideo="1" hasAudio="1">')
    lines.append(f'      <media-rep kind="original-media" src="file:///REPLACE_WITH_PATH/{orig_filename}"/>')
    lines.append(f'    </asset>')
    lines.append('  </resources>')
    lines.append('  <library>')
    lines.append('    <event name="AI Selections">')
    lines.append(f'    <project name="{orig_filename} - AI Edit">')
    lines.append(f'      <sequence format="{format_id}" duration="{seconds_to_rational(total_timeline, fps)}" '
                 f'tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">')
    lines.append('        <spine>')

    offset = 0.0
    for i, (start, seg_dur) in enumerate(segments):
        clip_offset = seconds_to_rational(offset, fps)
        clip_start = seconds_to_rational(start, fps)
        clip_dur = seconds_to_rational(seg_dur, fps)
        lines.append(f'          <asset-clip name="Segment {i+1}" ref="{asset_id}" '
                     f'offset="{clip_offset}" start="{clip_start}" duration="{clip_dur}" '
                     f'format="{format_id}" tcFormat="NDF">')
        lines.append(f'          </asset-clip>')
        offset += seg_dur

    lines.append('        </spine>')
    lines.append('      </sequence>')
    lines.append('    </project>')
    lines.append('    </event>')
    lines.append('  </library>')
    lines.append('</fcpxml>')

    return "\n".join(lines)


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

            service = drive()
            raw = INPUT / video["name"]
            download_file(service, video, raw)

            plog("Getting video info...")
            duration, width, height = get_duration(raw)
            plog(f"Duration: {duration:.1f}s | {width}x{height}")

            segments = find_best_segments(raw, duration)
            plog(f"Selected {len(segments)} segments totaling {sum(d for _,d in segments):.1f}s")

            for i, (start, dur) in enumerate(segments):
                plog(f"  Segment {i+1}: {start:.1f}s — {start+dur:.1f}s")

            plog("Generating FCPXML...")
            xml_content = generate_fcpxml(
                video, segments, duration, width, height, video["name"]
            )

            central = datetime.now(timezone(timedelta(hours=-5)))
            date_str = central.strftime("%m-%d-%Y %I%M %p")
            xml_name = f"{video['name'].rsplit('.', 1)[0]}_{date_str}.fcpxml"
            xml_path = OUTPUT / xml_name
            xml_path.write_text(xml_content, encoding="utf-8")

            upload_xml(xml_path)
            pipeline_status["done"] = True
            plog("Done! Open the FCPXML in Final Cut Pro.")
            plog("Note: Update the media path in the XML to match where your original file lives.")

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
<title>XML Editor</title>
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
#log{font-family:monospace;font-size:.68rem;line-height:1.8;color:#333;max-height:320px;overflow-y:auto;white-space:pre-wrap}
#log.active{color:#00a6ff}
#log.done{color:#00e676}
#log.error{color:#ff4d6d}
</style>
</head>
<body>
<div class="wrap">
  <h1>XML Editor</h1>
  <div class="sub">FCPXML GENERATOR</div>

  <div id="clip-card" class="clip-card">
    <div class="clip-thumb"></div>
    <div class="clip-info">
      <div class="clip-name">Loading latest clip...</div>
      <div class="clip-meta"></div>
    </div>
  </div>

  <button id="go-btn">GENERATE XML</button>
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
  fetch("/api/run", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({})})
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
    threading.Thread(target=run_pipeline, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(pipeline_status)


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 5000), debug=False)
