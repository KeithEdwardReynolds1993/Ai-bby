import os
import io
import json
import subprocess
import threading
import traceback
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import requests
from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account

app = Flask("worker_horizontal")

TMP = Path("/tmp/ai_xml")
INPUT = TMP / "input"
OUTPUT = TMP / "output"

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INTERVIEW_FOLDER_ID") or os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
TRANSCRIPTIONS_FOLDER = os.getenv("GOOGLE_DRIVE_TRANSCRIPTIONS_FOLDER_ID", "")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SEGMENT_DURATION = 30
MAX_SEGMENTS = 10

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


def get_video_info(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True
    )
    data = json.loads(r.stdout)
    duration = float(data["format"]["duration"])
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


def get_transcript(video_name):
    if not TRANSCRIPTIONS_FOLDER:
        return None
    base = video_name.rsplit(".", 1)[0]
    service = drive()
    for ext in [".json", ".txt", ".srt", ".vtt"]:
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
    return None


def download_audio_only(service, file_obj):
    """Download full file then extract low-res mono audio for analysis."""
    plog(f"Downloading {file_obj['name']} for audio extraction...")
    raw_video_tmp = INPUT / "raw_tmp.mp4"
    req = service.files().get_media(fileId=file_obj["id"])
    with open(raw_video_tmp, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                plog(f"  Download: {int(status.progress() * 100)}%")
    # Extract low-res mono audio for fast analysis
    raw_audio = INPUT / "raw_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw_video_tmp),
        "-vn", "-ar", "8000", "-ac", "1", "-ab", "16k",
        str(raw_audio)
    ], check=True, capture_output=True)
    raw_video_tmp.unlink()
    plog(f"Low-res audio extracted ({raw_audio.stat().st_size // 1024}KB)")
    return raw_audio


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
# AUDIO SILENCE DETECTION — find clean cut points
# =============================================================================

def find_clean_cut(audio_path, target_time, search_window=2.0, total_duration=None):
    """Find nearest silence near target_time using ffmpeg silencedetect."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(audio_path),
             "-af", f"atrim=start={max(0, target_time - search_window)}:end={min(total_duration or 99999, target_time + search_window)},silencedetect=noise=-40dB:d=0.15",
             "-f", "null", "-"],
            capture_output=True, text=True
        )
        output = result.stderr
        pauses = []
        for line in output.split("\n"):
            if "silence_start" in line:
                try:
                    t = float(line.split("silence_start:")[1].strip().split()[0])
                    original_t = max(0, target_time - search_window) + t
                    pauses.append(original_t)
                except Exception:
                    pass
        if pauses:
            best = min(pauses, key=lambda t: abs(t - target_time))
            plog(f"  Clean cut: {best:.2f}s (target {target_time:.2f}s)")
            return best
    except Exception as e:
        plog(f"  Silence detection failed: {e}")
    return target_time


# =============================================================================
# AI — pick best moments from transcript
# =============================================================================


def parse_transcript(raw, filename=""):
    """Parse transcript - handles JSON word-level format, SRT, or plain text."""
    if filename.endswith(".json") or (raw.strip().startswith("{")):
        try:
            data = json.loads(raw)
            # Word-level JSON format
            words = []
            for segment in data.get("segments", []):
                for word in segment.get("words", []):
                    if word.get("type") == "word" and word.get("text", "").strip():
                        words.append({
                            "text": word["text"],
                            "start": word["start"],
                            "end": word["start"] + word["duration"]
                        })
            # Build readable transcript with timestamps every ~30 words
            lines = []
            chunk = []
            chunk_start = None
            for w in words:
                if chunk_start is None:
                    chunk_start = w["start"]
                chunk.append(w["text"])
                if len(chunk) >= 30:
                    t = int(chunk_start)
                    lines.append(f"[{t//60:02d}:{t%60:02d}] {' '.join(chunk)}")
                    chunk = []
                    chunk_start = None
            if chunk:
                t = int(chunk_start or 0)
                lines.append(f"[{t//60:02d}:{t%60:02d}] {' '.join(chunk)}")
            return "\n".join(lines), words
        except Exception as e:
            print(f"JSON parse failed: {e}")
    # Plain text or SRT - no word timestamps
    return raw, []

def pick_moments_from_transcript(transcript, prompt=""):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    system = (
        f"You are a video editor. Given a transcript, find the {MAX_SEGMENTS} most compelling moments "
        f"that are roughly {SEGMENT_DURATION} seconds each.\n"
        "Return ONLY a JSON array of objects with:\n"
        "- start_time: float (seconds)\n"
        "- end_time: float (seconds)\n"
        "- quote: string (key phrase from this moment)\n"
        "- reason: string (why this moment is compelling)\n"
        "Order by importance, most compelling first."
    )

    user = f"Transcript:\n{transcript[:8000]}"
    if prompt:
        user += f"\n\nDirector note: {prompt}"

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.6
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


# =============================================================================
# FCPXML GENERATOR
# =============================================================================

def seconds_to_rational(seconds, fps=30):
    frames = round(seconds * fps)
    return f"{frames}/30s"


def generate_fcpxml(segments, duration, width, height, orig_filename):
    fps = 30
    uid = str(uuid.uuid4()).upper()
    total_timeline = sum(e - s for s, e, _ in segments)

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!DOCTYPE fcpxml>')
    lines.append('<fcpxml version="1.10">')
    lines.append('  <resources>')
    lines.append(f'    <format id="r2" name="FFVideoFormat{height}p{fps}" frameDuration="1/{fps}s" width="{width}" height="{height}"/>')
    lines.append(f'    <asset id="r1" name="{orig_filename}" uid="{uid}" start="0s" duration="{seconds_to_rational(duration, fps)}" hasVideo="1" hasAudio="1">')
    lines.append(f'      <media-rep kind="original-media" src="file:///REPLACE_WITH_PATH/{orig_filename}"/>')
    lines.append(f'    </asset>')
    lines.append('  </resources>')
    lines.append('  <library>')
    lines.append('    <event name="AI Selections">')
    lines.append(f'    <project name="{orig_filename} - AI Edit">')
    lines.append(f'      <sequence format="r2" duration="{seconds_to_rational(total_timeline, fps)}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">')
    lines.append('        <spine>')

    offset = 0.0
    for i, (start, end, quote) in enumerate(segments):
        seg_dur = end - start
        lines.append(f'          <!-- {quote[:80]} -->')
        lines.append(f'          <asset-clip name="Moment {i+1}" ref="r1" offset="{seconds_to_rational(offset, fps)}" start="{seconds_to_rational(start, fps)}" duration="{seconds_to_rational(seg_dur, fps)}" format="r2" tcFormat="NDF">')
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

def run_pipeline(prompt=""):
    global pipeline_status
    with pipeline_lock:
        pipeline_status = {"running": True, "log": [], "done": False, "error": None}
        try:
            ensure_dirs()
            clean_run_artifacts()

            if prompt:
                plog(f"Director note: {prompt}")

            plog("Getting latest video from Drive...")
            video = get_latest_video()
            if not video:
                raise ValueError("No videos found in incoming folder.")
            plog(f"Found: {video['name']}")

            plog("Looking for transcript...")
            transcript = get_transcript(video["name"])
            if not transcript:
                raise ValueError("No transcript found. Upload a matching .txt/.srt file to the transcriptions folder.")
            plog(f"Transcript loaded ({len(transcript)} chars)")

            plog("Parsing transcript...")
            parsed_text, word_timestamps = parse_transcript(transcript, video["name"])
            plog(f"Parsed: {len(word_timestamps)} words with timestamps" if word_timestamps else "Plain text transcript")

            plog("AI picking best moments from transcript...")
            moments = pick_moments_from_transcript(parsed_text, prompt)
            plog(f"AI selected {len(moments)} moments")
            for i, m in enumerate(moments):
                plog(f"  {i+1}. {m['start_time']:.1f}s-{m['end_time']:.1f}s — {m.get('quote', '')[:50]}")

            plog("Snapping to word boundaries from JSON timestamps...")
            segments = []
            for m in moments:
                raw_start = float(m["start_time"])
                raw_end = float(m["end_time"])

                if word_timestamps:
                    starts = [w["start"] for w in word_timestamps if abs(w["start"] - raw_start) < 5.0]
                    ends = [w["end"] for w in word_timestamps if abs(w["end"] - raw_end) < 5.0]
                    if starts:
                        raw_start = min(starts, key=lambda t: abs(t - raw_start))
                    if ends:
                        raw_end = min(ends, key=lambda t: abs(t - raw_end))

                if raw_end > raw_start:
                    segments.append((raw_start, raw_end, m.get("quote", "")))
                    plog(f"  Segment: {raw_start:.2f}s — {raw_end:.2f}s | {m.get('quote','')[:50]}")

            plog(f"Final segments: {len(segments)}")

            # Get duration from JSON transcript
            if word_timestamps:
                duration = word_timestamps[-1]["end"] + 1.0
            else:
                duration = max(e for _, e, _ in segments) + 1.0
            width, height = 1920, 1080

            plog("Generating FCPXML...")
            xml_content = generate_fcpxml(segments, duration, width, height, video["name"])

            central = datetime.now(timezone(timedelta(hours=-5)))
            date_str = central.strftime("%m-%d-%Y %I%M %p")
            xml_name = f"{video['name'].rsplit('.', 1)[0]}_{date_str}.fcpxml"
            xml_path = OUTPUT / xml_name
            xml_path.write_text(xml_content, encoding="utf-8")

            upload_xml(xml_path)
            pipeline_status["done"] = True
            plog("Done! Open the FCPXML in Final Cut Pro.")
            plog("Replace REPLACE_WITH_PATH with the folder containing your original file.")

        except Exception as e:
            plog(f"Error: {e}")
            plog(traceback.format_exc())
            pipeline_status["error"] = str(e)
        finally:
            pipeline_status["running"] = False


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
.prompt-row{position:relative;margin-bottom:10px}
#prompt{width:100%;background:#0d0d0d;border:1px solid #1c1c1c;border-radius:10px;padding:13px 48px 13px 14px;font-size:.9rem;color:#fff;outline:none;transition:border-color .15s}
#prompt:focus{border-color:#00a6ff}
#go-btn{position:absolute;right:8px;top:50%;transform:translateY(-50%);width:32px;height:32px;background:#00a6ff;color:#000;border:none;border-radius:7px;font-weight:900;cursor:pointer;font-size:.9rem}
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
  <div class="sub">TRANSCRIPT + AUDIO DRIVEN FCPXML</div>
  <div id="clip-card" class="clip-card">
    <div class="clip-thumb"></div>
    <div class="clip-info">
      <div class="clip-name">Loading latest clip...</div>
      <div class="clip-meta"></div>
    </div>
  </div>
  <div class="prompt-row">
    <input id="prompt" placeholder="Director note e.g. focus on community moments...">
    <button id="go-btn">&#9654;</button>
  </div>
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
      card.innerHTML = (data.clip.thumbnailLink ? "<img class='clip-thumb' src='" + data.clip.thumbnailLink + "'>" : "<div class='clip-thumb'></div>") +
        "<div class='clip-info'><div class='clip-name'>" + data.clip.name + "</div><div class='clip-meta'>" + mb + " &bull; latest</div></div>";
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
document.getElementById("prompt").addEventListener("keydown", function(e) { if (e.key === "Enter") go(); });
loadLatestClip();
</script>
</body>
</html>"""


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

@app.route("/api/status")
def api_status():
    return jsonify(pipeline_status)

if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 5000), debug=False)
