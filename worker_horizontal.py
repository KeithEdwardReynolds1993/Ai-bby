import os
import io
import json
import threading
import traceback
import requests
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account

app = Flask("worker_klap")

TMP = Path("/tmp/klap")
INPUT = TMP / "input"
OUTPUT = TMP / "output"

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INTERVIEW_FOLDER_ID") or os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
KLAP_API_KEY = os.getenv("KLAP_API_KEY", "kak_qwneAy8lteqZW3T0oFf0eFYD")

KLAP_API_URL = "https://api.klap.video/v2"
TARGET_CLIPS = 10
MIN_DURATION = 15
MAX_DURATION = 30
POLL_INTERVAL = 30

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


# =============================================================================
# DRIVE
# =============================================================================

def get_latest_video():
    service = drive()
    results = service.files().list(
        q=f"'{INCOMING_FOLDER}' in parents and trashed=false",
        fields="files(id,name,mimeType,size,thumbnailLink,modifiedTime,webContentLink)",
        orderBy="modifiedTime desc",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives"
    ).execute()
    files = [f for f in results.get("files", []) if f.get("mimeType", "").startswith("video/")]
    return files[0] if files else None


def get_public_video_url(file_obj):
    """Make the Drive file publicly readable and return a direct download URL."""
    service = drive()
    service.permissions().create(
        fileId=file_obj["id"],
        body={"role": "reader", "type": "anyone"},
        supportsAllDrives=True
    ).execute()
    return f"https://drive.google.com/uc?export=download&id={file_obj['id']}"


def upload_clip(local_path, clip_name):
    service = drive()
    uploaded = service.files().create(
        body={"name": clip_name, "parents": [OUTPUT_FOLDER]},
        media_body=MediaFileUpload(str(local_path), mimetype="video/mp4", resumable=True),
        fields="id,name",
        supportsAllDrives=True
    ).execute()
    plog(f"  Uploaded: {uploaded.get('name')} ({uploaded.get('id')})")
    return uploaded


# =============================================================================
# KLAP API
# =============================================================================

def klap_post(endpoint, body={}):
    resp = requests.post(
        f"{KLAP_API_URL}{endpoint}",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KLAP_API_KEY}"
        },
        json=body,
        timeout=30
    )
    if not resp.ok:
        raise ValueError(f"Klap POST {endpoint} failed {resp.status_code}: {resp.text}")
    return resp.json()


def klap_get(endpoint):
    resp = requests.get(
        f"{KLAP_API_URL}{endpoint}",
        headers={"Authorization": f"Bearer {KLAP_API_KEY}"},
        timeout=30
    )
    if not resp.ok:
        raise ValueError(f"Klap GET {endpoint} failed {resp.status_code}: {resp.text}")
    return resp.json()


def klap_poll(endpoint, check_key, check_value):
    while True:
        data = klap_get(endpoint)
        plog(f"  [{datetime.now().strftime('%H:%M:%S')}] status: {data.get(check_key)}")
        if data.get(check_key) != check_value:
            return data
        time.sleep(POLL_INTERVAL)


def generate_klap_shorts(video_url):
    plog("Submitting video to Klap...")
    task = klap_post("/tasks/video-to-shorts", {
        "source_video_url": video_url,
        "language": "en",
        "min_duration": MIN_DURATION,
        "max_duration": MAX_DURATION,
    })
    plog(f"Task created: {task['id']}")

    task = klap_poll(f"/tasks/{task['id']}", "status", "processing")
    if task.get("status") == "error":
        raise ValueError("Klap task processing failed.")

    folder_id = task["output_id"]
    plog(f"Klap folder: {folder_id}")

    projects = klap_get(f"/projects/{folder_id}")
    plog(f"Total clips available: {len(projects)}")

    # Sort by virality score, take top TARGET_CLIPS
    projects.sort(key=lambda p: p.get("virality_score", 0), reverse=True)
    return projects[:TARGET_CLIPS], folder_id


def export_klap_clip(folder_id, project_id):
    export = klap_post(f"/projects/{folder_id}/{project_id}/exports", {})
    plog(f"  Export started: {export['id']}")
    export = klap_poll(
        f"/projects/{folder_id}/{project_id}/exports/{export['id']}",
        "status", "processing"
    )
    if export.get("status") == "error":
        raise ValueError(f"Export failed for project {project_id}")
    return export.get("src_url") or export.get("stream_url")


def download_clip(url, dest_path):
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)


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

            plog("Getting latest video from Drive...")
            video = get_latest_video()
            if not video:
                raise ValueError("No videos found in incoming folder.")
            plog(f"Found: {video['name']}")

            plog("Making video publicly accessible for Klap...")
            video_url = get_public_video_url(video)
            plog(f"Video URL ready.")

            projects, folder_id = generate_klap_shorts(video_url)
            plog(f"Processing top {len(projects)} clips...")

            base_name = video["name"].rsplit(".", 1)[0]
            central = datetime.now(timezone(timedelta(hours=-5)))
            date_str = central.strftime("%m-%d-%Y_%I%M%p")

            for i, project in enumerate(projects):
                clip_num = i + 1
                score = project.get("virality_score", 0)
                name = project.get("name", f"clip_{clip_num}")
                plog(f"\nClip {clip_num}/{len(projects)}: \"{name}\" (score: {score})")

                try:
                    clip_url = export_klap_clip(folder_id, project["id"])
                    if not clip_url:
                        plog(f"  No URL returned, skipping.")
                        continue

                    clip_filename = f"{base_name}_{date_str}_clip{clip_num:02d}.mp4"
                    local_path = OUTPUT / clip_filename

                    plog(f"  Downloading clip...")
                    download_clip(clip_url, local_path)
                    plog(f"  Downloaded ({local_path.stat().st_size // 1024}KB)")

                    upload_clip(local_path, clip_filename)
                    local_path.unlink()

                except Exception as e:
                    plog(f"  Clip {clip_num} failed: {e}")
                    continue

            pipeline_status["done"] = True
            plog(f"\nDone! {len(projects)} clips uploaded to Drive output folder.")

        except Exception as e:
            plog(f"Error: {e}")
            plog(traceback.format_exc())
            pipeline_status["error"] = str(e)
        finally:
            pipeline_status["running"] = False


# =============================================================================
# HTML UI
# =============================================================================

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Klap Shorts Generator</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,sans-serif;background:#000;color:#fff;min-height:100vh;display:flex;align-items:center;justify-content:center}
.wrap{width:100%;max-width:480px;padding:40px 24px}
h1{font-size:1.4rem;font-weight:900;color:#00a6ff;margin-bottom:2px}
.sub{font-size:.65rem;color:#444;font-family:monospace;margin-bottom:28px}
.clip-card{background:#0d0d0d;border:1px solid #1c1c1c;border-radius:10px;padding:12px;display:flex;align-items:center;gap:12px;margin-bottom:20px;min-height:72px}
.clip-thumb{width:64px;height:64px;object-fit:cover;border-radius:6px;background:#111;flex-shrink:0}
.clip-info{flex:1;overflow:hidden}
.clip-name{font-size:.8rem;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#fff}
.clip-meta{font-size:.65rem;color:#444;font-family:monospace;margin-top:3px}
.btn-row{display:flex;gap:8px;margin-bottom:24px}
#go-btn{flex:1;background:#00a6ff;color:#000;border:none;border-radius:8px;font-weight:900;font-size:.9rem;padding:12px;cursor:pointer}
#go-btn:hover{opacity:.8}
#go-btn:disabled{opacity:.2;cursor:not-allowed}
#refresh-btn{background:none;border:1px solid #1a1a1a;border-radius:8px;color:#444;font-size:.65rem;font-family:monospace;padding:8px 14px;cursor:pointer;transition:border-color .15s,color .15s}
#refresh-btn:hover{border-color:#00a6ff;color:#00a6ff}
#log{font-family:monospace;font-size:.68rem;line-height:1.8;color:#444;max-height:360px;overflow-y:auto;white-space:pre-wrap}
#log.active{color:#00a6ff}
#log.done{color:#00e676}
#log.error{color:#ff4d6d}
</style>
</head>
<body>
<div class="wrap">
  <h1>Klap Shorts</h1>
  <div class="sub">GENERATES 10 CLIPS · 15–30 SECONDS · UPLOADS TO DRIVE</div>
  <div id="clip-card" class="clip-card">
    <div class="clip-thumb"></div>
    <div class="clip-info">
      <div class="clip-name">Loading latest clip...</div>
      <div class="clip-meta"></div>
    </div>
  </div>
  <div class="btn-row">
    <button id="go-btn">&#9654; Generate Shorts</button>
    <button id="refresh-btn">&#8635; Refresh</button>
  </div>
  <div id="log"></div>
</div>
<script>
var busy = false;
function setLog(text, cls) {
  var el = document.getElementById("log");
  el.textContent = text;
  el.className = cls || "";
  el.scrollTop = el.scrollHeight;
}
function loadLatestClip() {
  fetch("/api/latest-clip")
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (!data.clip) { document.querySelector(".clip-name").textContent = "No clips found."; return; }
      var card = document.getElementById("clip-card");
      var mb = data.clip.size ? (parseInt(data.clip.size)/1048576).toFixed(1)+" MB" : "";
      card.innerHTML = (data.clip.thumbnailLink ? "<img class='clip-thumb' src='"+data.clip.thumbnailLink+"'>" : "<div class='clip-thumb'></div>") +
        "<div class='clip-info'><div class='clip-name'>"+data.clip.name+"</div><div class='clip-meta'>"+mb+" &bull; latest</div></div>";
    });
}
function go() {
  if (busy) return;
  busy = true;
  document.getElementById("go-btn").disabled = true;
  setLog("Starting...", "active");
  fetch("/api/run", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({})})
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.error) { setLog("Error: "+data.error, "error"); busy=false; document.getElementById("go-btn").disabled=false; return; }
      pollLog();
    });
}
function pollLog() {
  var timer = setInterval(function() {
    fetch("/api/status").then(function(r) { return r.json(); }).then(function(data) {
      setLog((data.log||[]).join("\n"), data.running ? "active" : (data.done ? "done" : "error"));
      if (!data.running) { clearInterval(timer); busy=false; document.getElementById("go-btn").disabled=false; }
    });
  }, 2000);
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
