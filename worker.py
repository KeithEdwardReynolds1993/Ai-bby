import os
import io
import json
import hashlib
import subprocess
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account
import requests

app = Flask(__name__)

TMP = Path("/tmp/ai_bby")
INPUT = TMP / "input"
OUTPUT = TMP / "output"
MERGED = TMP / "merged.mp4"

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
ARCHIVE_FOLDER = os.getenv("GOOGLE_DRIVE_ARCHIVE_FOLDER_ID")
MAX_CAPTION_CHARS = int(os.getenv("MAX_CAPTION_CHARS") or "60")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pipeline_status = {"running": False, "log": [], "done": False, "error": None}

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
    if MERGED.exists():
        MERGED.unlink()
    for f in OUTPUT.glob("*.mp4"):
        f.unlink()

def log(msg):
    print(msg)
    pipeline_status["log"].append(msg)

def run(cmd):
    log(">>> " + " ".join(str(c) for c in cmd))
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
        body=file_metadata, media_body=media, fields="id, name", supportsAllDrives=True
    ).execute()
    log(f"Uploaded: {uploaded.get('name')} ({uploaded.get('id')})")
    return uploaded

def ask_openai(prompt, clip_names):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment variables")
    system = (
        "You are a video editing AI. Given a user prompt, return ONLY a JSON object with these fields:\n"
        "- caption: string (title to overlay on the video, max 60 chars)\n"
        "- speed: float (1.0=normal, 2.0=double speed, 0.5=half speed)\n"
        "- vibe: string (one of: normal, hype, cinematic, dreamy, gritty)\n"
        "- explanation: string (1-2 sentences explaining what you changed and why)\n"
        "Return only valid JSON, no markdown, no extra text."
    )
    user = f"Clips: {', '.join(clip_names)}\nPrompt: {prompt}"
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
        raise ValueError(f"OpenAI API error {resp.status_code}: {resp.text}")
    result = resp.json()
    content = result["choices"][0]["message"]["content"].strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())

def get_vibe_filters(vibe):
    filters = {
        "normal": "",
        "hype": ",eq=contrast=1.3:brightness=0.05:saturation=1.5",
        "cinematic": ",eq=contrast=1.1:brightness=-0.05:saturation=0.7,vignette",
        "dreamy": ",gblur=sigma=1.5,eq=brightness=0.08:saturation=0.8",
        "gritty": ",eq=contrast=1.4:brightness=-0.1:saturation=0.6,noise=alls=20:allf=t"
    }
    return filters.get(vibe, "")

def build_video(selected_files, caption, speed=1.0, vibe="normal"):
    ensure_dirs()
    clean_run_artifacts()
    if len(caption) > MAX_CAPTION_CHARS:
        raise ValueError(f"Caption too long ({len(caption)} chars). Max {MAX_CAPTION_CHARS}.")
    service = drive()
    local_clips = []
    for i, f in enumerate(selected_files):
        target = INPUT / f"clip{i+1:02d}.mp4"
        download_file(service, f, target)
        local_clips.append(target)

    if abs(speed - 1.0) > 0.01:
        sped_clips = []
        for i, clip in enumerate(local_clips):
            out = INPUT / f"sped{i+1:02d}.mp4"
            pts = round(1.0 / speed, 4)
            run(["ffmpeg", "-y", "-i", str(clip), "-vf", f"setpts={pts}*PTS", "-an", str(out)])
            sped_clips.append(out)
        local_clips = sped_clips

    list_file = TMP / "list.txt"
    list_file.write_text("\n".join([f"file '{c}'" for c in local_clips]), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(MERGED)])

    cap_hash = hashlib.sha256((caption + vibe + str(speed)).encode()).hexdigest()[:8]
    final_path = OUTPUT / f"final_{cap_hash}.mp4"
    safe_caption = ffmpeg_escape(caption)
    pad = 40
    vibe_filter = get_vibe_filters(vibe)

    drawtext = (
        f"drawtext=text='{safe_caption}':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
        f"fontcolor=white:"
        f"fontsize=54:"
        f"x=(w-text_w)/2:"
        f"y=(h-text_h)/2:"
        f"box=1:"
        f"boxcolor=black@0.72:"
        f"boxborderw={pad}"
    )

    vf = (
        f"scale=1080:1920:force_original_aspect_ratio=decrease,"
        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        f"{vibe_filter},"
        f"{drawtext}"
    )

    run(["ffmpeg", "-y", "-i", str(MERGED), "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", str(final_path)])
    return final_path

def run_pipeline(selected_files, caption, speed=1.0, vibe="normal"):
    global pipeline_status
    pipeline_status = {"running": True, "log": [], "done": False, "error": None}
    try:
        log("Starting pipeline...")
        log(f"Caption: {caption} | Speed: {speed}x | Vibe: {vibe}")
        final_path = build_video(selected_files, caption, speed, vibe)
        upload_output(final_path)
        log("Done! Video is in your output folder.")
        pipeline_status["done"] = True
    except Exception as e:
        log(f"Error: {e}")
        pipeline_status["error"] = str(e)
    finally:
        pipeline_status["running"] = False

HTML = b"""<!DOCTYPE html>
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
  --text: #e8e8f0; --muted: #5a5a7a; --danger: #ff4d6d; --radius: 12px;
}
body { font-family: 'Syne', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 40px 24px; }
.wrap { max-width: 900px; margin: 0 auto; }
header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 48px; }
h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; color: var(--accent); }
.subtitle { font-family: 'DM Mono', monospace; font-size: .75rem; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; }
section { margin-bottom: 40px; }
.section-label { font-family: 'DM Mono', monospace; font-size: .7rem; letter-spacing: .15em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; display: flex; align-items: center; gap: 10px; }
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.section-header .section-label { margin-bottom: 0; flex: 1; }
#clip-list { display: flex; flex-direction: column; gap: 8px; min-height: 60px; }
.clip-item { display: flex; align-items: center; gap: 14px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 18px; cursor: grab; transition: border-color .15s, background .15s; user-select: none; }
.clip-item:hover { border-color: var(--accent2); background: var(--surface2); }
.clip-item.dragging { opacity: .4; }
.clip-item.drag-over { border-color: var(--accent); }
.clip-item.unselected .clip-name { color: var(--muted); }
.thumb { width: 60px; height: 60px; object-fit: cover; border-radius: 6px; flex-shrink: 0; background: var(--surface2); }
.clip-check { width: 20px; height: 20px; accent-color: var(--accent); cursor: pointer; flex-shrink: 0; }
.drag-handle { color: var(--muted); flex-shrink: 0; }
.clip-name { flex: 1; font-size: .9rem; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
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
#log-wrap { display: none; margin-top: 32px; }
#log { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; font-family: 'DM Mono', monospace; font-size: .75rem; line-height: 1.8; color: var(--text); max-height: 320px; overflow-y: auto; white-space: pre-wrap; }
.empty { padding: 32px; text-align: center; font-family: 'DM Mono', monospace; font-size: .8rem; color: var(--muted); border: 1px dashed var(--border); border-radius: var(--radius); }
.refresh-btn { background: none; border: 1px solid var(--border); border-radius: 8px; color: var(--muted); font-family: 'DM Mono', monospace; font-size: .7rem; padding: 6px 14px; cursor: pointer; transition: border-color .15s, color .15s; }
.refresh-btn:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Clip Studio</h1>
    <span class="subtitle">Video Pipeline</span>
  </header>

  <section>
    <div class="section-header">
      <div class="section-label">Clips from Drive</div>
      <button class="refresh-btn" id="refresh-btn">&#8635; Refresh</button>
    </div>
    <div id="clip-list"><div class="empty">Loading clips...</div></div>
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
    <input id="ai-prompt" type="text" placeholder='e.g. "fast cuts, hype energy" or "make it cinematic and short"'>
    <div id="ai-plan">
      <div class="plan-label">AI Plan</div>
      <div id="ai-plan-text"></div>
    </div>
    <div class="btn-row">
      <button id="ask-ai-btn">Ask AI</button>
      <button id="run-btn" disabled>
        <span class="spinner" id="spinner"></span>
        <span id="btn-label">Run Pipeline</span>
      </button>
    </div>
  </section>

  <div id="log-wrap">
    <div class="section-label" style="margin-top:32px;margin-bottom:14px;">Pipeline Log</div>
    <div id="log"></div>
  </div>
</div>

<script>
var MAX_CHARS = 60;
var allFiles = [];
var dragSrc = null;
var aiSettings = { caption: "", speed: 1.0, vibe: "normal" };

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
    var mb = f.size ? (parseInt(f.size) / 1048576).toFixed(1) + " MB" : "";
    var thumb = f.thumbnailLink
      ? "<img class='thumb' src='" + f.thumbnailLink + "'>"
      : "<div class='thumb'></div>";
    el.innerHTML = "<span class='drag-handle'>&#8942;</span>"
      + thumb
      + "<input type='checkbox' class='clip-check' checked>"
      + "<span class='clip-name'>" + f.name + "</span>"
      + "<span class='clip-meta'>" + mb + "</span>";
    el.querySelector(".clip-check").addEventListener("change", function() {
      el.classList.toggle("unselected", !this.checked);
      updateRunBtn();
    });
    el.addEventListener("dragstart", function(e) { dragSrc = el; el.classList.add("dragging"); e.dataTransfer.effectAllowed = "move"; });
    el.addEventListener("dragover", function(e) { e.preventDefault(); document.querySelectorAll(".clip-item").forEach(function(x) { x.classList.remove("drag-over"); }); el.classList.add("drag-over"); });
    el.addEventListener("drop", function(e) {
      e.preventDefault(); if (dragSrc === el) return;
      var items = Array.from(list.querySelectorAll(".clip-item"));
      var si = items.indexOf(dragSrc), ti = items.indexOf(el);
      if (si < ti) list.insertBefore(dragSrc, el.nextSibling); else list.insertBefore(dragSrc, el);
      syncOrder();
    });
    el.addEventListener("dragend", function() { document.querySelectorAll(".clip-item").forEach(function(x) { x.classList.remove("dragging", "drag-over"); }); });
    list.appendChild(el);
  });
}

function syncOrder() {
  var ids = Array.from(document.querySelectorAll(".clip-item")).map(function(el) { return el.dataset.id; });
  allFiles = ids.map(function(id) { return allFiles.find(function(f) { return f.id === id; }); });
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

function askAI() {
  var prompt = document.getElementById("ai-prompt").value.trim();
  if (!prompt) return;
  var btn = document.getElementById("ask-ai-btn");
  btn.disabled = true;
  btn.textContent = "Thinking...";
  var clipNames = getSelectedFiles().map(function(f) { return f.name; });
  fetch("/api/ai-plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: prompt, clip_names: clipNames })
  })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      btn.disabled = false;
      btn.textContent = "Ask AI";
      if (data.error) { alert("AI error: " + data.error); return; }
      aiSettings = { caption: data.caption, speed: data.speed, vibe: data.vibe };
      document.getElementById("caption").value = data.caption;
      updateRunBtn();
      document.getElementById("ai-plan-text").innerHTML =
        "<b>Caption:</b> " + data.caption + "<br>" +
        "<b>Speed:</b> " + data.speed + "x<br>" +
        "<b>Vibe:</b> " + data.vibe + "<br>" +
        "<b>Plan:</b> " + data.explanation;
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
  document.getElementById("run-btn").disabled = true;
  document.getElementById("spinner").style.display = "block";
  document.getElementById("btn-label").textContent = "Running...";
  document.getElementById("log-wrap").style.display = "block";
  document.getElementById("log").textContent = "";
  fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files: selected, caption: caption, speed: aiSettings.speed, vibe: aiSettings.vibe })
  }).then(function() { pollLog(); });
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
document.getElementById("run-btn").addEventListener("click", runPipeline);
document.getElementById("ask-ai-btn").addEventListener("click", askAI);
document.getElementById("ai-prompt").addEventListener("keydown", function(e) {
  if (e.key === "Enter") askAI();
});

loadClips();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return app.response_class(HTML, mimetype="text/html")

@app.route("/api/clips")
def api_clips():
    try:
        files = list_incoming_files()
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai-plan", methods=["POST"])
def api_ai_plan():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        clip_names = data.get("clip_names", [])
        plan = ask_openai(prompt, clip_names)
        return jsonify(plan)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/run", methods=["POST"])
def api_run():
    global pipeline_status
    if pipeline_status.get("running"):
        return jsonify({"error": "Already running"}), 409
    data = request.json
    selected_files = data.get("files", [])
    caption = data.get("caption", "").strip()
    speed = float(data.get("speed") or 1.0)
    vibe = data.get("vibe") or "normal"
    if not selected_files:
        return jsonify({"error": "No files selected"}), 400
    if not caption:
        return jsonify({"error": "No caption provided"}), 400
    thread = threading.Thread(target=run_pipeline, args=(selected_files, caption, speed, vibe), daemon=True)
    thread.start()
    return jsonify({"ok": True})

@app.route("/api/status")
def api_status():
    return jsonify(pipeline_status)

if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 5000), debug=False)
