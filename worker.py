import os
import io
import json
import hashlib
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account

app = Flask(__name__)

TMP = Path("/tmp/ai_bby")
INPUT = TMP / "input"
OUTPUT = TMP / "output"
MERGED = TMP / "merged.mp4"
STATE_LOCAL = TMP / "processed_batches.json"

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
ARCHIVE_FOLDER = os.getenv("GOOGLE_DRIVE_ARCHIVE_FOLDER_ID")
MAX_CAPTION_CHARS = int(os.getenv("MAX_CAPTION_CHARS", "60"))

pipeline_status = {"running": False, "log": [], "done": False, "error": None}

def drive():
    creds = service_account.Credentials.from_service_account_info(
        SERVICE_ACCOUNT_INFO,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

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
        fields="files(id, name, modifiedTime, mimeType, size)",
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
        log(f"Archived: {f['name']}")

def build_video(selected_files, caption):
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
    list_file = TMP / "list.txt"
    list_file.write_text("\n".join([f"file '{c}'" for c in local_clips]), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(MERGED)])
    cap_hash = hashlib.sha256(caption.encode()).hexdigest()[:8]
    final_path = OUTPUT / f"final_{cap_hash}.mp4"
    safe_caption = ffmpeg_escape(caption)
    pad = 40
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
        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
        f"{drawtext}"
    )
    run(["ffmpeg", "-y", "-i", str(MERGED), "-vf", vf, "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "20", "-an", str(final_path)])
    return final_path

def run_pipeline(selected_files, caption):
    global pipeline_status
    pipeline_status = {"running": True, "log": [], "done": False, "error": None}
    try:
        log("Starting pipeline...")
        final_path = build_video(selected_files, caption)
        upload_output(final_path)
        archive_inputs(selected_files)
        log("Done! Video is in your output folder.")
        pipeline_status["done"] = True
    except Exception as e:
        log(f"Error: {e}")
        pipeline_status["error"] = str(e)
    finally:
        pipeline_status["running"] = False

HTML = (
    '<!DOCTYPE html><html lang="en"><head>'
    '<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">'
    '<title>Clip Studio</title>'
    '<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">'
    '<style>'
    '*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}'
    ':root{--bg:#0a0a0f;--surface:#13131a;--surface2:#1c1c27;--border:#2a2a3d;--accent:#c8ff00;--accent2:#7c4dff;--text:#e8e8f0;--muted:#5a5a7a;--danger:#ff4d6d;--radius:12px}'
    'body{font-family:\'Syne\',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:40px 24px}'
    '.wrap{max-width:860px;margin:0 auto}'
    'header{display:flex;align-items:baseline;gap:16px;margin-bottom:48px}'
    'h1{font-size:2.4rem;font-weight:800;letter-spacing:-0.03em;color:var(--accent)}'
    '.subtitle{font-family:\'DM Mono\',monospace;font-size:.75rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}'
    'section{margin-bottom:40px}'
    '.section-label{font-family:\'DM Mono\',monospace;font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:14px;display:flex;align-items:center;gap:10px}'
    '.section-label::after{content:\'\';flex:1;height:1px;background:var(--border)}'
    '.section-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}'
    '.section-header .section-label{margin-bottom:0;flex:1}'
    '#clip-list{display:flex;flex-direction:column;gap:8px;min-height:60px}'
    '.clip-item{display:flex;align-items:center;gap:14px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px 18px;cursor:grab;transition:border-color .15s,background .15s;user-select:none}'
    '.clip-item:hover{border-color:var(--accent2);background:var(--surface2)}'
    '.clip-item.dragging{opacity:.4}'
    '.clip-item.drag-over{border-color:var(--accent)}'
    '.clip-item.unselected .clip-name{color:var(--muted)}'
    '.clip-check{width:20px;height:20px;accent-color:var(--accent);cursor:pointer;flex-shrink:0}'
    '.drag-handle{color:var(--muted);font-size:1rem;flex-shrink:0}'
    '.clip-name{flex:1;font-size:.9rem;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}'
    '.clip-meta{font-family:\'DM Mono\',monospace;font-size:.68rem;color:var(--muted);flex-shrink:0}'
    '.caption-wrap{position:relative}'
    '#caption{width:100%;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;font-family:\'Syne\',sans-serif;font-size:1.1rem;font-weight:700;color:var(--text);outline:none;transition:border-color .15s}'
    '#caption:focus{border-color:var(--accent)}'
    '#caption.too-long{border-color:var(--danger)}'
    '.char-count{position:absolute;right:14px;top:50%;transform:translateY(-50%);font-family:\'DM Mono\',monospace;font-size:.7rem;color:var(--muted);pointer-events:none}'
    '.char-count.warn{color:var(--danger)}'
    '#run-btn{display:flex;align-items:center;gap:12px;background:var(--accent);color:#0a0a0f;border:none;border-radius:var(--radius);padding:16px 36px;font-family:\'Syne\',sans-serif;font-size:1rem;font-weight:800;cursor:pointer;transition:opacity .15s,transform .1s}'
    '#run-btn:hover:not(:disabled){opacity:.88;transform:translateY(-1px)}'
    '#run-btn:disabled{opacity:.35;cursor:not-allowed}'
    '.spinner{width:16px;height:16px;border:2px solid #0a0a0f;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;display:none}'
    '@keyframes spin{to{transform:rotate(360deg)}}'
    '#log-wrap{display:none;margin-top:32px}'
    '#log{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;font-family:\'DM Mono\',monospace;font-size:.75rem;line-height:1.8;color:var(--text);max-height:320px;overflow-y:auto;white-space:pre-wrap}'
    '.empty{padding:32px;text-align:center;font-family:\'DM Mono\',monospace;font-size:.8rem;color:var(--muted);border:1px dashed var(--border);border-radius:var(--radius)}'
    '.refresh-btn{background:none;border:1px solid var(--border);border-radius:8px;color:var(--muted);font-family:\'DM Mono\',monospace;font-size:.7rem;padding:6px 14px;cursor:pointer;transition:border-color .15s,color .15s}'
    '.refresh-btn:hover{border-color:var(--accent);color:var(--accent)}'
    '</style></head><body>'
    '<div class="wrap">'
    '<header><h1>Clip Studio</h1><span class="subtitle">Video Pipeline</span></header>'
    '<section>'
    '<div class="section-header"><div class="section-label">Clips from Drive</div><button class="refresh-btn" id="refresh-btn">&#8635; Refresh</button></div>'
    '<div id="clip-list"><div class="empty">Loading clips...</div></div>'
    '</section>'
    '<section>'
    '<div class="section-label">Title / Caption</div>'
    '<div class="caption-wrap">'
    '<input id="caption" type="text" placeholder="Enter title..." maxlength="100">'
    '<span class="char-count" id="char-count">0 / 60</span>'
    '</div></section>'
    '<button id="run-btn" disabled><span class="spinner" id="spinner"></span><span id="btn-label">Run Pipeline</span></button>'
    '<div id="log-wrap"><div class="section-label" style="margin-top:32px;margin-bottom:14px;">Pipeline Log</div><div id="log"></div></div>'
    '</div>'
    '<script>'
    'var MAX_CHARS=60,allFiles=[],dragSrc=null;'
    'function loadClips(){'
    '  var list=document.getElementById("clip-list");'
    '  list.innerHTML=\'<div class="empty">Fetching from Drive...</div>\';'
    '  fetch("/api/clips").then(function(r){return r.json();}).then(function(data){'
    '    allFiles=data.files||[];renderClips();updateRunBtn();'
    '  }).catch(function(){list.innerHTML=\'<div class="empty">Failed to load clips.</div>\';});'
    '}'
    'function renderClips(){'
    '  var list=document.getElementById("clip-list");'
    '  if(!allFiles.length){list.innerHTML=\'<div class="empty">No video files found.</div>\';return;}'
    '  list.innerHTML="";'
    '  allFiles.forEach(function(f,i){'
    '    var el=document.createElement("div");'
    '    el.className="clip-item";el.draggable=true;el.dataset.id=f.id;'
    '    var mb=f.size?(parseInt(f.size)/1048576).toFixed(1)+" MB":"";'
    '    el.innerHTML=\'<span class="drag-handle">&#8942;</span>\''
    '      +\'<input type="checkbox" class="clip-check" checked>\''
    '      +\'<span class="clip-name">\'+f.name+\'</span>\''
    '      +\'<span class="clip-meta">\'+mb+\'</span>\';'
    '    el.querySelector(".clip-check").addEventListener("change",function(){'
    '      el.classList.toggle("unselected",!this.checked);updateRunBtn();'
    '    });'
    '    el.addEventListener("dragstart",function(e){dragSrc=el;el.classList.add("dragging");e.dataTransfer.effectAllowed="move";});'
    '    el.addEventListener("dragover",function(e){e.preventDefault();document.querySelectorAll(".clip-item").forEach(function(x){x.classList.remove("drag-over");});el.classList.add("drag-over");});'
    '    el.addEventListener("drop",function(e){'
    '      e.preventDefault();if(dragSrc===el)return;'
    '      var items=Array.from(list.querySelectorAll(".clip-item"));'
    '      var si=items.indexOf(dragSrc),ti=items.indexOf(el);'
    '      if(si<ti)list.insertBefore(dragSrc,el.nextSibling);else list.insertBefore(dragSrc,el);'
    '      syncOrder();'
    '    });'
    '    el.addEventListener("dragend",function(){document.querySelectorAll(".clip-item").forEach(function(x){x.classList.remove("dragging","drag-over");});});'
    '    list.appendChild(el);'
    '  });'
    '}'
    'function syncOrder(){'
    '  var ids=Array.from(document.querySelectorAll(".clip-item")).map(function(el){return el.dataset.id;});'
    '  allFiles=ids.map(function(id){return allFiles.find(function(f){return f.id===id;});});'
    '}'
    'function updateRunBtn(){'
    '  var caption=document.getElementById("caption").value.trim();'
    '  var selected=getSelectedFiles();'
    '  document.getElementById("run-btn").disabled=!caption||caption.length>MAX_CHARS||selected.length===0;'
    '}'
    'function getSelectedFiles(){'
    '  return Array.from(document.querySelectorAll(".clip-item"))'
    '    .filter(function(el){return el.querySelector(".clip-check").checked;})'
    '    .map(function(el){return allFiles.find(function(f){return f.id===el.dataset.id;});})'
    '    .filter(Boolean);'
    '}'
    'function runPipeline(){'
    '  var caption=document.getElementById("caption").value.trim();'
    '  var selected=getSelectedFiles();'
    '  document.getElementById("run-btn").disabled=true;'
    '  document.getElementById("spinner").style.display="block";'
    '  document.getElementById("btn-label").textContent="Running...";'
    '  document.getElementById("log-wrap").style.display="block";'
    '  document.getElementById("log").textContent="";'
    '  fetch("/api/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({files:selected,caption:caption})})'
    '    .then(function(){pollLog();});'
    '}'
    'function pollLog(){'
    '  var logEl=document.getElementById("log");'
    '  var interval=setInterval(function(){'
    '    fetch("/api/status").then(function(r){return r.json();}).then(function(data){'
    '      logEl.textContent=data.log.join("\\n");'
    '      logEl.scrollTop=logEl.scrollHeight;'
    '      if(!data.running){'
    '        clearInterval(interval);'
    '        document.getElementById("spinner").style.display="none";'
    '        document.getElementById("run-btn").disabled=false;'
    '        document.getElementById("btn-label").textContent=data.done?"Done - Run Again":"Run Pipeline";'
    '      }'
    '    });'
    '  },1000);'
    '}'
    'document.getElementById("caption").addEventListener("input",function(){'
    '  var val=this.value;'
    '  var count=document.getElementById("char-count");'
    '  count.textContent=val.length+" / 60";'
    '  count.classList.toggle("warn",val.length>MAX_CHARS);'
    '  this.classList.toggle("too-long",val.length>MAX_CHARS);'
    '  updateRunBtn();'
    '});'
    'document.getElementById("refresh-btn").addEventListener("click",loadClips);'
    'document.getElementById("run-btn").addEventListener("click",runPipeline);'
    'loadClips();'
    '</script></body></html>'
)

@app.route("/")
def index():
    return HTML

@app.route("/api/clips")
def api_clips():
    try:
        files = list_incoming_files()
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/run", methods=["POST"])
def api_run():
    global pipeline_status
    if pipeline_status.get("running"):
        return jsonify({"error": "Already running"}), 409
    data = request.json
    selected_files = data.get("files", [])
    caption = data.get("caption", "").strip()
    if not selected_files:
        return jsonify({"error": "No files selected"}), 400
    if not caption:
        return jsonify({"error": "No caption provided"}), 400
    thread = threading.Thread(target=run_pipeline, args=(selected_files, caption), daemon=True)
    thread.start()
    return jsonify({"ok": True})

@app.route("/api/status")
def api_status():
    return jsonify(pipeline_status)

if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
