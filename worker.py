import os
import io
import json
import hashlib
import shutil
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, render_template_string, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account

app = Flask(__name__)

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
STATE_FILENAME = os.getenv("STATE_FILENAME", "processed_batches.json")
MIN_FILE_AGE_SECONDS = int(os.getenv("MIN_FILE_AGE_SECONDS", "30"))
MAX_CAPTION_CHARS = int(os.getenv("MAX_CAPTION_CHARS", "60"))

# =========================
# PIPELINE STATE
# =========================

pipeline_status = {"running": False, "log": [], "done": False, "error": None}

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

# =========================
# DRIVE
# =========================

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
    request = service.files().get_media(fileId=file_obj["id"])
    with open(target, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
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
        fields="id, name",
        supportsAllDrives=True
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

# =========================
# FFMPEG
# =========================

def build_video(selected_files, caption):
    ensure_dirs()
    clean_run_artifacts()

    if len(caption) > MAX_CAPTION_CHARS:
        raise ValueError(f"Caption too long ({len(caption)} chars). Max {MAX_CAPTION_CHARS}.")

    service = drive()

    # Download selected clips in order
    local_clips = []
    for i, f in enumerate(selected_files):
        target = INPUT / f"clip{i+1:02d}_{f['name']}"
        download_file(service, f, target)
        local_clips.append(target)

    # Concat
    list_file = TMP / "list.txt"
    list_file.write_text(
        "\n".join([f"file '{c}'" for c in local_clips]),
        encoding="utf-8"
    )

    run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(MERGED)
    ])

    # Build output name from caption hash
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
# PIPELINE THREAD
# =========================

def run_pipeline(selected_files, caption):
    global pipeline_status
    pipeline_status = {"running": True, "log": [], "done": False, "error": None}

    try:
        log("Starting pipeline...")
        final_path = build_video(selected_files, caption)
        upload_output(final_path)
        archive_inputs(selected_files)
        log("✓ Done! Video is in your output folder.")
        pipeline_status["done"] = True
    except Exception as e:
        log(f"✗ Error: {e}")
        pipeline_status["error"] = str(e)
    finally:
        pipeline_status["running"] = False

# =========================
# ROUTES
# =========================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clip Studio</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #1c1c27;
    --border: #2a2a3d;
    --accent: #c8ff00;
    --accent2: #7c4dff;
    --text: #e8e8f0;
    --muted: #5a5a7a;
    --danger: #ff4d6d;
    --radius: 12px;
  }

  body {
    font-family: 'Syne', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 40px 24px;
  }

  body::before {
    content: '';
    position: fixed;
    top: -200px; left: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(124,77,255,0.08) 0%, transparent 70%);
    pointer-events: none;
  }

  .wrap { max-width: 860px; margin: 0 auto; }

  header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 48px;
  }

  h1 {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--accent);
  }

  .subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  section { margin-bottom: 40px; }

  .section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* Clip list */
  #clip-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-height: 60px;
  }

  .clip-item {
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 18px;
    cursor: grab;
    transition: border-color 0.15s, background 0.15s, transform 0.15s;
    user-select: none;
  }

  .clip-item:hover { border-color: var(--accent2); background: var(--surface2); }
  .clip-item.dragging { opacity: 0.4; transform: scale(0.98); }
  .clip-item.drag-over { border-color: var(--accent); }

  .clip-check {
    width: 20px; height: 20px;
    accent-color: var(--accent);
    cursor: pointer;
    flex-shrink: 0;
  }

  .drag-handle {
    color: var(--muted);
    font-size: 1rem;
    flex-shrink: 0;
    cursor: grab;
  }

  .clip-name {
    flex: 1;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .clip-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    flex-shrink: 0;
  }

  .clip-item.unselected .clip-name { color: var(--muted); }

  /* Caption input */
  .caption-wrap {
    position: relative;
  }

  #caption {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    outline: none;
    transition: border-color 0.15s;
    letter-spacing: -0.01em;
  }

  #caption:focus { border-color: var(--accent); }
  #caption.too-long { border-color: var(--danger); }

  .char-count {
    position: absolute;
    right: 14px; top: 50%;
    transform: translateY(-50%);
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    pointer-events: none;
  }

  .char-count.warn { color: var(--danger); }

  /* Run button */
  #run-btn {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--accent);
    color: #0a0a0f;
    border: none;
    border-radius: var(--radius);
    padding: 16px 36px;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
  }

  #run-btn:hover:not(:disabled) { opacity: 0.88; transform: translateY(-1px); }
  #run-btn:disabled { opacity: 0.35; cursor: not-allowed; }

  .spinner {
    width: 16px; height: 16px;
    border: 2px solid #0a0a0f;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: none;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  /* Log */
  #log-wrap {
    display: none;
    margin-top: 32px;
  }

  #log {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.8;
    color: var(--text);
    max-height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
  }

  .log-done { color: var(--accent); font-weight: 500; }
  .log-error { color: var(--danger); }

  /* Empty state */
  .empty {
    padding: 32px;
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    border: 1px dashed var(--border);
    border-radius: var(--radius);
  }

  /* Refresh */
  .refresh-btn {
    background: none;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 6px 14px;
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
  }

  .refresh-btn:hover { border-color: var(--accent); color: var(--accent); }

  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
  }

  .section-header .section-label { margin-bottom: 0; flex: 1; }
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
      <button class="refresh-btn" onclick="loadClips()">↻ Refresh</button>
    </div>
    <div id="clip-list"><div class="empty">Loading clips...</div></div>
  </section>

  <section>
    <div class="section-label">Title / Caption</div>
    <div class="caption-wrap">
      <input id="caption" type="text" placeholder="Enter title..." maxlength="100" oninput="onCaptionInput()">
      <span class="char-count" id="char-count">0 / {{ max_chars }}</span>
    </div>
  </section>

  <button id="run-btn" onclick="runPipeline()" disabled>
    <span class="spinner" id="spinner"></span>
    <span id="btn-label">Run Pipeline</span>
  </button>

  <div id="log-wrap">
    <div class="section-label" style="margin-top:32px; margin-bottom:14px;">Pipeline Log</div>
    <div id="log"></div>
  </div>
</div>

<script>
  const MAX_CHARS = {{ max_chars }};
  let allFiles = [];
  let dragSrc = null;

  // ---- Load clips ----
  async function loadClips() {
    const list = document.getElementById('clip-list');
    list.innerHTML = '<div class="empty">Fetching from Drive...</div>';
    try {
      const res = await fetch('/api/clips');
      const data = await res.json();
      allFiles = data.files || [];
      renderClips();
    } catch(e) {
      list.innerHTML = '<div class="empty">Failed to load clips.</div>';
    }
    updateRunBtn();
  }

  function renderClips() {
    const list = document.getElementById('clip-list');
    if (!allFiles.length) {
      list.innerHTML = '<div class="empty">No video files found in incoming folder.</div>';
      return;
    }

    list.innerHTML = '';
    allFiles.forEach((f, i) => {
      const el = document.createElement('div');
      el.className = 'clip-item';
      el.draggable = true;
      el.dataset.id = f.id;
      el.dataset.index = i;

      const sizeMB = f.size ? (parseInt(f.size) / 1048576).toFixed(1) + ' MB' : '';

      el.innerHTML = `
        <span class="drag-handle">⠿</span>
        <input type="checkbox" class="clip-check" checked onchange="onCheck(this, '${f.id}')">
        <span class="clip-name">${f.name}</span>
        <span class="clip-meta">${sizeMB}</span>
      `;

      el.addEventListener('dragstart', onDragStart);
      el.addEventListener('dragover', onDragOver);
      el.addEventListener('drop', onDrop);
      el.addEventListener('dragend', onDragEnd);

      list.appendChild(el);
    });
  }

  function onCheck(checkbox, id) {
    const item = checkbox.closest('.clip-item');
    item.classList.toggle('unselected', !checkbox.checked);
    updateRunBtn();
  }

  // ---- Drag to reorder ----
  function onDragStart(e) {
    dragSrc = this;
    this.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
  }

  function onDragOver(e) {
    e.preventDefault();
    document.querySelectorAll('.clip-item').forEach(el => el.classList.remove('drag-over'));
    this.classList.add('drag-over');
    e.dataTransfer.dropEffect = 'move';
  }

  function onDrop(e) {
    e.preventDefault();
    if (dragSrc === this) return;
    const list = document.getElementById('clip-list');
    const items = [...list.querySelectorAll('.clip-item')];
    const srcIdx = items.indexOf(dragSrc);
    const tgtIdx = items.indexOf(this);
    if (srcIdx < tgtIdx) list.insertBefore(dragSrc, this.nextSibling);
    else list.insertBefore(dragSrc, this);
    syncFilesOrder();
  }

  function onDragEnd() {
    document.querySelectorAll('.clip-item').forEach(el => {
      el.classList.remove('dragging', 'drag-over');
    });
  }

  function syncFilesOrder() {
    const items = document.querySelectorAll('.clip-item');
    const idOrder = [...items].map(el => el.dataset.id);
    allFiles = idOrder.map(id => allFiles.find(f => f.id === id));
  }

  // ---- Caption ----
  function onCaptionInput() {
    const val = document.getElementById('caption').value;
    const count = document.getElementById('char-count');
    const input = document.getElementById('caption');
    count.textContent = `${val.length} / ${MAX_CHARS}`;
    const over = val.length > MAX_CHARS;
    count.classList.toggle('warn', over);
    input.classList.toggle('too-long', over);
    updateRunBtn();
  }

  // ---- Run button ----
  function updateRunBtn() {
    const btn = document.getElementById('run-btn');
    const caption = document.getElementById('caption').value.trim();
    const selected = getSelectedFiles();
    btn.disabled = !caption || caption.length > MAX_CHARS || selected.length === 0;
  }

  function getSelectedFiles() {
    const items = document.querySelectorAll('.clip-item');
    return [...items]
      .filter(el => el.querySelector('.clip-check').checked)
      .map(el => allFiles.find(f => f.id === el.dataset.id))
      .filter(Boolean);
  }

  // ---- Pipeline ----
  async function runPipeline() {
    const caption = document.getElementById('caption').value.trim();
    const selected = getSelectedFiles();

    document.getElementById('run-btn').disabled = true;
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('btn-label').textContent = 'Running...';
    document.getElementById('log-wrap').style.display = 'block';
    document.getElementById('log').textContent = '';

    await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ files: selected, caption })
    });

    pollLog();
  }

  function pollLog() {
    const logEl = document.getElementById('log');
    const interval = setInterval(async () => {
      const res = await fetch('/api/status');
      const data = await res.json();

      logEl.textContent = data.log.join('\n');
      logEl.scrollTop = logEl.scrollHeight;

      if (!data.running) {
        clearInterval(interval);
        document.getElementById('spinner').style.display = 'none';
        document.getElementById('run-btn').disabled = false;

        if (data.done) {
          document.getElementById('btn-label').textContent = '✓ Done — Run Again';
          const last = logEl.lastElementChild || logEl;
          logEl.innerHTML += `\n`;
        } else if (data.error) {
          document.getElementById('btn-label').textContent = 'Run Pipeline';
        } else {
          document.getElementById('btn-label').textContent = 'Run Pipeline';
        }
      }
    }, 1000);
  }

  // Init
  loadClips();
  document.getElementById('caption').addEventListener('input', updateRunBtn);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, max_chars=MAX_CAPTION_CHARS)

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
    app.run(host="0.0.0.0", port=5000, debug=False)
