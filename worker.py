import os
import json
import subprocess
import threading
import traceback
from pathlib import Path

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
MERGED = TMP / "merged.mp4"
MERGED_CAPPED = TMP / "merged_capped.mp4"

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
ARCHIVE_FOLDER = os.getenv("GOOGLE_DRIVE_ARCHIVE_FOLDER_ID")
MUSIC_FOLDER = os.getenv("GOOGLE_DRIVE_MUSIC_FOLDER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_CAPTION_CHARS = int(os.getenv("MAX_CAPTION_CHARS") or "60")
MAX_OUTPUT_DURATION = 30

pipeline_status = {"running": False, "log": [], "done": False, "error": None}
pipeline_lock = threading.Lock()


def log(msg):
    print(msg)
    pipeline_status["log"].append(str(msg))


def safe_float(val, default=1.0):
    try:
        return float(str(val).replace("x", "").strip())
    except Exception:
        return default


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
    for p in [MERGED, MERGED_CAPPED]:
        if p.exists():
            p.unlink()
    for f in INPUT.glob("*"):
        if f.is_file():
            f.unlink()
    for f in OUTPUT.glob("*"):
        if f.is_file():
            f.unlink()
    for f in MUSIC_DIR.glob("*"):
        if f.is_file():
            f.unlink()


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


def get_video_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True,
        text=True,
        check=True
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def get_vibe_filters(vibe):
    return {
        "normal": "",
        "hype": ",eq=contrast=1.25:brightness=0.04:saturation=1.35",
        "cinematic": ",eq=contrast=1.08:brightness=-0.04:saturation=0.8",
        "dreamy": ",gblur=sigma=1.2,eq=brightness=0.06:saturation=0.85",
        "gritty": ",eq=contrast=1.35:brightness=-0.08:saturation=0.65",
        "retro": ",eq=saturation=0.8"
    }.get(vibe, "")


def list_incoming_files():
    service = drive()
    results = service.files().list(
        q=f"'{INCOMING_FOLDER}' in parents and trashed=false",
        fields="files(id,name,mimeType,size,thumbnailLink)",
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
        fields="files(id,name,mimeType,size)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives"
    ).execute()
    files = results.get("files", [])
    return [f for f in files if f.get("mimeType", "").startswith(("audio/", "video/mp4"))]


def download_file(service, file_obj, target):
    req = service.files().get_media(fileId=file_obj["id"])
    with open(target, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    log(f"Downloaded: {file_obj['name']}")


def upload_output(final_path):
    service = drive()
    uploaded = service.files().create(
        body={"name": final_path.name, "parents": [OUTPUT_FOLDER]},
        media_body=MediaFileUpload(str(final_path), mimetype="video/mp4"),
        fields="id,name",
        supportsAllDrives=True
    ).execute()
    log(f"Uploaded: {uploaded.get('name')} ({uploaded.get('id')})")
    return uploaded


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


def ask_openai(prompt, clip_names, music_files=None):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    music_list = "\n".join([f["name"] for f in (music_files or [])]) or "No music available"

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        },
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return ONLY JSON with keys: "
                        "caption, speed, vibe, music_file, cut_style, "
                        "caption_fade_in, caption_fade_out, ken_burns, explanation."
                    )
                },
                {
                    "role": "user",
                    "content": f"Clips: {', '.join(clip_names)}\nMusic:\n{music_list}\nPrompt: {prompt}"
                }
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


def build_video(selected_files, caption, speed=1.0, vibe="normal",
                music_file_obj=None, cut_style="every_downbeat",
                caption_fade_in=0.5, caption_fade_out=1.0,
                ken_burns=False):
    ensure_dirs()
    clean_run_artifacts()

    if len(caption) > MAX_CAPTION_CHARS:
        raise ValueError(f"Caption too long ({len(caption)} chars). Max {MAX_CAPTION_CHARS}.")

    service = drive()
    local_clips = []

    for i, f in enumerate(selected_files):
        raw = INPUT / f"raw{i+1:02d}.mp4"
        target = INPUT / f"clip{i+1:02d}.mp4"
        download_file(service, f, raw)

        run([
            "ffmpeg", "-y", "-i", str(raw),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(target)
        ])
        local_clips.append(target)

    capped = []
    for i, clip in enumerate(local_clips):
        out = INPUT / f"capped{i+1:02d}.mp4"
        run(["ffmpeg", "-y", "-i", str(clip), "-t", str(MAX_OUTPUT_DURATION), "-c", "copy", str(out)])
        capped.append(out)
    local_clips = capped

    if abs(speed - 1.0) > 0.01:
        sped = []
        for i, clip in enumerate(local_clips):
            out = INPUT / f"sped{i+1:02d}.mp4"
            pts = round(1.0 / speed, 4)
            run(["ffmpeg", "-y", "-i", str(clip), "-vf", f"setpts={pts}*PTS", "-an", str(out)])
            sped.append(out)
        local_clips = sped

    list_file = TMP / "list.txt"
    list_file.write_text("\n".join([f"file '{c}'" for c in local_clips]), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(MERGED)])

    raw_merged_duration = get_video_duration(MERGED)
    if raw_merged_duration > MAX_OUTPUT_DURATION:
        run(["ffmpeg", "-y", "-i", str(MERGED), "-t", str(MAX_OUTPUT_DURATION), "-c", "copy", str(MERGED_CAPPED)])
        merged_input = MERGED_CAPPED
    else:
        merged_input = MERGED

    merged_duration = min(raw_merged_duration, MAX_OUTPUT_DURATION)
    final_path = OUTPUT / "final.mp4"

    music_path = None
    if music_file_obj:
        music_path = MUSIC_DIR / music_file_obj["name"]
        download_file(service, music_file_obj, music_path)

    safe_caption = ffmpeg_escape(caption)
    vibe_filter = get_vibe_filters(vibe)
    kb_filter = ",scale=iw*1.05:ih*1.05,crop=iw/1.05:ih/1.05" if ken_burns else ""

    fade_in_end = caption_fade_in + 0.5
    fade_out_start = max(fade_in_end + 0.5, merged_duration - caption_fade_out - 0.5)
    alpha_expr = (
        f"if(lt(t,{caption_fade_in}),0,"
        f"if(lt(t,{fade_in_end}),(t-{caption_fade_in})/0.5,"
        f"if(lt(t,{fade_out_start}),1,"
        f"if(lt(t,{fade_out_start+0.5}),({fade_out_start+0.5}-t)/0.5,0))))"
    )

    vf = (
        f"scale=1080:1920:force_original_aspect_ratio=decrease,"
        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black{kb_filter}{vibe_filter},"
        f"drawtext=text='{safe_caption}':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontcolor=white:fontsize=54:x=(w-text_w)/2:y=(h-text_h)/2:"
        f"box=1:boxcolor=black@0.72:boxborderw=40:alpha='{alpha_expr}'"
    )

    if music_path and music_path.exists():
        music_trimmed = TMP / "music_trimmed.wav"
        run([
            "ffmpeg", "-y", "-i", str(music_path), "-t", str(merged_duration),
            "-af", f"afade=t=out:st={max(0, merged_duration - 2)}:d=2",
            str(music_trimmed)
        ])
        run([
            "ffmpeg", "-y",
            "-i", str(merged_input),
            "-i", str(music_trimmed),
            "-t", str(merged_duration),
            "-filter_complex", f"[0:v]{vf}[vout];[1:a]volume=0.85[aout]",
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

    return final_path


def run_pipeline(selected_files, caption, speed=1.0, vibe="normal",
                 music_file_obj=None, cut_style="every_downbeat",
                 caption_fade_in=0.5, caption_fade_out=1.0,
                 ken_burns=False):
    global pipeline_status

    with pipeline_lock:
        pipeline_status = {"running": True, "log": [], "done": False, "error": None}
        try:
            log("Starting pipeline...")
            log(f"Caption: {caption} | Speed: {speed}x | Vibe: {vibe}")

            final_path = build_video(
                selected_files,
                caption,
                speed,
                vibe,
                music_file_obj,
                cut_style,
                caption_fade_in,
                caption_fade_out,
                ken_burns
            )

            uploaded = upload_output(final_path)
            archive_clips(selected_files)

            pipeline_status["done"] = True
            pipeline_status["drive_file_id"] = uploaded.get("id")
            pipeline_status["drive_file_name"] = uploaded.get("name")
            log("Done!")

        except Exception as e:
            log(f"Error: {e}")
            log(traceback.format_exc())
            pipeline_status["error"] = str(e)
        finally:
            pipeline_status["running"] = False


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clip Studio</title>
<style>
body{font-family:Arial,sans-serif;background:#0a0a0f;color:#fff;max-width:760px;margin:0 auto;padding:32px}
h1{color:#c8ff00}
.section{margin-bottom:24px}
.box{border:1px solid #333;padding:16px;border-radius:12px;background:#13131a}
.item{display:flex;gap:10px;align-items:center;padding:10px;border:1px solid #2a2a3d;border-radius:10px;margin-bottom:8px}
.thumb{width:52px;height:52px;object-fit:cover;border-radius:6px;background:#222}
button{padding:12px 18px;border:none;border-radius:10px;cursor:pointer;font-weight:bold}
#ask-ai-btn{background:#7c4dff;color:#fff}
#run-btn{background:#c8ff00;color:#000}
input{width:100%;padding:14px;border-radius:10px;border:1px solid #333;background:#13131a;color:#fff}
#log{white-space:pre-wrap;font-family:monospace;max-height:260px;overflow:auto}
</style>
</head>
<body>
<h1>Clip Studio</h1>

<div class="section">
  <h3>Clips</h3>
  <div id="clip-list" class="box">Loading clips...</div>
</div>

<div class="section">
  <h3>Music</h3>
  <div id="music-list" class="box">Loading music...</div>
</div>

<div class="section">
  <h3>Title / Caption</h3>
  <input id="caption" placeholder="Enter title...">
</div>

<div class="section">
  <h3>AI Director</h3>
  <input id="ai-prompt" placeholder='e.g. "hype reel, fast cuts"'>
</div>

<div class="section">
  <button id="ask-ai-btn">Ask AI</button>
  <button id="run-btn">Run Pipeline</button>
</div>

<div class="section box">
  <h3>Pipeline Log</h3>
  <div id="log"></div>
</div>

<script>
let allFiles = [];
let allMusic = [];
let selectedMusicId = null;
let aiSettings = {
  caption: "",
  speed: 1.0,
  vibe: "normal",
  cut_style: "every_downbeat",
  caption_fade_in: 0.5,
  caption_fade_out: 1.0,
  ken_burns: false
};

function getSelectedFiles() {
  return Array.from(document.querySelectorAll(".clip-check"))
    .filter(x => x.checked)
    .map(x => allFiles.find(f => f.id === x.dataset.id))
    .filter(Boolean);
}

function getSelectedMusic() {
  if (!selectedMusicId) return null;
  return allMusic.find(f => f.id === selectedMusicId) || null;
}

function loadClips() {
  fetch("/api/clips").then(r => r.json()).then(data => {
    allFiles = data.files || [];
    const list = document.getElementById("clip-list");
    if (!allFiles.length) {
      list.innerHTML = "No video files found.";
      return;
    }
    list.innerHTML = allFiles.map(f => {
      const thumb = f.thumbnailLink ? `<img class="thumb" src="${f.thumbnailLink}">` : `<div class="thumb"></div>`;
      return `<label class="item">${thumb}<input class="clip-check" data-id="${f.id}" type="checkbox" checked> <span>${f.name}</span></label>`;
    }).join("");
  }).catch(() => {
    document.getElementById("clip-list").textContent = "Failed to load clips.";
  });
}

function loadMusic() {
  fetch("/api/music").then(r => r.json()).then(data => {
    allMusic = data.files || [];
    const list = document.getElementById("music-list");
    let html = `<label class="item"><input type="radio" name="music" checked onclick="selectedMusicId=null"> <span>No music</span></label>`;
    html += allMusic.map(f =>
      `<label class="item"><input type="radio" name="music" onclick="selectedMusicId='${f.id}'"> <span>${f.name}</span></label>`
    ).join("");
    list.innerHTML = html;
  }).catch(() => {
    document.getElementById("music-list").textContent = "Failed to load music.";
  });
}

function askAI() {
  const prompt = document.getElementById("ai-prompt").value.trim();
  if (!prompt) return;

  fetch("/api/ai-plan", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      prompt,
      clip_names: getSelectedFiles().map(f => f.name)
    })
  }).then(r => r.json()).then(data => {
    if (data.error) {
      document.getElementById("log").textContent = "AI error: " + data.error;
      return;
    }
    aiSettings = {
      caption: data.caption || "",
      speed: parseFloat(String(data.speed || "1").replace("x", "")) || 1.0,
      vibe: data.vibe || "normal",
      cut_style: data.cut_style || "every_downbeat",
      caption_fade_in: data.caption_fade_in || 0.5,
      caption_fade_out: data.caption_fade_out || 1.0,
      ken_burns: !!data.ken_burns
    };
    document.getElementById("caption").value = data.caption || "";
    document.getElementById("log").textContent = JSON.stringify(aiSettings, null, 2);
  }).catch(err => {
    document.getElementById("log").textContent = "AI request failed: " + err;
  });
}

function runPipeline() {
  const caption = document.getElementById("caption").value.trim();
  const files = getSelectedFiles();
  const music = getSelectedMusic();
  document.getElementById("log").textContent = "";

  fetch("/api/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      files,
      caption,
      music_file: music,
      speed: parseFloat(String(aiSettings.speed || "1").replace("x", "")) || 1.0,
      vibe: aiSettings.vibe,
      cut_style: aiSettings.cut_style,
      caption_fade_in: aiSettings.caption_fade_in,
      caption_fade_out: aiSettings.caption_fade_out,
      ken_burns: aiSettings.ken_burns
    })
  }).then(r => r.json()).then(data => {
    if (data.error) {
      document.getElementById("log").textContent = "Error: " + data.error;
      return;
    }
    pollLog();
  }).catch(err => {
    document.getElementById("log").textContent = "Run failed: " + err;
  });
}

function pollLog() {
  const timer = setInterval(() => {
    fetch("/api/status").then(r => r.json()).then(data => {
      document.getElementById("log").textContent = (data.log || []).join("\\n");
      if (!data.running) clearInterval(timer);
    });
  }, 1000);
}

document.getElementById("ask-ai-btn").addEventListener("click", askAI);
document.getElementById("run-btn").addEventListener("click", runPipeline);

loadClips();
loadMusic();
</script>
</body>
</html>
"""


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
        data = request.json or {}
        music_files = list_music_files()
        plan = ask_openai(data.get("prompt", ""), data.get("clip_names", []), music_files)
        return jsonify(plan)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/run", methods=["POST"])
def api_run():
    global pipeline_status
    try:
        if pipeline_status.get("running"):
            return jsonify({"error": "Already running"}), 409

        data = request.json or {}
        selected_files = data.get("files", [])
        caption = data.get("caption", "").strip()

        if not selected_files or not caption:
            return jsonify({"error": "Missing files or caption"}), 400

        thread = threading.Thread(
            target=run_pipeline,
            args=(
                selected_files,
                caption,
                safe_float(data.get("speed"), 1.0),
                data.get("vibe") or "normal",
                data.get("music_file"),
                data.get("cut_style") or "every_downbeat",
                safe_float(data.get("caption_fade_in"), 0.5),
                safe_float(data.get("caption_fade_out"), 1.0),
                bool(data.get("ken_burns") or False),
            ),
            daemon=True
        )
        thread.start()
        return jsonify({"ok": True})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def api_status():
    return jsonify(pipeline_status)


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 5000), debug=False)
