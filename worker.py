import os
import io
import json
import hashlib
import subprocess
import threading
import traceback
import tempfile
from pathlib import Path

import numpy as np
import librosa
from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account
import requests

app = Flask(__name__)

TMP = Path("/tmp/ai_bby")
INPUT = TMP / "input"
OUTPUT = TMP / "output"
MUSIC_DIR = TMP / "music"
MERGED = TMP / "merged.mp4"

SERVICE_ACCOUNT_INFO = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
INCOMING_FOLDER = os.getenv("GOOGLE_DRIVE_INCOMING_FOLDER_ID")
OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
ARCHIVE_FOLDER = os.getenv("GOOGLE_DRIVE_ARCHIVE_FOLDER_ID")
MUSIC_FOLDER = os.getenv("GOOGLE_DRIVE_MUSIC_FOLDER_ID")
MAX_CAPTION_CHARS = int(os.getenv("MAX_CAPTION_CHARS") or "60")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pipeline_status = {"running": False, "log": [], "done": False, "error": None}


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
        body=file_metadata, media_body=media, fields="id, name", supportsAllDrives=True
    ).execute()
    log(f"Uploaded: {uploaded.get('name')} ({uploaded.get('id')})")
    return uploaded


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
        capture_output=True, text=True
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
    """Return dict with bpm, downbeat timestamps, energy profile."""
    log(f"Analyzing music: {music_path.name}")
    y, sr = librosa.load(str(music_path), sr=None, mono=True)

    # BPM + beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Downbeats (every 4 beats)
    downbeat_times = beat_times[::4].tolist()

    # Energy (RMS per frame)
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_max = float(np.max(rms))

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.mean(centroid))

    duration = float(len(y) / sr)

    log(f"BPM: {float(tempo):.1f} | Downbeats: {len(downbeat_times)} | Duration: {duration:.1f}s")

    return {
        "bpm": float(tempo),
        "beat_times": beat_times.tolist(),
        "downbeat_times": downbeat_times,
        "duration": duration,
        "energy_mean": energy_mean,
        "energy_max": energy_max,
        "brightness": brightness,
    }

def get_cut_times(music_analysis, cut_style, num_clips, total_video_duration):
    """
    Return list of timestamps (seconds) at which to cut to next clip.
    cut_style: 'every_downbeat' | 'every_2_downbeats' | 'every_4_downbeats' | 'every_beat' | 'phrase'
    """
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

    # Filter to within video duration, skip first beat (start at 0)
    times = [t for t in times if 0 < t < total_video_duration]
    return times


# ─── AI Director ──────────────────────────────────────────────────────────────

def ask_openai(prompt, clip_names, music_files=None):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    music_list = "\n".join([f["name"] for f in music_files]) if music_files else "No music available"

    system = (
        "You are a professional AI video director. Given a user's creative prompt, "
        "analyze it and return a JSON editing plan.\n\n"
        "Think carefully about the emotional tone, energy, and pacing the user wants. "
        "Use your reasoning to pick settings that best serve their vision.\n\n"
        "Return ONLY a JSON object with these fields:\n"
        "- caption: string (title overlay, max 60 chars)\n"
        "- speed: float (1.0=normal, 1.5=fast, 0.75=slow)\n"
        "- vibe: string (one of: normal, hype, cinematic, dreamy, gritty, retro)\n"
        "- music_file: string (exact filename from the music list, or null)\n"
        "- cut_style: string (one of: every_beat, every_downbeat, every_2_downbeats, every_4_downbeats, phrase)\n"
        "- transition: string (one of: cut, fade, dissolve, wipeleft, wiperight, slideleft, slideright, zoom, pixelize, radial)\n"
        "- caption_fade_in: float (seconds before caption fades in, e.g. 0.5)\n"
        "- caption_fade_out: float (seconds before end caption fades out, e.g. 1.0)\n"
        "- ken_burns: boolean (slow zoom pan effect for cinematic feel)\n"
        "- explanation: string (2-3 sentences of your creative reasoning)\n\n"
        "cut_style guide:\n"
        "  every_beat = very fast cuts (hype reels)\n"
        "  every_downbeat = fast cuts (energetic)\n"
        "  every_2_downbeats = medium cuts (balanced)\n"
        "  every_4_downbeats = slow cuts (cinematic)\n"
        "  phrase = very slow, 8-bar phrases (emotional/documentary)\n\n"
        "Return only valid JSON, no markdown, no extra text."
    )

    user = (
        f"Clips: {', '.join(clip_names)}\n\n"
        f"Available music tracks:\n{music_list}\n\n"
        f"User prompt: {prompt}"
    )

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
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

    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


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

    # ── Download clips ──
    local_clips = []
    for i, f in enumerate(selected_files):
        target = INPUT / f"clip{i+1:02d}.mp4"
        download_file(service, f, target)
        local_clips.append(target)

    # ── Download music ──
    music_path = None
    music_analysis = None
    if music_file_obj:
        music_path = MUSIC_DIR / music_file_obj["name"]
        download_file(service, music_file_obj, music_path)
        music_analysis = analyze_music(music_path)

    # ── Apply speed ──
    if abs(speed - 1.0) > 0.01:
        sped_clips = []
        for i, clip in enumerate(local_clips):
            out = INPUT / f"sped{i+1:02d}.mp4"
            pts = round(1.0 / speed, 4)
            audio_filter = f"atempo={speed}" if speed <= 2.0 else f"atempo=2.0,atempo={speed/2.0:.4f}"
            run(["ffmpeg", "-y", "-i", str(clip),
                 "-vf", f"setpts={pts}*PTS",
                 "-af", audio_filter,
                 str(out)])
            sped_clips.append(out)
        local_clips = sped_clips

    # ── Get durations ──
    durations = [get_video_duration(c) for c in local_clips]
    total_duration = sum(durations)
    log(f"Total video duration: {total_duration:.1f}s")

    # ── Beat-synced cutting ──
    if music_analysis and cut_style != "cut":
        cut_times = get_cut_times(music_analysis, cut_style, len(local_clips), total_duration)
        log(f"Beat cut points: {[f'{t:.2f}s' for t in cut_times]}")

        # Trim/loop each clip to fit between cut points
        # Build segment boundaries
        boundaries = [0.0] + cut_times + [total_duration]
        segment_durations = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]

        trimmed_clips = []
        for i, (clip, seg_dur) in enumerate(zip(
            (local_clips * ((len(segment_durations) // len(local_clips)) + 1))[:len(segment_durations)],
            segment_durations
        )):
            out = INPUT / f"trim{i+1:02d}.mp4"
            clip_dur = get_video_duration(clip)
            # If clip is shorter than segment, loop it; if longer, trim it
            if clip_dur < seg_dur:
                loop_count = int(seg_dur / clip_dur) + 1
                run(["ffmpeg", "-y", "-stream_loop", str(loop_count), "-i", str(clip),
                     "-t", str(seg_dur), "-c", "copy", str(out)])
            else:
                run(["ffmpeg", "-y", "-i", str(clip),
                     "-t", str(seg_dur), "-c", "copy", str(out)])
            trimmed_clips.append(out)
        local_clips = trimmed_clips
        durations = segment_durations

    # ── Ken Burns zoom pan ──
    if ken_burns:
        kb_clips = []
        for i, clip in enumerate(local_clips):
            out = INPUT / f"kb{i+1:02d}.mp4"
            dur = durations[i] if i < len(durations) else get_video_duration(clip)
            frames = int(dur * 25)
            direction = i % 2  # alternate zoom in / zoom out
            if direction == 0:
                zoompan = f"zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1080x1920:fps=25"
            else:
                zoompan = f"zoompan=z='if(lte(zoom,1.0),1.5,max(1.0,zoom-0.0015))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1080x1920:fps=25"
            run(["ffmpeg", "-y", "-i", str(clip), "-vf", zoompan, "-an", str(out)])
            kb_clips.append(out)
        local_clips = kb_clips

    # ── Concat with xfade transitions ──
    xfade_name = XFADE_TRANSITIONS.get(transition)

    if xfade_name and len(local_clips) > 1:
        # Build complex filter for xfade between all clips
        xfade_dur = 0.3
        filter_parts = []
        inputs = []
        for i, clip in enumerate(local_clips):
            inputs += ["-i", str(clip)]

        # Chain xfades
        prev = "[0:v]"
        for i in range(1, len(local_clips)):
            clip_dur = durations[i-1] if i-1 < len(durations) else 2.0
            offset = max(0.1, clip_dur - xfade_dur)
            out_label = f"[xf{i}]" if i < len(local_clips) - 1 else "[vout]"
            filter_parts.append(
                f"{prev}[{i}:v]xfade=transition={xfade_name}:duration={xfade_dur}:offset={offset}{out_label}"
            )
            prev = f"[xf{i}]"

        filter_complex = ";".join(filter_parts)
        merged_xfade = TMP / "merged_xfade.mp4"
        run(["ffmpeg", "-y"] + inputs +
            ["-filter_complex", filter_complex,
             "-map", "[vout]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             str(merged_xfade)])
        merged_input = merged_xfade
    else:
        # Simple concat
        list_file = TMP / "list.txt"
        list_file.write_text("\n".join([f"file '{c}'" for c in local_clips]), encoding="utf-8")
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(MERGED)])
        merged_input = MERGED

    # ── Final render: scale + vibe + caption ──
    cap_hash = hashlib.sha256((caption + vibe + str(speed) + transition).encode()).hexdigest()[:8]
    final_path = OUTPUT / f"final_{cap_hash}.mp4"
    safe_caption = ffmpeg_escape(caption)
    pad = 40
    vibe_filter = get_vibe_filters(vibe)
    merged_duration = get_video_duration(merged_input)

    # Animated caption: fade in then fade out
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
        f"fontcolor=white:"
        f"fontsize=54:"
        f"x=(w-text_w)/2:"
        f"y=(h-text_h)/2:"
        f"box=1:"
        f"boxcolor=black@0.72:"
        f"boxborderw={pad}:"
        f"alpha='{alpha_expr}'"
    )

    vf = (
        f"scale=1080:1920:force_original_aspect_ratio=decrease,"
        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        f"{vibe_filter},"
        f"{drawtext}"
    )

    # ── Mix music ──
    if music_path and music_path.exists():
        log(f"Mixing music: {music_path.name}")
        # Trim or loop music to video duration
        music_trimmed = TMP / "music_trimmed.wav"
        music_dur = music_analysis["duration"] if music_analysis else 999
        if music_dur < merged_duration:
            loop_count = int(merged_duration / music_dur) + 1
            run(["ffmpeg", "-y", "-stream_loop", str(loop_count), "-i", str(music_path),
                 "-t", str(merged_duration), "-af", "afade=t=out:st=" + str(merged_duration - 2) + ":d=2",
                 str(music_trimmed)])
        else:
            run(["ffmpeg", "-y", "-i", str(music_path),
                 "-t", str(merged_duration),
                 "-af", f"afade=t=out:st={merged_duration - 2}:d=2",
                 str(music_trimmed)])

        # Render video + mixed audio
        run(["ffmpeg", "-y",
             "-i", str(merged_input),
             "-i", str(music_trimmed),
             "-filter_complex",
             f"[0:v]{vf}[vout];"
             f"[1:a]volume=0.85,afade=t=in:st=0:d=1[aout]",
             "-map", "[vout]", "-map", "[aout]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-c:a", "aac", "-b:a", "192k",
             str(final_path)])
    else:
        run(["ffmpeg", "-y", "-i", str(merged_input), "-vf", vf,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an",
             str(final_path)])

    return final_path


def run_pipeline(selected_files, caption, speed=1.0, vibe="normal",
                 music_file_obj=None, cut_style="every_downbeat",
                 transition="cut", caption_fade_in=0.5, caption_fade_out=1.0,
                 ken_burns=False):
    global pipeline_status
    pipeline_status = {"running": True, "log": [], "done": False, "error": None}
    try:
        log("Starting pipeline...")
        log(f"Caption: {caption} | Speed: {speed}x | Vibe: {vibe} | Cut: {cut_style} | Transition: {transition}")
        if music_file_obj:
            log(f"Music: {music_file_obj['name']}")
        final_path = build_video(
            selected_files, caption, speed, vibe,
            music_file_obj, cut_style, transition,
            caption_fade_in, caption_fade_out, ken_burns
        )
        upload_output(final_path)
        log("Done! Video is in your output folder.")
        pipeline_status["done"] = True
    except Exception as e:
        log(f"Error: {e}")
        log(traceback.format_exc())
        pipeline_status["error"] = str(e)
    finally:
        pipeline_status["running"] = False


# ─── HTML UI ──────────────────────────────────────────────────────────────────

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
.music-radio { width: 18px; height: 18px; accent-color: var(--accent); flex-shrink: 0; }
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
#log-wrap { display: none; margin-top: 32px; }
#log { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; font-family: 'DM Mono', monospace; font-size: .75rem; line-height: 1.8; color: var(--text); max-height: 320px; overflow-y: auto; white-space: pre-wrap; }
.empty { padding: 32px; text-align: center; font-family: 'DM Mono', monospace; font-size: .8rem; color: var(--muted); border: 1px dashed var(--border); border-radius: var(--radius); }
.refresh-btn { background: none; border: 1px solid var(--border); border-radius: 8px; color: var(--muted); font-family: 'DM Mono', monospace; font-size: .7rem; padding: 6px 14px; cursor: pointer; transition: border-color .15s, color .15s; }
.refresh-btn:hover { border-color: var(--accent); color: var(--accent); }
.no-music-btn { background: none; border: 1px dashed var(--border); border-radius: var(--radius); color: var(--muted); font-family: 'DM Mono', monospace; font-size: .75rem; padding: 10px 18px; cursor: pointer; width: 100%; text-align: left; transition: border-color .15s; }
.no-music-btn:hover, .no-music-btn.selected { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Clip Studio</h1>
    <span class="subtitle">AI Video Director</span>
  </header>

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
    <input id="ai-prompt" type="text" placeholder='e.g. "hype reel, fast cuts" or "slow cinematic, emotional"'>
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
var allMusic = [];
var dragSrc = null;
var selectedMusicId = null;
var aiSettings = {
  caption: "", speed: 1.0, vibe: "normal",
  cut_style: "every_downbeat", transition: "cut",
  caption_fade_in: 0.5, caption_fade_out: 1.0, ken_burns: false
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
    .catch(function() { list.innerHTML = "<div class='empty'>Failed to load clips.</div>"; });
}

function loadMusic() {
  var list = document.getElementById("music-list");
  list.innerHTML = "<div class='empty'>Fetching music...</div>";
  fetch("/api/music")
    .then(function(r) { return r.json(); })
    .then(function(data) {
      allMusic = data.files || [];
      renderMusic();
    })
    .catch(function() { list.innerHTML = "<div class='empty'>Failed to load music.</div>"; });
}

function renderClips() {
  var list = document.getElementById("clip-list");
  if (!allFiles.length) { list.innerHTML = "<div class='empty'>No video files found.</div>"; return; }
  list.innerHTML = "";
  allFiles.forEach(function(f) {
    var el = document.createElement("div");
    el.className = "clip-item";
    el.draggable = true;
    el.dataset.id = f.id;
    var mb = f.size ? (parseInt(f.size) / 1048576).toFixed(1) + " MB" : "";
    var thumb = f.thumbnailLink ? "<img class='thumb' src='" + f.thumbnailLink + "'>" : "<div class='thumb'></div>";
    el.innerHTML = "<span class='drag-handle'>::</span>" + thumb +
      "<input type='checkbox' class='clip-check' checked>" +
      "<span class='clip-name'>" + f.name + "</span>" +
      "<span class='clip-meta'>" + mb + "</span>";
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

function renderMusic() {
  var list = document.getElementById("music-list");
  list.innerHTML = "";

  // No music option
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
    em.textContent = "No music files found in Drive music folder.";
    list.appendChild(em);
    return;
  }

  allMusic.forEach(function(f) {
    var el = document.createElement("div");
    el.className = "music-item" + (selectedMusicId === f.id ? " selected" : "");
    el.dataset.id = f.id;
    var mb = f.size ? (parseInt(f.size) / 1048576).toFixed(1) + " MB" : "";
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

      // Auto-select AI's music pick
      if (data.music_file) {
        var match = allMusic.find(function(f) { return f.name === data.music_file; });
        if (match) { selectedMusicId = match.id; renderMusic(); }
      }

      document.getElementById("ai-plan-text").innerHTML =
        "<b>Caption:</b> " + data.caption + "<br>" +
        "<b>Music:</b> " + (data.music_file || "none") + "<br>" +
        "<b>Cut style:</b> " + aiSettings.cut_style + "<br>" +
        "<b>Transition:</b> " + aiSettings.transition + "<br>" +
        "<b>Vibe:</b> " + data.vibe + " | <b>Speed:</b> " + data.speed + "x | <b>Ken Burns:</b> " + (data.ken_burns ? "yes" : "no") + "<br>" +
        "<b>Reasoning:</b> " + data.explanation;
      document.getElementById("ai-plan").style.display = "block";
    })
    .catch(function() { btn.disabled = false; btn.textContent = "Ask AI"; alert("Failed to reach AI."); });
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
document.getElementById("refresh-music-btn").addEventListener("click", loadMusic);
document.getElementById("run-btn").addEventListener("click", runPipeline);
document.getElementById("ask-ai-btn").addEventListener("click", askAI);
document.getElementById("ai-prompt").addEventListener("keydown", function(e) { if (e.key === "Enter") askAI(); });

loadClips();
loadMusic();
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
    data = request.json
    selected_files = data.get("files", [])
    caption = data.get("caption", "").strip()
    if not selected_files or not caption:
        return jsonify({"error": "Missing files or caption"}), 400

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


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 5000), debug=False)
