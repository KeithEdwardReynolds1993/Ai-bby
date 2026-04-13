import os
import shlex
import shutil
import subprocess
from pathlib import Path

APP_DIR = Path("/app")
TMP_DIR = Path("/tmp/ai_bby")
INPUT_DIR = APP_DIR / "input"
OUTPUT_DIR = APP_DIR / "output"

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAPTION_TEXT = os.getenv("CAPTION_TEXT", "Fun times in AI village")
FONT_FILE = os.getenv("FONT_FILE", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

INPUTS = [
    INPUT_DIR / "clip1.mp4",
    INPUT_DIR / "clip2.mp4",
    INPUT_DIR / "clip3.mp4",
]

NORMALIZED = [
    TMP_DIR / "clip1_norm.mp4",
    TMP_DIR / "clip2_norm.mp4",
    TMP_DIR / "clip3_norm.mp4",
]

CONCAT_LIST = TMP_DIR / "clips.txt"
MERGED = TMP_DIR / "merged.mp4"
FINAL_TMP = TMP_DIR / "final.mp4"
FINAL_OUT = OUTPUT_DIR / "final.mp4"


def log(*parts):
    print(*parts, flush=True)


def run_cmd(cmd):
    log("\n>>>", " ".join(shlex.quote(str(x)) for x in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    log(">>> return code:", result.returncode)
    if result.stdout.strip():
        log(">>> stdout:\n" + result.stdout[-4000:])
    if result.stderr.strip():
        log(">>> stderr:\n" + result.stderr[-4000:])
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    return result


def ensure_ffmpeg():
    run_cmd(["ffmpeg", "-version"])


def validate_inputs():
    missing = [str(p) for p in INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input clips. Expected these files:\n"
            + "\n".join(missing)
            + "\n\nPut 3 clips in /app/input as clip1.mp4, clip2.mp4, clip3.mp4"
        )


def normalize_clip(src: Path, dst: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-vf",
        (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "fps=30,"
            "format=yuv420p"
        ),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-ar", "48000",
        "-ac", "2",
        "-b:a", "128k",
        str(dst),
    ]
    run_cmd(cmd)


def write_concat_list():
    lines = [f"file '{p}'" for p in NORMALIZED]
    CONCAT_LIST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_clips():
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(CONCAT_LIST),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        str(MERGED),
    ]
    run_cmd(cmd)


def escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", r"\\")
            .replace(":", r"\:")
            .replace("'", r"\'")
            .replace("%", r"\%")
            .replace(",", r"\,")
            .replace("[", r"\[")
            .replace("]", r"\]")
    )


def add_caption():
    text = escape_drawtext(CAPTION_TEXT)
    vf = (
        "drawbox=x=120:y=1450:w=840:h=180:color=black@0.88:t=fill,"
        f"drawtext=fontfile='{FONT_FILE}':"
        f"text='{text}':"
        "fontcolor=white:fontsize=52:"
        "x=(w-text_w)/2:y=1507"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(MERGED),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(FINAL_TMP),
    ]
    run_cmd(cmd)


def publish_output():
    shutil.copy2(FINAL_TMP, FINAL_OUT)
    log(f"FINAL VIDEO: {FINAL_OUT}")
    log(f"TMP VIDEO:   {FINAL_TMP}")


def main():
    log("Ai-bby V1 starting...")
    log("Input dir:", INPUT_DIR)
    log("Output dir:", OUTPUT_DIR)
    log("Caption:", CAPTION_TEXT)

    ensure_ffmpeg()
    validate_inputs()

    for src, dst in zip(INPUTS, NORMALIZED):
        log(f"Normalizing: {src.name}")
        normalize_clip(src, dst)

    log("Writing concat list...")
    write_concat_list()

    log("Concatenating clips...")
    concat_clips()

    log("Adding caption...")
    add_caption()

    publish_output()
    log("Done.")


if __name__ == "__main__":
    main()
