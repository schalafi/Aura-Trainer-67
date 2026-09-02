"""Run reference-dance pose extraction (+ a debug pose-overlay video) as a
Modal Function instead of on your local machine.

Why: local extraction has proven unreliable across environments (sandboxed
network blocks on youtube.com, yt-dlp bot/PO-token checks, a macOS
GPU-delegate crash in MediaPipe). This uploads your local clip to a Modal
Function that runs the same MediaPipe pose extraction in a consistent Linux
container, forcing CPU inference explicitly (see pose_extraction.py) so the
GPU-delegate crash class can't happen there either.

One-time setup (on your own machine; this is your own Modal account, nothing
is shared with or stored by this chat):
    pip install modal
    modal setup   # opens a browser login

Usage:
    modal run tools/modal_extract.py \\
        --local-file clips/Howard-The-Alien-ORIGINAL-VIDEO.mp4 \\
        --start 0 --end 30 \\
        --slug howard_the_alien \\
        --title "Howard the Alien"

This uploads the local video's bytes to Modal for this one run only (nothing
is kept remotely afterward -- the function is stateless), runs pose
extraction there, and:
  - writes data/dances/<slug>.json + updates data/dances/dances.json locally
  - saves a debug overlay video (skeleton drawn over the original footage) to
    clips/<slug>_pose_overlay.avi (gitignored) so you can eyeball detection
    quality before it's wired into the app -- .avi/MJPG rather than .mp4,
    since OpenCV's mp4v encoder produces corrupted output in this container

Note: Modal automatically routes function argument/return payloads over 2MiB
through its blob storage rather than the direct RPC path, transparently --
no special handling needed here. That's the documented pattern for passing
local files to a remote function (read bytes, pass as an argument). A
future dataset-scale need (hundreds of clips, huge files) would call for a
modal.Volume instead, but a single 30s clip doesn't.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
DATA_DIR = REPO_ROOT / "data" / "dances"
CLIPS_DIR = REPO_ROOT / "clips"

sys.path.insert(0, str(TOOLS_DIR))


def _bake_model():
    import pose_extraction

    pose_extraction.ensure_model()


image = (
    modal.Image.debian_slim(python_version="3.11")
    # mediapipe's bundled OpenCV/OpenGL-ES components dynamically link against
    # these even though we use opencv-python-headless ourselves and never
    # render anything -- Debian's slim image doesn't ship any of them. Full
    # set (not just libgl1) to avoid finding the next missing .so one at a
    # time: libGL, EGL/GLES (mediapipe's GPU-capable calculators), and the
    # X11/GL runtime libs OpenCV's GUI-capable build otherwise expects.
    .apt_install(
        "libgl1",
        "libegl1",
        "libgles2",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
    )
    .pip_install("mediapipe>=0.10.14", "opencv-python-headless", "numpy", "scipy")
    # copy=True: _bake_model (a build step below) needs to import pose_extraction,
    # so it must be copied into the image layer now rather than mounted at
    # container startup (Modal's default for add_local_* when it's the last step).
    .add_local_python_source("pose_extraction", copy=True)
    .run_function(_bake_model)
)

app = modal.App("aura-trainer-extract", image=image)


@app.function(timeout=900)
def extract_and_overlay(video_bytes: bytes, start: float, end: float) -> dict:
    import tempfile

    import pose_extraction

    model_path = pose_extraction.ensure_model()

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "input.mp4"
        in_path.write_bytes(video_bytes)
        out_path = Path(tmp) / "overlay.avi"

        # Pass 1: detect raw landmarks over every frame in [start, end].
        indexed, fps = pose_extraction.detect_landmarks(
            in_path, model_path, start_seconds=start, end_seconds=end
        )
        # Smooth over the whole timeline (reduces per-frame jitter) before
        # both exporting the JSON and drawing the debug overlay, so the
        # overlay video actually reflects what gets saved.
        indexed = pose_extraction.smooth_landmark_sequence(indexed)

        frames = pose_extraction.build_output_frames(indexed)
        # Pass 2: re-read the video and draw the smoothed skeleton per frame.
        pose_extraction.render_overlay_video(in_path, indexed, fps, out_path, start_seconds=start)

        overlay_bytes = out_path.read_bytes()

    duration = frames[-1]["t"] if frames else 0.0
    return {
        "fps": fps,
        "duration": duration,
        "frames": frames,
        "overlay_mp4": overlay_bytes,
    }


@app.local_entrypoint()
def main(local_file: str, start: float, end: float, slug: str, title: str, youtube_url: str = ""):
    from pose_extraction import extract_youtube_id, update_dance_index

    video_path = Path(local_file).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"--local-file not found: {video_path}")

    print(f"Uploading {video_path.name} to Modal and running extraction [{start}, {end}]s ...")
    result = extract_and_overlay.remote(video_path.read_bytes(), start, end)

    frames = result["frames"]
    if not frames:
        raise RuntimeError("No pose detected in any frame of the clip")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{slug}.json"
    out_path.write_text(
        json.dumps(
            {"fps": result["fps"], "duration": result["duration"], "frames": frames}, indent=2
        )
    )
    youtube_id = extract_youtube_id(youtube_url) if youtube_url else ""
    update_dance_index(slug, title, youtube_id, start, end)
    print(f"Wrote {len(frames)} frames to {out_path}")
    print(f"Updated {DATA_DIR / 'dances.json'}")

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = CLIPS_DIR / f"{slug}_pose_overlay.avi"
    overlay_path.write_bytes(result["overlay_mp4"])
    print(f"Wrote debug pose-overlay video to {overlay_path} (not committed -- gitignored)")
