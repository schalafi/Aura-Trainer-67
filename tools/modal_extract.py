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
    clips/<slug>_pose_overlay.mp4 (gitignored) so you can eyeball detection
    quality before it's wired into the app

Note: Modal's function argument/return payload limit is ~100MB, comfortably
enough for a 30s clip at reasonable resolution. A much longer/higher-res
reference clip would need a modal.Volume instead of passing bytes inline.
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
    # libgl1/libglib2.0-0: mediapipe's bundled OpenCV components dynamically
    # link against libGL even though we use opencv-python-headless ourselves;
    # Debian's slim image doesn't ship it by default.
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("mediapipe>=0.10.14", "opencv-python-headless", "numpy")
    # copy=True: _bake_model (a build step below) needs to import pose_extraction,
    # so it must be copied into the image layer now rather than mounted at
    # container startup (Modal's default for add_local_* when it's the last step).
    .add_local_python_source("pose_extraction", copy=True)
    .run_function(_bake_model)
)

app = modal.App("aura-trainer-extract", image=image)


@app.function(timeout=600)
def extract_and_overlay(video_bytes: bytes, start: float, end: float) -> dict:
    import tempfile

    import cv2
    import pose_extraction

    model_path = pose_extraction.ensure_model()

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "input.mp4"
        in_path.write_bytes(video_bytes)
        out_path = Path(tmp) / "overlay.mp4"

        probe = cv2.VideoCapture(str(in_path))
        fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.release()

        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            def on_frame(frame, raw_landmarks):
                pose_extraction.draw_pose_overlay(frame, raw_landmarks)
                writer.write(frame)

            frames, detected_fps = pose_extraction.extract_keypoints(
                in_path, model_path, start_seconds=start, end_seconds=end, on_frame=on_frame
            )
        finally:
            writer.release()

        overlay_bytes = out_path.read_bytes()

    duration = frames[-1]["t"] if frames else 0.0
    return {
        "fps": detected_fps,
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
    overlay_path = CLIPS_DIR / f"{slug}_pose_overlay.mp4"
    overlay_path.write_bytes(result["overlay_mp4"])
    print(f"Wrote debug pose-overlay video to {overlay_path} (not committed -- gitignored)")
