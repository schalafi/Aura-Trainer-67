#!/usr/bin/env python3
"""Extract a reference pose-keypoint sequence from a YouTube clip.

This is a one-time, offline developer tool: it downloads a short section of a
YouTube video *transiently* (to a temp directory), runs MediaPipe's Pose
Landmarker over every frame, writes out only the derived numeric keypoints as
JSON, and deletes the downloaded video afterward. The raw video is never
committed to the repo or served by the app -- only this small JSON file is.

Usage:
    python tools/extract_reference_pose.py \\
        --url https://youtu.be/WxrQ3SqSt6Q \\
        --start 12.5 --end 42.0 \\
        --slug howard_the_alien \\
        --title "Howard the Alien"

Requires `yt-dlp`, `opencv-python`, and `mediapipe` (see tools/requirements.txt)
plus `ffmpeg` on PATH for precise section downloads.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "dances"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# Indices into MediaPipe's 33-point pose landmark list.
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose landmarker model to {MODEL_PATH} ...", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def download_clip(
    url: str,
    start: float,
    end: float,
    out_dir: Path,
    cookies_from_browser: str | None = None,
    youtube_player_client: str | None = None,
) -> Path:
    """Download only the [start, end] section of the video into out_dir."""
    import yt_dlp

    out_template = str(out_dir / "clip.%(ext)s")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]/mp4/best",
        "outtmpl": out_template,
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "quiet": False,
        "noplaylist": True,
    }
    if cookies_from_browser:
        # Lets yt-dlp reuse cookies from a local browser profile to get past
        # YouTube's bot-check ("Sign in to confirm you're not a bot"). Cookies
        # never leave the local machine -- yt-dlp reads them directly from the
        # browser's local cookie store. See yt-dlp's --cookies-from-browser docs.
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if youtube_player_client:
        # Escape hatch for YouTube extractor breakage (e.g. the "SABR-only
        # streaming experiment" / "page needs to be reloaded" errors) -- lets
        # you force yt-dlp to use a specific player client, equivalent to its
        # own --extractor-args "youtube:player_client=..." flag.
        ydl_opts["extractor_args"] = {
            "youtube": {"player_client": youtube_player_client.split(",")}
        }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    candidates = list(out_dir.glob("clip.*"))
    if not candidates:
        raise RuntimeError("yt-dlp did not produce an output file")
    return candidates[0]


def normalize_landmarks(landmarks) -> list[dict]:
    """Hip-center origin, torso-length scale -- makes scoring robust to the
    user's distance from the camera and body proportions."""
    hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2
    hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
    shoulder_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
    shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
    torso = max(((shoulder_x - hip_x) ** 2 + (shoulder_y - hip_y) ** 2) ** 0.5, 1e-6)

    return [
        {
            "x": (lm.x - hip_x) / torso,
            "y": (lm.y - hip_y) / torso,
            "z": lm.z / torso,
            "visibility": getattr(lm, "visibility", 1.0),
        }
        for lm in landmarks
    ]


def extract_keypoints(video_path: Path, model_path: Path) -> tuple[list[dict], float]:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    frames: list[dict] = []
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int((frame_idx / fps) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = normalize_landmarks(result.pose_landmarks[0])
                frames.append({"t": frame_idx / fps, "landmarks": landmarks})

            frame_idx += 1

    cap.release()
    return frames, fps


def update_dance_index(slug: str, title: str, youtube_id: str, start: float, end: float) -> None:
    index_path = DATA_DIR / "dances.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [d for d in index if d["slug"] != slug]
    index.append(
        {
            "slug": slug,
            "title": title,
            "youtube_id": youtube_id,
            "start_seconds": start,
            "end_seconds": end,
            "keypoints_file": f"{slug}.json",
        }
    )
    index_path.write_text(json.dumps(index, indent=2) + "\n")


def extract_youtube_id(url: str) -> str:
    import re

    match = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="YouTube URL of the reference clip")
    parser.add_argument("--start", type=float, required=True, help="Section start, seconds")
    parser.add_argument("--end", type=float, required=True, help="Section end, seconds")
    parser.add_argument("--slug", required=True, help="Short id, e.g. howard_the_alien")
    parser.add_argument("--title", required=True, help="Display name, e.g. 'Howard the Alien'")
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER",
        help="Pass browser cookies to yt-dlp to bypass YouTube bot-checks, e.g. firefox, chrome",
    )
    parser.add_argument(
        "--youtube-player-client",
        default=None,
        metavar="CLIENT[,CLIENT...]",
        help=(
            "Override yt-dlp's YouTube player client selection if extraction fails with "
            "'SABR-only streaming' / 'page needs to be reloaded' errors, e.g. tv or tv,web"
        ),
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ensure_model()
    youtube_id = extract_youtube_id(args.url)

    tmp_dir = Path(tempfile.mkdtemp(prefix="aura-extract-"))
    try:
        print(f"Downloading section [{args.start}, {args.end}]s from {args.url} ...")
        video_path = download_clip(
            args.url,
            args.start,
            args.end,
            tmp_dir,
            args.cookies_from_browser,
            args.youtube_player_client,
        )

        print("Running pose landmarker over extracted frames ...")
        frames, fps = extract_keypoints(video_path, model_path)
        if not frames:
            raise RuntimeError("No pose detected in any frame of the clip")

        duration = frames[-1]["t"] if frames else 0.0
        out_path = DATA_DIR / f"{args.slug}.json"
        out_path.write_text(
            json.dumps({"fps": fps, "duration": duration, "frames": frames}, indent=2)
        )
        update_dance_index(args.slug, args.title, youtube_id, args.start, args.end)

        print(f"Wrote {len(frames)} frames to {out_path}")
        print(f"Updated {DATA_DIR / 'dances.json'}")
    finally:
        # Never keep the downloaded video around -- only derived keypoints are committed.
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
