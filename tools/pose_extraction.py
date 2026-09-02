"""Shared pose-extraction logic used by both the local CLI tool
(extract_reference_pose.py) and the Modal-based tool (modal_extract.py).

Kept dependency-light at import time: mediapipe/cv2 imports are deferred into
the functions that need them so this module can be imported in contexts
(like argument parsing) without requiring those packages yet.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

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

# Skeleton connections (landmark index pairs) for drawing an overlay.
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms + shoulders
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
]


def ensure_model(model_path: Path = MODEL_PATH) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose landmarker model to {model_path} ...", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


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


def make_landmarker(model_path: Path, delegate_cpu: bool = True):
    """Create a MediaPipe PoseLandmarker in VIDEO mode.

    delegate_cpu=True forces CPU-only inference, which avoids a known
    MediaPipe Tasks crash on macOS (GPU/Metal calculator-graph service
    unavailable: `Check failed: service_ Service is unavailable` inside
    DrishtiMetalHelper) and is also the right choice in a GPU-less Modal
    container.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options_kwargs = {"model_asset_path": str(model_path)}
    if delegate_cpu:
        base_options_kwargs["delegate"] = mp_python.BaseOptions.Delegate.CPU

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(**base_options_kwargs),
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_keypoints(
    video_path: Path,
    model_path: Path,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    on_frame=None,
) -> tuple[list[dict], float]:
    """Run pose detection over [start_seconds, end_seconds] of video_path.

    Output frame timestamps are relative to start_seconds (i.e. the first
    processed frame is always t=0), regardless of where in the source file
    that window falls -- this lets it work identically whether video_path is
    a pre-trimmed download (start_seconds=0) or a full local file
    (start_seconds/end_seconds are absolute offsets into that file).

    If on_frame is given, it's called as on_frame(bgr_frame, landmarks_or_None)
    for every processed frame -- used to build a debug overlay video without
    a second pass over the source.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

    frames: list[dict] = []
    with make_landmarker(model_path) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            relative_t = frame_idx / fps
            absolute_t = start_seconds + relative_t
            if end_seconds is not None and absolute_t > end_seconds:
                break

            timestamp_ms = int(absolute_t * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            raw_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            if raw_landmarks:
                frames.append({"t": relative_t, "landmarks": normalize_landmarks(raw_landmarks)})

            if on_frame is not None:
                on_frame(frame, raw_landmarks)

            frame_idx += 1

    cap.release()
    return frames, fps


def draw_pose_overlay(frame, raw_landmarks) -> None:
    """Draw the skeleton (in-place) on a BGR frame using raw (pixel-normalized
    0..1) MediaPipe landmarks, for a human-checkable debug video."""
    import cv2

    if raw_landmarks is None:
        return
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks]

    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 120), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 200, 255), -1)


def update_dance_index(
    slug: str, title: str, youtube_id: str, start: float, end: float, data_dir: Path = DATA_DIR
) -> None:
    index_path = data_dir / "dances.json"
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
    match = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")
    return match.group(1)
