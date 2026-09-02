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
# "full" (vs. "lite"): better accuracy, still fast enough for an offline
# extraction job. "heavy" is a one-line swap (change both constants below) if
# even more accuracy is wanted at the cost of extraction time.
MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
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


class _Landmark:
    """Minimal stand-in for MediaPipe's landmark objects (just the x/y/z/
    visibility attributes normalize_landmarks() and draw_pose_overlay() read),
    so smoothed values can flow through the same code paths as raw ones."""

    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x: float, y: float, z: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


# (frame_idx, relative_t, raw_landmarks_or_None) in frame order.
IndexedLandmarks = list


def detect_landmarks(
    video_path: Path,
    model_path: Path,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> tuple[IndexedLandmarks, float]:
    """Pass 1: run pose detection over [start_seconds, end_seconds] of
    video_path. Returns (indexed, fps) where indexed has one entry per
    processed frame -- (frame_idx, relative_t, raw_landmarks_or_None),
    frame_idx starting at 0 for the first processed frame and relative_t
    relative to start_seconds (so the first processed frame is t=0)
    regardless of where in the source file that window falls.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

    indexed: IndexedLandmarks = []
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
            indexed.append((frame_idx, relative_t, raw_landmarks))
            frame_idx += 1

    cap.release()
    return indexed, fps


def smooth_landmark_sequence(
    indexed: IndexedLandmarks, window_length: int = 7, polyorder: int = 2
) -> IndexedLandmarks:
    """Reduce per-frame jitter with a Savitzky-Golay filter applied per
    landmark, per x/y/z axis, across the timeline -- standard technique for
    smoothing noisy pose/hand-tracking sequences. Since extraction is
    offline (the whole clip is already in memory), a global/non-causal
    filter over the full sequence gives cleaner results than a real-time
    filter would.

    Frames with no detection are left as None. Detected frames are treated
    as one contiguous sequence for smoothing purposes (ignoring any gaps
    from brief dropouts) -- an acceptable approximation as long as dropouts
    are rare. Falls back to returning the input unchanged if there aren't
    enough detected frames to fill one smoothing window.
    """
    import numpy as np
    from scipy.signal import savgol_filter

    detected = [(i, t, lm) for i, t, lm in indexed if lm is not None]
    if len(detected) < window_length:
        return indexed

    n_landmarks = len(detected[0][2])
    coords = np.array(
        [[[lm.x, lm.y, lm.z] for lm in landmarks] for _, _, landmarks in detected]
    )  # shape: (n_detected_frames, n_landmarks, 3)
    visibilities = [
        [getattr(lm, "visibility", 1.0) for lm in landmarks] for _, _, landmarks in detected
    ]

    wl = window_length if window_length % 2 == 1 else window_length + 1
    po = min(polyorder, wl - 1)
    smoothed = savgol_filter(coords, window_length=wl, polyorder=po, axis=0)

    smoothed_by_frame_idx = {
        frame_idx: [
            _Landmark(
                x=float(smoothed[row, j, 0]),
                y=float(smoothed[row, j, 1]),
                z=float(smoothed[row, j, 2]),
                visibility=visibilities[row][j],
            )
            for j in range(n_landmarks)
        ]
        for row, (frame_idx, _, _) in enumerate(detected)
    }

    return [(i, t, smoothed_by_frame_idx.get(i)) for i, t, _ in indexed]


def build_output_frames(indexed: IndexedLandmarks) -> list[dict]:
    """Convert detected (+ optionally smoothed) landmarks into the exported
    JSON shape, dropping frames with no detection."""
    return [
        {"t": t, "landmarks": normalize_landmarks(lm)} for _, t, lm in indexed if lm is not None
    ]


def extract_keypoints(
    video_path: Path,
    model_path: Path,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    smooth: bool = True,
) -> tuple[list[dict], float]:
    """Convenience wrapper: detect landmarks, optionally smooth them, and
    build the exported frame list -- used by the simple local-CLI path,
    which doesn't need a debug overlay video."""
    indexed, fps = detect_landmarks(video_path, model_path, start_seconds, end_seconds)
    if smooth:
        indexed = smooth_landmark_sequence(indexed)
    return build_output_frames(indexed), fps


def render_overlay_video(
    video_path: Path, indexed: IndexedLandmarks, fps: float, out_path: Path, start_seconds: float = 0.0
) -> None:
    """Pass 2: re-read video_path and draw the (already smoothed) landmarks
    matching each frame_idx onto it, writing a proper H.264 .mp4 to out_path.
    `indexed` must use the same frame indexing detect_landmarks() produced.

    Encodes via the `ffmpeg` CLI (piping raw BGR24 frames over stdin) rather
    than OpenCV's built-in VideoWriter: OpenCV's mp4v fourcc isn't a real
    H.264 implementation and produces visibly corrupted (glitching,
    degrading) output in minimal Linux containers without a full ffmpeg
    backend. `ffmpeg` + libx264 gives a normal, small, widely-compatible
    mp4 instead. Requires the `ffmpeg` binary on PATH (apt package `ffmpeg`).
    """
    import subprocess

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if start_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

    by_frame_idx = {frame_idx: lm for frame_idx, _, lm in indexed}
    max_idx = indexed[-1][0] if indexed else -1
    try:
        frame_idx = 0
        while frame_idx <= max_idx:
            ok, frame = cap.read()
            if not ok:
                break
            draw_pose_overlay(frame, by_frame_idx.get(frame_idx))
            proc.stdin.write(frame.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        proc.stdin.close()
        stderr_output = proc.stderr.read()
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {returncode}): {stderr_output.decode(errors='replace')}"
            )


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
