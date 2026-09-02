#!/usr/bin/env python3
"""Generate a SYNTHETIC placeholder reference pose sequence.

This is NOT the real "Howard the Alien" choreography -- it's a simple looping
arm-wave + hip-sway animation used only so the frontend, scoring engine, and
API can be built and tested end-to-end before real footage can be processed.

Network access to youtube.com is blocked in this sandbox, so
tools/extract_reference_pose.py could not be run here. Once you have normal
internet access, run it against the real clip and it will overwrite the file
this script produces:

    python tools/extract_reference_pose.py \\
        --url https://youtu.be/WxrQ3SqSt6Q --start <s> --end <e> \\
        --slug howard_the_alien --title "Howard the Alien"
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "dances"

FPS = 30
DURATION_S = 8.0

# Base standing pose in hip-centered, torso-scaled space (same convention as
# extract_reference_pose.normalize_landmarks): hip midpoint = (0, 0), y grows
# downward (image convention), torso (hip->shoulder) length ~= 1 unit.
BASE = {
    0: (0.0, -1.75),  # nose
    1: (0.03, -1.78), 2: (0.05, -1.78), 3: (0.07, -1.77),
    4: (-0.03, -1.78), 5: (-0.05, -1.78), 6: (-0.07, -1.77),
    7: (0.1, -1.7), 8: (-0.1, -1.7),
    9: (0.03, -1.68), 10: (-0.03, -1.68),
    11: (0.35, -1.0), 12: (-0.35, -1.0),   # shoulders
    13: (0.55, -0.5), 14: (-0.55, -0.5),   # elbows
    15: (0.6, -0.05), 16: (-0.6, -0.05),   # wrists
    17: (0.63, 0.05), 18: (-0.63, 0.05),
    19: (0.62, 0.0), 20: (-0.62, 0.0),
    21: (0.58, -0.02), 22: (-0.58, -0.02),
    23: (0.18, 0.0), 24: (-0.18, 0.0),     # hips
    25: (0.2, 1.0), 26: (-0.2, 1.0),       # knees
    27: (0.2, 2.0), 28: (-0.2, 2.0),       # ankles
    29: (0.2, 2.1), 30: (-0.2, 2.1),
    31: (0.25, 2.15), 32: (-0.25, 2.15),
}

ARM_JOINTS = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
HIP_JOINTS = {23, 24, 25, 26}


def landmark_at(idx: int, t: float) -> dict:
    x, y = BASE[idx]
    wave = math.sin(2 * math.pi * t / 1.6)  # ~1.6s per arm-wave cycle
    sway = math.sin(2 * math.pi * t / 2.4)

    if idx in ARM_JOINTS:
        # Raise/lower arms over the wave cycle.
        y = y - 0.9 * max(wave, 0) * (1.0 if idx % 2 == 1 else 1.0)
        x = x + 0.15 * wave * (1 if x > 0 else -1)
    if idx in HIP_JOINTS or idx in (11, 12):
        x = x + 0.06 * sway

    return {"x": round(x, 4), "y": round(y, 4), "z": 0.0, "visibility": 1.0}


def main() -> None:
    n_frames = int(FPS * DURATION_S)
    frames = []
    for i in range(n_frames):
        t = i / FPS
        landmarks = [landmark_at(idx, t) for idx in range(33)]
        frames.append({"t": round(t, 4), "landmarks": landmarks})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "howard_the_alien.json"
    out_path.write_text(json.dumps({"fps": FPS, "duration": DURATION_S, "frames": frames}, indent=2))

    index_path = DATA_DIR / "dances.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [d for d in index if d["slug"] != "howard_the_alien"]
    index.append(
        {
            "slug": "howard_the_alien",
            "title": "Howard the Alien",
            "youtube_id": "WxrQ3SqSt6Q",
            "start_seconds": 0.0,
            "end_seconds": DURATION_S,
            "keypoints_file": "howard_the_alien.json",
            "placeholder": True,
        }
    )
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Wrote {n_frames} SYNTHETIC placeholder frames to {out_path}")
    print("Replace with real data via tools/extract_reference_pose.py once you have internet access to youtube.com.")


if __name__ == "__main__":
    main()
