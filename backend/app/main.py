"""FastAPI app: serves the built frontend SPA plus a small JSON API describing
available reference dances and their pre-extracted pose keypoint sequences.

No video/audio is ever served from here -- reference playback happens via the
official YouTube IFrame Player directly in the browser. This backend only ever
serves derived numeric pose data (JSON) and static app files.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "dances"
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

app = FastAPI(title="Aura Farming Trainer API")


def _load_dance_index() -> list[dict]:
    index_path = DATA_DIR / "dances.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text())


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/dances")
def list_dances() -> list[dict]:
    """Metadata only (no keypoints) -- enough for the UI to build a dance picker."""
    return [
        {k: v for k, v in dance.items() if k != "keypoints_file"}
        for dance in _load_dance_index()
    ]


@app.get("/api/dances/{slug}")
def get_dance(slug: str) -> dict:
    dance = next((d for d in _load_dance_index() if d["slug"] == slug), None)
    if dance is None:
        raise HTTPException(status_code=404, detail=f"Unknown dance '{slug}'")

    keypoints_path = DATA_DIR / dance["keypoints_file"]
    if not keypoints_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Reference keypoints for '{slug}' have not been generated yet",
        )

    reference = json.loads(keypoints_path.read_text())
    return {**{k: v for k, v in dance.items() if k != "keypoints_file"}, "reference": reference}


# Serve the built SPA (frontend/dist) if present. In local dev the frontend
# usually runs on its own Vite dev server instead, so this is a no-op until
# `npm run build` has produced frontend/dist.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
