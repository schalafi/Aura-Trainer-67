"""Modal entrypoint for the Aura Farming Trainer web app.

Local dev:
    modal serve backend/modal_app.py

Deploy:
    modal deploy backend/modal_app.py

This wraps the FastAPI app (backend/app/main.py) as a Modal ASGI app. Modal
handles TLS, autoscaling, and scale-to-zero -- no servers to manage. No
secrets are required for this app (no auth, no third-party API keys), so
none are declared here. If that changes, use `modal.Secret` rather than
hardcoding values.
"""

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REPO_ROOT / "backend" / "requirements.txt"))
    .add_local_dir(str(REPO_ROOT / "backend" / "app"), remote_path="/root/app")
    .add_local_dir(str(REPO_ROOT / "data" / "dances"), remote_path="/root/data/dances")
    .add_local_dir(
        str(REPO_ROOT / "frontend" / "dist"),
        remote_path="/root/frontend/dist",
        ignore=[".gitkeep"],
    )
)

app = modal.App("aura-farming-trainer", image=image)


@app.function()
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/root")
    from app.main import app as fastapi_app

    return fastapi_app
