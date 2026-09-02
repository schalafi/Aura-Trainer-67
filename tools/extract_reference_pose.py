#!/usr/bin/env python3
"""Extract a reference pose-keypoint sequence from a YouTube clip, locally.

This is a one-time, offline developer tool: it downloads a short section of a
YouTube video *transiently* (to a temp directory), runs MediaPipe's Pose
Landmarker over every frame, writes out only the derived numeric keypoints as
JSON, and deletes the downloaded video afterward. The raw video is never
committed to the repo or served by the app -- only this small JSON file is.

NOTE: if YouTube extraction or local MediaPipe inference is unreliable in
your environment (bot/PO-token checks, GPU-delegate crashes, etc.), prefer
`tools/modal_extract.py`, which runs the same extraction in a Modal cloud
container instead of on your machine.

Usage (via yt-dlp):
    python tools/extract_reference_pose.py \\
        --url https://youtu.be/WxrQ3SqSt6Q \\
        --start 12.5 --end 42.0 \\
        --slug howard_the_alien \\
        --title "Howard the Alien"

Usage (from an already-downloaded local file):
    python tools/extract_reference_pose.py \\
        --local-file ~/Downloads/howard_the_alien.mp4 \\
        --start 0 --end 30 \\
        --slug howard_the_alien \\
        --title "Howard the Alien"

Requires `yt-dlp`, `opencv-python`, and `mediapipe` (see tools/requirements.txt)
plus `ffmpeg` on PATH for precise section downloads (yt-dlp path only).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from pose_extraction import (
    DATA_DIR,
    ensure_model,
    extract_keypoints,
    extract_youtube_id,
    update_dance_index,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=None, help="YouTube URL of the reference clip (omit if using --local-file)"
    )
    parser.add_argument(
        "--local-file",
        default=None,
        metavar="PATH",
        help=(
            "Use an already-downloaded local video file instead of fetching via yt-dlp "
            "(e.g. if YouTube extraction is blocked). --start/--end are absolute offsets "
            "into this file. Only pose keypoints are extracted; the file itself is never "
            "copied into the repo."
        ),
    )
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

    if not args.local_file and not args.url:
        parser.error("one of --url or --local-file is required")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ensure_model()
    youtube_id = extract_youtube_id(args.url) if args.url else ""

    tmp_dir = Path(tempfile.mkdtemp(prefix="aura-extract-")) if not args.local_file else None
    try:
        if args.local_file:
            video_path = Path(args.local_file).expanduser().resolve()
            if not video_path.exists():
                raise FileNotFoundError(f"--local-file not found: {video_path}")
            print(f"Reading local file {video_path}, section [{args.start}, {args.end}]s ...")
            print("Running pose landmarker over extracted frames ...")
            frames, fps = extract_keypoints(
                video_path, model_path, start_seconds=args.start, end_seconds=args.end
            )
        else:
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
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
