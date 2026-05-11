"""
Save one frame from a video to a PNG (OpenCV only; no FFmpeg).

From the project root:
    python tools/extract_frame.py
    python tools/extract_frame.py --ms 5000
    python tools/extract_frame.py --video data/other.mp4 --output data/frame.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    root = _project_root()
    default_video = root / "data" / "sample.mp4"
    default_out = root / "data" / "frame.png"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video",
        type=Path,
        default=default_video,
        help=f"Input video (default: {default_video})",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_out,
        help=f"PNG path to write (default: {default_out})",
    )
    p.add_argument(
        "--ms",
        type=float,
        default=0.0,
        help="Seek to this timestamp in milliseconds before reading (default: 0 = first frame)",
    )
    args = p.parse_args()

    video = args.video.resolve()
    out = args.output.resolve()

    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"Cannot open video: {video}", file=sys.stderr)
        return 1

    if args.ms > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.ms)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("Could not read a frame (try a smaller --ms or check the file).", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), frame):
        print(f"Failed to write: {out}", file=sys.stderr)
        return 1

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
