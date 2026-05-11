"""
Mark parking spot rectangles on a reference image.

  Left click   — add a spot (40x80 box from the click origin)
  Right click  — remove a spot whose box contains the cursor
  Q            — quit (positions are saved after each edit)
  S            — save a screenshot of the current overlay

Use data/frame.png as the reference image (or pass --image).
From the project root, run python tools/extract_frame.py first to grab a frame
from data/sample.mp4, then:

    python tools/parking_generator.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROI_W = 40
ROI_H = 80
WINDOW_NAME = "Parking spots"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_spots(path: Path) -> list[tuple[int, int]]:
    if not path.is_file():
        return []
    spots: list[tuple[int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        xs, ys = line.split(",")
        spots.append((int(xs), int(ys)))
    return spots


def save_spots(path: Path, spots: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x, y in spots:
            f.write(f"{x},{y}\n")


def spot_index_at(spots: list[tuple[int, int]], x: int, y: int) -> int | None:
    for i in range(len(spots) - 1, -1, -1):
        x0, y0 = spots[i]
        if x0 < x < x0 + ROI_W and y0 < y < y0 + ROI_H:
            return i
    return None


def draw_overlay(base: object, spots: list[tuple[int, int]]) -> None:
    for i, pos in enumerate(spots):
        x0, y0 = pos
        cv2.rectangle(base, pos, (x0 + ROI_W, y0 + ROI_H), (255, 0, 255), 2)
        cv2.putText(
            base,
            str(i + 1),
            (x0 + 2, y0 + 14),
            cv2.FONT_HERSHEY_PLAIN,
            0.9,
            (255, 255, 255),
            1,
        )
    cv2.putText(
        base,
        f"Spots: {len(spots)} | L=add R=remove Q=quit S=screenshot",
        (5, 20),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (0, 200, 0),
        1,
    )


def run(image_path: Path, slots_path: Path) -> int:
    spots = load_spots(slots_path)
    if spots:
        print(f"Loaded {len(spots)} spots from {slots_path}")
    else:
        print(f"No existing file at {slots_path} — starting with zero spots.")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Cannot read image: {image_path}", file=sys.stderr)
        return 1

    state = {"spots": spots}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["spots"].append((x, y))
            save_spots(slots_path, state["spots"])
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = spot_index_at(state["spots"], x, y)
            if idx is not None:
                state["spots"].pop(idx)
                save_spots(slots_path, state["spots"])

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        display = image.copy()
        draw_overlay(display, state["spots"])
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out = Path("generator_preview.jpg")
            cv2.imwrite(str(out), display)
            print(f"Screenshot saved as {out.resolve()}")

    cv2.destroyAllWindows()
    print(f"Done. {len(state['spots'])} spots in {slots_path}")
    return 0


def main() -> int:
    root = _project_root()
    default_image = root / "data" / "frame.png"
    default_slots = root / "data" / "sample"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--image",
        type=Path,
        default=default_image,
        help=f"Reference frame (default: {default_image})",
    )
    p.add_argument(
        "--slots",
        type=Path,
        default=default_slots,
        help=f"Output coordinates file (default: {default_slots})",
    )
    args = p.parse_args()

    return run(args.image.resolve(), args.slots.resolve())


if __name__ == "__main__":
    sys.exit(main())
