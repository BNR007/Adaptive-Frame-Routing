"""Measure compositor-visible URFTS frame identifiers from an OBS recording."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visual_frame_marker import decode_visual_frame_marker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open recording: {args.video}")
    recording_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, int | float | str]] = []
    previous: int | None = None
    unwrapped: int | None = None
    transitions = repeats = invalid = skipped_ids = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        marker = decode_visual_frame_marker(frame)
        status = "invalid"
        if marker is None:
            invalid += 1
        elif previous is None:
            unwrapped = marker
            status = "first"
            previous = marker
        else:
            delta = (marker - previous) & 0xFFFF
            if delta == 0:
                repeats += 1
                status = "repeat"
            else:
                transitions += 1
                skipped_ids += max(0, delta - 1)
                assert unwrapped is not None
                unwrapped += delta
                status = "transition"
                previous = marker
        samples.append(
            {
                "recording_frame": index,
                "time_seconds": index / recording_fps if recording_fps > 0 else "",
                "marker_id": "" if marker is None else marker,
                "unwrapped_marker_id": "" if unwrapped is None else unwrapped,
                "status": status,
            }
        )
        index += 1
    capture.release()
    duration = index / recording_fps if recording_fps > 0 else 0.0
    visible_events = (1 + transitions) if any(row["marker_id"] != "" for row in samples) else 0
    summary = {
        "recording": str(args.video.resolve()),
        "recording_fps": recording_fps,
        "recording_frames": index,
        "duration_seconds": duration,
        "decoded_samples": index - invalid,
        "invalid_samples": invalid,
        "visible_frame_events": visible_events,
        "visible_unique_fps": visible_events / duration if duration > 0 else 0.0,
        "repeated_recording_frames": repeats,
        "skipped_marker_ids": skipped_ids,
        "scope": "OBS/compositor-visible submissions; not physical display scanout",
    }
    with (args.output_directory / "visual_frames.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=samples[0].keys() if samples else ("recording_frame",))
        writer.writeheader()
        writer.writerows(samples)
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
