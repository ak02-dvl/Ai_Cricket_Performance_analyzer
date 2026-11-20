
import argparse
import csv
from pathlib import Path

import cv2

from pipeline.analyzer import CricketAnalyzer


JOINTS = [
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.start < 1 or args.end < args.start:
        parser.error("Expected 1 <= start <= end")

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    analyzer = CricketAnalyzer()
    analyzer.footwork.set_batting_stance("Right-handed")

    rows = []
    frame_number = 0
    previous_phase = None

    try:
        while frame_number < args.end:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Process all preceding frames to preserve
            # detector history and temporal state.
            _, report = analyzer.process_frame(
                frame,
                source_frame_number=frame_number
            )

            if not isinstance(report, dict):
                report = {}

            phase = str(report.get("phase", "Unknown"))

            if args.start <= frame_number <= args.end:

                angles = report.get("angles") or {}
                impact = report.get("impact") or {}

                if not isinstance(angles, dict):
                    angles = {}

                if not isinstance(impact, dict):
                    impact = {}

                row = {
                    "frame": frame_number,
                    "phase": phase,
                    "shot": report.get("shot", "Unknown"),
                    "analysis_status": report.get(
                        "analysis_status", ""
                    ),
                    "impact_event": bool(
                        impact.get("event", False)
                    ),
                    "impact_frame": impact.get(
                        "impact_frame", ""
                    ),
                }

                for joint in JOINTS:
                    row[joint] = angles.get(joint, "")
                                    # Experimental wrist-phase diagnostics.
                # Reads state only; does not change classification.
                detector = analyzer.phase_detector

                row.update({
                    "wrist_arm": getattr(
                        detector, "wrist_signal_arm", None
                    ),
                    "wrist_relative_y": getattr(
                        detector, "wrist_last_relative_y", None
                    ),
                    "wrist_signal_available": getattr(
                        detector, "wrist_signal_available", False
                    ),
                    "wrist_up_counter": getattr(
                        detector, "wrist_up_counter", 0
                    ),
                    "wrist_down_counter": getattr(
                        detector, "wrist_down_counter", 0
                    ),
                })

                rows.append(row)

                if (
                    phase != previous_phase
                    or row["impact_event"]
                ):
                    print(
                        f"Frame {frame_number}: "
                        f"Phase={phase}, "
                        f"LeftElbow={row['left_elbow']}, "
                        f"RightElbow={row['right_elbow']}, "
                        f"Impact={row['impact_event']}"
                    )

            previous_phase = phase

    finally:
        cap.release()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fields = [
        "frame", "phase", "shot",
        "analysis_status",
        *JOINTS,
        "impact_event", "impact_frame",
         "wrist_arm",
    "wrist_relative_y",
    "wrist_signal_available",
    "wrist_up_counter",
    "wrist_down_counter",
    ]

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
