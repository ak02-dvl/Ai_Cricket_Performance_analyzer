"""CLI for impact-centered annotation clips and unbiased shot-label evaluation."""

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow direct execution from project root without changing Python path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shot_dataset_v40 import (  # noqa: E402
    SHOT_LABELS, evaluate_predictions, load_csv, manifest_from_reports,
    read_reports, validate_manifest, write_csv,
)


def extract_clips(rows, videos_dir, out_dir):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is needed for extraction: pip install opencv-python") from exc
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        video_path = Path(videos_dir) / row["source_video"]
        result = {"clip_id": row["clip_id"], "video": str(video_path)}
        if not video_path.is_file():
            result["status"] = "Video file missing"
            results.append(result)
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            result["status"] = "Cannot open video"
            results.append(result)
            continue
        writer = None
        try:
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(row["fps"])
            if w <= 0 or h <= 0 or fps <= 0:
                raise RuntimeError("Video metadata is invalid")
            output_path = out / (row["clip_id"] + ".mp4")
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Unable to write MP4; check OpenCV codecs")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
            count = 0
            for _ in range(start, end + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                count += 1
            result.update({"status": "Extracted" if count == end - start + 1 else "Partial extraction",
                           "frames_written": count, "file": str(output_path)})
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shot classification v4.0A ground-truth tools")
    subs = parser.add_subparsers(dest="command", required=True)
    init = subs.add_parser("init", help="Generate an UNLABELED manifest from JSON reports")
    init.add_argument("--reports", nargs="+", required=True)
    init.add_argument("--output", default="data/shot_labels_v40.csv")
    init.add_argument("--pre-frames", type=int, default=20)
    init.add_argument("--post-frames", type=int, default=16)
    validate = subs.add_parser("validate", help="Validate manually reviewed labels and leakage")
    validate.add_argument("--manifest", required=True)
    extract = subs.add_parser("extract", help="Extract annotation-review clips from the original videos")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--videos-dir", default="uploads")
    extract.add_argument("--output", default="data/shot_review_v40")
    eval_cmd = subs.add_parser("evaluate", help="Evaluate predictions on reviewed test clips ONLY")
    eval_cmd.add_argument("--manifest", required=True)
    eval_cmd.add_argument("--predictions", required=True)
    args = parser.parse_args(argv)
    if args.command == "init":
        rows = manifest_from_reports(read_reports(args.reports), args.pre_frames, args.post_frames)
        write_csv(args.output, rows)
        print(f"Created {args.output} with {len(rows)} UNLABELED clip(s).")
        print("Do not assign shot labels without inspecting the original clips.")
    elif args.command == "validate":
        result = validate_manifest(load_csv(args.manifest))
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            return 1
    elif args.command == "extract":
        rows = load_csv(args.manifest)
        result = validate_manifest(rows)
        if not result["valid"]:
            raise ValueError("Invalid manifest: " + "; ".join(result["errors"]))
        print(json.dumps(extract_clips(rows, args.videos_dir, args.output), indent=2))
    elif args.command == "evaluate":
        rows = load_csv(args.manifest)
        with Path(args.predictions).open("r", encoding="utf-8-sig", newline="") as f:
            pred_rows = list(csv.DictReader(f))
        print(json.dumps(evaluate_predictions(rows, pred_rows), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
