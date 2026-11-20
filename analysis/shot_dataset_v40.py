"""Shot Classification v4.0A: human-ground-truth dataset and honest evaluation.

This module is intentionally independent of CricketAnalyzer and makes no predictions.
The only evidence for Drive/Cut/Pull/Defense labels is human-reviewed video.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SHOT_LABELS = ("Drive", "Cut", "Pull", "Defense")
ALL_LABELS = (*SHOT_LABELS, "Other", "Ambiguous", "Unlabeled")
PREDICTION_LABELS = (*SHOT_LABELS, "Uncertain")
VIEWS = ("Unknown", "Front", "Side", "Behind", "Oblique", "Broadcast")
HAND = ("Unknown", "Right-handed", "Left-handed")
REVIEW = ("Unreviewed", "Reviewed", "Disputed")
SPLITS = ("", "train", "val", "test")
FIELDNAMES = (
    "clip_id", "recording_id", "source_video", "impact_frame",
    "start_frame", "end_frame", "total_frames", "fps",
    "shot_label", "review_status", "camera_view", "batting_stance",
    "split", "annotator", "notes",
)


def load_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV requires a header row")
        missing = set(FIELDNAMES) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        return [{key: (row.get(key) or "").strip() for key in FIELDNAMES}
                for row in reader]


def write_csv(path: str | Path, rows: list[dict], fields=FIELDNAMES) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return slug or "video"


def _positive_int(value: str, key: str) -> int:
    try:
        if not str(value).strip().isdigit():
            raise ValueError()
        n = int(value)
    except ValueError:
        raise ValueError(f"{key} must be a positive integer") from None
    if n < 1:
        raise ValueError(f"{key} must be a positive integer")
    return n


def _positive_float(value: str, key: str) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a positive finite number") from None
    if not math.isfinite(n) or n <= 0:
        raise ValueError(f"{key} must be a positive finite number")
    return n


def manifest_from_reports(reports: list[dict], pre_frames=20, post_frames=16) -> list[dict]:
    """Create UNLABELED clips from confirmed events; never infer a shot type."""
    if pre_frames < 0 or post_frames < 0:
        raise ValueError("Frame windows cannot be negative")
    rows = []
    for report in reports:
        video = Path(str(report.get("video_name", ""))).name
        if not video or video == ".":
            raise ValueError("Report missing video_name")
        metadata = report.get("source_video") or {}
        total = _positive_int(str(metadata.get("total_frames", "")), "total_frames")
        fps = _positive_float(str(metadata.get("fps", "")), "fps")
        events = report.get("impact_events") or []
        if not isinstance(events, list):
            raise ValueError(f"impact_events must be list: {video}")
        for event_index, event in enumerate(events, start=1):
            frame = _positive_int(str(event.get("impact_frame", "")), "impact_frame")
            if frame > total:
                raise ValueError(f"Impact frame beyond video length: {video}")
            rows.append({
                "clip_id": f"{_safe_id(Path(video).stem)}_f{frame:06d}_e{event_index}",
                "recording_id": _safe_id(Path(video).stem),
                "source_video": video,
                "impact_frame": frame,
                "start_frame": max(1, frame - pre_frames),
                "end_frame": min(total, frame + post_frames),
                "total_frames": total,
                "fps": fps,
                "shot_label": "Unlabeled",  # intentionally no invented labels
                "review_status": "Unreviewed",
                "camera_view": "Unknown",
                "batting_stance": "Unknown",
                "split": "",
                "annotator": "",
                "notes": "Estimated impact: check video before assigning shot label",
            })
    return rows


def validate_manifest(rows: list[dict]) -> dict:
    errors, warnings = [], []
    ids = set()
    recording_to_split = defaultdict(set)
    video_to_recording = defaultdict(set)
    label_counts = Counter()
    reviewed = 0

    for i, row in enumerate(rows, start=2):
        label = row.get("shot_label", "")
        state = row.get("review_status", "")
        split = row.get("split", "")
        clip_id = row.get("clip_id", "")
        recording = row.get("recording_id", "")
        video = row.get("source_video", "")
        prefix = f"CSV row {i}"
        if not clip_id or not re.fullmatch(r"[A-Za-z0-9_-]+", clip_id):
            errors.append(f"{prefix}: clip_id must be a safe nonempty identifier")
        elif clip_id in ids:
            errors.append(f"{prefix}: duplicate clip_id '{clip_id}'")
        ids.add(clip_id)
        if not recording or not re.fullmatch(r"[A-Za-z0-9_-]+", recording):
            errors.append(f"{prefix}: invalid recording_id")
        if not video or Path(video).name != video:
            errors.append(f"{prefix}: source_video must be a plain filename")
        if label not in ALL_LABELS:
            errors.append(f"{prefix}: invalid shot_label '{label}'")
        if state not in REVIEW:
            errors.append(f"{prefix}: invalid review_status '{state}'")
        if row.get("camera_view") not in VIEWS:
            errors.append(f"{prefix}: invalid camera_view")
        if row.get("batting_stance") not in HAND:
            errors.append(f"{prefix}: invalid batting_stance")
        if split not in SPLITS:
            errors.append(f"{prefix}: invalid split '{split}'")
        try:
            start = _positive_int(str(row.get("start_frame", "")), "start_frame")
            impact = _positive_int(str(row.get("impact_frame", "")), "impact_frame")
            end = _positive_int(str(row.get("end_frame", "")), "end_frame")
            total = _positive_int(str(row.get("total_frames", "")), "total_frames")
            _positive_float(str(row.get("fps", "")), "fps")
            if not 1 <= start <= impact <= end <= total:
                errors.append(f"{prefix}: frame range must contain impact within video")
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")
        if state == "Reviewed":
            reviewed += 1
            if label == "Unlabeled":
                errors.append(f"{prefix}: reviewed clip cannot remain Unlabeled")
            if not row.get("annotator"):
                warnings.append(f"{prefix}: reviewed clip has no annotator")
        elif label in SHOT_LABELS:
            errors.append(f"{prefix}: class '{label}' needs human review_status=Reviewed")
        if split:
            if state != "Reviewed" or label not in SHOT_LABELS:
                errors.append(f"{prefix}: split requires a reviewed Drive/Cut/Pull/Defense label")
            recording_to_split[recording].add(split)
        if video and recording:
            video_to_recording[video].add(recording)
        label_counts[label] += 1
    for recording, split_set in recording_to_split.items():
        if len(split_set) > 1:
            errors.append(f"Data leakage: recording_id '{recording}' is in {sorted(split_set)}")
    for video, recording_set in video_to_recording.items():
        if len(recording_set) > 1:
            errors.append(f"Data leakage: source_video '{video}' has multiple recording IDs")
    if not any(row.get("split") == "test" and row.get("shot_label") in SHOT_LABELS
               and row.get("review_status") == "Reviewed" for row in rows):
        warnings.append("No reviewed test clips; no held-out accuracy can be reported")
    if not rows:
        warnings.append("Manifest contains no impact events")
    return {
        "valid": not errors,
        "clips": len(rows),
        "reviewed_clips": reviewed,
        "label_counts": dict(label_counts),
        "errors": errors,
        "warnings": warnings,
    }


def evaluate_predictions(rows: list[dict], predictions: list[dict]) -> dict:
    """Held-out evaluation. 'Uncertain' is an explicit abstention, not success."""
    status = validate_manifest(rows)
    if status["errors"]:
        raise ValueError("Manifest invalid: " + "; ".join(status["errors"]))
    gold = {
        row["clip_id"]: row["shot_label"]
        for row in rows if row["split"] == "test"
        and row["review_status"] == "Reviewed"
        and row["shot_label"] in SHOT_LABELS
    }
    if not gold:
        return {"status": "Insufficient labeled held-out data", "evaluated_clips": 0,
                "accuracy_all": None, "accuracy_when_classified": None,
                "coverage": None, "macro_f1_supported": None}
    preds = {}
    for pred in predictions:
        clip_id = pred.get("clip_id", "")
        if clip_id in preds:
            raise ValueError(f"Duplicate prediction: {clip_id}")
        label = pred.get("predicted_label", "")
        if label not in PREDICTION_LABELS:
            raise ValueError(f"Invalid prediction '{label}' for clip '{clip_id}'")
        preds[clip_id] = label
    all_n = len(gold)
    covered = sum(preds.get(i, "Uncertain") != "Uncertain" for i in gold)
    correct = sum(preds.get(i, "Uncertain") == y for i, y in gold.items())
    metrics = {}
    for label in SHOT_LABELS:
        tp = sum(y == label and preds.get(i, "Uncertain") == label for i, y in gold.items())
        fp = sum(y != label and preds.get(i, "Uncertain") == label for i, y in gold.items())
        fn = sum(y == label and preds.get(i, "Uncertain") != label for i, y in gold.items())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
        metrics[label] = {"support": tp + fn, "precision": precision,
                          "recall": recall, "f1": f1}
    present_f1 = [v["f1"] for v in metrics.values() if v["support"] > 0]
    confusion = sorted([{"actual": y, "predicted": preds.get(i, "Uncertain")}
                        for i, y in gold.items()], key=lambda x: (x["actual"], x["predicted"]))
    return {"status": "Evaluated (pilot; dataset size/view diversity still matter)",
            "evaluated_clips": all_n, "classified_clips": covered,
            "abstained_clips": all_n - covered,
            "coverage": round(covered / all_n, 4),
            "accuracy_all": round(correct / all_n, 4),
            "accuracy_when_classified": round(correct / covered, 4) if covered else None,
            "macro_f1_supported": round(sum(present_f1) / len(present_f1), 4),
            "per_class": metrics, "confusion": confusion,
            "note": "Only reviewed clips in test are evaluated; Uncertain counts as an abstention."}


def read_reports(paths: list[str | Path]) -> list[dict]:
    result = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            result.append(json.load(f))
    return result
