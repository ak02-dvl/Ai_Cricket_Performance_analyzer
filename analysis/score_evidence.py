"""Read-only evidence summary for joint-based batting scores.

Does not change joint scoring, weights, visibility filtering or phases.
Only used when the existing joint_measurement_reliability field is present.
"""

import math
from analysis.joint_config import JOINTS


def build_score_evidence(report):
    """Describe the input coverage of an existing impact-frame score."""
    if not isinstance(report, dict):
        return None

    coverage = report.get("joint_measurement_reliability")
    if not isinstance(coverage, dict):
        return None

    scores = report.get("scores")
    if not isinstance(scores, dict):
        scores = {}

    expected_names = tuple(JOINTS.keys())
    valid = {}
    for joint in expected_names:
        if joint not in scores:
            continue
        try:
            value = float(scores[joint])
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and 0.0 <= value <= 100.0:
            valid[joint] = value

    expected = len(expected_names)
    scored = len(valid)
    missing = [joint for joint in expected_names if joint not in valid]
    try:
        measured = int(coverage.get("measured_joints", -1))
    except (TypeError, ValueError, OverflowError):
        measured = -1

    warnings = []
    if measured != scored:
        warnings.append(
            "Measured angle count and valid joint score count differ."
        )
    coverage_missing = set(coverage.get("missing_joints") or [])
    if coverage_missing != set(missing):
        warnings.append(
            "Missing angle names and unscored joint names differ."
        )

    if scored == 0:
        status = "No valid joint scores"
        scenario = None
    else:
        status = "Partial joint evidence" if missing else "Complete joint coverage"
        score_sum = sum(valid.values())
        # Illustrative only. Missing joints are NOT assigned these values.
        scenario = {
            "minimum": round(score_sum / expected, 1),
            "maximum": round((score_sum + len(missing) * 100) / expected, 1),
        }

    if warnings:
        status = "Check score inputs"

    return {
        "status": status,
        "expected_joints": expected,
        "scored_joints": scored,
        "score_input_coverage_percent": round(scored / expected * 100, 1),
        "missing_joint_scores": missing,
        "reported_score_unchanged": report.get("overall_score"),
        "illustrative_full_joint_average_range": scenario,
        "warnings": warnings,
        "note": (
            "The reported score is based on available joint scores only. "
            "The illustrative range assumes missing joints could score from "
            "0 to 100 while existing joint scores remain constant. "
            "It is not a confidence interval, a corrected score, or a "
            "validated measure of batting quality."
        ),
    }
