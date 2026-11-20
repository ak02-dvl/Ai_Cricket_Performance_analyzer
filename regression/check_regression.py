"""Regression guard for AI Cricket Performance Analyzer JSON exports.

Run from the project root:
    python regression/check_regression.py --vk regression/results/vk.json --rohit regression/results/rohit.json

Demo using the previously verified reference exports:
    python regression/check_regression.py --demo

This tool does NOT run video analysis and does NOT measure physical accuracy.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE = ROOT / "baselines.json"
MISSING = object()


def read_json(path):
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def at(data, dotted_path):
    value = data
    for key in dotted_path.split("."):
        if isinstance(value, dict):
            value = value.get(key, MISSING)
        elif isinstance(value, list) and key.isdigit():
            idx = int(key)
            value = value[idx] if idx < len(value) else MISSING
        else:
            return MISSING
        if value is MISSING:
            return MISSING
    return value


def equal(actual, expected):
    # bool must not silently compare equal to 0 or 1.
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)
    return actual == expected


def describe(value):
    return "<MISSING>" if value is MISSING else repr(value)


def check_report(report, baseline):
    """Return failure messages; never modifies the report or baseline."""
    failures = []

    def assert_equal(label, actual, expected):
        if not equal(actual, expected):
            failures.append(f"{label}: expected {describe(expected)}, got {describe(actual)}")

    def assert_true(condition, message):
        if not condition:
            failures.append(message)

    # 1. Known-good, exact output regression values.
    for path, expected in baseline["checks"].items():
        assert_equal(path, at(report, path), expected)

    # 2. Cross-field invariants (catch inconsistent newly generated reports).
    impact_count = at(report, "impact_count")
    events = at(report, "impact_events")
    if isinstance(events, list):
        assert_equal("impact_count vs impact_events length", impact_count, len(events))
    else:
        failures.append("impact_events: missing or not a list")

    samples = at(report, "swing_summary.samples")
    by_phase = at(report, "swing_summary.samples_by_phase")
    if isinstance(by_phase, dict) and all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0
        for v in by_phase.values()
    ):
        assert_equal("swing sample total vs sum of phases", samples, sum(by_phase.values()))
    else:
        failures.append("swing_summary.samples_by_phase: missing or invalid")

    coverage = at(report, "swing_summary.phase_coverage")
    if not isinstance(coverage, dict):
        failures.append("swing_summary.phase_coverage: missing or invalid")
        return failures

    phase_data = coverage.get("phases")
    others = coverage.get("other_observed_phases")
    if not isinstance(phase_data, dict) or not isinstance(others, dict):
        failures.append("phase_coverage.phases or other_observed_phases: missing/invalid")
        return failures

    valid_others = all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0
        for v in others.values()
    )
    assert_true(valid_others, "phase_coverage.other_observed_phases has invalid counts")
    sum_observed = sum(others.values()) if valid_others else 0
    sum_samples = 0

    for name, data in phase_data.items():
        if not isinstance(data, dict):
            failures.append(f"phase_coverage.{name}: not an object")
            continue
        observed = data.get("observed_frames")
        wrists = data.get("wrist_landmark_frames")
        accepted = data.get("qualifying_motion_samples")
        missing = data.get("frames_without_motion_sample")

        counts = (observed, wrists, accepted, missing)
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in counts):
            failures.append(f"phase_coverage.{name}: invalid numeric counts")
            continue
        assert_true(
            0 <= accepted <= wrists <= observed,
            f"phase_coverage.{name}: impossible observed/wrist/sample counts",
        )
        assert_equal(f"phase_coverage.{name}.frames_without_motion_sample", missing, observed - accepted)
        percent = data.get("motion_sample_coverage_percent", MISSING)
        expected_percent = round(100 * accepted / observed, 1) if observed else None
        assert_equal(f"phase_coverage.{name}.motion_sample_coverage_percent", percent, expected_percent)
        expected_status = (
            "Not observed" if observed == 0 else
            "No qualifying motion samples" if accepted == 0 else
            "Limited evidence" if accepted < 3 else "Measured"
        )
        assert_equal(f"phase_coverage.{name}.status", data.get("status", MISSING), expected_status)
        assert_equal(f"phase_coverage.{name}.unmatched_motion_sample_frames",
                     data.get("unmatched_motion_sample_frames", MISSING), [])
        first, last = data.get("first_observed_frame"), data.get("last_observed_frame")
        if observed == 0:
            assert_true(first is None and last is None, f"phase_coverage.{name}: unobserved phase has frame range")
        else:
            assert_true(isinstance(first, int) and isinstance(last, int) and first <= last,
                        f"phase_coverage.{name}: missing/invalid observed frame range")
        if isinstance(by_phase, dict):
            assert_equal(f"phase_coverage.{name} vs swing phase samples",
                         accepted, by_phase.get(name, 0))
        sum_observed += observed
        sum_samples += accepted

    assert_equal("phase_observation_frames vs labeled+other frames",
                 coverage.get("phase_observation_frames", MISSING), sum_observed)
    assert_equal("swing samples vs phase coverage accepted samples", samples, sum_samples)

    alignment = at(report, "swing_summary.impact_alignment")
    event_frame = at(report, "impact_events.0.impact_frame")
    if isinstance(alignment, dict):
        assert_equal("impact alignment frame vs first impact event",
                     alignment.get("confirmed_impact_frame", MISSING), event_frame)
        if isinstance(event_frame, int):
            near = alignment.get("samples_near_impact", [])
            impact_samples = [item for item in near
                              if isinstance(item, dict) and item.get("source_frame") == event_frame]
            assert_true(len(impact_samples) == 1 and
                        impact_samples[0].get("phase_passed_to_swing") == "impact",
                        "impact_alignment: missing or mislabeled impact-frame motion sample")
    else:
        failures.append("impact_alignment: missing or invalid")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vk", type=Path, help="Fresh exported JSON for vk.mp4")
    parser.add_argument("--rohit", type=Path, help="Fresh exported JSON for rohit.mp4")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--demo", action="store_true", help="Test against bundled known-good snapshots (not fresh video tests)")
    args = parser.parse_args(argv)

    if args.demo:
        if args.vk or args.rohit:
            parser.error("Use --demo by itself, or pass both --vk and --rohit")
        inputs = {
            "vk.mp4": ROOT / "samples" / "vk_reference.json",
            "rohit.mp4": ROOT / "samples" / "rohit_reference.json",
        }
        print("DEMO ONLY: testing previously verified JSON snapshots, not new video runs.\n")
    else:
        if not args.vk or not args.rohit:
            parser.error("Both --vk and --rohit are required for a complete regression run")
        inputs = {"vk.mp4": args.vk, "rohit.mp4": args.rohit}

    try:
        manifest = read_json(args.baseline)
        baselines = manifest["videos"]
        reports = {name: read_json(path) for name, path in inputs.items()}
        for name, report in reports.items():
            if report.get("video_name") != name:
                raise ValueError(f"Wrong report for {name}: video_name={report.get('video_name')!r}")
    except (ValueError, KeyError, TypeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    failure_count = 0
    for name, report in reports.items():
        problems = check_report(report, baselines[name])
        if problems:
            failure_count += len(problems)
            print(f"[FAIL] {name}: {len(problems)} issue(s)")
            for problem in problems:
                print(f"       - {problem}")
        else:
            print(f"[PASS] {name}: {len(baselines[name]['checks'])} baseline checks + consistency checks")

    print("\n" + ("DEMO PASSED" if args.demo and not failure_count else
                  "REGRESSION PASSED: VK + ROHIT" if not failure_count else
                  f"REGRESSION FAILED: {failure_count} problem(s)"))
    return 1 if failure_count else 0


if __name__ == "__main__":
    sys.exit(main())
