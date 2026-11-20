# Automated regression tests — AI Cricket Performance Analyzer

**Target folder:** `<project root>/regression/` next to `app.py` and the `pipeline/` folder.

This checker uses *only Python standard-library modules*. It compares JSON reports from fresh runs against protected, previously verified values. It **does not** run the video analyzer, verify ground-truth physical impact or measure coaching accuracy.

## 1. Install

Copy the entire `regression` folder into:

`C:\Users\91636\Desktop\Ai_cricket_performance_analyzer\regression\`

**Do not edit:** `app.py`, `pipeline/analyzer.py`, `analysis/phase_detector.py`, or `analysis/swing_analyzer.py` for this enhancement.

## 2. Check the checker (demo and self-tests)

In VS Code's terminal, from your project root:

```powershell
python regression/check_regression.py --demo
python -m unittest discover -s regression -p "test_regression_checker.py" -v
```

`--demo` validates the bundled **old known-good reports**. It does *not* establish whether a new code change passes regression. The unit tests deliberately alter old reports to verify the checker catches failures.

## 3. Run real regression tests after any change

1. Run `streamlit run app.py` and analyze **VK.mp4**, selecting **Right-handed**.
2. Download the *newly generated* `cricket_analysis_summary.json` and save as `regression/results/vk.json`.
3. Analyze **Rohit.mp4**, selecting **Right-handed**, and save its *newly generated* JSON as `regression/results/rohit.json`.
4. Compare both new reports:

```powershell
python regression/check_regression.py --vk regression/results/vk.json --rohit regression/results/rohit.json
```

A successful result ends with `REGRESSION PASSED: VK + ROHIT`. A failed result lists the exact mismatched fields and exits with status `1`. Missing/unreadable/wrong-video files exit with status `2`.

**Important:** The downloaded Streamlit file is named the same for both videos, so rename the **fresh** files before running the next video. Do not copy the bundled `samples/` files into `results/`; that would test old snapshots rather than your current code.

## 4. What is protected?

The manifest `baselines.json` contains **95 VK + 94 Rohit exact checks** for video metadata, processed counts, impact frame/event/confirmation, stopping reason, scores, footwork, phase-wise swing samples, impact alignment, and phase coverage. The checker additionally validates invariants (coverage percentages, sample counts, observed-frame accounting, and consistency between impact labeling and event frame).

Values are intentionally strict to catch changes. A genuinely beneficial future enhancement may change a protected value; in that case, investigate and verify both videos, then intentionally update the baseline **only after confirming the new result is correct**. Never automatically overwrite protected baselines.

## Files

- `check_regression.py` — full regression checker (run on fresh exports).
- `baselines.json` — protected expected values from the most recent approved reports.
- `test_regression_checker.py` — tests that the checker itself works.
- `samples/vk_reference.json`, `samples/rohit_reference.json` — historical known-good reports used **only for demonstration and self-tests**.
- `results/` — place your **fresh** app exports here (not included in the bundle).
