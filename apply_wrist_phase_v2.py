
from pathlib import Path
from datetime import datetime
import shutil


# =========================================================
# TARGET FILE
# =========================================================

PHASE = Path("analysis/phase_detector.py")

if not PHASE.is_file():
    raise SystemExit(
        "Run this script from your AI Cricket project root."
    )

source = PHASE.read_text(encoding="utf-8")

if "wrist_gap_recent_down" in source:
    raise SystemExit(
        "Wrist phase guard v2 is already installed."
    )


# =========================================================
# SAFE REPLACEMENT FUNCTION
# =========================================================

def replace_once(old, new, section):
    global source

    count = source.count(old)

    if count != 1:
        raise RuntimeError(
            f"{section}: expected 1 match, found {count}. "
            "No files have been changed."
        )

    source = source.replace(old, new, 1)


# =========================================================
# 1. PhaseDetector.__init__
# Add experimental state
# =========================================================

replace_once(
    """        self.wrist_up_counter = 0
        self.wrist_down_counter = 0


    # =====================================================
    # RESET""",
    """        self.wrist_up_counter = 0
        self.wrist_down_counter = 0
        self.wrist_gap_frames = 0
        self.wrist_gap_recent_down = 0
        self.backlift_first_elbow = None
        self.backlift_peak_elbow = None


    # =====================================================
    # RESET""",
    "__init__"
)


# =========================================================
# 2. PhaseDetector.reset
# Clear experimental state
# =========================================================

replace_once(
    """        self.wrist_up_counter = 0
        self.wrist_down_counter = 0

    # =====================================================
    # SAFE ANGLE""",
    """        self.wrist_up_counter = 0
        self.wrist_down_counter = 0
        self.wrist_gap_frames = 0
        self.wrist_gap_recent_down = 0
        self.backlift_first_elbow = None
        self.backlift_peak_elbow = None

    # =====================================================
    # SAFE ANGLE""",
    "reset"
)


# =========================================================
# 3. PhaseDetector.observe_wrist_motion
# Reset state when tracked arm changes
# =========================================================

replace_once(
    """            self.wrist_last_relative_y = None
            self.wrist_up_counter = 0
            self.wrist_down_counter = 0

        try:
            wrist""",
    """            self.wrist_last_relative_y = None
            self.wrist_up_counter = 0
            self.wrist_down_counter = 0
            self.wrist_gap_frames = 0
            self.wrist_gap_recent_down = 0

        try:
            wrist""",
    "wrist arm switching"
)


# =========================================================
# 4. PhaseDetector.observe_wrist_motion
# Detect wrist movement after short tracking gaps
#
# FIXED: Blank line in matching pattern
# =========================================================

replace_once(
    """        except (TypeError, ValueError, KeyError, AttributeError):
            self.wrist_last_relative_y = None
            self.wrist_up_counter = 0
            self.wrist_down_counter = 0
            return

        self.wrist_signal_available = True

        if self.wrist_last_relative_y is not None:
""",
    """        except (TypeError, ValueError, KeyError, AttributeError):
            # Preserve a reference through short tracking gaps.
            # Do not count missing frames as observed movement.
            self.wrist_gap_frames += 1

            if self.wrist_gap_frames > 4:
                self.wrist_last_relative_y = None

            self.wrist_gap_recent_down = 0
            self.wrist_up_counter = 0
            self.wrist_down_counter = 0
            return

        self.wrist_signal_available = True

        if (
            2 <= self.wrist_gap_frames <= 4
            and self.wrist_last_relative_y is not None
        ):
            displacement = (
                relative_y - self.wrist_last_relative_y
            )

            self.wrist_gap_recent_down = (
                2 if displacement >= 0.06 else 0
            )

        else:
            self.wrist_gap_recent_down = max(
                0,
                self.wrist_gap_recent_down - 1
            )

        self.wrist_gap_frames = 0

        if self.wrist_last_relative_y is not None:
""",
    "wrist tracking gap"
)


# =========================================================
# 5. PhaseDetector.update -> BACKLIFT
#
# FIXED: Match the existing multiline v1 block
# =========================================================

replace_once(
    """            if (
                self.experimental_wrist_phase_guard
                and self.wrist_signal_available
            ):
                if self.wrist_down_counter >= 2:
                    print(
                        "[WRIST PHASE] Backlift -> Downswing "
                        "(wrist lowering)"
                    )
                    return self.change_phase("Downswing")

                return self.change_phase("Backlift")
""",
    """            if (
                self.experimental_wrist_phase_guard
                and self.wrist_signal_available
            ):

                if self.backlift_first_elbow is None:
                    self.backlift_first_elbow = average_elbow
                    self.backlift_peak_elbow = average_elbow

                self.backlift_peak_elbow = max(
                    self.backlift_peak_elbow,
                    average_elbow
                )

                elbow_rose = (
                    self.backlift_peak_elbow
                    - self.backlift_first_elbow >= 6.0
                )

                # A wrist dip alone must not end Backlift.
                # Require a sustained elbow-angle reversal
                # after the elbow has opened sufficiently.
                if elbow_rose and falling:
                    self.downswing_counter += 1
                else:
                    self.downswing_counter = 0

                elbow_reversal = (
                    self.downswing_counter
                    >= self.transition_frames_required
                )

                # Supporting evidence after short tracking gap.
                recovered_wrist_descent = (
                    self.wrist_gap_recent_down > 0
                    and self.wrist_down_counter >= 2
                )

                if elbow_reversal or recovered_wrist_descent:

                    reason = (
                        "elbow reversal"
                        if elbow_reversal
                        else "wrist recovery"
                    )

                    print(
                        f"[WRIST PHASE] Backlift -> Downswing "
                        f"({reason})"
                    )

                    return self.change_phase("Downswing")

                return self.change_phase("Backlift")
""",
    "Backlift transition"
)


# =========================================================
# 6. PhaseDetector.change_phase
# Reset evidence for a new batting cycle
# =========================================================

replace_once(
    """            self.current_phase = new_phase
            self.phase_frames = 0""",
    """            self.current_phase = new_phase

            if (
                new_phase == "Backlift"
                and self.experimental_wrist_phase_guard
            ):
                self.backlift_first_elbow = None
                self.backlift_peak_elbow = None

            self.phase_frames = 0""",
    "change_phase"
)


# =========================================================
# VALIDATE BEFORE SAVING
# =========================================================

compile(source, str(PHASE), "exec")


# =========================================================
# CREATE BACKUP
# =========================================================

backup = PHASE.with_name(
    "phase_detector_backup_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".py"
)

shutil.copy2(PHASE, backup)


# =========================================================
# WRITE UPDATED FILE
# =========================================================

PHASE.write_text(
    source,
    encoding="utf-8"
)


# =========================================================
# SUCCESS
# =========================================================

print("=" * 60)
print("Wrist phase guard v2 installed successfully.")
print("Experiment remains disabled by default.")
print("Backup:", backup)
print("=" * 60)
