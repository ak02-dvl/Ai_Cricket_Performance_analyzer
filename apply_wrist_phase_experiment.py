
from datetime import datetime
from pathlib import Path
import shutil

PHASE = Path("analysis/phase_detector.py")
ANALYZER = Path("pipeline/analyzer.py")


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one match, found {count}. "
            "No files changed."
        )
    return text.replace(old, new, 1)


def main():
    if not PHASE.is_file() or not ANALYZER.is_file():
        raise SystemExit("Run from your cricket project root.")

    phase = PHASE.read_text(encoding="utf-8")
    analyzer = ANALYZER.read_text(encoding="utf-8")

    if "experimental_wrist_phase_guard" in phase:
        raise SystemExit(
            "Wrist phase experiment already installed."
        )

    # FILE: analysis/phase_detector.py
    # IMPORTS
    phase = replace_once(
        phase,
        "from collections import deque\n",
        "from collections import deque\nimport math\n",
        "Phase import",
    )

    # CLASS: PhaseDetector
    # METHOD: __init__
    phase = replace_once(
        phase,
        """        self.stance_falling_counter = 0


    # =====================================================
    # RESET""",
        """        self.stance_falling_counter = 0

        # Experimental: disabled by default
        self.experimental_wrist_phase_guard = False
        self.wrist_last_relative_y = None
        self.wrist_signal_arm = None
        self.wrist_signal_available = False
        self.wrist_up_counter = 0
        self.wrist_down_counter = 0


    # =====================================================
    # RESET""",
        "PhaseDetector.__init__",
    )

    # METHOD: reset
    phase = replace_once(
        phase,
        """        self.stance_falling_counter = 0

    # =====================================================
    # SAFE ANGLE""",
        """        self.stance_falling_counter = 0
        self.wrist_last_relative_y = None
        self.wrist_signal_arm = None
        self.wrist_signal_available = False
        self.wrist_up_counter = 0
        self.wrist_down_counter = 0

    # =====================================================
    # SAFE ANGLE""",
        "PhaseDetector.reset",
    )

    # NEW METHOD: observe_wrist_motion
    # INSERT BEFORE: MAIN UPDATE
    phase = replace_once(
        phase,
        """    # MAIN UPDATE
    # =====================================================

    def update(self, angles):""",
        """    # EXPERIMENTAL WRIST MOTION
    # =====================================================

    def observe_wrist_motion(self, landmarks, arm):
        # Wrist height relative to the same-side shoulder.
        # This is not a direct bat-tip measurement.

        if not self.experimental_wrist_phase_guard:
            return

        self.wrist_signal_available = False

        if arm != self.wrist_signal_arm:
            self.wrist_signal_arm = arm
            self.wrist_last_relative_y = None
            self.wrist_up_counter = 0
            self.wrist_down_counter = 0

        try:
            wrist = landmarks[f"{arm}_wrist"]
            shoulder = landmarks[f"{arm}_shoulder"]

            relative_y = (
                float(wrist["y"])
                - float(shoulder["y"])
            )

            if not math.isfinite(relative_y):
                raise ValueError("Invalid wrist position")

        except (TypeError, ValueError, KeyError, AttributeError):
            self.wrist_last_relative_y = None
            self.wrist_up_counter = 0
            self.wrist_down_counter = 0
            return

        self.wrist_signal_available = True

        if self.wrist_last_relative_y is not None:
            delta = (
                relative_y - self.wrist_last_relative_y
            )

            if delta < -0.005:
                self.wrist_up_counter += 1
                self.wrist_down_counter = 0

            elif delta > 0.005:
                self.wrist_down_counter += 1
                self.wrist_up_counter = 0

            else:
                self.wrist_up_counter = 0
                self.wrist_down_counter = 0

        self.wrist_last_relative_y = relative_y

    # =====================================================
    # MAIN UPDATE
    # =====================================================

    def update(self, angles, landmarks=None):""",
        "PhaseDetector.observe_wrist_motion",
    )

    # METHOD: update
    # Observe wrist even when the elbow disappears
    phase = replace_once(
        phase,
        """        # If the selected arm is temporarily unavailable,
        # hold the existing phase instead of switching
        # measurements and creating a false trend.
        if tracked_elbow is None:
""",
        """        # Experimental wrist evidence
        self.observe_wrist_motion(
            landmarks, tracked_arm
        )

        # If the selected arm is temporarily unavailable,
        # hold the existing phase instead of switching
        # measurements and creating a false trend.
        if tracked_elbow is None:

            if (
                self.experimental_wrist_phase_guard
                and self.current_phase == "Backlift"
                and self.wrist_signal_available
                and self.wrist_down_counter >= 2
            ):
                print(
                    "[WRIST PHASE] Backlift -> Downswing "
                    "(elbow unavailable)"
                )
                return self.change_phase("Downswing")
""",
        "PhaseDetector.update elbow visibility",
    )

    # METHOD: update
    # SECTION: STANCE
    phase = replace_once(
        phase,
        """        if self.current_phase == "Stance":
            
            # =============================================
            # EXPERIMENTAL PIXEL-MODE DIRECT DOWNSWING""",
        """        if self.current_phase == "Stance":

            # Rising wrist can indicate Backlift
            # even while the elbow is closing.
            if (
                self.experimental_wrist_phase_guard
                and self.wrist_signal_available
                and self.wrist_up_counter >= 2
            ):
                print(
                    "[WRIST PHASE] Stance -> Backlift "
                    "(wrist rising)"
                )
                self.stance_falling_counter = 0
                return self.change_phase("Backlift")

            # =============================================
            # EXPERIMENTAL PIXEL-MODE DIRECT DOWNSWING""",
        "PhaseDetector.update Stance",
    )

    # METHOD: update
    # SECTION: EXPERIMENTAL PIXEL-MODE DIRECT DOWNSWING
    phase = replace_once(
        phase,
        """                if sustained_closure:
                    self.stance_falling_counter += 1""",
        """                # Prevent elbow closure alone from
                # forcing an early Downswing when
                # reliable wrist data is available.
                if (
                    self.experimental_wrist_phase_guard
                    and self.wrist_signal_available
                ):
                    sustained_closure = (
                        sustained_closure
                        and self.wrist_down_counter >= 2
                    )

                if sustained_closure:
                    self.stance_falling_counter += 1""",
        "PhaseDetector.update Downswing entry",
    )

    # METHOD: update
    # SECTION: BACKLIFT
    phase = replace_once(
        phase,
        """        if self.current_phase == "Backlift":

            self.backlift_counter += 1

            # -------------------------------------------------""",
        """        if self.current_phase == "Backlift":

            self.backlift_counter += 1

            # Experimental wrist reversal:
            # a closing elbow alone is insufficient.
            if (
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

            # -------------------------------------------------""",
        "PhaseDetector.update Backlift",
    )

    # FILE: pipeline/analyzer.py
    # CLASS: CricketAnalyzer
    # METHOD: __init__
    analyzer = replace_once(
        analyzer,
        """        self.phase_detector.experimental_stance_downswing_entry = (
            self.angle_geometry_mode == "pixel"
        )""",
        """        self.phase_detector.experimental_stance_downswing_entry = (
            self.angle_geometry_mode == "pixel"
        )

        # Opt-in wrist phase experiment.
        # Default behavior remains unchanged.
        self.phase_detector.experimental_wrist_phase_guard = (
            self.angle_geometry_mode == "pixel"
            and os.getenv(
                "CRICKET_WRIST_PHASE_GUARD", "0"
            ) == "1"
        )""",
        "CricketAnalyzer.__init__",
    )

    # FILE: pipeline/analyzer.py
    # METHOD: process_frame
    # SECTION: Step 12 - Phase
    analyzer = replace_once(
        analyzer,
        """                phase = self.phase_detector.update(
                    angles
                )""",
        """                phase = self.phase_detector.update(
                    angles,
                    landmarks=landmarks
                )""",
        "CricketAnalyzer.process_frame Step 12",
    )

    # Validate before modifying either project file.
    compile(phase, str(PHASE), "exec")
    compile(analyzer, str(ANALYZER), "exec")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup = (
        Path("regression_outputs")
        / f"wrist_guard_backup_{stamp}"
    )

    backup.mkdir(parents=True, exist_ok=False)

    shutil.copy2(
        PHASE, backup / "phase_detector.py"
    )
    shutil.copy2(
        ANALYZER, backup / "analyzer.py"
    )

    PHASE.write_text(phase, encoding="utf-8")
    ANALYZER.write_text(analyzer, encoding="utf-8")

    print("Experimental wrist phase guard installed.")
    print("Default status: DISABLED")
    print("Backups:", backup.resolve())


if __name__ == "__main__":
    main()
