class ShotEventAnalyzer:

    def __init__(self):

        self.active = False

        self.frames = []

        self.phase_frames = {}

        self.started_frame = 0

        self.ended_frame = 0

        self.frame_counter = 0

    # =====================================================
    # START SHOT
    # =====================================================

    def start(self):

        if self.active:
            return

        self.active = True

        self.frames = []

        self.phase_frames = {}

        self.started_frame = self.frame_counter

    # =====================================================
    # UPDATE
    # =====================================================

    def update(self, report):

        self.frame_counter += 1

        if not self.active:
            return

        if report is None:
            return

        if not report.get("angles"):

            return

        self.frames.append(report)

        phase = report.get(
            "phase",
            "Unknown"
        )

        if phase not in self.phase_frames:

            self.phase_frames[phase] = []

        self.phase_frames[phase].append(report)

    # =====================================================
    # CHECK SHOT END
    # =====================================================

    def should_end(self, report):

        if not self.active:
            return False

        if report is None:
            return False

        phase = report.get(
            "phase",
            ""
        )

        if phase == "Follow Through":

            return True

        return False

    # =====================================================
    # FINISH SHOT
    # =====================================================

    def finish(self):

        if not self.active:

            return None

        self.active = False

        self.ended_frame = self.frame_counter

        if not self.frames:

            return None

        # =================================================
        # BEST FRAME
        # =================================================

        best_report = max(

            self.frames,

            key=lambda x:
            x.get(
                "overall_score",
                0
            )

        )

        # =================================================
        # AVERAGE SCORE
        # =================================================

        scores = []

        for report in self.frames:

            score = report.get(
                "overall_score",
                0
            )

            if score > 0:

                scores.append(score)

        if scores:

            average_score = round(

                sum(scores) /
                len(scores),

                1

            )

        else:

            average_score = 0

        # =================================================
        # SHOT
        # =================================================

        shots = []

        for report in self.frames:

            shot = report.get(
                "shot",
                "Unknown"
            )

            if shot not in [
                None,
                "Unknown",
                "Waiting"
            ]:

                shots.append(shot)

        shot = self.most_common(shots)

        # =================================================
        # MAIN PHASE
        # =================================================

        phases = []

        for report in self.frames:

            phase = report.get(
                "phase",
                "Unknown"
            )

            if phase not in [
                None,
                "Unknown",
                "Waiting"
            ]:

                phases.append(phase)

        main_phase = self.most_common(
            phases
        )

        # =================================================
        # SWING
        # =================================================

        swing_speeds = []

        swing_directions = []

        for report in self.frames:

            swing = report.get(
                "swing",
                {}
            )

            speed = swing.get(
                "speed",
                "Unknown"
            )

            direction = swing.get(
                "direction",
                "Unknown"
            )

            if speed != "Unknown":

                swing_speeds.append(
                    speed
                )

            if direction != "Unknown":

                swing_directions.append(
                    direction
                )

        swing_speed = self.most_common(
            swing_speeds
        )

        swing_direction = self.most_common(
            swing_directions
        )

        # =================================================
        # FEEDBACK
        # =================================================

        strengths = []

        improvements = []

        for report in self.frames:

            mistakes = report.get(
                "mistakes",
                []
            )

            for feedback in mistakes:

                if not feedback:

                    continue

                feedback = str(
                    feedback
                ).strip()

                # -----------------------------------------
                # Positive feedback
                # -----------------------------------------

                if (
                    feedback.startswith("✓")
                    or
                    feedback.lower().startswith(
                        "excellent"
                    )
                    or
                    feedback.lower().startswith(
                        "good"
                    )
                ):

                    clean = self.remove_symbol(
                        feedback
                    )

                    if clean not in strengths:

                        strengths.append(
                            clean
                        )

                # -----------------------------------------
                # Negative feedback
                # -----------------------------------------

                else:

                    clean = self.remove_symbol(
                        feedback
                    )

                    if (
                        "insufficient landmarks"
                        in clean.lower()
                    ):

                        continue

                    if (
                        "failed"
                        in clean.lower()
                    ):

                        continue

                    if clean not in improvements:

                        improvements.append(
                            clean
                        )

        # =================================================
        # FINAL RESULT
        # =================================================

        result = {

            "frames":
                len(self.frames),

            "start_frame":
                self.started_frame,

            "end_frame":
                self.ended_frame,

            "average_score":
                average_score,

            "best_score":
                best_report.get(
                    "overall_score",
                    0
                ),

            "shot":
                shot,

            "phase":
                main_phase,

            "swing_speed":
                swing_speed,

            "swing_direction":
                swing_direction,

            "strengths":
                strengths,

            "improvements":
                improvements,

            "best_report":
                best_report

        }

        # =================================================
        # RESET
        # =================================================

        self.frames = []

        self.phase_frames = {}

        return result

    # =====================================================
    # MOST COMMON
    # =====================================================

    def most_common(self, values):

        if not values:

            return "Unknown"

        counts = {}

        for value in values:

            counts[value] = (
                counts.get(
                    value,
                    0
                ) + 1
            )

        return max(
            counts,
            key=counts.get
        )

    # =====================================================
    # REMOVE SYMBOL
    # =====================================================

    def remove_symbol(self, text):

        text = str(text).strip()

        if text.startswith("✓"):

            text = text[1:].strip()

        if text.startswith("✗"):

            text = text[1:].strip()

        return text