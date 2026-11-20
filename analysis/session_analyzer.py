class SessionAnalyzer:

    def __init__(self):

        self.frames_analyzed = 0

        self.scores = []
        self.phases = []
        self.shots = []

        self.swing_speeds = []
        self.swing_directions = []

        # Store positive and negative feedback separately
        self.strengths = []
        self.improvements = []

    # =====================================================
    # ADD FRAME RESULT
    # =====================================================

    def update(self, report):

        if report is None:
            return

        angles = report.get("angles", {})

        # Don't count frames where pose analysis failed
        if not angles:
            return

        self.frames_analyzed += 1

        # =================================================
        # SCORE
        # =================================================

        score = report.get("overall_score", 0)

        if score > 0:
            self.scores.append(score)

        # =================================================
        # PHASE
        # =================================================

        phase = report.get("phase")

        if phase not in [
            None,
            "Waiting",
            "Unknown",
            "Pose Failed",
            "No Landmarks",
            "Angle Failed",
            "Crop Failed"
        ]:

            self.phases.append(phase)

        # =================================================
        # SHOT
        # =================================================

        shot = report.get("shot")

        if shot not in [
            None,
            "Waiting",
            "Unknown"
        ]:

            self.shots.append(shot)

        # =================================================
        # SWING
        # =================================================

        swing = report.get("swing", {})

        speed = swing.get(
            "speed",
            "Unknown"
        )

        direction = swing.get(
            "direction",
            "Unknown"
        )

        if speed != "Unknown":
            self.swing_speeds.append(speed)

        if direction != "Unknown":
            self.swing_directions.append(direction)

        # =================================================
        # TECHNIQUE FEEDBACK
        # =================================================

        mistakes = report.get(
            "mistakes",
            []
        )

        for item in mistakes:

            self.add_feedback(item)

        # =================================================
        # FOOTWORK FEEDBACK
        # =================================================

        footwork = report.get(
            "footwork",
            []
        )

        for item in footwork:

            self.add_feedback(item)

    # =====================================================
    # CLASSIFY FEEDBACK
    # =====================================================

    def add_feedback(self, item):

        if not item:
            return

        item = str(item).strip()

        # Ignore technical/system messages
        ignored = [

            "Insufficient landmarks",

            "Unknown",

            "Pose Failed",

            "No Landmarks",

            "Angle Failed",

            "Crop Failed"

        ]

        if item in ignored:
            return

        # ---------------------------------------------
        # Positive feedback
        # ---------------------------------------------

        if (
            item.startswith("✓")
            or
            "Excellent" in item
            or
            "Good" in item
            or
            "stable" in item.lower()
            or
            "correctly" in item.lower()
            or
            "balanced" in item.lower()
        ):

            if item not in self.strengths:

                self.strengths.append(item)

        # ---------------------------------------------
        # Negative feedback
        # ---------------------------------------------

        elif (
            item.startswith("✗")
            or
            "too low" in item.lower()
            or
            "too straight" in item.lower()
            or
            "dropping" in item.lower()
            or
            "collapsing" in item.lower()
            or
            "bends too much" in item.lower()
            or
            "excessively" in item.lower()
            or
            "improve" in item.lower()
            or
            "shift weight" in item.lower()
        ):

            if item not in self.improvements:

                self.improvements.append(item)

    # =====================================================
    # MOST COMMON VALUE
    # =====================================================

    def most_common(self, values):

        if not values:

            return "Unknown"

        counts = {}

        for value in values:

            counts[value] = (
                counts.get(value, 0) + 1
            )

        return max(
            counts,
            key=counts.get
        )

    # =====================================================
    # AVERAGE SCORE
    # =====================================================

    def average_score(self):

        if not self.scores:

            return 0

        return round(
            sum(self.scores) /
            len(self.scores),
            1
        )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    def get_summary(self):

        return {

            "frames_analyzed":
                self.frames_analyzed,

            "average_score":
                self.average_score(),

            "shot":
                self.most_common(
                    self.shots
                ),

            "phase":
                self.most_common(
                    self.phases
                ),

            "swing_speed":
                self.most_common(
                    self.swing_speeds
                ),

            "swing_direction":
                self.most_common(
                    self.swing_directions
                ),

            "strengths":
                self.strengths,

            "improvements":
                self.improvements

        }

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.frames_analyzed = 0

        self.scores.clear()

        self.phases.clear()

        self.shots.clear()

        self.swing_speeds.clear()

        self.swing_directions.clear()

        self.strengths.clear()

        self.improvements.clear()