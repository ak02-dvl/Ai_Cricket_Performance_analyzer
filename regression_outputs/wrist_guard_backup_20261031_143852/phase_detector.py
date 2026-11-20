
from collections import deque


class PhaseDetector:

    def __init__(
        self,
        history_size=7,
        stance_frames=2,
        transition_frames=2,
        follow_through_frames=2,
        recovery_frames=3
    ):

        self.history_size = int(history_size)

        self.stance_frames_required = int(stance_frames)
        self.transition_frames_required = int(transition_frames)
        self.follow_through_frames_required = int(
            follow_through_frames
        )
        self.recovery_frames_required = int(
            recovery_frames
        )

        # -------------------------------------------------
        # Angle history
        # -------------------------------------------------

        self.left_history = deque(
            maxlen=self.history_size
        )

        self.right_history = deque(
            maxlen=self.history_size
        )

        self.average_history = deque(
            maxlen=self.history_size
        )

        # -------------------------------------------------
        # State
        # -------------------------------------------------

        self.current_phase = "Unknown"

        self.stance_counter = 0
        self.backlift_counter = 0
        self.downswing_counter = 0
        self.follow_through_counter = 0
        self.recovery_counter = 0

        # Number of frames spent in current phase
        self.phase_frames = 0

        # Used to detect movement reversal
        self.previous_average = None

        self.previous_trend = 0
        self.primary_arm = None

        # Number of consecutive frames where the selected
        # primary arm was unavailable.
        self.primary_arm_missing = 0

        # Do not switch arms because of one bad MediaPipe frame.
        self.arm_switch_frames = 3
        self.follow_through_requires_impact = False
         # -------------------------------------------------
        # Delivery-entry synchronization
        # -------------------------------------------------

        # The analyzer will arm this when a real delivery
        # has been confirmed.
        self.delivery_entry_armed = False

        # First reliable elbow angle after delivery activation.
        self.delivery_entry_elbow = None

        # Number of valid phase samples since activation.
        self.delivery_entry_samples = 0

        # Consecutive falling-motion frames.
        self.delivery_entry_falling_counter = 0

        # If phase analysis begins while the elbow is already
        # relatively closed, the batsman may already be in the
        # downswing rather than starting from stance/backlift.
        self.direct_downswing_entry_angle = 110.0

        # Require sustained falling motion before direct entry.
        self.direct_downswing_entry_frames = 2

        # Direct entry is only allowed near the beginning of
        # the newly activated delivery sequence.
        self.delivery_entry_window = 10
        
        # Experimental pixel-mode phase entry.
        # Disabled by default to protect legacy behavior.
        self.experimental_stance_downswing_entry = False
        self.stance_falling_counter = 0


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.left_history.clear()
        self.right_history.clear()
        self.average_history.clear()

        self.current_phase = "Unknown"

        self.stance_counter = 0
        self.backlift_counter = 0
        self.downswing_counter = 0
        self.follow_through_counter = 0
        self.recovery_counter = 0

        self.phase_frames = 0

        self.previous_average = None
        self.previous_trend = 0
        self.primary_arm = None
        self.primary_arm_missing = 0
        self.delivery_entry_armed = False
        self.delivery_entry_elbow = None
        self.delivery_entry_samples = 0
        self.delivery_entry_falling_counter = 0
        self.stance_falling_counter = 0

    # =====================================================
    # SAFE ANGLE
    # =====================================================

    def get_angle(self, angles, name):

        try:

            value = angles.get(name)

            if value is None:
                return None

            value = float(value)

            if value < 0 or value > 180:
                return None

            return value

        except Exception:
            return None

        # =====================================================
    # SELECT STABLE PRIMARY ARM
    # =====================================================

    def select_primary_elbow(
        self,
        left_elbow,
        right_elbow
    ):
        """
        Select one elbow as the continuous phase signal.

        Rules:
        1. Once an arm is selected, keep using it.
        2. Ignore temporary appearance of the other arm.
        3. Do not immediately switch if the primary arm disappears.
        4. Switch only after several consecutive missing frames.
        5. Clear movement history after an actual arm switch so
           two different elbow signals are never compared.
        """

        # -------------------------------------------------
        # INITIAL ARM SELECTION
        # -------------------------------------------------

        if self.primary_arm is None:

            if (
                left_elbow is not None
                and
                right_elbow is None
            ):
                self.primary_arm = "left"

            elif (
                right_elbow is not None
                and
                left_elbow is None
            ):
                self.primary_arm = "right"

            elif (
                left_elbow is not None
                and
                right_elbow is not None
            ):
                # When both are initially visible, use the
                # more extended elbow as the initial signal.
                #
                # More importantly, once chosen it remains
                # locked instead of averaging both arms.
                if left_elbow >= right_elbow:
                    self.primary_arm = "left"
                else:
                    self.primary_arm = "right"

            else:
                return None, None

            self.primary_arm_missing = 0

            print(
                f"[PHASE ARM] Primary arm selected: "
                f"{self.primary_arm}"
            )


        # -------------------------------------------------
        # GET CURRENT PRIMARY / SECONDARY VALUES
        # -------------------------------------------------

        if self.primary_arm == "left":

            primary_value = left_elbow
            secondary_value = right_elbow
            secondary_arm = "right"

        else:

            primary_value = right_elbow
            secondary_value = left_elbow
            secondary_arm = "left"


        # -------------------------------------------------
        # PRIMARY ARM AVAILABLE
        # -------------------------------------------------
        if primary_value is not None:

            # Was the primary arm missing in previous frames?
            recovered_after_gap = (
                self.primary_arm_missing > 0
            )

            self.primary_arm_missing = 0


            # -------------------------------------------------
            # PRIMARY ARM RECOVERED AFTER LANDMARK GAP
            # -------------------------------------------------
            #
            # Never compare the newly recovered elbow against
            # stale pre-gap movement history.
            #
            # Otherwise a temporary MediaPipe dropout can create
            # an artificial reversal and trigger Follow Through
            # too early.
            # -------------------------------------------------

            if recovered_after_gap:

                self.average_history.clear()

                self.follow_through_counter = 0

                print(
                    f"[PHASE ARM RECOVERY] "
                    f"{self.primary_arm} arm recovered; "
                    f"movement history reset"
                )


            return (
                primary_value,
                self.primary_arm
            )



        # -------------------------------------------------
        # PRIMARY ARM TEMPORARILY MISSING
        # -------------------------------------------------

        self.primary_arm_missing += 1

        print(
            f"[PHASE ARM] "
            f"{self.primary_arm} arm missing "
            f"({self.primary_arm_missing}/"
            f"{self.arm_switch_frames})"
        )


        # Do NOT immediately use the other arm.
        #
        # Holding the current phase for a couple of frames
        # is much safer than creating an artificial trend.
        if (
            self.primary_arm_missing
            <
            self.arm_switch_frames
        ):
            return None, self.primary_arm


        # -------------------------------------------------
        # SWITCH ONLY IF OTHER ARM IS ACTUALLY AVAILABLE
        # -------------------------------------------------

        if secondary_value is None:

            return None, self.primary_arm


        old_arm = self.primary_arm

        self.primary_arm = secondary_arm
        self.primary_arm_missing = 0


        # CRITICAL:
        #
        # Never compare elbow history from two different arms.
        self.average_history.clear()


        print(
            f"[PHASE ARM SWITCH] "
            f"{old_arm} -> {self.primary_arm}"
        )


        return (
            secondary_value,
            self.primary_arm
        )  
    # =====================================================
    # ARM DELIVERY ENTRY
    # =====================================================

    def arm_delivery_entry(self):
        """
        Tell PhaseDetector that a real delivery has just
        activated phase analysis.

        This allows the detector to recognize when analysis
        begins while the batsman is already moving through
        the downswing.
        """

        self.delivery_entry_armed = True

        self.delivery_entry_elbow = None

        self.delivery_entry_samples = 0

        self.delivery_entry_falling_counter = 0

        print(
            "[PHASE ENTRY] "
            "Delivery entry armed"
        )  

    # =====================================================
    # CHANGE PHASE
    # =====================================================

    def change_phase(self, new_phase):
        if new_phase != self.current_phase:
            print(
                f"[PHASE CHANGE] "
                f"{self.current_phase} -> {new_phase}"
            )
            self.current_phase = new_phase
            self.phase_frames = 0

        self.phase_frames += 1
        return self.current_phase

    def register_impact(self):    
        if self.current_phase != "Impact":
            print(
                f"[PHASE SYNC] "
                f"{self.current_phase} -> Impact "
                f"(bat-ball contact)"
            )

        self.current_phase = "Impact"

        self.phase_frames = 0

        self.stance_counter = 0
        self.backlift_counter = 0
        self.downswing_counter = 0
        self.follow_through_counter = 0
        self.recovery_counter = 0

        return self.current_phase

    # =====================================================
    # GET TREND
    # =====================================================

    def calculate_trend(self):

        if len(self.average_history) < 3:
            return 0.0

        values = list(self.average_history)

        recent = values[-1]
        previous = values[-3]

        return recent - previous

    # =====================================================
    # GET SHORT-TERM MOVEMENT
    # =====================================================

    def calculate_short_change(self):

        if len(self.average_history) < 2:
            return 0.0

        values = list(self.average_history)

        return values[-1] - values[-2]

    # =====================================================
    # MAIN UPDATE
    # =====================================================

    def update(self, angles):
        print("🔥 PHASE DETECTOR UPDATE CALLED 🔥")

        if angles is None:
            return self.current_phase

        left_elbow = self.get_angle(
            angles,
            "left_elbow"
        )

        right_elbow = self.get_angle(
            angles,
            "right_elbow"
        )
        print(
        f"[ANGLE DEBUG] "
        f"left_elbow={left_elbow} "
        f"right_elbow={right_elbow} "
        f"angles={angles}"
        )

        # -------------------------------------------------
        # Need at least one valid arm
        # -------------------------------------------------

        # -------------------------------------------------
        # Stable primary-arm phase signal
        # -------------------------------------------------

        (
            tracked_elbow,
            tracked_arm
        ) = self.select_primary_elbow(
            left_elbow,
            right_elbow
        )


        # If the selected arm is temporarily unavailable,
        # hold the existing phase instead of switching
        # measurements and creating a false trend.
        if tracked_elbow is None:

            print(
                f"[PHASE DEBUG] "
                f"PrimaryArm={tracked_arm} "
                f"TrackedElbow=None "
                f"CurrentPhase={self.current_phase} "
                f"(phase held)"
            )

            return self.current_phase


        # Keep the old variable name below so the rest of
        # PhaseDetector does not need a risky rewrite.
        #
        # It now represents ONE consistent elbow rather
        # than an average of whichever elbows are visible.
        average_elbow = tracked_elbow

        # -------------------------------------------------
        # Store history
        # -------------------------------------------------

        if left_elbow is not None:
            self.left_history.append(left_elbow)

        if right_elbow is not None:
            self.right_history.append(right_elbow)

        self.average_history.append(average_elbow)

        # -------------------------------------------------
        # Calculate movement
        # -------------------------------------------------

        trend = self.calculate_trend()
        short_change = self.calculate_short_change()

        # =================================================
        # DELIVERY ENTRY CONTEXT
        # =================================================

        if self.delivery_entry_armed:

            self.delivery_entry_samples += 1

            # Store first reliable elbow value after
            # the delivery gate activates.
            if self.delivery_entry_elbow is None:

                self.delivery_entry_elbow = average_elbow

                print(
                    f"[PHASE ENTRY] "
                    f"InitialElbow="
                    f"{self.delivery_entry_elbow:.2f}"
                )

                # If the elbow is still relatively open,
                # this is a normal beginning of the batting
                # sequence. Let the existing state machine
                # handle it normally.
                if (
                    self.delivery_entry_elbow
                    >=
                    self.direct_downswing_entry_angle
                ):

                    print(
                        "[PHASE ENTRY] "
                        "Normal phase entry"
                    )

                    self.delivery_entry_armed = False


            # Do not allow special entry detection forever.
            if (
                self.delivery_entry_samples
                >
                self.delivery_entry_window
            ):

                self.delivery_entry_armed = False

                self.delivery_entry_falling_counter = 0
        print(
                f"[PHASE DEBUG] "
                f"Arm={tracked_arm} "
                f"TrackedElbow={average_elbow:.2f} "
                f"Trend={trend:.2f} "
                f"ShortChange={short_change:.2f} "
                f"CurrentPhase={self.current_phase}"
                )        
       

        # -------------------------------------------------
        # Movement classification
        #
        # Positive trend:
        #     arms becoming more open
        #
        # Negative trend:
        #     arms becoming more closed
        # -------------------------------------------------
        # A trend needs at least 3 samples because
        # calculate_trend() compares current with [-3].
        trend_ready = (
            len(self.average_history) >= 3
        )

        rising = (
            trend_ready
            and
            trend > 1.0
        )

        falling = (
            trend_ready
            and
            trend < -1.0
        )

        stable = (
            trend_ready
            and
            abs(trend) <= 1.0
        )


        # =================================================
        # UNKNOWN
        # =================================================

        if self.current_phase == "Unknown":

            self.stance_counter += 1
            # =================================================
            # DIRECT MID-SWING ENTRY
            # =================================================
            #
            # If delivery analysis begins while the elbow is
            # already relatively closed and continues closing,
            # the batsman may already be in Downswing.
            # =================================================

            if (
                self.delivery_entry_armed
                and
                self.delivery_entry_elbow is not None
                and
                self.delivery_entry_elbow
                <
                self.direct_downswing_entry_angle
            ):

                if falling:

                    self.delivery_entry_falling_counter += 1

                else:

                    self.delivery_entry_falling_counter = 0


                if (
                    self.delivery_entry_falling_counter
                    >=
                    self.direct_downswing_entry_frames
                ):

                    print(
                        f"[PHASE ENTRY] "
                        f"Mid-swing entry detected -> Downswing "
                        f"(initial_elbow="
                        f"{self.delivery_entry_elbow:.2f}, "
                        f"current_elbow="
                        f"{average_elbow:.2f})"
                    )

                    self.delivery_entry_armed = False

                    self.delivery_entry_falling_counter = 0

                    return self.change_phase(
                        "Downswing"
                    )

            # Initial stable position
            if (
                average_elbow >= 120
                and abs(short_change) < 4
            ):

                if (
                    self.stance_counter
                    >= self.stance_frames_required
                ):

                    return self.change_phase("Stance")

            # Movement has already started
            if rising:

                self.backlift_counter += 1

                if self.backlift_counter >= 1:

                    return self.change_phase(
                        "Backlift"
                    )

            return self.change_phase("Unknown")

        # =================================================
        # STANCE
        # =================================================

        if self.current_phase == "Stance":
            
            # =============================================
            # EXPERIMENTAL PIXEL-MODE DIRECT DOWNSWING
            # =============================================

            if self.experimental_stance_downswing_entry:

                initial_elbow = self.delivery_entry_elbow

                sustained_closure = (
                    initial_elbow is not None
                    and 110.0 <= initial_elbow <= 130.0
                    and (initial_elbow - average_elbow) >= 12.0
                    and falling
                    and short_change < -1.0
                )

                if sustained_closure:
                    self.stance_falling_counter += 1
                else:
                    self.stance_falling_counter = 0

                if self.stance_falling_counter >= 2:

                    print(
                        "[PHASE ENTRY] "
                        "Experimental Stance -> Downswing "
                        f"(initial={initial_elbow:.2f}, "
                        f"current={average_elbow:.2f})"
                    )

                    self.stance_falling_counter = 0

                    return self.change_phase("Downswing")


            # -------------------------------------------------
            # Backlift starts when the upper-body/arms begin
            # moving away from the settled position.
            # -------------------------------------------------

            if rising:

                self.backlift_counter += 1

            else:

                self.backlift_counter = 0

            if self.backlift_counter >= 1:

                return self.change_phase(
                    "Backlift"
                )

            return self.change_phase("Stance")

        # =================================================
        # BACKLIFT
        # =================================================

        if self.current_phase == "Backlift":

            self.backlift_counter += 1

            # -------------------------------------------------
            # IMPORTANT:
            #
            # Downswing is detected primarily by REVERSAL.
            #
            # We do not require a specific elbow angle.
            # -------------------------------------------------

            if falling:

                self.downswing_counter += 1

            else:

                self.downswing_counter = 0

            # -------------------------------------------------
            # Strong reversal
            # -------------------------------------------------

            if (
                self.downswing_counter
                >= self.transition_frames_required
            ):

                return self.change_phase(
                    "Downswing"
                )

            # -------------------------------------------------
            # Safety condition
            #
            # If the angle has stopped increasing and the
            # arms have already reached a relatively closed
            # position, consider movement to be downswing.
            # -------------------------------------------------

            if (
                average_elbow < 105
                and short_change < -0.5
            ):

                return self.change_phase(
                    "Downswing"
                )

            return self.change_phase("Backlift")

        # =================================================
        # DOWNSWING
        # =================================================

        if self.current_phase == "Downswing":

            self.downswing_counter += 1

            # -------------------------------------------------
            # During downswing we wait for the movement to
            # slow/reverse before declaring follow-through.
            # -------------------------------------------------

            if stable:

                self.follow_through_counter += 1

            elif rising:

                self.follow_through_counter += 1

            else:

                self.follow_through_counter = 0

            
            # ============================================
            # FOLLOW-THROUGH CONTACT GATE
            # ============================================

            if (
                self.follow_through_counter
                >= self.follow_through_frames_required
            ):

                # Legacy behavior when the gate is disabled.
                if not self.follow_through_requires_impact:

                    return self.change_phase(
                        "Follow Through"
                    )

                # With the gate enabled, elbow reversal
                # alone does not confirm Follow Through.
                #
                # The pipeline will call register_impact()
                # when bat-ball contact is confirmed.

                if (
                    self.follow_through_counter
                    == self.follow_through_frames_required
                ):

                    print(
                        "[PHASE HOLD] Reversal detected; "
                        "waiting for confirmed impact"
                    )

            return self.change_phase("Downswing")

        # =================================================
        # IMPACT
        # =================================================

        if self.current_phase == "Impact":

            self.phase_frames += 1

            # Impact is intentionally short-lived.
            # Once contact has happened, subsequent movement
            # belongs to the follow-through.
            if self.phase_frames >= 1:
                return self.change_phase(
                    "Follow Through"
                )

            return self.change_phase(
                "Impact"
            )

        # =================================================
        # FOLLOW THROUGH
        # =================================================

        if self.current_phase == "Follow Through":

            self.follow_through_counter += 1

            # -------------------------------------------------
            # Recovery = movement settles again
            # -------------------------------------------------

            if (
                average_elbow >= 115
                and abs(trend) < 2.0
            ):

                self.recovery_counter += 1

            else:

                self.recovery_counter = 0

            if (
                self.recovery_counter
                >= self.recovery_frames_required
            ):

                return self.change_phase(
                    "Recovery"
                )

            return self.change_phase(
                "Follow Through"
            )

        # =================================================
        # RECOVERY
        # =================================================

        if self.current_phase == "Recovery":

            # -------------------------------------------------
            # New movement can begin another batting cycle.
            # -------------------------------------------------

            if rising:

                return self.change_phase(
                    "Backlift"
                )

            return self.change_phase(
                "Recovery"
            )

        return self.current_phase


# =============================================================
# GLOBAL DETECTOR
# =============================================================

_default_detector = PhaseDetector()


# =============================================================
# BACKWARD COMPATIBILITY
# =============================================================

def detect_phase(angles):

    """
    Backward-compatible interface.

    Existing code can still call:

        detect_phase(angles)

    The detector maintains temporal state internally.
    """

    return _default_detector.update(angles)


# =============================================================
# RESET FUNCTION
# =============================================================

def reset_phase_detector():

    """
    Reset the global detector.
    """

    _default_detector.reset()