from analysis.ideal_ranges import IDEAL_RANGES


def score_joint(angle, ideal_min, ideal_max):

    if ideal_min <= angle <= ideal_max:
        return 100

    distance = min(
        abs(angle - ideal_min),
        abs(angle - ideal_max)
    )

    score = max(0, 100 - distance * 2)

    return round(score, 1)



def calculate_score(angles, batting_stance="Right-handed"):

    scores = {}

    stance = str(
        batting_stance or "Right-handed"
    ).strip().lower()

    is_left_handed = stance in (
        "left-handed",
        "left handed"
    )

    for joint, angle in angles.items():

        # Keep the original anatomical joint name
        # in the returned scores dictionary.
        reference_joint = joint

        # For left-handed batting, use the equivalent
        # right-handed joint's ideal scoring range.
        # This preserves existing right-handed rules.
        if is_left_handed:

            if joint.startswith("left_"):

                reference_joint = (
                    "right_" + joint[5:]
                )

            elif joint.startswith("right_"):

                reference_joint = (
                    "left_" + joint[6:]
                )

        if reference_joint not in IDEAL_RANGES:
            continue

        low, high = IDEAL_RANGES[reference_joint]

        scores[joint] = score_joint(
            angle,
            low,
            high
        )

    return scores

def calculate_technique_score(scores, mistakes):
    """
    Calculate a separate technique score.

    Starts with the average joint score and applies
    a small penalty for detected technique mistakes.
    """

    if not scores:
        return 0.0

    # Average of joint scores
    joint_score = sum(scores.values()) / len(scores)

    
    # Count actual technique mistakes only.
    # Positive feedback must not reduce the score.

    actual_mistakes = [
        mistake
        for mistake in (mistakes or [])
        if isinstance(mistake, str)
        and mistake.strip()
        and mistake.strip().lower()
        != "excellent batting technique."
    ]

    mistake_penalty = len(actual_mistakes) * 5


    technique_score = max(
        0,
        joint_score - mistake_penalty
    )

    return round(technique_score, 1)