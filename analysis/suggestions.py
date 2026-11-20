from analysis.ideal_ranges import IDEAL_RANGES


def generate_suggestions(angles):

    suggestions = []

    for joint, angle in angles.items():

        if joint not in IDEAL_RANGES:
            continue

        low, high = IDEAL_RANGES[joint]

        if angle < low:

            suggestions.append(
                f"{joint.replace('_',' ').title()} angle is too low. Try increasing it."
            )

        elif angle > high:

            suggestions.append(
                f"{joint.replace('_',' ').title()} angle is too high. Try reducing it."
            )

        else:

            suggestions.append(
                f"{joint.replace('_',' ').title()} is within the ideal range."
            )

    return suggestions