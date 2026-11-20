"""Diagnostic only: compare a posture with its left/right mirrored equivalent.
Run from the AI Cricket Performance Analyzer project root:
    python regression/check_stance_consistency.py
Does not change production files, baseline data, or scores.
"""

from __future__ import annotations

from analysis.technique_analyzer import TechniqueAnalyzer
from analysis.scoring import calculate_score, calculate_technique_score


def mirror_left_right(angles):
    mirrored = {}
    for key, value in angles.items():
        if key.startswith('left_'):
            mirrored['right_' + key[5:]] = value
        elif key.startswith('right_'):
            mirrored['left_' + key[6:]] = value
        else:
            mirrored[key] = value
    return mirrored


def evaluate(label, stance, angles):
    print(f'\n--- {label} / {stance} ---')
    print('Angles:', angles)
    analyzer = TechniqueAnalyzer()

    try:
        mistakes = analyzer.analyze(angles, batting_stance=stance)
        print('Feedback:', mistakes)
    except Exception as exc:
        print('Technique analysis ERROR:', repr(exc))
        return None

    try:
        joint_scores = calculate_score(
            angles,
            batting_stance=stance
        )
        print('Joint scores:', joint_scores)
    except Exception as exc:
        print('Joint scoring ERROR:', repr(exc))
        return None

    try:
        score = calculate_technique_score(joint_scores, mistakes)
        print('Technique score:', score)
    except Exception as exc:
        print('Technique score ERROR:', repr(exc))
        return None

    return {'mistakes': mistakes, 'scores': joint_scores, 'technique_score': score}


def main():
    # Synthetic example, NOT footage or a ground-truth coaching assessment.
    # Its purpose is to expose hard-coded anatomical left/right assumptions.
    right = {
        'left_elbow': 101.33,
        'right_elbow': 145.0,
        'left_knee': 170.26,
        'right_knee': 141.04,
        'left_shoulder': 62.08,
        'right_shoulder': 98.0,
        'left_hip': 113.89,
        'right_hip': 94.56,
    }
    left = mirror_left_right(right)

    result_right = evaluate('Synthetic right-handed posture', 'Right-handed', right)
    result_left = evaluate('Mirrored left-handed posture', 'Left-handed', left)

    if result_right is not None and result_left is not None:
        print('\n--- COMPARISON (inspection only) ---')
        print('Right technique score:', result_right['technique_score'])
        print('Left technique score: ', result_left['technique_score'])
        print('Feedback identical:', result_right['mistakes'] == result_left['mistakes'])
        
        mirrored_right_scores = mirror_left_right(
            result_right["scores"]
        )

        print(
            "Mirrored joint scores identical:",
            mirrored_right_scores == result_left["scores"]
        )

        print(
            "Technique score difference:",
            round(
                abs(
                    result_right["technique_score"]
                    - result_left["technique_score"]
                ),
                1
            )
        )

        print('Note: Differences call for review, not automatic proof of an error.')
        print('This does not replace regression tests or real left-handed footage.')


if __name__ == '__main__':
    main()
