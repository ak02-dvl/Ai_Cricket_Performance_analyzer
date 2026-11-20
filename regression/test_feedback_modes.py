
from analysis.technique_analyzer import TechniqueAnalyzer
from analysis.scoring import (
    calculate_score,
    calculate_technique_score
)


def run_tests():

    analyzer = TechniqueAnalyzer()

    # Test 1: Legacy shoulder behavior preserved.
    legacy = analyzer.analyze(
        {"left_shoulder": 55},
        batting_stance="Right-handed",
        feedback_mode="legacy"
    )

    assert "Front shoulder is dropping." in legacy

    # Test 2: In-range shoulder angle has no
    # experimental warning.
    aligned = analyzer.analyze(
        {"left_shoulder": 55},
        batting_stance="Right-handed",
        feedback_mode="aligned"
    )

    assert aligned == []

    # Test 3: Back elbow in ideal range.
    aligned = analyzer.analyze(
        {"right_elbow": 85},
        batting_stance="Right-handed",
        feedback_mode="aligned"
    )

    assert aligned == []

    # Test 4: Out-of-range shoulder angle
    # produces anatomically appropriate feedback.
    aligned = analyzer.analyze(
        {"left_shoulder": 20},
        batting_stance="Right-handed",
        feedback_mode="aligned"
    )

    assert any(
        "upper-arm-to-torso angle" in item
        for item in aligned
    )

    # Test 5: Stance mirroring is preserved.
    aligned = analyzer.analyze(
        {"right_elbow": 150},
        batting_stance="Left-handed",
        feedback_mode="aligned"
    )

    assert aligned == []

    scores = calculate_score(
        {"right_elbow": 150},
        batting_stance="Left-handed"
    )

    assert scores["right_elbow"] == 100

    # Test 6: Positive feedback has no penalty.
    assert calculate_technique_score(
        {"left_elbow": 100},
        ["Excellent batting technique."]
    ) == 100.0

    print("ALL FEEDBACK MODE TESTS PASSED")


if __name__ == "__main__":
    run_tests()
