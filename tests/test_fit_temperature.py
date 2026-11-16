"""Temperature for the cause probabilities: fitted on the development split so the percentages mean what they say."""
import math
import random

from llm.fit_temperature import fit, nll


def test_an_overconfident_scorer_gets_a_temperature_above_one():
    rng = random.Random(0)
    rows = []
    for _ in range(400):           # right 60% of the time but log-odds of +-6: far too sure
        truth = "A" if rng.random() < 0.6 else "B"
        rows.append(({"A": 6.0, "B": -6.0}, truth))
    t = fit(rows)
    assert 3.0 < t < 40.0
    assert nll(rows, t) < nll(rows, 1.0)


def test_a_calibrated_scorer_keeps_about_one():
    rng = random.Random(1)
    rows = []
    for _ in range(3000):
        lo = rng.uniform(-3, 3)
        p = 1 / (1 + math.exp(-lo))
        rows.append(({"A": lo, "B": 0.0}, "A" if rng.random() < p else "B"))
    assert 0.8 < fit(rows) < 1.25
