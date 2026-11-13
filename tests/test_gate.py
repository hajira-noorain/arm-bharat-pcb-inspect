import math

import numpy as np
import pytest

from pcbinspect.gate import (ConformalGate, clopper_pearson, evaluate, false_reject_threshold,
                             miss_threshold)


def test_miss_threshold_is_kth_smallest():
    s = np.arange(1, 100)  # n = 99 -> alpha*(n+1) = 5 for alpha=0.05
    assert miss_threshold(s, 0.05) == 5.0


def test_too_few_defective_boards_never_auto_passes():
    assert miss_threshold(np.array([0.9, 0.8, 0.7]), 0.05) == -math.inf


def test_too_few_good_boards_never_auto_fails():
    assert false_reject_threshold(np.array([0.1, 0.2]), 0.01) == math.inf


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_marginal_miss_rate_guarantee_holds(alpha):
    """Monte-Carlo check of the split-conformal guarantee E[miss] <= alpha.

    Defective-board scores are drawn from an arbitrary (skewed) distribution; the guarantee is
    distribution-free so it must hold regardless of the shape.
    """
    rng = np.random.default_rng(0)
    n_cal, n_test, reps = 200, 400, 2000
    miss = []
    for _ in range(reps):
        cal = rng.beta(5, 1.5, n_cal)
        test = rng.beta(5, 1.5, n_test)
        lam = miss_threshold(cal, alpha)
        miss.append(np.mean(test < lam))
    mean_miss = float(np.mean(miss))
    # standard error of the mean over reps is tiny; allow 3 SE of slack
    se = float(np.std(miss) / math.sqrt(reps))
    assert mean_miss <= alpha + 3 * se, (mean_miss, alpha)
    # and it is not wildly conservative (k/(n+1) is close to alpha for n=200)
    assert mean_miss >= alpha * 0.6


def test_false_reject_guarantee_holds():
    rng = np.random.default_rng(1)
    beta, reps = 0.02, 2000
    fr = []
    for _ in range(reps):
        cal = rng.beta(1.2, 6, 300)
        test = rng.beta(1.2, 6, 300)
        tau = false_reject_threshold(cal, beta)
        fr.append(np.mean(test >= tau))
    se = float(np.std(fr) / math.sqrt(reps))
    assert float(np.mean(fr)) <= beta + 3 * se


def test_verdict_zones_and_evaluate():
    gate = ConformalGate(defective_scores=np.linspace(0.3, 1.0, 99),
                         good_scores=np.linspace(0.0, 0.5, 199))
    th = gate.thresholds(alpha=0.05, beta=0.01)
    assert th.lam < th.tau  # overlapping distributions -> a REVIEW band exists
    assert th.verdict(th.lam - 1e-6) == "PASS"
    assert th.verdict(th.tau) == "FAIL"
    assert th.verdict((th.lam + th.tau) / 2) == "REVIEW"
    rep = evaluate(th, defective_scores=[0.2, 0.9, 0.95], good_scores=[0.01, 0.6])
    assert rep["miss_rate"] == pytest.approx(1 / 3)
    assert rep["n_defective"] == 3 and rep["n_good"] == 2


def test_save_load_roundtrip(tmp_path):
    g = ConformalGate([0.5, 0.6, 0.7], [0.1, 0.2], design="deeppcb")
    g.save(tmp_path / "g.json")
    g2 = ConformalGate.load(tmp_path / "g.json")
    assert g2.design == "deeppcb"
    assert list(g2.defective) == [0.5, 0.6, 0.7]


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 100)
    assert lo == 0.0 and hi == pytest.approx(0.0362, abs=1e-3)
    lo, hi = clopper_pearson(5, 100)
    assert lo == pytest.approx(0.0164, abs=1e-3) and hi == pytest.approx(0.1128, abs=1e-3)


def test_random_defect_miss_weights_boards_not_defects():
    """Board A: 1 of 1 found; board B: 1 of 4 found. Pooled miss = 3/5; the guaranteed quantity
    (one random defect per board) = (0 + 3/4) / 2. Boards with no defects are ignored."""
    from pcbinspect.gate import random_defect_miss
    assert random_defect_miss([[True], [True, False, False, False], []]) == pytest.approx(0.375)
    assert random_defect_miss([]) is None


def test_random_defect_miss_interval_covers_the_estimate_and_narrows_with_boards():
    from pcbinspect.gate import random_defect_miss, random_defect_miss_ci
    rng = np.random.default_rng(1)
    small = [list(rng.random(rng.integers(1, 6)) > 0.06) for _ in range(40)]
    big = [list(rng.random(rng.integers(1, 6)) > 0.06) for _ in range(800)]
    for boards in (small, big):
        lo, hi = random_defect_miss_ci(boards)
        assert lo <= random_defect_miss(boards) <= hi
    assert (lambda c: c[1] - c[0])(random_defect_miss_ci(big)) < (lambda c: c[1] - c[0])(random_defect_miss_ci(small))
    assert random_defect_miss_ci([[], []]) is None


def test_miss_rate_setting_is_validated():
    from pcbinspect.gate import miss_threshold
    for bad in (0.0, -0.1, 0.9, 1.0, 1.5):
        with pytest.raises(ValueError):
            miss_threshold(np.linspace(0, 1, 50), bad)
