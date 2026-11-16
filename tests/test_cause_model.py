"""The cause model: evidence likelihoods per cause, read from kb/evidence.yaml, combined by Bayes' rule."""
import pytest

from pcbinspect.causes import FEATURES, CauseModel, features_from
from pcbinspect.kb import default_kb


@pytest.fixture(scope="module")
def cm():
    return CauseModel(default_kb())


def test_every_cause_has_evidence_a_source_provenance_and_a_check(cm):
    kb = default_kb()
    for cls, e in kb.defects.items():
        for c in e["causes"]:
            ev = cm.evidence[c["id"]]
            assert ev["provenance"] in ("sourced", "inferred") and ev["source"].strip() and ev["confirm"].strip()
            assert set(ev["p"]) == set(FEATURES)
            assert all(0.0 < v < 1.0 for v in ev["p"].values())
        assert abs(sum(cm.evidence[c["id"]]["prior"] for c in e["causes"]) - 1.0) < 1e-6, cls


def test_posterior_sums_to_one_and_unknown_evidence_gives_the_prior(cm):
    post = cm.posterior("open", {})
    assert abs(sum(post.values()) - 1.0) < 1e-9
    assert post == pytest.approx({k: cm.evidence[k]["prior"] for k in post})


def test_same_spot_on_board_after_board_points_to_the_phototool(cm):
    post = cm.posterior("open", {"same_spot": True, "long_thin": True, "many": False})
    assert max(post, key=post.get) == "OPEN-C2"           # dust or hair on the phototool
    post = cm.posterior("open", {"same_spot": False, "long_thin": True, "near_edge": True, "many": False})
    assert max(post, key=post.get) == "OPEN-C4"           # handling scratch


def test_features_from_station_evidence():
    ev = {"repeat_same_spot": {"boards": 5, "of": 12}, "near_edge": False, "shape": "long_thin",
          "count_on_board": 1, "trend": "unknown", "co_occurring": []}
    f = features_from(ev)
    assert f == {"same_spot": True, "near_edge": False, "long_thin": True, "round": False, "many": False}
    assert features_from({"repeat_same_spot": "unknown", "near_edge": "unknown", "shape": "round",
                          "count_on_board": 4, "trend": "rising"}) == {"round": True, "long_thin": False,
                                                                       "many": True, "rising": True}
    assert features_from({"repeat_same_spot": {"boards": 1, "of": 12}})["same_spot"] is False


def test_new_clues_become_features_and_a_false_roll_repeat_says_nothing():
    from pcbinspect.causes import features_from
    ev = {"repeat_in_line": {"boards": 4, "of": 10}, "central": True, "roll_pitch": False}
    f = features_from(ev)
    assert f["lane"] is True and f["central"] is True and "roll_pitch" not in f
    assert features_from({"roll_pitch": True})["roll_pitch"] is True


def test_every_cause_states_every_clue():
    from pcbinspect.causes import FEATURES, CauseModel
    from pcbinspect.kb import default_kb
    cm = CauseModel(default_kb())
    for c, e in cm.evidence.items():
        assert set(e["p"]) == set(FEATURES), c


def test_a_lane_points_to_the_nozzle_or_conveyor_and_the_centre_to_under_etch():
    from pcbinspect.causes import CauseModel
    from pcbinspect.kb import default_kb
    cm = CauseModel(default_kb())
    top = lambda cls, f: max(cm.posterior(cls, f).items(), key=lambda x: x[1])[0]
    assert top("spur", {"lane": True, "same_spot": False}) == "SPUR-C2"
    assert top("scratch", {"lane": True, "long_thin": True, "round": False}) == "SCR-C2"
    assert top("short", {"central": True, "many": True, "same_spot": False}) == "SHORT-C1"
