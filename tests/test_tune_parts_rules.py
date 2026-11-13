"""The rule applied to raw parts measurements (training/tune_parts_rules.py)."""
from training.tune_parts_rules import CURRENT, decide, score

R = {"present_conf": 0.25, "cap": 0.7, "margin": 0.15, "changed_extra": 0.2, "extra_conf": 0.6, "extra_like": 0.5}


def test_missing_needs_no_detection_and_an_unlike_place():
    rec = {"parts": [[0.0, 0.3, 0.9, True],      # not found, looks unlike: missing (on the labelled area)
                     [0.0, 0.9, 0.95, False],    # not found, but looks the same: the detector blinked
                     [0.8, 0.9, 0.95, False],    # found and alike: fine
                     [0.0, None, None, False]],  # not found, cannot be compared: the detector decides
           "others": []}
    d = decide(rec, R)
    assert d["missing"] == 2 and d["missing_on"] and d["changed"] == 0 and d["unexpected"] == 0


def test_changed_needs_a_measured_range_and_a_clearly_lower_likeness():
    rec = {"parts": [[0.9, 0.4, 0.9, True],      # threshold 0.7 - 0.2 = 0.5: 0.4 is clearly unlike
                     [0.9, 0.6, 0.9, False],     # unlike enough for missing, not for changed
                     [0.9, 0.1, None, False]],   # no measured range: never called changed
           "others": []}
    d = decide(rec, R)
    assert d["changed"] == 1 and d["changed_on"] and d["missing"] == 0


def test_unexpected_needs_confidence_an_unlike_place_and_no_ignore_zone():
    rec = {"parts": [], "others": [[0.9, 0.2, False, True], [0.9, 0.8, False, False], [0.5, 0.2, False, False],
                                   [0.9, 0.2, True, False]]}
    d = decide(rec, R)
    assert d["unexpected"] == 1 and d["unexpected_on"]


def test_photos_that_do_not_line_up_are_counted_but_not_judged():
    photos = [{"split": "test", "types": ["good"], "cc": 0.9, "parts": [[0.0, 0.2, 0.9, False]], "others": []},
              {"split": "test", "types": ["good"], "cc": 0.3, "parts": [[0.0, 0.2, 0.9, False]], "others": []},
              {"split": "test", "types": ["missing", "scratch"], "cc": 0.9, "parts": [[0.0, 0.2, 0.9, True]], "others": []}]
    s = score(photos, CURRENT, 0.6)
    assert s["good"] == {"photos": 2, "lined_up": 1, "anything": 1, "missing_calls": 1, "changed_calls": 0,
                         "unexpected_calls": 0, "on_area": 0, "missing_on_area": 0, "unexpected_on_area": 0,
                         "changed_on_area": 0}
    assert s["missing"]["missing_on_area"] == 1 and s["scratch"]["photos"] == 1
