"""Learning during use: causes the operator confirmed move the station's view of this line away from the rule table."""
import pytest

from pcbinspect.causes import CauseModel
from pcbinspect.kb import default_kb
from pcbinspect.learned import LineCauses
from pcbinspect.store import Store

SAME = {"repeat_same_spot": {"boards": 5, "of": 10}, "near_edge": False, "shape": "long_thin", "count_on_board": 1,
        "trend": "steady"}


@pytest.fixture(scope="module")
def cm():
    return CauseModel(default_kb())


def test_with_nothing_confirmed_the_line_is_the_rule_table(cm):
    line = LineCauses(cm, [])
    f = {"same_spot": True}
    assert line.posterior("open", f) == pytest.approx(cm.posterior("open", f))
    p = {"OPEN-C1": 0.1, "OPEN-C2": 0.6, "OPEN-C3": 0.2, "OPEN-C4": 0.1}
    assert line.adjust("open", f, p) == pytest.approx(p)


def test_confirmations_move_the_line_and_llamas_probabilities_with_it(cm):
    rows = [("open", "OPEN-C1", SAME)] * 30           # on this line, same-spot opens kept being resist adhesion
    line = LineCauses(cm, rows)
    f = {"same_spot": True, "long_thin": True, "round": False}
    assert line.posterior("open", f)["OPEN-C1"] > cm.posterior("open", f)["OPEN-C1"] + 0.2
    p = {"OPEN-C1": 0.2, "OPEN-C2": 0.6, "OPEN-C3": 0.1, "OPEN-C4": 0.1}
    q = line.adjust("open", f, p)
    assert q["OPEN-C1"] > p["OPEN-C1"] and sum(q.values()) == pytest.approx(1.0)
    assert line.confirmed("open") == 30 and line.confirmed("short") == 0


def test_other_defect_types_are_untouched(cm):
    line = LineCauses(cm, [("open", "OPEN-C1", SAME)] * 30)
    f = {"many": True}
    assert line.posterior("short", f) == pytest.approx(cm.posterior("short", f))


def test_the_store_keeps_confirmations_and_refuses_a_cause_of_another_type(tmp_path):
    from test_evidence import _log
    s = Store(tmp_path)
    iid = _log(s, [("open", (10, 10, 20, 20))])
    kb = default_kb()
    s.confirm_cause(iid, "D1", "OPEN-C2", SAME, "asha", kb)
    with pytest.raises(ValueError):
        s.confirm_cause(iid, "D1", "SHORT-C1", SAME, "asha", kb)
    rows = s.confirmed_causes()
    assert rows == [("open", "OPEN-C2", SAME)]
    s.confirm_cause(iid, "D1", "OPEN-C1", SAME, "ravi", kb)       # a later answer replaces the earlier one
    assert s.confirmed_causes() == [("open", "OPEN-C1", SAME)]


def test_the_confirm_endpoint_records_and_the_station_learns(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pcbinspect.app import create_app
    from test_app_demo import BASE_CFG
    from test_evidence import _log
    (tmp_path / "designs").mkdir()
    c = TestClient(create_app({**BASE_CFG, "root": str(tmp_path)}))
    s = Store(tmp_path / "var")
    iid = _log(s, [("open", (10, 10, 20, 20))])
    r = c.post("/api/cause/confirm", json={"inspection_id": iid, "defect_key": "D1", "cause_id": "OPEN-C2",
                                           "operator": "asha"})
    assert r.status_code == 200 and r.json()["learned_from"] == 1
    assert c.post("/api/cause/confirm", json={"inspection_id": iid, "defect_key": "D1",
                                              "cause_id": "SHORT-C1"}).status_code == 400
