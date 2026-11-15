import time

import pytest

from pcbinspect import ask
from pcbinspect.kb import KnowledgeBase
from pcbinspect.record import Defect, Inspection
from pcbinspect.spc import drift_alerts, p_chart, pareto
from pcbinspect.store import Store


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def test_drift_alert_fires_on_injected_stencil_fault(kb):
    boards = [set() for _ in range(60)]
    for i in range(0, 60, 15):          # baseline: an occasional short
        boards[i] = {"short"}
    boards += [{"short"} if i % 3 == 0 else set() for i in range(20)]   # 7/20 recent shorts
    alerts = drift_alerts(boards, "deeppcb", kb)
    assert [a.cls for a in alerts] == ["short"]
    a = alerts[0]
    assert a.recent_count == 7 and a.ratio >= 3
    assert "etching" in a.message.lower()


def test_no_alert_without_drift_or_history(kb):
    steady = [{"spur"} if i % 5 == 0 else set() for i in range(100)]
    assert drift_alerts(steady, "d", kb) == []
    assert drift_alerts([{"short"}] * 10, "d", kb) == []    # not enough baseline


def test_p_chart_limits():
    flags = [0] * 95 + [1] * 5 + [1] * 10 + [0] * 10
    pc = p_chart(flags, subgroup=20)
    assert pc["pbar"] == pytest.approx(sum(pc["p"]) / len(pc["p"]))
    assert 5 in pc["out_of_control"]                    # the 10/20 subgroup


def test_pareto_cumulative():
    p = pareto(["short", "short", "spur", "open", "short"])
    assert p[0] == ("short", 3, 0.6) and p[-1][2] == pytest.approx(1.0)


def test_keyword_routing():
    assert ask.keyword_route("how many shorts in the last 3 hours")[0] == "defect_counts"
    assert ask.keyword_route("how many shorts in the last 3 hours")[1] == {"hours": 3}
    assert ask.keyword_route("history of board B-0042")[0] == "board_history"
    assert ask.keyword_route("what was the yield this shift")[0] == "yield"
    assert ask.keyword_route("any drift alerts?")[0] == "open_alerts"


def test_answer_runs_whitelisted_query_only(tmp_path):
    s = Store(tmp_path)
    d = Defect("D1", "short", "detector", (1, 1, 2, 2), 0.9, severity="critical")
    s.log_inspection(Inspection("B-1", "deeppcb", "", [d], 0.9, "FAIL", 0.05, None), None)
    s.log_inspection(Inspection("B-2", "deeppcb", "", [], 0.0, "PASS", 0.05, None), None)
    out = ask.answer("what was the yield in the last 2 hours", s.conn, backend=None)
    assert out["query"] == "yield" and out["rows"][0]["boards"] == 2
    assert out["rows"][0]["failed"] == 1
    # SQL injection attempt is just a parameter value
    out = ask.answer("history of board x'; DROP TABLE inspections;--", s.conn)
    assert s.conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 2


class LyingBackend:
    name, model = "llama-executorch", "fake"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, n):
        from pcbinspect.advisor import GenResult
        self.calls += 1
        if self.calls == 1:
            return GenResult('yield", "params": {"hours": 2}}', 1)
        return GenResult("We inspected 57 boards today.", 1)   # 57 is not in the rows


def test_llama_summary_with_invented_numbers_is_rejected(tmp_path):
    s = Store(tmp_path)
    s.log_inspection(Inspection("B-1", "deeppcb", "", [], 0.0, "PASS", 0.05, None), None)
    out = ask.answer("yield?", s.conn, backend=LyingBackend())
    assert out["route"] == "llama" and out["summary_by"] == "template"
    assert out["summary_rejected"] and "57" not in out["answer"]       # a typed number is never shown


class GluedNumberBackend(LyingBackend):
    """Seen on the station: the constrained decoder let '2' start the allowed '24', then ended the number with
    'boards'. '2boards' has no word boundary, so a word-bounded number check never saw the 2."""
    def generate(self, prompt, n):
        from pcbinspect.advisor import GenResult
        self.calls += 1
        if self.calls == 1:
            return GenResult('yield", "params": {"hours": 2}}', 1)
        return GenResult("2boards failed today.", 1)


def test_a_number_glued_to_a_word_is_still_checked(tmp_path):
    s = Store(tmp_path)
    s.log_inspection(Inspection("B-1", "deeppcb", "", [], 0.0, "PASS", 0.05, None), None)
    out = ask.answer("how many boards failed today?", s.conn, backend=GluedNumberBackend())
    assert out["summary_by"] == "template" and out["summary_rejected"]
    assert "2boards" not in out["answer"]
