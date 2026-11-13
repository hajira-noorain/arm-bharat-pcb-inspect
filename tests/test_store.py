import csv
import sqlite3

import pytest

from pcbinspect.record import Defect, Inspection
from pcbinspect.store import Store


def make_insp(verdict="FAIL", n=2):
    defects = [Defect(id=f"D{i+1}", cls="short", source="detector", bbox=(1, 2, 3, 4),
                      confidence=0.9, severity="critical", ipc={"class2": "reject"})
               for i in range(n)]
    return Inspection(board_id="B1", design="deeppcb", image_path="x.jpg", defects=defects,
                      board_score=0.9, verdict=verdict, alpha=0.05, mm_per_px=0.02,
                      latency_ms={"vision": 120.0})


def test_inspection_roundtrip(tmp_path):
    s = Store(tmp_path)
    sha = s.put_image(b"fake-jpeg-bytes")
    assert s.image_path(sha).read_bytes() == b"fake-jpeg-bytes"
    iid = s.log_inspection(make_insp(), sha)
    got = s.inspection(iid)
    assert got["verdict"] == "FAIL" and len(got["defects"]) == 2
    assert s.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_inspection_is_atomic(tmp_path):
    """A defect row that violates the schema must roll back the whole inspection."""
    s = Store(tmp_path)
    insp = make_insp()
    insp.defects[1].bbox = (1, 2, 3)  # wrong arity -> insert fails
    with pytest.raises(sqlite3.ProgrammingError):
        s.log_inspection(insp, None)
    assert s.conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 0
    assert s.conn.execute("SELECT COUNT(*) FROM defects").fetchone()[0] == 0


def test_feedback_and_undo(tmp_path):
    s = Store(tmp_path)
    iid = s.log_inspection(make_insp(n=1), None)
    did = s.inspection(iid)["defects"][0]["id"]
    ev = s.log_feedback("false_alarm", iid, did)
    assert s.inspection(iid)["defects"][0]["operator_label"] == "false_alarm"
    s.undo_feedback(ev)
    assert s.inspection(iid)["defects"][0]["operator_label"] is None
    with pytest.raises(ValueError):
        s.undo_feedback(ev)


def test_verdict_check_constraint(tmp_path):
    s = Store(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        s.log_inspection(make_insp(verdict="MAYBE"), None)


def test_model_version_single_active(tmp_path):
    s = Store(tmp_path)
    s.record_model_version("anomaly_bank", "v1", {"auroc": 0.9}, active=True)
    s.record_model_version("anomaly_bank", "v2", {"auroc": 0.92}, active=True)
    active = [v["version"] for v in s.model_versions("anomaly_bank") if v["active"]]
    assert active == ["v2"]


def test_csv_export(tmp_path):
    s = Store(tmp_path)
    s.log_inspection(make_insp(n=2), None)
    s.log_inspection(make_insp(verdict="PASS", n=0), None)
    n = s.export_csv(tmp_path / "out.csv")
    rows = list(csv.DictReader(open(tmp_path / "out.csv", encoding="utf-8")))
    assert n == 3 and len(rows) == 3
    assert {r["verdict"] for r in rows} == {"FAIL", "PASS"}


def test_a_persons_decision_records_who_and_why_and_reaches_the_export(tmp_path):
    s = Store(tmp_path)
    iid = s.log_inspection(make_insp(verdict="REVIEW", n=1), None)
    s.set_final_verdict(iid, "PASS", operator="  Asha  ", reason="dust on the lens, board is clean")
    got = s.inspection(iid)
    assert (got["final_verdict"], got["decided_by"], got["decision_reason"]) == \
        ("PASS", "Asha", "dust on the lens, board is clean") and got["decided_ts"] > 0
    with pytest.raises(ValueError):
        s.set_final_verdict(iid, "REVIEW")
    with pytest.raises(KeyError):
        s.set_final_verdict(iid + 99, "FAIL", operator="Asha")
    out = tmp_path / "log.csv"
    s.export_csv(out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert rows[0]["decided_by"] == "Asha" and rows[0]["decision_reason"].startswith("dust")


def test_a_database_made_before_decisions_had_names_is_upgraded(tmp_path):
    s = Store(tmp_path)
    s.conn.execute("ALTER TABLE inspections DROP COLUMN decided_by")
    s.conn.close()
    s2 = Store(tmp_path)                                    # _migrate() adds the column back
    iid = s2.log_inspection(make_insp(verdict="REVIEW", n=1), None)
    s2.set_final_verdict(iid, "FAIL", operator="Ravi")
    assert s2.inspection(iid)["decided_by"] == "Ravi"
