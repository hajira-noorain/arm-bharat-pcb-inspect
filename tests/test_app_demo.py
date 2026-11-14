"""Demo mode through the web API: test boards with their known answer, uploads kept in memory, shift figures."""
import json
import threading
import time

import cv2
import numpy as np
import pytest

from pcbinspect.teach import recapture
from test_teach import photo

fastapi = pytest.importorskip("fastapi")

BASE_CFG = {"data_dir": "var", "designs_dir": "designs", "camera": "replay", "replay_glob": "demo/boards/*.png",
            "vision_backend": "executorch", "detector_pte": "none.pte", "detector_pt": "none.pt",
            "backbone_pte": "none.pte", "llm_pte": "none.pte", "llm_tokenizer": "none.model", "gpio": False}


def _client(cfg):
    from fastapi.testclient import TestClient

    from pcbinspect.app import create_app
    return TestClient(create_app(cfg))


@pytest.fixture(scope="module")
def taught_client(tmp_path_factory):
    """A station that has been taught one bare board design ('demo-board') and has two test boards: the same
    board with a trace cut through (one marked defect) and a good one."""
    root = tmp_path_factory.mktemp("demo")
    rng = np.random.default_rng(0)
    base = photo()
    cut = base.copy()
    cv2.line(cut, (700, 40), (700, 100), (45, 100, 60), 9)
    boards = root / "demo" / "boards"
    boards.mkdir(parents=True)
    for i, im in enumerate([base, recapture(base, rng), recapture(base, rng), recapture(base, rng),
                            recapture(cut, rng)]):
        cv2.imwrite(str(boards / f"b{i}.png"), im)
    (root / "designs").mkdir()
    c = _client({**BASE_CFG, "root": str(root), "teach_windows": 30, "teach_good": 20, "demo": True,
                 "upload_design": "demo-board"})
    for _ in range(3):
        assert c.post("/api/teach/capture", json={"name": "demo-board"}).json()["ok"]
    assert c.post("/api/teach/finish", json={"name": "demo-board", "mm_per_px": 0.025}).json()["running"]
    for _ in range(600):
        s = c.get("/api/teach/demo-board").json()
        if not s.get("running"):
            break
        time.sleep(0.5)
    assert s.get("done"), s
    (root / "demo" / "manifest.json").write_text(json.dumps([
        {"file": "b4.png", "design": "demo-board", "good": False,
         "boxes": [{"cls": "open", "bbox": [690, 30, 710, 110]}]},
        {"file": "b0.png", "design": "demo-board", "good": True, "boxes": []}]))
    c.root = root
    return c


@pytest.fixture(scope="module")
def plain_client(tmp_path_factory):
    root = tmp_path_factory.mktemp("plain")
    (root / "demo" / "boards").mkdir(parents=True)
    cv2.imwrite(str(root / "demo" / "boards" / "b0.png"), photo(size=600))
    (root / "designs").mkdir()
    return _client({**BASE_CFG, "root": str(root)})


def test_samples_listed_only_in_demo_mode(taught_client, plain_client):
    assert [s["file"] for s in taught_client.get("/api/samples").json()] == ["b4.png", "b0.png"]
    assert plain_client.get("/api/samples").status_code == 404
    assert plain_client.post("/api/inspect", json={"design": "x", "sample": "b0.png"}).status_code == 404
    state = taught_client.get("/api/state").json()
    assert state["demo"] is True and state["queue"] == 0 and "shared cloud CPU" in state["disclosure"]
    assert plain_client.get("/api/state").json()["disclosure"] is None


def test_inspecting_a_sample_returns_its_known_answer(taught_client):
    r = taught_client.post("/api/inspect", json={"design": "demo-board", "sample": "b4.png"}).json()
    assert r["verdict"] == "FAIL" and r["sample"] == "b4.png"
    assert r["known"] == {"marked": 1, "found": 1, "extra": 0}
    img = taught_client.get(f"/api/image/{r['id']}.jpg")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    thumb = cv2.imdecode(np.frombuffer(taught_client.get("/api/samples/b4.png.jpg").content, np.uint8),
                         cv2.IMREAD_COLOR)
    assert max(thumb.shape[:2]) == 160


def test_unknown_sample_and_path_tricks_are_404(taught_client):
    for f in ["nope.png", "../designs/x.yaml", "..%2Fdesigns%2Fx.yaml"]:
        assert taught_client.post("/api/inspect", json={"design": "demo-board", "sample": f}).status_code == 404
        assert taught_client.get(f"/api/samples/{f}.jpg").status_code == 404
    assert taught_client.get("/api/image/99999.jpg").status_code == 404


def test_stats_reflect_inspections(taught_client):
    taught_client.post("/api/inspect", json={"design": "demo-board", "sample": "b0.png"})
    st = taught_client.get("/api/stats").json()
    assert st["boards"] >= 1 and st["shift_hours"] == 8 and st["verdict_ms_median"] > 0


def test_a_new_inspection_does_not_wait_for_the_previous_advice(taught_client, monkeypatch):
    st = taught_client.app.state.station
    gate = threading.Event()
    real_template = st.advisor._template

    def slow_advise(record, history=None, **kw):     # Llama "still writing" until the gate opens
        gate.wait(5)
        return real_template(record, [])
    monkeypatch.setattr(st.advisor, "advise", slow_advise)
    t0 = time.perf_counter()
    taught_client.post("/api/inspect", json={"design": "demo-board", "sample": "b4.png"})
    taught_client.post("/api/inspect", json={"design": "demo-board", "sample": "b4.png"})
    took = time.perf_counter() - t0
    gate.set()
    assert took < 4.0


def _jpeg(im):
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def _upload(client, data, name="board.jpg", ctype="image/jpeg"):
    return client.post("/api/inspect/upload", files={"file": (name, data, ctype)})


def test_upload_is_inspected_and_never_logged(taught_client):
    st = taught_client.app.state.station
    n_before = st.store.conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    files_before = sorted((st.data / "images").rglob("*"))
    r = _upload(taught_client, _jpeg(photo(size=800)))
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["verdict"] == "REVIEW" and p["uncalibrated"] is True and p["upload"] is True and p["token"]
    assert all(d["size_mm"] is None and d["size_px"] > 0 for d in p["defects"])
    assert st.store.conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == n_before
    assert sorted((st.data / "images").rglob("*")) == files_before
    img = taught_client.get(f"/api/upload/{p['token']}.jpg")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"


def test_upload_rejects_non_images_and_oversize(taught_client):
    assert _upload(taught_client, b"just text", "a.txt", "text/plain").status_code == 400
    assert _upload(taught_client, b"\xff\xd8\xff" + b"0" * 10_000_000).status_code == 413
    gif = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    assert _upload(taught_client, gif, "a.gif", "image/gif").status_code == 400


def test_unknown_upload_token_is_404(taught_client, plain_client):
    assert taught_client.get("/api/upload/abc").status_code == 404
    assert taught_client.get("/api/upload/abc.jpg").status_code == 404
    assert _upload(plain_client, _jpeg(photo(size=400))).status_code == 404


def test_upload_advice_arrives_in_memory(taught_client):
    """Advice for an upload is written into the in-memory result, never into the log. (The test station has no
    detector, so the advice job is fed a real defect record for the upload's token.)"""
    st = taught_client.app.state.station
    n_advice = st.store.conn.execute("SELECT COUNT(*) FROM advice").fetchone()[0]
    p = _upload(taught_client, _jpeg(photo(size=600))).json()
    record = {"board_id": "upload", "design": "demo-board", "verdict": "REVIEW", "defects": [
        {"id": "D1", "type": "open", "severity": "critical", "confidence": 0.9, "size_mm": None,
         "location": "centre", "ipc": {}}]}
    st.uploads[p["token"]]["advice_pending"] = True
    st.jobs.put((p["token"], record, "demo-board"))
    for _ in range(60):
        p = taught_client.get(f"/api/upload/{p['token']}").json()
        if p["advice"]:
            break
        time.sleep(0.5)
    assert p["advice"] and p["advice"][0]["advice"]["steps"] and p["advice_pending"] is False
    assert st.store.conn.execute("SELECT COUNT(*) FROM advice").fetchone()[0] == n_advice


def test_reset_demo_empties_the_log(taught_client):
    st = taught_client.app.state.station
    taught_client.post("/api/inspect", json={"design": "demo-board", "sample": "b0.png"})
    assert taught_client.get("/api/stats").json()["boards"] >= 1
    st.reset_demo()
    assert taught_client.get("/api/stats").json()["boards"] == 0
    assert taught_client.get("/api/state").json()["recent"] == [] and not st.uploads and not st.cache


def test_log_lists_recent_with_counts(taught_client):
    taught_client.post("/api/inspect", json={"design": "demo-board", "sample": "b4.png"})
    rows = taught_client.get("/api/log?limit=200").json()
    assert rows[0]["n_defects"] >= 1 and rows[0]["verdict"] in ("FAIL", "REVIEW") and "final_verdict" in rows[0]
    assert rows[0]["advice_by"] in (None, "template", "llama-executorch")


def test_test_boards_and_thumbnails_are_read_once(taught_client, monkeypatch):
    """The page asks for ~40 thumbnails at once; each must not decode a full photo again."""
    import pcbinspect.app as app_mod
    st = taught_client.app.state.station
    taught_client.get("/api/samples"); taught_client.get("/api/samples/b4.png.jpg")
    calls = {"load": 0, "read": 0}
    real_load, real_read = app_mod.load_samples, app_mod.cv2.imread
    monkeypatch.setattr(app_mod, "load_samples", lambda *a, **k: (calls.__setitem__("load", calls["load"] + 1), real_load(*a, **k))[1])
    monkeypatch.setattr(app_mod.cv2, "imread", lambda *a, **k: (calls.__setitem__("read", calls["read"] + 1), real_read(*a, **k))[1])
    for _ in range(3):
        assert taught_client.get("/api/samples").status_code == 200
        assert taught_client.get("/api/samples/b4.png.jpg").status_code == 200
    assert calls == {"load": 0, "read": 0}


def test_demo_advice_queue_keeps_only_the_newest_boards_waiting_for_llama(taught_client, monkeypatch):
    """Online, a burst of inspections must not make the next visitor wait minutes for Llama: only the newest few
    boards wait; older queued ones get the knowledge base's standard advice at once, marked as such."""
    st = taught_client.app.state.station
    gate = threading.Event()
    real_template = st.advisor._template

    def slow_advise(record, history=None, **kw):
        gate.wait(10)
        return real_template(record, [])
    monkeypatch.setattr(st.advisor, "advise", slow_advise)
    monkeypatch.setitem(st.cfg, "demo_advice_queue", 2)
    ids = [taught_client.post("/api/inspect", json={"design": "demo-board", "sample": "b4.png"}).json()["id"]
           for _ in range(6)]
    busy = [i for i in ids if any("busy" in (a["rejected_reason"] or "")
                                  for a in taught_client.get(f"/api/inspection/{i}").json()["advice"])]
    assert len(busy) >= 3 and ids[-1] not in busy and ids[-2] not in busy
    gate.set()
    for _ in range(80):
        if all(taught_client.get(f"/api/inspection/{i}").json()["advice"] for i in ids):
            break
        time.sleep(0.25)
    assert all(taught_client.get(f"/api/inspection/{i}").json()["advice"] for i in ids)
