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
    c = _client({**BASE_CFG, "root": str(root), "teach_windows": 30, "teach_good": 20, "demo": True})
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
