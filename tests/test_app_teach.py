"""The teach-a-bare-board flow through the web API: capture good boards, calibrate, then inspect."""
import time

import cv2
import numpy as np
import pytest

from pcbinspect.teach import recapture
from test_teach import photo

fastapi = pytest.importorskip("fastapi")


def test_teach_then_inspect_through_the_api(tmp_path):
    from fastapi.testclient import TestClient

    from pcbinspect.app import create_app
    rng = np.random.default_rng(0)
    base = photo()
    cut = base.copy()
    cv2.line(cut, (700, 40), (700, 100), (45, 100, 60), 9)            # a trace cut right through
    boards = tmp_path / "demo" / "boards"
    boards.mkdir(parents=True)
    for i, im in enumerate([base, recapture(base, rng), recapture(base, rng),
                            recapture(base, rng), recapture(cut, rng)]):
        cv2.imwrite(str(boards / f"b{i}.png"), im)
    (tmp_path / "designs").mkdir()
    cfg = {"root": str(tmp_path), "data_dir": "var", "designs_dir": "designs", "camera": "replay",
           "replay_glob": "demo/boards/*.png", "vision_backend": "executorch", "detector_pte": "none.pte",
           "detector_pt": "none.pt", "backbone_pte": "none.pte", "llm_pte": "none.pte",
           "llm_tokenizer": "none.model", "gpio": False, "teach_windows": 30, "teach_good": 20}
    c = TestClient(create_app(cfg))

    assert c.post("/api/teach/finish", json={"name": "demo-board"}).status_code == 409     # nothing captured
    for k in (1, 2, 3):
        r = c.post("/api/teach/capture", json={"name": "demo-board"}).json()
        assert r["ok"] and r["captures"] == k and r["golden"] == (k == 1)
    assert c.post("/api/teach/finish", json={"name": "demo-board"}).status_code == 409     # no scale yet
    assert c.post("/api/teach/finish", json={"name": "demo-board", "mm_per_px": 0.025}).json()["running"]
    for _ in range(600):
        s = c.get("/api/teach/demo-board").json()
        if not s.get("running"):
            break
        time.sleep(0.5)
    assert s.get("done"), s
    assert s["can_fail"] and s["can_pass"] and "open" in s["limits"]
    assert (tmp_path / "models" / "golden" / "demo-board.png").exists()
    assert c.post("/api/teach/capture", json={"name": "demo-board"}).status_code == 409    # name is taken now

    good = c.post("/api/inspect", json={"design": "demo-board"}).json()                    # b3: good board
    assert good["verdict"] == "PASS" and not good["defects"]
    bad = c.post("/api/inspect", json={"design": "demo-board"}).json()                     # b4: cut trace
    assert bad["verdict"] == "FAIL"
    opens = [d for d in bad["defects"] if d["cls"] == "open"]
    assert opens and opens[0]["x1"] <= 705 and opens[0]["x2"] >= 695
    lim = c.get("/api/eyetest/demo-board").json()
    assert lim and "planted" in str(lim)
