"""Operator station: FastAPI server + single-page dashboard. Fully offline.

Run:  python -m pcbinspect.app --config config/app.yaml
Open: http://<pi-ip>:8000 on the Pi's screen or a phone on the Pi's hotspot.
"""
from __future__ import annotations

import argparse
import glob
import logging
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import ask as ask_mod
from .advisor import Advisor
from .anomaly import enrol
from .camera import ReplayCamera, open_camera
from .gpio import StackLight
from .kb import default_kb
from .learning import FeedbackBuffer, crop_embedding
from .overlay import render
from .pipeline import Design, Inspector, load_designs, save_design
from .spc import drift_alerts, p_chart, pareto
from .store import Store

log = logging.getLogger("pcbinspect")
STATIC = Path(__file__).resolve().parent / "static"


class Station:
    """Everything the web handlers need, created once at start-up."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = Path(cfg.get("root", ".")).resolve()
        self.data = self.root / cfg.get("data_dir", "var")
        self.store = Store(self.data)
        self.kb = default_kb()
        self.designs = load_designs(self.root / cfg["designs_dir"])
        self.feature_fn, self.detector = self._load_vision()
        self.light = StackLight(cfg.get("gpio", True))
        self.inspector = Inspector(self.root, self.designs, self.detector, self.feature_fn,
                                   self.store, self.light)
        for name, d in self.designs.items():
            self.store.upsert_design(name, d.kind, d.rules.min_trace_width_mm, d.rules.min_spacing_mm)
        self.llm = self._load_llm()
        self.advisor = Advisor(self.llm, self.kb)
        self.camera = self._open_camera()
        self.cache: OrderedDict[int, tuple] = OrderedDict()   # iid -> (insp, analysis, bgr)
        self.buffers: dict[str, FeedbackBuffer] = {}
        self.jobs: queue.Queue = queue.Queue()
        self.lock = threading.Lock()                           # one inspection at a time
        self.llm_lock = threading.Lock()                       # one Llama generation at a time
        threading.Thread(target=self._advice_worker, daemon=True).start()

    # ---------------- loading ----------------
    def _load_vision(self):
        cfg, root = self.cfg, self.root
        feature_fn, det = None, None
        from .detect_bare import BareDetector
        if cfg.get("vision_backend") == "ultralytics":
            from .anomaly import torch_feature_fn
            from .detect_bare import UltralyticsTiles
            feature_fn = torch_feature_fn("cuda" if _cuda() else "cpu")
            if (root / cfg["detector_pt"]).exists():
                det = BareDetector(UltralyticsTiles(root / cfg["detector_pt"]))
        else:
            from .anomaly import executorch_feature_fn
            from .detect_bare import ExecuTorchTiles
            if (root / cfg["backbone_pte"]).exists():
                feature_fn = executorch_feature_fn(root / cfg["backbone_pte"])
            if (root / cfg["detector_pte"]).exists():
                det = BareDetector(ExecuTorchTiles(root / cfg["detector_pte"]))
        log.info("vision: detector=%s backbone=%s", det is not None, feature_fn is not None)
        return feature_fn, det

    def _load_llm(self):
        p = self.root / self.cfg.get("llm_pte", "")
        if not p.is_file():
            log.warning("Llama .pte not found at %s: advice will use templates", p)
            return None
        from .llm import ExecuTorchLlama
        try:
            return ExecuTorchLlama(p, self.root / self.cfg["llm_tokenizer"],
                                   threads=self.cfg.get("llm_threads", 4))
        except Exception as exc:
            log.exception("Llama failed to load (%s): advice will use templates", exc)
            return None

    def _open_camera(self):
        kind = self.cfg.get("camera", "auto")
        if kind == "replay":
            paths = sorted(glob.glob(str(self.root / self.cfg["replay_glob"])))
            return ReplayCamera(paths) if paths else None
        try:
            return open_camera(kind)
        except Exception as exc:
            log.warning("camera %s unavailable (%s); trying replay", kind, exc)
            paths = sorted(glob.glob(str(self.root / self.cfg["replay_glob"])))
            return ReplayCamera(paths) if paths else None

    def buffer(self, design: str) -> FeedbackBuffer:
        if design not in self.buffers:
            (self.data / "learning").mkdir(parents=True, exist_ok=True)
            self.buffers[design] = FeedbackBuffer(self.data / "learning" / f"{design}_feedback.npz")
        return self.buffers[design]

    # ---------------- advice worker ----------------
    def _history(self, design: str, window: int = 20) -> dict[str, int]:
        rows = self.store.conn.execute(
            "SELECT id FROM inspections WHERE design=? ORDER BY id DESC LIMIT ?", (design, window)).fetchall()
        ids = [r["id"] for r in rows]
        if not ids:
            return {}
        q = ("SELECT cls, COUNT(DISTINCT inspection_id) AS n FROM defects WHERE inspection_id IN (%s) "
             "GROUP BY cls" % ",".join("?" * len(ids)))
        return {r["cls"]: r["n"] for r in self.store.conn.execute(q, ids)}

    def _advice_worker(self):
        while True:
            iid, record, design = self.jobs.get()
            try:
                with self.llm_lock:
                    t0 = time.perf_counter()
                    adv = self.advisor.advise(record, self._history(design))
                    ms = (time.perf_counter() - t0) * 1e3
                g = adv.gen
                self.store.log_advice(iid, adv.backend, adv.model, adv.to_json(), adv.grounded,
                                      "; ".join(adv.rejected_reasons) or None, ms,
                                      g.prefill_tps if g else None, g.decode_tps if g else None)
                self._check_drift(design)
            except Exception:
                log.exception("advice failed for inspection %s", iid)

    def _check_drift(self, design: str):
        rows = self.store.conn.execute(
            "SELECT i.id, GROUP_CONCAT(d.cls) AS cls FROM inspections i LEFT JOIN defects d "
            "ON d.inspection_id=i.id WHERE i.design=? GROUP BY i.id ORDER BY i.id", (design,)).fetchall()
        per_board = [set((r["cls"] or "").split(",")) - {""} for r in rows]
        for a in drift_alerts(per_board, design, self.kb):
            recent = self.store.conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE design=? AND rule=? AND ts>?",
                (design, f"drift:{a.cls}", time.time() - 600)).fetchone()[0]
            if not recent:
                self.store.log_alert(design, f"drift:{a.cls}", a.message, a.__dict__)

    # ---------------- inspection ----------------
    def inspect(self, design: str, alpha: float | None, board_id: str | None) -> dict:
        if self.camera is None:
            raise HTTPException(503, "no camera or replay images available")
        if design not in self.designs:
            raise HTTPException(404, f"unknown design {design}")
        with self.lock:
            t0 = time.perf_counter()
            bgr = self.camera.capture()
            cap_ms = (time.perf_counter() - t0) * 1e3
            insp, analysis, iid = self.inspector.inspect(bgr, design, board_id, alpha)
        insp.latency_ms = {"capture": round(cap_ms, 2), **insp.latency_ms}
        self.store.conn.execute("UPDATE inspections SET latency_json=? WHERE id=?",
                                (__import__("json").dumps(insp.latency_ms), iid))
        self.cache[iid] = (insp, analysis, bgr)
        while len(self.cache) > 30:
            self.cache.popitem(last=False)
        cv2.imwrite(str(self._overlay_path(iid)), render(bgr, insp, analysis))
        if insp.defects:
            self.jobs.put((iid, insp.to_llm_record(), design))
        if isinstance(self.camera, ReplayCamera):
            self.camera.next()
        return self.inspection_json(iid)

    def _overlay_path(self, iid: int) -> Path:
        p = self.data / "overlays"
        p.mkdir(exist_ok=True)
        return p / f"{iid}.jpg"

    def inspection_json(self, iid: int) -> dict:
        r = self.store.inspection(iid)
        if r is None:
            raise HTTPException(404)
        import json
        for d in r["defects"]:
            d["ipc"] = json.loads(d["ipc_json"] or "{}")
            d["name"] = self.kb.display_name(d["cls"])
        for a in r["advice"]:
            a["advice"] = json.loads(a["advice_json"])
        r["latency"] = json.loads(r["latency_json"] or "{}")
        th = self.inspector.thresholds(r["design"], r["alpha"])
        r["gate"] = None if th is None else {
            "alpha": th.alpha, "beta": th.beta, "lam": th.lam, "tau": th.tau,
            "n_defective": th.n_defective, "n_good": th.n_good}
        r["advice_pending"] = bool(r["defects"]) and not r["advice"]
        return r

    # ---------------- feedback / learning ----------------
    def feedback(self, iid: int, defect_row_id: int, action: str) -> dict:
        if iid not in self.cache:
            raise HTTPException(409, "inspection no longer in memory; feedback must follow soon after")
        insp, analysis, bgr = self.cache[iid]
        row = self.store.conn.execute("SELECT * FROM defects WHERE id=? AND inspection_id=?",
                                      (defect_row_id, iid)).fetchone()
        if row is None:
            raise HTTPException(404, "defect not found")
        design = insp.design
        result = {"action": action}
        if row["source"] == "anomaly" and action == "false_alarm":
            am = self.inspector.anomaly.get(design)
            region = _match_region(analysis, (row["x1"], row["y1"], row["x2"], row["y2"]))
            if am is None or region is None:
                raise HTTPException(409, "anomaly region not available")
            confirmed = action == "false_alarm" and getattr(self, "_pending_confirm", None) == defect_row_id
            res = am.false_alarm(analysis.anomaly, region, confirmed_twice=confirmed)
            if not res["accepted"]:
                self._pending_confirm = defect_row_id
                return {**result, **res}
            self._pending_confirm = None
            am.save(self.root / self.designs[design].anomaly_dir)
            ev = self.store.log_feedback(action, iid, defect_row_id, {"anomaly": res})
            self.store.record_model_version(f"anomaly_bank:{design}", str(am.version),
                                            {"bank_size": int(len(am.bank))}, True,
                                            f"false alarm from inspection {iid}")
            return {**result, "event_id": ev, "learned": "added to the good-board memory",
                    "bank_size": int(len(am.bank))}
        # detector findings (and anomaly confirmations): store a labelled crop embedding
        ev = self.store.log_feedback(action, iid, defect_row_id)
        if row["source"] == "detector" and self.feature_fn is not None:
            label = 0 if action == "false_alarm" else 1
            emb = crop_embedding(self.feature_fn, bgr, (row["x1"], row["y1"], row["x2"], row["y2"]))
            self.buffer(design).add(emb, label, ev)
            npos, nneg = self.buffer(design).counts()
            result.update({"learned": "stored for the next model update", "examples": {
                "confirmed": npos, "false_alarms": nneg}})
        return {**result, "event_id": ev}

    def undo(self, event_id: int) -> dict:
        ev = self.store.undo_feedback(event_id)
        import json
        payload = json.loads(ev["payload_json"] or "{}")
        insp_row = self.store.inspection(ev["inspection_id"]) if ev["inspection_id"] else None
        design = insp_row["design"] if insp_row else None
        if "anomaly" in payload and design in self.inspector.anomaly:
            am = self.inspector.anomaly[design]
            if am.history and am.history[-1]["version"] == payload["anomaly"]["version"]:
                am.undo_last()
                am.save(self.root / self.designs[design].anomaly_dir)
            else:
                raise HTTPException(409, "only the most recent memory change can be undone")
        elif design:
            self.buffer(design).remove_event(event_id)
        return {"undone": event_id}

    def enrol(self, name: str, kind: str, n_images: int) -> dict:
        if self.feature_fn is None:
            raise HTTPException(503, "vision backbone not loaded")
        if not 3 <= n_images <= 30:
            raise HTTPException(400, "use 3-30 good boards")
        imgs = []
        for _ in range(n_images):
            imgs.append(self.camera.capture())
            if isinstance(self.camera, ReplayCamera):
                self.camera.next()
        t0 = time.perf_counter()
        m = enrol(self.feature_fn, imgs, name, coreset_ratio=0.25, max_bank=16384)
        secs = time.perf_counter() - t0
        rel = f"models/anomaly/{name}"
        m.save(self.root / rel)
        d = Design(name, kind, kind == "bare" and self.detector is not None, rel, None, 20.0,
                   self.inspector.severity.default_rules, None,
                   description=f"enrolled on the device from {n_images} good boards")
        save_design(self.root / self.cfg["designs_dir"], d)
        self.designs[name] = d
        self.inspector.designs[name] = d
        self.inspector.reload_design(name)
        self.store.upsert_design(name, kind)
        self.store.record_model_version(f"anomaly_bank:{name}", "1", {"bank_size": int(len(m.bank)),
                                        "images": n_images, "seconds": secs}, True, "enrolled")
        return {"design": name, "images": n_images, "seconds": round(secs, 2),
                "bank_size": int(len(m.bank)),
                "note": "No calibrated gate yet: boards with evidence go to REVIEW until it is calibrated."}


def _match_region(analysis, bbox):
    if analysis is None or analysis.anomaly is None:
        return None
    for r in analysis.anomaly.regions:
        if tuple(r.bbox) == tuple(bbox):
            return r
    return None


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ======================================================================= web
class InspectReq(BaseModel):
    design: str
    alpha: float | None = None
    board_id: str | None = None


class FeedbackReq(BaseModel):
    inspection_id: int
    defect_id: int
    action: str


class ReviewReq(BaseModel):
    inspection_id: int
    verdict: str


class EnrolReq(BaseModel):
    name: str
    kind: str = "assembled"
    images: int = 8


class AskReq(BaseModel):
    question: str


def create_app(cfg: dict) -> FastAPI:
    st = Station(cfg)
    app = FastAPI(title="PCB inspection station")
    app.state.station = st

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    def state():
        designs = [{"name": n, "kind": d.kind, "alpha": d.alpha, "description": d.description,
                    "calibrated": n in st.inspector.gates,
                    "detector": d.use_detector and st.detector is not None,
                    "anomaly": n in st.inspector.anomaly} for n, d in st.designs.items()]
        return {"designs": designs,
                "camera": None if st.camera is None else st.camera.name,
                "llm": None if st.llm is None else st.llm.model,
                "vision": {"detector": None if st.detector is None else st.detector.tiles.name,
                           "backbone": st.feature_fn is not None},
                "light": st.light.state,
                "recent": st.store.recent(12),
                "alerts": st.store.open_alerts()}

    @app.get("/api/preview.jpg")
    def preview():
        if st.camera is None:
            raise HTTPException(503)
        f = st.camera.preview()
        if f is None:
            raise HTTPException(503)
        h, w = f.shape[:2]
        s = 960 / max(h, w)
        if s < 1:
            f = cv2.resize(f, (int(w * s), int(h * s)))
        ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(jpg.tobytes(), media_type="image/jpeg")

    @app.post("/api/inspect")
    def inspect(req: InspectReq):
        return st.inspect(req.design, req.alpha, req.board_id)

    @app.get("/api/inspection/{iid}")
    def inspection(iid: int):
        return st.inspection_json(iid)

    @app.get("/api/overlay/{iid}.jpg")
    def overlay(iid: int):
        p = st._overlay_path(iid)
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="image/jpeg")

    @app.post("/api/feedback")
    def feedback(req: FeedbackReq):
        if req.action not in ("confirmed", "false_alarm") and not req.action.startswith("relabel:"):
            raise HTTPException(400, "action must be confirmed, false_alarm or relabel:<type>")
        return st.feedback(req.inspection_id, req.defect_id, req.action)

    @app.post("/api/undo/{event_id}")
    def undo(event_id: int):
        try:
            return st.undo(event_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/review")
    def review(req: ReviewReq):
        st.store.set_final_verdict(req.inspection_id, req.verdict)
        st.light.show(req.verdict)
        return {"ok": True}

    @app.post("/api/advice/{advice_id}/rate/{rating}")
    def rate(advice_id: int, rating: int):
        st.store.rate_advice(advice_id, rating)
        return {"ok": True}

    @app.post("/api/enrol")
    def enrol_design(req: EnrolReq):
        with st.lock:
            return st.enrol(req.name, req.kind, req.images)

    @app.get("/api/spc/{design}")
    def spc(design: str):
        rows = st.store.conn.execute(
            "SELECT verdict, final_verdict FROM inspections WHERE design=? ORDER BY id", (design,)).fetchall()
        flags = [int((r["final_verdict"] or r["verdict"]) != "PASS") for r in rows]
        classes = [r["cls"] for r in st.store.conn.execute(
            "SELECT d.cls FROM defects d JOIN inspections i ON i.id=d.inspection_id WHERE i.design=? "
            "AND COALESCE(d.operator_label,'') != 'false_alarm'", (design,))]
        return {"p_chart": p_chart(flags), "pareto": pareto(classes), "boards": len(flags),
                "alerts": [a for a in st.store.open_alerts() if a["design"] == design]}

    @app.post("/api/alerts/{aid}/ack")
    def ack(aid: int):
        st.store.conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (aid,))
        return {"ok": True}

    @app.post("/api/ask")
    def ask(req: AskReq):
        with st.llm_lock:
            return ask_mod.answer(req.question, st.store.conn, st.llm)

    @app.get("/api/export.csv")
    def export():
        p = st.data / "export.csv"
        st.store.export_csv(p)
        return FileResponse(p, media_type="text/csv", filename="inspections.csv")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/app.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="override config: key=value")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg.get("host", "0.0.0.0"), port=int(cfg.get("port", 8000)))


if __name__ == "__main__":
    main()
