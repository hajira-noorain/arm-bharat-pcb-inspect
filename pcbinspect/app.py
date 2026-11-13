"""Operator station: FastAPI server + single-page dashboard. Fully offline.

Run:  python -m pcbinspect.app --config config/app.yaml
Open: http://<pi-ip>:8000 on the Pi's screen or a phone on the Pi's hotspot.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
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
from fastapi.responses import FileResponse, HTMLResponse, Response
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
from .spec import Rule, capability, keyword_compile, render as render_rule, same_rule, verify
from .store import Store

log = logging.getLogger("pcbinspect")
CACHE_FRAMES = 6      # full frames kept for feedback/overlay (a 12 MP frame is ~36 MB; Llama needs the RAM)
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
        self.teach_sessions: dict[str, list] = {}     # design name -> good captures (first = golden)
        self.teach_status: dict[str, dict] = {}
        self.jobs: queue.Queue = queue.Queue()
        self.lock = threading.Lock()                           # one inspection at a time
        self.llm_lock = threading.Lock()                       # one Llama generation at a time
        self.spec_compiler = None                              # loaded on first use (setup-time task)
        self._load_customer_rules()
        threading.Thread(target=self._advice_worker, daemon=True).start()

    def _load_customer_rules(self):
        for name in self.designs:
            self.inspector.customer_rules[name] = [(r["id"], Rule.from_dict(r["rule"]))
                                                   for r in self.store.rules(name)]

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
            if cfg.get("vision_backend") == "arm-runtime":
                # Arm's optimised YOLO26n runtime for the Pi 5 (falls back to ExecuTorch if absent)
                from .detect_bare import ArmRuntimeTiles, resolve_arm_runtime
                found = resolve_arm_runtime(root, cfg.get("arm_runtime_lib", ""), cfg.get("arm_runtime_model", ""))
                if found is None:
                    log.warning("Arm runtime not found under %s; using ExecuTorch for vision", root)
                else:
                    try:
                        det = BareDetector(ArmRuntimeTiles(*found))
                        log.info("vision on the Arm runtime: %s, %s", *found)
                    except Exception as exc:
                        log.warning("Arm runtime unavailable (%s); using ExecuTorch for vision", exc)
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
            paths = [p for p in sorted(glob.glob(str(self.root / self.cfg["replay_glob"])))
                     if not p.endswith(".golden.jpg")]
            return ReplayCamera(paths) if paths else None
        try:
            return open_camera(kind)
        except Exception as exc:
            log.warning("camera %s unavailable (%s); trying replay", kind, exc)
            paths = [p for p in sorted(glob.glob(str(self.root / self.cfg["replay_glob"])))
                     if not p.endswith(".golden.jpg")]
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
            except Exception as exc:    # noqa: BLE001
                log.exception("advice failed for inspection %s", iid)
                try:    # the operator must still get advice, and must see that Llama did not write it
                    adv = self.advisor._template(record, [f"Llama could not answer ({type(exc).__name__})"])
                    self.store.log_advice(iid, adv.backend, adv.model, adv.to_json(), False,
                                          "; ".join(adv.rejected_reasons) or None, 0.0, None, None)
                except Exception:   # noqa: BLE001
                    log.exception("fallback advice failed for inspection %s", iid)

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
        if alpha is not None and not 0.0 < alpha <= 0.5:
            raise HTTPException(400, "allowed miss rate must be above 0 and at most 50%")
        with self.lock:
            t0 = time.perf_counter()
            live = not isinstance(self.camera, ReplayCamera)
            bgr = self.camera.capture(n_average=self.cfg.get("average_frames", 1)) if live else self.camera.capture()
            cap_ms = (time.perf_counter() - t0) * 1e3
            golden = self.camera.golden() if not live else None
            force_review, capture_info = None, None
            if live:
                bgr, force_review, capture_info = self._check_capture(bgr, design)
            if board_id is None:
                n = self.store.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM inspections").fetchone()[0]
                board_id = f"B{n:05d}"
            try:
                insp, analysis, iid = self.inspector.inspect(bgr, design, board_id, alpha, golden=golden,
                                                             force_review=force_review, capture_info=capture_info)
            finally:   # a frame that fails must not be served again on the next press
                if isinstance(self.camera, ReplayCamera):
                    self.camera.next()
            self.cache[iid] = (insp, analysis, bgr)
            while len(self.cache) > CACHE_FRAMES:
                self.cache.popitem(last=False)
        insp.latency_ms = {"capture": round(cap_ms, 2), **insp.latency_ms}
        self.store.conn.execute("UPDATE inspections SET latency_json=? WHERE id=?",
                                (json.dumps(insp.latency_ms), iid))
        cv2.imwrite(str(self._overlay_path(iid)), render(bgr, insp, analysis))
        if insp.defects:
            self.jobs.put((iid, insp.to_llm_record(), design))
        return self.inspection_json(iid)

    # ---------------- detection-limit eye test ----------------
    def _eyetest_dir(self, design: str) -> Path:
        p = self.data / "eyetest" / design
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _eyetest_manifest(self) -> dict:
        from .eyetest import load_manifest
        mp = self.root / self.cfg.get("eyetest_dir", "assets/eyetest") / "manifest.json"
        if not mp.exists():
            raise HTTPException(404, f"eye test card manifest not found at {mp}")
        return load_manifest(mp)

    def eyetest_capture(self, design: str, path: str | None = None) -> dict:
        """Photograph one card sheet (or read a file in replay/demo mode), rectify it, identify it."""
        from .eyetest import rectify
        if design not in self.designs:
            raise HTTPException(404, f"unknown design {design}")
        if path:
            fp = (self.root / path).resolve()
            if self.root not in fp.parents:
                raise HTTPException(400, "the photo must be inside the station folder")
            bgr = cv2.imread(str(fp))
            if bgr is None:
                raise HTTPException(404, f"cannot read {path}")
        else:
            if self.camera is None:
                raise HTTPException(503, "no camera")
            with self.lock:
                bgr = self.camera.capture()
        r = rectify(bgr)
        d = self._eyetest_dir(design)
        if not r.ok or r.sheet is None:
            return {"ok": False, "reason": r.reason or "sheet number marker not visible",
                    "captured": sorted(int(x.stem.split("_")[1]) for x in d.glob("sheet_*.png"))}
        cv2.imwrite(str(d / f"sheet_{r.sheet:02d}.png"), r.img)
        # where this sheet came from, so a result read from demo files is never shown as a camera result
        (d / f"sheet_{r.sheet:02d}.src").write_text("file" if path else "camera")
        man = self._eyetest_manifest()
        have = sorted(int(x.stem.split("_")[1]) for x in d.glob("sheet_*.png"))
        need = [s["sheet"] for s in man["sheets"]]
        return {"ok": True, "sheet": r.sheet, "golden": r.sheet == 0, "captured": have,
                "missing": [k for k in need if k not in have]}

    def eyetest_finish(self, design: str, alpha: float | None = None) -> dict:
        from .eyetest import read_sheet, summarise
        if self.detector is None:
            raise HTTPException(503, "no detector loaded")
        man = self._eyetest_manifest()
        d = self._eyetest_dir(design)
        imgs = {int(x.stem.split("_")[1]): cv2.imread(str(x)) for x in d.glob("sheet_*.png")}
        if 0 not in imgs:
            raise HTTPException(409, "capture the GOLDEN sheet (sheet 0) first")
        tests = [s for s in man["sheets"] if not s["golden"] and s["sheet"] in imgs]
        if not tests:
            raise HTTPException(409, "capture at least one test sheet")
        use_golden = design in self.inspector.gates_golden
        th = self.inspector.thresholds(design, alpha, golden=use_golden)
        if th is None:
            raise HTTPException(409, "design has no calibrated gate")
        t0 = time.perf_counter()
        trials = []
        with self.lock:
            for s in tests:
                trials += read_sheet(imgs[s["sheet"]], imgs[0], s, self.detector, use_golden=use_golden)
        res = summarise(trials, float(th.lam))
        res.update({"design": design, "alpha": th.alpha, "evidence": "detector+golden" if use_golden else "detector",
                    "sheets_read": [s["sheet"] for s in tests], "sheets_total": len(man["sheets"]) - 1,
                    "source": ("simulated photos" if any(
                        (d / f"sheet_{k:02d}.src").exists() and (d / f"sheet_{k:02d}.src").read_text() == "file"
                        for k in [0] + [s["sheet"] for s in tests]) else "camera"),
                    "seconds": round(time.perf_counter() - t0, 1)})
        res["id"] = self.store.log_eyetest(design, res, rig=str(self.camera.name if self.camera else ""))
        return res

    # ---------------- customer acceptance rules ----------------
    def _spec_backend(self):
        """The Llama used to read acceptance clauses: a dedicated (larger) .pte if configured, else the
        advice model. Loaded once, on first use: compiling rules is a setup-time task."""
        if self.spec_compiler is not None:
            return self.spec_compiler
        from .spec_llm import ConstrainedSpecCompiler, LlamaSpecCompiler
        be = None
        p = self.root / self.cfg.get("spec_llm_pte", "")
        if self.cfg.get("spec_llm_pte") and p.is_file() and self.cfg.get("spec_reader", "grounded") == "grounded":
            # Constrained decoding on the same .pte through executorch.runtime: Llama can only answer from
            # option lists the clause's own words allow (62.5% of held-out clauses exactly right against
            # 47.5% for free generation, never "no rule"; llm/eval_spec.py --constrained --grounded).
            try:
                from .llm_logits import LogitLlama
                self.spec_compiler = ConstrainedSpecCompiler(
                    LogitLlama(p, self.root / self.cfg["llm_tokenizer"]), grounded=True)
                self.spec_reader_name = "grounded choice (constrained decoding)"
                return self.spec_compiler
            except Exception as exc:    # noqa: BLE001
                log.exception("constrained rule reader unavailable (%s); using free generation", exc)
        if self.cfg.get("spec_llm_pte") and p.is_file():
            from .llm import ExecuTorchLlama
            try:
                be = ExecuTorchLlama(p, self.root / self.cfg["llm_tokenizer"], threads=self.cfg.get("llm_threads", 4))
            except Exception as exc:
                log.exception("spec Llama failed to load (%s)", exc)
        be = be or self.llm
        self.spec_compiler = LlamaSpecCompiler(be) if be is not None else False
        return self.spec_compiler

    def compile_rule(self, design: str, clause: str) -> dict:
        if design not in self.designs:
            raise HTTPException(404, f"unknown design {design}")
        d = self.designs[design]
        limits = self.store.detection_limits(design)
        out = {"clause": clause, "readers": []}
        rules = {}
        comp = self._spec_backend()
        if comp:
            with self.llm_lock:
                r = comp.compile(clause)
            rules["llama"] = r.rule
            out["llama_ms"] = round(r.latency_ms)
            out["llama_reader"] = getattr(self, "spec_reader_name", "free generation")
            out["llama_margins"] = (r.answers or {}).get("_margins")
            out["llama_error"] = r.error
        rules["keyword"] = keyword_compile(clause)
        for name, rule in rules.items():
            if rule is None:
                out["readers"].append({"reader": name, "rule": None})
                continue
            out["readers"].append({
                "reader": name, "rule": rule.to_dict(), "sentence": render_rule(rule), "verify": verify(rule, clause),
                "capability": capability(rule, limits, d.rules.min_trace_width_mm, d.rules.min_spacing_mm)})
        lr, kr = rules.get("llama"), rules.get("keyword")
        out["readers_agree"] = bool(lr is not None and same_rule(lr, kr))
        first = out["readers"][0]
        out["status"] = ("verified" if out["readers_agree"] and first.get("verify", {}).get("status") == "verified"
                         else "operator_must_choose")
        out["detection_limits_from"] = "latest eye test" if limits else "no eye test yet"
        return out

    def save_rule(self, design: str, clause: str, rule: dict, reader: str) -> dict:
        r = Rule.from_dict(rule)
        from .spec import schema_errors
        errs = schema_errors(r)
        if errs:
            raise HTTPException(422, "; ".join(errs))
        d = self.designs[design]
        cap = capability(r, self.store.detection_limits(design), d.rules.min_trace_width_mm, d.rules.min_spacing_mm)
        rid = self.store.add_rule(design, clause, r.to_dict(), reader, verify(r, clause), cap)
        self._load_customer_rules()
        return {"id": rid, "sentence": render_rule(r), "capability": cap}

    # ---------------- capture envelope (live cameras only) ----------------
    def _flatfield_path(self) -> Path:
        return self.data / "flatfield.npz"

    def _profile_path(self, design: str) -> Path:
        return self.data / f"capture_profile_{design}.json"

    def _check_capture(self, bgr, design: str):
        """Flat-field correction (if recorded), then: is this capture inside the envelope the station was
        calibrated for? Generic checks (blur, clipping, exposure) plus, when recorded, the design's reference
        profile (sharpness, brightness, colour balance). Returns (image, reason or None, info)."""
        from .imaging import CaptureProfile, FlatField, capture_quality
        if self._flatfield_path().exists():
            bgr = FlatField.load(self._flatfield_path()).apply(bgr)
        q = capture_quality(bgr)
        reasons = list(q["reasons"])
        if self._profile_path(design).exists():
            reasons += CaptureProfile.load(self._profile_path(design)).check(bgr)["reasons"]
        info = {"sharpness": round(q["sharpness"], 1), "mean": round(q["mean"], 1)}
        try:
            info["camera"] = self.camera.settings()
        except Exception:   # noqa: BLE001
            pass
        reason = ("capture outside the calibrated envelope: " + "; ".join(dict.fromkeys(reasons))) if reasons else None
        return bgr, reason, info

    def record_flatfield(self, n: int = 8) -> dict:
        """Blank matte grey/white card under the same light, filling the view."""
        from .imaging import FlatField
        if self.camera is None or isinstance(self.camera, ReplayCamera):
            raise HTTPException(409, "needs a live camera")
        with self.lock:
            frames = [self.camera.capture() for _ in range(n)]
        FlatField.from_frames(frames).save(self._flatfield_path())
        return {"ok": True, "frames": n}

    def record_profile(self, design: str, n: int = 8) -> dict:
        """With the GOLDEN board in place: the reference look of a good capture for this design."""
        from .imaging import CaptureProfile, FlatField
        if self.camera is None or isinstance(self.camera, ReplayCamera):
            raise HTTPException(409, "needs a live camera")
        with self.lock:
            frames = [self.camera.capture() for _ in range(n)]
        if self._flatfield_path().exists():
            ff = FlatField.load(self._flatfield_path())
            frames = [ff.apply(f) for f in frames]
        prof = CaptureProfile.from_images(frames)
        prof.save(self._profile_path(design))
        return {"ok": True, **prof.to_dict()}

    def _overlay_path(self, iid: int) -> Path:
        p = self.data / "overlays"
        p.mkdir(exist_ok=True)
        return p / f"{iid}.jpg"

    def inspection_json(self, iid: int) -> dict:
        r = self.store.inspection(iid)
        if r is None:
            raise HTTPException(404)
        for d in r["defects"]:
            d["ipc"] = json.loads(d["ipc_json"] or "{}")
            d["name"] = self.kb.display_name(d["cls"])
        for a in r["advice"]:
            a["advice"] = json.loads(a["advice_json"])
        r["latency"] = json.loads(r["latency_json"] or "{}")
        versions = json.loads(r.get("model_versions_json") or "{}") if isinstance(r, dict) else {}
        th = self.inspector.thresholds(r["design"], r["alpha"], golden=versions.get("gate") == "golden")
        r["gate"] = None if th is None else {
            "alpha": th.alpha, "beta": th.beta, "lam": th.lam, "tau": th.tau,
            "n_defective": th.n_defective, "n_good": th.n_good,
            "planted": "planted defects" in (self.designs[r["design"]].description or "")
            if r["design"] in self.designs else False}
        r["advice_pending"] = bool(r["defects"]) and not r["advice"]
        # why a board that looked clean was still sent to a person (blurred picture, golden board not matched)
        r["check_reason"] = versions.get("capture_check")
        r["evidence"] = versions.get("evidence")
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
            self.buffer(design).add(emb, label, ev, conf=float(row["confidence"] or 0.5), group=iid)
            npos, nneg = self.buffer(design).counts()
            result.update({"learned": "stored for the next model update", "examples": {
                "confirmed": npos, "false_alarms": nneg}})
        return {**result, "event_id": ev}

    def undo(self, event_id: int) -> dict:
        pre = self.store.feedback_event(event_id)
        if pre is not None:   # refuse BEFORE touching the log, so a refusal leaves everything as it was
            p0 = json.loads(pre["payload_json"] or "{}")
            row0 = self.store.inspection(pre["inspection_id"]) if pre["inspection_id"] else None
            am0 = self.inspector.anomaly.get(row0["design"]) if row0 else None
            if "anomaly" in p0 and am0 is not None and not (
                    am0.history and am0.history[-1]["version"] == p0["anomaly"]["version"]):
                raise HTTPException(409, "only the most recent memory change can be undone")
        ev = self.store.undo_feedback(event_id)
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
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
            raise HTTPException(400, "design name: letters, digits, - and _ only")
        if name in self.designs:   # its gate was calibrated for the old memory; never silently replace it
            raise HTTPException(409, f"a design named {name} already exists; pick another name")
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


    # ---------------- teach a bare board: golden image + gate calibrated on planted defects ----------------
    def teach_capture(self, name: str) -> dict:
        """One good board per call (lift it and put it down again, or use another good board). The first
        capture becomes the golden image; later ones must show the same board in the same place."""
        from .golden import MIN_ALIGN_CC, align, copper_mask
        from .imaging import capture_quality
        if not re.fullmatch(r"[A-Za-z0-9_-]{2,40}", name):
            raise HTTPException(400, "design name: 2-40 letters, digits, - and _")
        if name in self.designs:
            raise HTTPException(409, f"a design named {name} already exists; pick another name")
        if self.camera is None:
            raise HTTPException(503, "no camera")
        with self.lock:
            live = not isinstance(self.camera, ReplayCamera)
            bgr = self.camera.capture(n_average=self.cfg.get("average_frames", 1)) if live else self.camera.capture()
            if not live:
                self.camera.next()
        caps = self.teach_sessions.setdefault(name, [])
        if live:
            q = capture_quality(bgr)
            if q["reasons"]:
                return {"ok": False, "captures": len(caps), "reason": "retake: " + "; ".join(q["reasons"])}
        frac = float((copper_mask(bgr) > 0).mean())
        if not 0.03 <= frac <= 0.85:
            return {"ok": False, "captures": len(caps),
                    "reason": "no copper pattern visible: is a bare board under the camera, in focus?"}
        if caps:
            _, cc = align(bgr, caps[0])
            if cc < MIN_ALIGN_CC:
                return {"ok": False, "captures": len(caps),
                        "reason": "this does not match the first board: same design, against the corner stops?"}
        if len(caps) >= 8:
            return {"ok": False, "captures": len(caps), "reason": "8 good boards are enough: finish teaching"}
        caps.append(bgr)
        return {"ok": True, "captures": len(caps), "golden": len(caps) == 1, "ready": len(caps) >= 2}

    def teach_cancel(self, name: str) -> dict:
        self.teach_sessions.pop(name, None)
        self.teach_status.pop(name, None)
        return {"cancelled": name}

    def teach_finish(self, name: str, mm_per_px: float | None = None) -> dict:
        caps = self.teach_sessions.get(name) or []
        if len(caps) < 2:
            raise HTTPException(409, "capture the good board at least twice (3 or more is better)")
        if self.teach_status.get(name, {}).get("running"):
            return self.teach_status[name]
        from .calib import detect_scale
        sc = detect_scale(caps[0], 20.0)
        if sc is not None and sc.source == "aruco":
            mmpp, scale_src = float(sc.mm_per_px), "the printed marker"
        elif mm_per_px:
            mmpp, scale_src = float(mm_per_px), "the value you entered"
        else:
            raise HTTPException(409, "no scale: put the printed 20 mm marker beside the board for the first "
                                     "capture, or enter millimetres per pixel")
        self.teach_status[name] = {"running": True, "progress": 0.0, "started": time.time()}

        def job():
            from .teach import calibrate_bare
            try:
                def score(t, g, aligned):
                    with self.lock:     # inspections of other designs can interleave between trials
                        return self.inspector.golden_evidence(t, g, aligned=aligned, guided=True)

                def progress(f):
                    self.teach_status[name]["progress"] = round(float(f), 3)
                t0 = time.perf_counter()
                res = calibrate_bare(score, caps[0], caps[1:], 1.0 / mmpp, np.random.default_rng(), progress=progress,
                                     n_windows=int(self.cfg.get("teach_windows", 120)),
                                     n_good=int(self.cfg.get("teach_good", 24)))
                gdir, gate_dir = self.root / "models" / "golden", self.root / "models" / "gate"
                gdir.mkdir(parents=True, exist_ok=True)
                gate_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(gdir / f"{name}.png"), caps[0])
                res.gate.design = f"{name}_golden"
                res.gate.save(gate_dir / f"{name}_golden.json")
                d = Design(name, "bare", True, None, mmpp, 20.0,
                           self.inspector.severity.default_rules, None, 0.05, 0.05,
                           f"taught on the station from {len(caps)} good captures; gate calibrated on planted defects",
                           f"models/golden/{name}.png", f"models/gate/{name}_golden.json", "guided")
                with self.lock:
                    save_design(self.root / self.cfg["designs_dir"], d)
                    self.designs[name] = d
                    self.inspector.designs[name] = d
                    self.inspector.reload_design(name)
                    self.store.upsert_design(name, "bare")
                summary = res.summary()
                summary.update({"design": name, "sheets_read": [], "sheets_total": 0, "scale_from": scale_src,
                                "seconds": round(time.perf_counter() - t0, 1)})
                summary["id"] = self.store.log_eyetest(name, summary, rig="planted on the taught board")
                self.store.record_model_version(f"golden_gate:{name}", "1", {
                    "captures": len(caps), "planted_windows": res.n_windows, "good_scores": len(res.good_scores),
                    "lambda_5pct": res.lam05}, True, "taught")
                th = res.gate.thresholds(0.05, 0.05)
                self.teach_status[name] = {
                    "running": False, "done": True, "design": name, "captures": len(caps),
                    "seconds": summary["seconds"], "scale_from": scale_src, "mm_per_px": round(mmpp, 5),
                    "can_fail": bool(np.isfinite(th.tau)), "can_pass": bool(th.lam > 0),
                    "limits": {k: v["sentence"] for k, v in res.limits.items()}, "notes": res.notes}
                self.teach_sessions.pop(name, None)
            except Exception as exc:    # noqa: BLE001
                log.exception("teaching %s failed", name)
                self.teach_status[name] = {"running": False, "done": False, "reason": str(exc)}
        threading.Thread(target=job, daemon=True).start()
        return self.teach_status[name]


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


class TeachReq(BaseModel):
    name: str
    mm_per_px: float | None = None


class EyeCaptureReq(BaseModel):
    design: str
    path: str | None = None      # replay/demo: a photo file instead of the camera


class EyeFinishReq(BaseModel):
    design: str
    alpha: float | None = None


class RuleCompileReq(BaseModel):
    design: str
    clause: str


class RuleSaveReq(BaseModel):
    design: str
    clause: str
    rule: dict
    reader: str


class AskReq(BaseModel):
    question: str


def _gate_of(st, name):
    return st.inspector.gates.get(name) or st.inspector.gates_golden.get(name)


def create_app(cfg: dict) -> FastAPI:
    st = Station(cfg)
    app = FastAPI(title="PCB inspection station")
    app.state.station = st
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    def state():
        designs = [{"name": n, "kind": d.kind, "alpha": d.alpha, "description": d.description,
                    "calibrated": n in st.inspector.gates or n in st.inspector.gates_golden,
                    "taught": "planted defects" in (d.description or ""),
                    "n_defective": len(_gate_of(st, n).defective) if _gate_of(st, n) else 0,
                    "n_good": len(_gate_of(st, n).good) if _gate_of(st, n) else 0,
                    "customer_class": d.rules.customer_class,
                    "rules_from_design": d.rules.rules_from_design,
                    "detector": d.use_detector and st.detector is not None,
                    "anomaly": n in st.inspector.anomaly} for n, d in st.designs.items()]
        return {"designs": designs,
                "camera": None if st.camera is None else st.camera.name,
                "llm": None if st.llm is None else st.llm.model,
                "vision": {"detector": None if st.detector is None else st.detector.tiles.name,
                           "backbone": st.feature_fn is not None},
                "light": st.light.state,
                "eyetest_demo": sorted(str(x.relative_to(st.root)).replace("\\", "/")
                                       for x in (st.root / "demo" / "eyetest").glob("*.jpg")),
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

    @app.post("/api/teach/capture")
    def teach_capture(req: TeachReq):
        return st.teach_capture(req.name)

    @app.post("/api/teach/finish")
    def teach_finish(req: TeachReq):
        return st.teach_finish(req.name, req.mm_per_px)

    @app.post("/api/teach/cancel")
    def teach_cancel(req: TeachReq):
        return st.teach_cancel(req.name)

    @app.get("/api/teach/{name}")
    def teach_state(name: str):
        return {"captures": len(st.teach_sessions.get(name, [])), **st.teach_status.get(name, {})}

    learn_status: dict[str, dict] = {}

    @app.post("/api/learn/{design}")
    def learn(design: str):
        if design not in st.designs:
            raise HTTPException(404)
        if learn_status.get(design, {}).get("running"):
            return learn_status[design]
        from .promote import try_promote
        learn_status[design] = {"running": True, "started": time.time()}

        def job():
            try:
                with st.lock:      # no inspections while the model is being swapped
                    rep = try_promote(st.inspector, design, st.buffer(design), st.root, st.store)
                learn_status[design] = {"running": False, **rep}
            except Exception as exc:
                log.exception("learning update failed")
                learn_status[design] = {"running": False, "promoted": False, "reason": repr(exc)}
        threading.Thread(target=job, daemon=True).start()
        return learn_status[design]

    @app.get("/api/learn/{design}")
    def learn_state(design: str):
        n_pos, n_neg = st.buffer(design).counts()
        versions = [dict(v) for v in st.store.model_versions(f"rescore_head:{design}")]
        for v in versions:
            v["metrics"] = json.loads(v.pop("metrics_json") or "{}")
        return {"status": learn_status.get(design), "confirmed": n_pos, "false_alarms": n_neg,
                "active_version": getattr(st.inspector.heads.get(design), "version", None),
                "versions": versions[-10:][::-1]}

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

    @app.post("/api/capture/flatfield")
    def capture_flatfield():
        return st.record_flatfield()

    @app.post("/api/capture/profile/{design}")
    def capture_profile(design: str):
        return st.record_profile(design)

    @app.get("/api/eyetest/card.pdf")
    def eyetest_card():
        f = st.root / cfg.get("eyetest_dir", "assets/eyetest") / "eyetest_card.pdf"
        if not f.exists():
            raise HTTPException(404, "card PDF not in this bundle")
        return FileResponse(f, media_type="application/pdf", filename="eyetest_card.pdf")

    @app.post("/api/eyetest/capture")
    def eyetest_capture(req: EyeCaptureReq):
        return st.eyetest_capture(req.design, req.path)

    @app.post("/api/eyetest/finish")
    def eyetest_finish(req: EyeFinishReq):
        return st.eyetest_finish(req.design, req.alpha)

    @app.post("/api/eyetest/reset/{design}")
    def eyetest_reset(design: str):
        for x in st._eyetest_dir(design).glob("sheet_*.*"):
            x.unlink()
        return {"ok": True}

    @app.get("/api/eyetest/{design}")
    def eyetest_latest(design: str):
        d = st._eyetest_dir(design)
        return {"latest": st.store.latest_eyetest(design),
                "captured": sorted(int(x.stem.split("_")[1]) for x in d.glob("sheet_*.png"))}

    @app.post("/api/rules/compile")
    def rules_compile(req: RuleCompileReq):
        return st.compile_rule(req.design, req.clause)

    @app.post("/api/rules")
    def rules_save(req: RuleSaveReq):
        return st.save_rule(req.design, req.clause, req.rule, req.reader)

    @app.get("/api/rules/{design}")
    def rules_list(design: str):
        return [{**r, "sentence": render_rule(Rule.from_dict(r["rule"]))} for r in st.store.rules(design)]

    @app.delete("/api/rules/{rule_id}")
    def rules_delete(rule_id: int):
        st.store.deactivate_rule(rule_id)
        st._load_customer_rules()
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
