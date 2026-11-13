"""The inspection pipeline: one call from a camera frame to a logged, graded, gated verdict.

analyse()  pure scoring (used identically for conformal calibration and live inspection)
inspect()  analyse + gate + severity + log + stack light
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .anomaly import AnomalyModel, FeatureFn
from .calib import detect_scale, read_board_id
from .detect_bare import BareDetector
from .gate import ConformalGate, GateThresholds
from .golden import GoldenMeasurer
from .learning import RescoreHead, combine, crop_embedding
from .record import ANOMALY_CLASS, Defect, Inspection, zone_of
from .severity import DesignRules, SeverityEngine

MIN_SHOW_CONF = 0.25      # display floor used only when the gate has no calibration yet
RESCORE_TOP_K = 20        # rescore at most this many detections per board (Pi time budget)


@dataclass
class Design:
    name: str
    kind: str                              # 'bare' | 'assembled'
    use_detector: bool
    anomaly_dir: str | None
    mm_per_px: float | None                # fallback scale when no ArUco marker is visible
    marker_mm: float
    rules: DesignRules
    gate_file: str | None
    alpha: float = 0.05
    beta: float = 0.01
    description: str = ""
    golden: str | None = None              # known-good image of this design (for mm measurement)

    @classmethod
    def from_yaml(cls, p: Path) -> "Design":
        d = yaml.safe_load(Path(p).read_text())
        r = d.get("rules", {})
        return cls(d["name"], d["kind"], d.get("detector", d["kind"] == "bare"),
                   d.get("anomaly_dir"), d.get("mm_per_px"), d.get("marker_mm", 20.0),
                   DesignRules(r.get("min_trace_width_mm", 0.2), r.get("min_spacing_mm", 0.2)),
                   d.get("gate"), d.get("alpha", 0.05), d.get("beta", 0.01), d.get("description", ""),
                   d.get("golden"))


@dataclass
class Candidate:
    cls: str
    source: str
    bbox: tuple[int, int, int, int]
    conf: float                      # after re-scoring, if a head is active
    raw_conf: float
    embedding: np.ndarray | None = None
    region_idx: int | None = None    # anomaly region index (for feedback)


@dataclass
class Analysis:
    candidates: list[Candidate]
    board_score: float
    mm_per_px: float | None
    scale_source: str
    board_id: str | None
    anomaly: object | None           # AnomalyResult
    latency_ms: dict[str, float] = field(default_factory=dict)


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class Inspector:
    def __init__(self, root: Path, designs: dict[str, Design], detector: BareDetector | None,
                 feature_fn: FeatureFn | None, store=None, light=None,
                 severity: SeverityEngine | None = None):
        self.root = Path(root)
        self.designs = designs
        self.detector = detector
        self.feature_fn = feature_fn
        self.store = store
        self.light = light
        self.severity = severity or SeverityEngine()
        self.anomaly: dict[str, AnomalyModel] = {}
        self.gates: dict[str, ConformalGate] = {}
        self.heads: dict[str, RescoreHead] = {}
        self._auto_id = itertools.count(1)
        for name, d in designs.items():
            self.reload_design(name)

    # ------------------------------------------------------------ models
    def reload_design(self, name: str) -> None:
        d = self.designs[name]
        if d.anomaly_dir and self.feature_fn is not None and (self.root / d.anomaly_dir).exists():
            self.anomaly[name] = AnomalyModel.load(self.root / d.anomaly_dir, self.feature_fn)
        if d.gate_file and (self.root / d.gate_file).exists():
            self.gates[name] = ConformalGate.load(self.root / d.gate_file)
        head_p = self.root / "models" / "heads" / f"{name}.json"
        if head_p.exists():
            self.heads[name] = RescoreHead.load(head_p)

    def thresholds(self, design: str, alpha: float | None = None) -> GateThresholds | None:
        g = self.gates.get(design)
        if g is None:
            return None
        d = self.designs[design]
        return g.thresholds(alpha if alpha is not None else d.alpha, d.beta)

    # ------------------------------------------------------------ scoring
    def analyse(self, bgr: np.ndarray, design: str, head: RescoreHead | None | str = "active",
                read_id: bool = True) -> Analysis:
        d = self.designs[design]
        lat: dict[str, float] = {}
        t = time.perf_counter()
        board_id = read_board_id(bgr) if read_id else None
        lat["board_id"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        sc = detect_scale(bgr, d.marker_mm)
        if sc is not None and sc.source == "aruco":
            mm_per_px, scale_src = sc.mm_per_px, "aruco"
        else:
            mm_per_px, scale_src = d.mm_per_px, ("design-default" if d.mm_per_px else "none")
        lat["scale"] = (time.perf_counter() - t) * 1e3

        cands: list[Candidate] = []
        if d.use_detector and self.detector is not None:
            dets, dl = self.detector.detect(bgr)
            lat.update(dl)
            cands += [Candidate(x.cls, "detector", x.bbox, x.conf, x.conf) for x in dets]

        anomaly_res = None
        am = self.anomaly.get(design)
        if am is not None:
            anomaly_res = am.score(bgr)
            lat.update(anomaly_res.latency_ms)
            for i, r in enumerate(anomaly_res.regions):
                if any(_overlaps(r.bbox, c.bbox) for c in cands if c.source == "detector"
                       and c.conf >= 0.25):
                    continue  # the detector already names this area
                cands.append(Candidate(ANOMALY_CLASS, "anomaly", r.bbox, r.norm, r.norm,
                                       region_idx=i))
            if not anomaly_res.regions:
                # image-level evidence still counts toward the board score
                cands.append(Candidate(ANOMALY_CLASS, "anomaly-image", (0, 0, 0, 0),
                                       anomaly_res.norm_score, anomaly_res.norm_score))

        h = self.heads.get(design) if head == "active" else head
        if h is not None and self.feature_fn is not None:
            t = time.perf_counter()
            for c in sorted((c for c in cands if c.source == "detector"),
                            key=lambda c: -c.raw_conf)[:RESCORE_TOP_K]:
                c.embedding = crop_embedding(self.feature_fn, bgr, c.bbox)
                c.conf = combine(c.raw_conf, h.p_real(c.embedding))
            lat["rescore"] = (time.perf_counter() - t) * 1e3

        score = max((c.conf for c in cands), default=0.0)
        return Analysis(cands, score, mm_per_px, scale_src, board_id, anomaly_res, lat)

    # ------------------------------------------------------------ full inspection
    def golden_for(self, design: str) -> np.ndarray | None:
        d = self.designs[design]
        if d.golden and (self.root / d.golden).exists():
            return cv2.imread(str(self.root / d.golden))
        return None

    def inspect(self, bgr: np.ndarray, design: str, board_id: str | None = None,
                alpha: float | None = None, log: bool = True,
                golden: np.ndarray | None = None) -> tuple[Inspection, Analysis, int | None]:
        t_all = time.perf_counter()
        d = self.designs[design]
        a = self.analyse(bgr, design)
        th = self.thresholds(design, alpha)
        if th is not None:
            verdict = th.verdict(a.board_score)
            show_thr = th.lam if np.isfinite(th.lam) else MIN_SHOW_CONF
        else:  # uncalibrated design: never auto-pass anything with evidence
            verdict = "REVIEW" if a.board_score >= MIN_SHOW_CONF else "PASS"
            show_thr = MIN_SHOW_CONF
        shown = [] if verdict == "PASS" else [
            c for c in a.candidates if c.conf >= show_thr and c.source != "anomaly-image"]

        t = time.perf_counter()
        h, w = bgr.shape[:2]
        golden = golden if golden is not None else self.golden_for(design)
        gm = None
        if golden is not None and a.mm_per_px and any(c.source == "detector" for c in shown):
            gm = GoldenMeasurer(bgr, golden, a.mm_per_px, d.rules.min_spacing_mm)
            a.latency_ms["golden_align"] = (time.perf_counter() - t) * 1e3
        defects = []
        for i, c in enumerate(sorted(shown, key=lambda c: -c.conf)):
            x1, y1, x2, y2 = c.bbox
            mmpp = a.mm_per_px
            defect = Defect(
                id=f"D{i + 1}", cls=c.cls, source="detector" if c.source == "detector" else "anomaly",
                bbox=c.bbox, confidence=round(c.conf, 4),
                size_mm=None if mmpp is None else round(max(x2 - x1, y2 - y1) * mmpp, 3),
                width_mm=None if mmpp is None else round(min(x2 - x1, y2 - y1) * mmpp, 3),
                area_mm2=None if mmpp is None else round((x2 - x1) * (y2 - y1) * mmpp * mmpp, 3),
                location=zone_of(c.bbox, w, h))
            meas = gm.measure(c.cls, c.bbox) if gm is not None and c.source == "detector" else None
            defects.append(self.severity.grade(defect, d.rules, measurement=meas))
        a.latency_ms["severity"] = (time.perf_counter() - t) * 1e3

        bid = board_id or a.board_id or f"{design}-{int(time.time())}-{next(self._auto_id)}"
        insp = Inspection(
            board_id=bid, design=design, image_path="", defects=defects,
            board_score=round(a.board_score, 4), verdict=verdict,
            alpha=th.alpha if th else (alpha or d.alpha), mm_per_px=a.mm_per_px,
            latency_ms={}, model_versions=self.model_versions(design))
        a.latency_ms["total_to_verdict"] = (time.perf_counter() - t_all) * 1e3

        iid = None
        if log and self.store is not None:
            t = time.perf_counter()
            ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            sha = self.store.put_image(jpg.tobytes()) if ok else None
            insp.image_path = str(self.store.image_path(sha)) if sha else ""
            insp.latency_ms = {k: round(v, 2) for k, v in a.latency_ms.items()}
            iid = self.store.log_inspection(insp, sha)
            a.latency_ms["log"] = (time.perf_counter() - t) * 1e3
        insp.latency_ms = {k: round(v, 2) for k, v in a.latency_ms.items()}
        if self.light is not None:
            self.light.show(verdict)
        return insp, a, iid

    def model_versions(self, design: str) -> dict[str, str]:
        v = {}
        if design in self.anomaly:
            v["anomaly_bank"] = str(self.anomaly[design].version)
        if design in self.heads:
            v["rescore_head"] = str(self.heads[design].version)
        if self.detector is not None:
            v["detector"] = self.detector.tiles.name
        return v


def load_designs(config_dir: Path) -> dict[str, Design]:
    out = {}
    for p in sorted(Path(config_dir).glob("*.yaml")):
        d = Design.from_yaml(p)
        out[d.name] = d
    return out


def save_design(config_dir: Path, d: Design) -> None:
    Path(config_dir).mkdir(parents=True, exist_ok=True)
    doc = {"name": d.name, "kind": d.kind, "detector": d.use_detector, "anomaly_dir": d.anomaly_dir,
           "mm_per_px": d.mm_per_px, "marker_mm": d.marker_mm,
           "rules": {"min_trace_width_mm": d.rules.min_trace_width_mm,
                     "min_spacing_mm": d.rules.min_spacing_mm},
           "gate": d.gate_file, "alpha": d.alpha, "beta": d.beta, "description": d.description,
           "golden": d.golden}
    (Path(config_dir) / f"{d.name}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    json  # noqa: B018 (keep import for callers that serialise designs)
