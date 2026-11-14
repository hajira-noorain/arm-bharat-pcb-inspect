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
from .fusion import fuse
from .golden import MIN_ALIGN_CC, GoldenMeasurer, align
from .golden_diff import diff_candidates
from .learning import RescoreHead, crop_embedding, rescore
from .record import ANOMALY_CLASS, Defect, Inspection, zone_of
from .severity import DesignRules, SeverityEngine
from .spec import Rule, apply_rules, render

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
    golden: str | None = None              # known-good image of this design (mm measurement + difference channel)
    gate_golden: str | None = None         # gate calibrated on detector+golden-difference evidence
    detector_tiles: str = "all"            # 'all' | 'guided' (golden-first: only tiles holding a difference;
    #                                        the gate must be calibrated under the same policy)
    parts_map: str | None = None           # assembled boards: the taught good board's parts (pcbinspect/parts.py)

    @classmethod
    def from_yaml(cls, p: Path) -> "Design":
        d = yaml.safe_load(Path(p).read_text())
        r = d.get("rules") or {}
        rules = DesignRules(r.get("min_trace_width_mm", 0.2), r.get("min_spacing_mm", 0.2),
                            int(d.get("customer_class", 2)),
                            "min_trace_width_mm" in r and "min_spacing_mm" in r)
        return cls(d["name"], d["kind"], d.get("detector", d["kind"] == "bare"),
                   d.get("anomaly_dir"), d.get("mm_per_px"), d.get("marker_mm", 20.0), rules,
                   d.get("gate"), d.get("alpha", 0.05), d.get("beta", 0.01), d.get("description", ""),
                   d.get("golden"), d.get("gate_golden"), d.get("detector_tiles", "all"), d.get("parts_map"))


@dataclass
class Candidate:
    cls: str
    source: str
    bbox: tuple[int, int, int, int]
    conf: float                      # after re-scoring, if a head is active
    raw_conf: float
    embedding: np.ndarray | None = None
    region_idx: int | None = None    # anomaly region index (for feedback)
    diff_score: float | None = None  # golden-difference evidence for this area, if any
    topo_cls: str | None = None      # what the golden comparison says it is (netlist / shape)
    diff_size_px: float | None = None  # characteristic size of the golden difference (px)
    part: str | None = None            # parts check: the kind of the missing / extra part


@dataclass
class Analysis:
    candidates: list[Candidate]
    board_score: float
    mm_per_px: float | None
    scale_source: str
    board_id: str | None
    anomaly: object | None           # AnomalyResult
    latency_ms: dict[str, float] = field(default_factory=dict)
    golden_used: bool = False        # the golden-difference channel contributed (selects the gate)
    golden_note: str = ""
    golden_aligned: np.ndarray | None = None   # the golden image warped onto this capture (reused for mm)
    parts_only: list = field(default_factory=list)   # parts findings no calibrated evidence backs: shown, and
    #                                                   the board goes to a person; they never change the score
    parts_note: str = ""                             # why the parts check did not run, if it could not


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


FRAGMENT_GAP_MM = 0.5     # golden-comparison pieces of one defect closer than this are shown as one finding


def merge_fragments(cands: list, mm_per_px: float | None) -> list:
    """One defect often reaches the golden comparison as several pieces (a bridge between two thin traces
    breaks into a chain of small regions). For DISPLAY only, golden-only pieces of the same type that touch or
    lie within FRAGMENT_GAP_MM of each other, or of a detection of the same type, become one finding: the
    strongest one, with a box around all of them. The verdict and the board score were decided before this and
    are not changed."""
    gap = FRAGMENT_GAP_MM / mm_per_px if mm_per_px else 12.0

    def near(a, b):
        return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])
    out: list = []
    for c in sorted(cands, key=lambda c: (c.source == "golden", -c.conf)):   # detections first, strongest first
        if c.source == "golden":
            host = next((o for o in out if o.cls == c.cls and o.source in ("golden", "detector")
                         and near(o.bbox, c.bbox)), None)
            if host is not None:
                if host.source == "golden":     # grow a golden finding's box to cover its pieces
                    host.bbox = (min(host.bbox[0], c.bbox[0]), min(host.bbox[1], c.bbox[1]),
                                 max(host.bbox[2], c.bbox[2]), max(host.bbox[3], c.bbox[3]))
                continue
        out.append(c)
    return out


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
        self.gates_golden: dict[str, ConformalGate] = {}
        self.customer_rules: dict[str, list[tuple[int, Rule]]] = {}   # confirmed by the operator
        self.heads: dict[str, RescoreHead] = {}
        self._auto_id = itertools.count(1)
        self._golden_cache: dict[str, tuple] = {}
        self.parts_detector = None          # one-class part detector (set by the app when its model is present)
        self.parts: dict[str, tuple] = {}   # design -> (PartsMap, the good board's picture)
        for name, d in designs.items():
            self.reload_design(name)

    # ------------------------------------------------------------ models
    def reload_design(self, name: str) -> None:
        d = self.designs[name]
        if d.anomaly_dir and self.feature_fn is not None and (self.root / d.anomaly_dir).exists():
            self.anomaly[name] = AnomalyModel.load(self.root / d.anomaly_dir, self.feature_fn)
        if d.gate_file and (self.root / d.gate_file).exists():
            self.gates[name] = ConformalGate.load(self.root / d.gate_file)
        if d.gate_golden and (self.root / d.gate_golden).exists():
            self.gates_golden[name] = ConformalGate.load(self.root / d.gate_golden)
        if d.parts_map and (self.root / d.parts_map).exists():
            from .parts import PartsMap
            ref = cv2.imread(str((self.root / d.parts_map).with_name("parts_ref.jpg")))
            if ref is not None:
                self.parts[name] = (PartsMap.load(self.root / d.parts_map), ref)
        head_p = self.root / "models" / "heads" / f"{name}.json"
        if head_p.exists():
            self.heads[name] = RescoreHead.load(head_p)

    def thresholds(self, design: str, alpha: float | None = None, golden: bool = False) -> GateThresholds | None:
        """Thresholds of the gate calibrated on the SAME evidence the board was judged with."""
        g = self.gates_golden.get(design) if golden else self.gates.get(design)
        if g is None:
            return None
        d = self.designs[design]
        return g.thresholds(alpha if alpha is not None else d.alpha, d.beta)

    # ------------------------------------------------------------ scoring
    def analyse(self, bgr: np.ndarray, design: str, head: RescoreHead | None | str = "active",
                read_id: bool = True, golden: np.ndarray | None = None) -> Analysis:
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

        # Golden comparison FIRST, aligned once: it sees any copper change, and it tells the detector where
        # to look (golden-first designs run the detector only on tiles that hold a difference).
        diffs, golden_ok, golden_note, golden_aligned = [], False, "", None
        use_golden = golden is not None and d.use_detector and design in self.gates_golden
        if use_golden:
            t = time.perf_counter()
            golden_aligned, cc = align(bgr, golden)
            lat["golden_align"] = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            if cc >= MIN_ALIGN_CC:
                diffs, info = diff_candidates(bgr, golden_aligned, aligned=True)
                golden_ok = bool(info["ok"])
                golden_note = "" if golden_ok else info.get("reason", "golden comparison refused")
            else:
                golden_note = f"golden alignment too poor (cc {cc:.2f})"
            lat["golden_diff"] = (time.perf_counter() - t) * 1e3
            if not golden_ok:
                golden_aligned = None
        elif golden is not None and design not in self.gates_golden:
            golden_note = "golden comparison not calibrated for this design"

        cands: list[Candidate] = []
        if d.use_detector and self.detector is not None:
            guided = golden_ok and d.detector_tiles == "guided"
            dets, dl = self.detector.detect(bgr, only_near=[c.bbox for c in diffs] if guided else None)
            lat.update(dl)
            cands += [Candidate(x.cls, "detector", x.bbox, x.conf, x.conf) for x in dets]

        anomaly_res = None
        am = self.anomaly.get(design)
        if am is not None:
            anomaly_res = am.score(bgr)
            lat.update(anomaly_res.latency_ms)
            for i, r in enumerate(anomaly_res.regions):
                if any(_overlaps(r.bbox, c.bbox) for c in cands if c.source == "detector"
                       and c.conf >= max(0.25, r.norm)):
                    continue  # the detector already names this area at least as confidently
                cands.append(Candidate(ANOMALY_CLASS, "anomaly", r.bbox, r.norm, r.norm,
                                       region_idx=i))
            if not anomaly_res.regions:
                # image-level evidence still counts toward the board score
                cands.append(Candidate(ANOMALY_CLASS, "anomaly-image", (0, 0, 0, 0),
                                       anomaly_res.norm_score, anomaly_res.norm_score))

        parts_only, parts_note = self._parts_check(bgr, design, mm_per_px, cands, lat)

        h = self.heads.get(design) if head == "active" else head
        if h is not None and self.feature_fn is not None:
            t = time.perf_counter()
            for c in sorted((c for c in cands if c.source == "detector"),
                            key=lambda c: -c.raw_conf)[:RESCORE_TOP_K]:
                c.embedding = crop_embedding(self.feature_fn, bgr, c.bbox)
                c.conf = rescore(c.cls, c.raw_conf, h.p_real(c.embedding))
            lat["rescore"] = (time.perf_counter() - t) * 1e3

        golden_used = golden_ok
        if golden_ok:
            fz = fuse([{"bbox": c.bbox, "conf": c.conf, "cls": c.cls, "_c": c} for c in cands
                       if c.source == "detector"], diffs)
            for f in fz:
                if "_c" in f:  # an existing detection, possibly strengthened by the difference
                    c = f["_c"]
                    c.conf, c.diff_score, c.topo_cls = f["conf"], f["diff_score"], f["topo_cls"]
                    c.diff_size_px = f["diff_size_px"]
                else:
                    cands.append(Candidate(f["cls"], "golden", f["bbox"], f["conf"], f["conf"],
                                           diff_score=f["diff_score"], topo_cls=f["topo_cls"],
                                           diff_size_px=f["diff_size_px"]))

        score = max((c.conf for c in cands), default=0.0)
        return Analysis(cands, score, mm_per_px, scale_src, board_id, anomaly_res, lat, golden_used, golden_note,
                        golden_aligned, parts_only, parts_note)

    def _parts_check(self, bgr: np.ndarray, design: str, mm_per_px: float | None, cands: list[Candidate],
                     lat: dict[str, float]) -> tuple[list[Candidate], str]:
        """Assembled boards with a taught parts map: is every part still there, is anything extra?

        A finding that falls on an area the anomaly memory already flags NAMES that area ("missing capacitor"
        instead of "unknown anomaly") and keeps the anomaly's calibrated score. A finding nothing else backs is
        returned separately: it is shown and sends the board to a person, but it never enters the board score,
        because the part detector's confidence is not on the calibrated scale."""
        if design not in self.parts or self.parts_detector is None:
            return [], ""
        from .golden import align_warp
        from .parts import MIN_ALIGN_CC_PARTS, MISMATCH_SHARE, check_board, find_parts
        pmap, ref = self.parts[design]
        t = time.perf_counter()
        warp, cc = align_warp(bgr, ref)
        lat["parts_align"] = (time.perf_counter() - t) * 1e3
        if warp is None or cc < MIN_ALIGN_CC_PARTS:
            return [], f"parts check skipped: the board could not be lined up with the taught one (cc {cc:.2f})"
        dets, ms = find_parts(self.parts_detector, bgr, scale=pmap.scale)   # the scale it was taught at
        lat["parts_detector"] = ms
        h, w = bgr.shape[:2]
        alone: list[Candidate] = []
        found = check_board(pmap, dets, warp, (w, h), ref=ref, test=bgr)
        differ = sum(f.what in ("missing_part", "changed_part") for f in found)
        if len(pmap.parts) >= 4 and differ >= MISMATCH_SHARE * len(pmap.parts):
            return [], (f"parts check: {differ} of {len(pmap.parts)} taught parts are missing or look different; "
                        "this is not the taught board, or it is badly placed or unpopulated")
        for f in found:
            box = tuple(int(round(min(max(v, 0), lim))) for v, lim in zip(f.box, (w, h, w, h)))
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            hit = next((c for c in cands if c.source == "anomaly" and _overlaps(c.bbox, box)), None)
            if hit is not None:
                hit.cls, hit.part = f.what, f.kind
            else:
                alone.append(Candidate(f.what, "parts", box, f.confidence, f.confidence, part=f.kind))
        return alone, ""

    def golden_evidence(self, bgr: np.ndarray, golden: np.ndarray, aligned: bool = False,
                        guided: bool = True) -> list[tuple[tuple, float]]:
        """(bbox, confidence) of everything the golden-first channel reports for a capture: the golden
        comparison, the detector (on the tiles that differ when `guided`), fused. This is the evidence a
        taught bare board is judged with, so its gate is calibrated with exactly this function."""
        if aligned:
            g_al, cc = golden, 1.0
        else:
            g_al, cc = align(bgr, golden)
        if cc < MIN_ALIGN_CC:
            return []
        diffs, info = diff_candidates(bgr, g_al, aligned=True)
        if not info["ok"]:
            return []
        dets = []
        if self.detector is not None:
            dd, _ = self.detector.detect(bgr, only_near=[c.bbox for c in diffs] if guided else None)
            dets = [{"bbox": x.bbox, "conf": x.conf, "cls": x.cls} for x in dd]
        return [(tuple(f["bbox"]), float(f["conf"])) for f in fuse(dets, diffs)]

    # ------------------------------------------------------------ full inspection
    def golden_for(self, design: str) -> np.ndarray | None:
        d = self.designs[design]
        if not d.golden or not (self.root / d.golden).exists():
            return None
        p = self.root / d.golden
        key = (str(p), p.stat().st_mtime_ns)
        if self._golden_cache.get(design, (None, None))[0] != key:   # decode once, not on every board
            self._golden_cache[design] = (key, cv2.imread(str(p)))
        return self._golden_cache[design][1]

    def inspect(self, bgr: np.ndarray, design: str, board_id: str | None = None,
                alpha: float | None = None, log: bool = True,
                golden: np.ndarray | None = None, force_review: str | None = None,
                capture_info: dict | None = None) -> tuple[Inspection, Analysis, int | None]:
        """force_review: a reason why this capture cannot be trusted (blur, glare, a lighting shift
        outside the calibrated envelope). A board that would PASS goes to a person instead: the
        miss-rate guarantee only holds for captures like the calibration ones."""
        t_all = time.perf_counter()
        d = self.designs[design]
        golden = golden if golden is not None else self.golden_for(design)
        a = self.analyse(bgr, design, golden=golden)
        if golden is not None and not a.golden_used and design in self.gates_golden:
            # This design is judged WITH its golden board. If the comparison was refused (board shifted,
            # rotated, wrong board), the detector alone must not clear it: a person looks instead.
            force_review = force_review or f"golden comparison refused: {a.golden_note or 'no alignment'}"
        th = self.thresholds(design, alpha, golden=a.golden_used)
        if th is not None:
            verdict = th.verdict(a.board_score)
            show_thr = th.lam if np.isfinite(th.lam) else MIN_SHOW_CONF
        else:  # uncalibrated design: never auto-pass anything with evidence
            verdict = "REVIEW" if a.board_score >= MIN_SHOW_CONF else "PASS"
            show_thr = MIN_SHOW_CONF
        if a.parts_note and not force_review:
            # the parts check could not run (the board would not line up with the taught one: wrong board, badly
            # placed, or so much of it changed that it no longer matches). That is not a pass: a person looks.
            force_review = a.parts_note
        if a.parts_only and not force_review:
            n = {k: sum(c.cls == k for c in a.parts_only) for k in ("missing_part", "changed_part", "extra_part")}
            force_review = "parts check: " + ", ".join(x for x in (
                f"{n['missing_part']} part(s) look missing" if n["missing_part"] else "",
                f"{n['changed_part']} part(s) look different" if n["changed_part"] else "",
                f"{n['extra_part']} unexpected part(s)" if n["extra_part"] else "") if x)
        if force_review and verdict == "PASS":
            verdict = "REVIEW"
        shown = [] if verdict == "PASS" else [
            c for c in a.candidates if c.conf >= show_thr and c.source != "anomaly-image"]
        if verdict != "PASS":
            shown += a.parts_only
        shown = merge_fragments(shown, a.mm_per_px)

        t = time.perf_counter()
        h, w = bgr.shape[:2]
        gm = None
        insp_golden_used = a.golden_used
        if golden is not None and a.mm_per_px and any(c.source in ("detector", "golden") for c in shown):
            if a.golden_aligned is not None:   # aligned once in analyse(): no second ECC, no second warp
                gm = GoldenMeasurer(bgr, a.golden_aligned, a.mm_per_px, d.rules.min_spacing_mm, aligned=True)
            else:
                gm = GoldenMeasurer(bgr, golden, a.mm_per_px, d.rules.min_spacing_mm)
            a.latency_ms["golden_measure"] = (time.perf_counter() - t) * 1e3
        defects, features = [], []
        for i, c in enumerate(sorted(shown, key=lambda c: -c.conf)):
            x1, y1, x2, y2 = c.bbox
            mmpp = a.mm_per_px
            defect = Defect(
                id=f"D{i + 1}", cls=c.cls, part=c.part,
                source=c.source if c.source in ("detector", "golden", "parts") else "anomaly",
                bbox=c.bbox, confidence=round(c.conf, 4),
                size_mm=None if mmpp is None else round(max(x2 - x1, y2 - y1) * mmpp, 3),
                width_mm=None if mmpp is None else round(min(x2 - x1, y2 - y1) * mmpp, 3),
                area_mm2=None if mmpp is None else round((x2 - x1) * (y2 - y1) * mmpp * mmpp, 3),
                location=zone_of(c.bbox, w, h))
            meas = gm.measure(c.cls, c.bbox) if gm is not None and c.source in ("detector", "golden") else None
            self.severity.grade(defect, d.rules, measurement=meas)
            size = (c.diff_size_px * mmpp if c.diff_size_px and mmpp else defect.size_mm)
            features.append({"cls": c.cls, "size_mm": size,
                             "width_loss_pct": meas.pct if meas is not None and meas.kind == "width" else None,
                             "spacing_loss_pct": meas.pct if meas is not None and meas.kind == "spacing" else None})
            if gm is not None and meas is None and gm.last_reason and c.source in ("detector", "golden"):
                defect.severity_reason += f"; no mm measurement: {gm.last_reason}"
            if not d.rules.rules_from_design and meas is not None:
                defect.severity_reason += "; design rules are generic defaults, set them from the Gerber"
            defects.append(defect)
        self._apply_customer_rules(design, defects, features)
        a.latency_ms["severity"] = (time.perf_counter() - t) * 1e3

        bid = board_id or a.board_id or f"{design}-{int(time.time())}-{next(self._auto_id)}"
        insp = Inspection(
            board_id=bid, design=design, image_path="", defects=defects,
            board_score=round(a.board_score, 4), verdict=verdict,
            alpha=th.alpha if th else (alpha or d.alpha), mm_per_px=a.mm_per_px,
            latency_ms={}, model_versions=self.model_versions(design),
            disposition=SeverityEngine.board_disposition(defects) if verdict != "PASS" else "ACCEPT",
            customer_class=d.rules.customer_class)
        if force_review:
            insp.model_versions["capture_check"] = force_review
        insp.model_versions["gate"] = "golden" if insp_golden_used else "detector"
        if capture_info:
            insp.model_versions["capture"] = json.dumps(capture_info, default=str)[:500]
        parts = (["detector"] if d.use_detector and self.detector is not None else []) +                 (["golden"] if a.golden_used else []) + (["anomaly memory"] if a.anomaly is not None else [])
        if design in self.parts and self.parts_detector is not None:
            parts.append("parts check" if not a.parts_note else f"parts check not run ({a.parts_note})")
        insp.model_versions["evidence"] = ("+".join(parts) or "none") + (
            f" ({a.golden_note})" if a.golden_note and not a.golden_used else "")
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

    CUSTOMER_DISPOSITION = {"reject": "SCRAP", "rework": "REWORK", "review": "REVIEW", "accept": "ACCEPT"}

    def _apply_customer_rules(self, design: str, defects, features) -> None:
        """Operator-confirmed customer rules decide the disposition of the defect types they cover; the
        IPC-style defaults decide the rest. A rule that needs a measurement the station could not make
        cannot clear a defect (it fires)."""
        rules = self.customer_rules.get(design) or []
        if not rules or not defects:
            return
        acts, _ = apply_rules([r for _, r in rules], features)
        for defect, f, act in zip(defects, features, acts):
            if act is None:
                continue
            defect.disposition = self.CUSTOMER_DISPOSITION[act]
            for rid, r in rules:
                if apply_rules([r], [f])[0][0] == act:
                    defect.severity_reason += f"; customer rule #{rid}: {render(r)}"
                    break

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
           "customer_class": d.rules.customer_class,
           "rules": ({"min_trace_width_mm": d.rules.min_trace_width_mm,
                      "min_spacing_mm": d.rules.min_spacing_mm} if d.rules.rules_from_design else None),
           "gate": d.gate_file, "alpha": d.alpha, "beta": d.beta, "description": d.description,
           "golden": d.golden, "gate_golden": d.gate_golden,
           "detector_tiles": d.detector_tiles, "parts_map": d.parts_map}
    (Path(config_dir) / f"{d.name}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    json  # noqa: B018 (keep import for callers that serialise designs)
