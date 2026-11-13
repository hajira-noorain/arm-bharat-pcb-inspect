"""Parts check for assembled boards: is every part of the good board still there, and is anything extra?

When a good assembled board is taught, the station finds its parts once and stores a PARTS MAP: where each part
sits (in the good board's own picture) and what kind it is. At inspection the new picture is lined up with the
good one, parts are found again, and the two lists are compared:

  missing   a part on the map with no part found at its place
  extra     a part found where the map has none

The detector answers only "is there a part here?" (one class, training/prepare_components.py). Part kinds come
from a small classifier that runs when the board is taught, never during inspection, so naming costs nothing
per board. A part whose kind the classifier is not sure of is reported simply as "part".

Nothing here loads a model: detections come in as plain boxes, so this is testable without one.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MIN_KIND_CONF = 0.6      # below this the kind is not shown: "part missing", not a guessed "capacitor missing"
MATCH_IOU = 0.2          # a found part this much over a mapped part is that part (alignment is never perfect)
MIN_MAP_CONF = 0.5       # only confidently found parts go on the map: a doubtful one would later "go missing"
EDGE_MM = 1.0            # parts closer than this to the picture edge are not judged (they may be cut off)


@dataclass
class Part:
    box: list[float]                 # x0, y0, x1, y1 in the good board's picture
    kind: str = "part"
    kind_conf: float = 0.0
    conf: float = 1.0                # how sure the detector was when the board was taught

    @property
    def label(self) -> str:
        return self.kind if self.kind_conf >= MIN_KIND_CONF and self.kind != "other" else "part"

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


@dataclass
class PartsMap:
    size: list[int]                  # width, height of the good board's picture
    px_per_mm: float | None
    parts: list[Part] = field(default_factory=list)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(path: Path) -> "PartsMap":
        d = json.loads(Path(path).read_text())
        return PartsMap(d["size"], d.get("px_per_mm"), [Part(**p) for p in d["parts"]])

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for p in self.parts:
            kinds[p.label] = kinds.get(p.label, 0) + 1
        return {"parts": len(self.parts), "by_kind": dict(sorted(kinds.items(), key=lambda kv: -kv[1]))}


@dataclass
class PartFinding:
    what: str                        # "missing_part" or "extra_part"
    box: list[float]                 # in the good board's picture
    kind: str                        # "capacitor", ... or "part"
    confidence: float                # missing: how sure the map was of this part; extra: the detector's confidence
    size_mm: float | None = None


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def _same_place(a, b) -> bool:
    """Two boxes are the same part if they overlap enough, or one's centre lies inside the other (a small part
    under a slightly shifted picture can have a low IoU and still plainly be there)."""
    if iou(a, b) >= MATCH_IOU:
        return True
    for p, q in ((a, b), (b, a)):
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        if q[0] <= cx <= q[2] and q[1] <= cy <= q[3]:
            return True
    return False


def dedupe(dets: list[dict], thr: float = 0.5) -> list[dict]:
    """One box per part: tiles overlap, so the same part can be found twice."""
    keep: list[dict] = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if all(iou(d["bbox"], k["bbox"]) < thr for k in keep):
            keep.append(d)
    return keep


def build_map(dets: list[dict], size: tuple[int, int], px_per_mm: float | None = None,
              kinds: list[tuple[str, float]] | None = None) -> PartsMap:
    """dets: [{"bbox": [x0,y0,x1,y1], "conf": c}] found on the good board. kinds: for each det, the classifier's
    (kind, confidence), or None when no classifier is available (every part is then just "part")."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["conf"])
    parts: list[Part] = []
    for i in order:
        d = dets[i]
        if d["conf"] < MIN_MAP_CONF or any(iou(d["bbox"], p.box) >= 0.5 for p in parts):
            continue
        k, kc = kinds[i] if kinds else ("part", 0.0)
        parts.append(Part([float(v) for v in d["bbox"]], k, float(kc), float(d["conf"])))
    parts.sort(key=lambda p: (p.box[1], p.box[0]))
    return PartsMap([int(size[0]), int(size[1])], px_per_mm, parts)


def compare(pmap: PartsMap, dets: list[dict], present_conf: float = 0.25, extra_conf: float = 0.6,
            valid: tuple[float, float, float, float] | None = None) -> list[PartFinding]:
    """dets: parts found on the board under inspection, already in the good board's picture coordinates.

    present_conf: a weak detection is enough to say a mapped part is still there (missing a real part is the
                  costly mistake, so the part gets the benefit of the doubt only from an actual detection).
    extra_conf:   calling something extra needs a confident detection.
    valid:        x0, y0, x1, y1 of the area the aligned picture actually covers; parts outside it, or within
                  EDGE_MM of its edge, are not judged."""
    margin = EDGE_MM * pmap.px_per_mm if pmap.px_per_mm else 8.0
    x0, y0, x1, y1 = valid if valid else (0.0, 0.0, float(pmap.size[0]), float(pmap.size[1]))

    def judged(box) -> bool:
        return box[0] >= x0 + margin and box[1] >= y0 + margin and box[2] <= x1 - margin and box[3] <= y1 - margin

    def mm(box) -> float | None:
        return round(max(box[2] - box[0], box[3] - box[1]) / pmap.px_per_mm, 2) if pmap.px_per_mm else None
    seen = [d for d in dets if d["conf"] >= present_conf]
    out: list[PartFinding] = []
    for p in pmap.parts:
        if judged(p.box) and not any(_same_place(p.box, d["bbox"]) for d in seen):
            out.append(PartFinding("missing_part", list(p.box), p.label, p.conf, mm(p.box)))
    for d in dedupe([d for d in dets if d["conf"] >= extra_conf]):
        if judged(d["bbox"]) and not any(_same_place(d["bbox"], p.box) for p in pmap.parts):
            out.append(PartFinding("extra_part", [float(v) for v in d["bbox"]], "part", float(d["conf"]), mm(d["bbox"])))
    return out


def warp_boxes(dets: list[dict], matrix) -> list[dict]:
    """Detections found in the inspected picture -> the good board's picture, through the 2x3 or 3x3 alignment
    matrix (inspected -> good). Each box becomes the upright box around its four moved corners."""
    import numpy as np
    m = np.asarray(matrix, float)
    if m.shape == (2, 3):
        m = np.vstack([m, [0, 0, 1]])
    out = []
    for d in dets:
        x0, y0, x1, y1 = d["bbox"]
        pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], float) @ m.T
        pts = pts[:, :2] / pts[:, 2:3]
        out.append({**d, "bbox": [float(pts[:, 0].min()), float(pts[:, 1].min()),
                                  float(pts[:, 0].max()), float(pts[:, 1].max())]})
    return out
