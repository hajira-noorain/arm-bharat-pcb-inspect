"""Parts check for assembled boards: is every part of the good board still there, and is anything extra?

When a good assembled board is taught, the station finds its parts once and stores a PARTS MAP: where each part
sits (in the good board's own picture) and what kind it is. At inspection the new picture is lined up with the
good one, parts are found again, and the two lists are compared:

  missing   a part on the map with no part found at its place, AND the place no longer looks like the good board
  extra     a part found where the map has none
  changed   a part is found at its place, but the place looks clearly unlike the good board (a different part,
            a part turned round or standing up)

Two independent cues are used, as in the component-window comparison of classic optical inspection: the part
detector (is something part-like here?) and a direct comparison of each part's picture with the good board's
(does this place look the same?). A part is called missing only when both agree, so one cue having a bad moment
does not raise a false call. Comparing component by component is the published remedy for "unusual area" models
missing absent or swapped parts (Kim et al. 2024, arXiv 2312.13783; Bergmann et al. 2022, MVTec LOCO AD).

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
    sim_min: float | None = None     # the lowest likeness to the reference this part showed on OTHER good boards

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
    scale: float = 1.0               # the picture is resampled by this before the detector looks at it; the
    #                                  SAME factor is used when the board is taught and when it is inspected
    ignore: list[list[float]] = field(default_factory=list)   # places where good boards themselves sometimes
    #                                  show a "part" (a mark, a pad, a label): never reported as unexpected
    taught_from: int = 1             # good pictures the map was checked against
    dropped: int = 0                 # parts left off because the good boards did not show them every time

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(path: Path) -> "PartsMap":
        d = json.loads(Path(path).read_text())
        return PartsMap(d["size"], d.get("px_per_mm"), [Part(**p) for p in d["parts"]], d.get("scale", 1.0),
                        d.get("ignore", []), d.get("taught_from", 1), d.get("dropped", 0))

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for p in self.parts:
            kinds[p.label] = kinds.get(p.label, 0) + 1
        return {"parts": len(self.parts), "by_kind": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
                "good_pictures": self.taught_from, "left_off_as_unreliable": self.dropped}


@dataclass
class PartFinding:
    what: str                        # "missing_part" or "extra_part"
    box: list[float]                 # in the good board's picture
    kind: str                        # "capacitor", ... or "part"
    confidence: float                # missing: how sure the map was of this part; extra: the detector's confidence
    size_mm: float | None = None
    likeness: float | None = None    # how much the place looks like the good board there (1 = identical)


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


LIKE_DEFAULT = 0.5       # likeness below this = "does not look like the good board here" (no good-board range known)
LIKE_MARGIN = 0.25       # ... or this far below the lowest likeness the part showed across the good boards
CHANGED_EXTRA = 0.2      # a part that IS detected must look this much worse still before it is called changed


def unlike_below(p: Part) -> float:
    """The likeness under which this part's place counts as not looking like the good board."""
    if p.sim_min is None:
        return LIKE_DEFAULT
    return min(0.6, max(0.2, p.sim_min - LIKE_MARGIN))


def likeness(ref_gray, test_gray, box, search: int | None = None) -> float | None:
    """How much the inspected picture looks like the good one at this part: normalised cross-correlation of the
    good board's picture of the part against the inspected picture, best over a small shift (alignment is never
    perfect, and a tall part leans with its distance from the camera axis). Both pictures are grey and in the
    same frame (grey or colour: both work). None when the part is too small or too plain to compare."""
    import cv2
    h, w = test_gray.shape[:2]
    # the part with a rim of board around it: the rim holds the part's outline, and the pads and print that
    # show when the part is gone (a plain-topped part alone has nothing to compare)
    gx, gy = 0.2 * (box[2] - box[0]) + 4, 0.2 * (box[3] - box[1]) + 4
    x0, y0 = int(max(box[0] - gx, 0)), int(max(box[1] - gy, 0))
    x1, y1 = int(min(box[2] + gx, w)), int(min(box[3] + gy, h))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    tpl = ref_gray[y0:y1, x0:x1]
    if float(tpl.std()) < 4.0:
        return None
    s = search if search is not None else int(min(max(0.15 * max(x1 - x0, y1 - y0), 4), 20))
    win = test_gray[max(y0 - s, 0):min(y1 + s, h), max(x0 - s, 0):min(x1 + s, w)]
    if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
        return None
    return float(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED).max())


def compare(pmap: PartsMap, dets: list[dict], present_conf: float = 0.25, extra_conf: float = 0.6,
            valid: tuple[float, float, float, float] | None = None,
            likes: list[float | None] | None = None) -> list[PartFinding]:
    """dets: parts found on the board under inspection, already in the good board's picture coordinates.
    likes: for each part of the map, how much its place looks like the good board (likeness()), if the
           pictures were compared; without it the detector alone decides.

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
    for i, p in enumerate(pmap.parts):
        if not judged(p.box):
            continue
        like = likes[i] if likes is not None else None
        if not any(_same_place(p.box, d["bbox"]) for d in seen):
            # nothing part-like found here. If the place still looks like the good board, the detector had a
            # bad moment and nothing is reported; if the pictures could not be compared, the detector decides.
            if like is None or like < unlike_below(p):
                out.append(PartFinding("missing_part", list(p.box), p.label, p.conf, mm(p.box), like))
        elif like is not None and like < unlike_below(p) - CHANGED_EXTRA:
            out.append(PartFinding("changed_part", list(p.box), p.label, p.conf, mm(p.box), like))
    for d in dedupe([d for d in dets if d["conf"] >= extra_conf]):
        if any(_same_place(d["bbox"], z) for z in pmap.ignore):
            continue
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


# ------------------------------------------------------------------ running the models
TRAIN_PX_PER_MM = 16.0   # the scale the part detector was trained at (training/prepare_components.py)


def scale_for(mm_per_px: float | None) -> float:
    """The resampling factor that brings a picture to the detector's training scale (1.0 if unknown or close)."""
    if not mm_per_px:
        return 1.0
    s = TRAIN_PX_PER_MM * mm_per_px                           # (px/mm wanted) / (px/mm here)
    return 1.0 if 0.8 <= s <= 1.25 else min(max(s, 0.25), 3.0)


def find_parts(detector, bgr, mm_per_px: float | None = None, scale: float | None = None) -> tuple[list[dict], float]:
    """Parts on a picture, as [{"bbox", "conf"}] in that picture's own pixels, and the time taken (ms).
    The picture is resampled first so a part looks the size the detector learned it at: by `scale` when the
    taught map fixes it, else from the picture's known mm scale."""
    import time

    import cv2
    t = time.perf_counter()
    s = float(scale) if scale else scale_for(mm_per_px)
    img = bgr if s == 1.0 else cv2.resize(bgr, None, fx=s, fy=s,
                                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    dets, _ = detector.detect(img)
    out = [{"bbox": [v / s for v in d.bbox], "conf": float(d.conf)} for d in dets]
    return dedupe(out), (time.perf_counter() - t) * 1e3


def part_crops(bgr, dets: list[dict], size: int = 64, margin: float = 1.4):
    """Square crops around each part, the way the classifier's training crops were cut."""
    import cv2
    import numpy as np
    out = []
    for d in dets:
        x0, y0, x1, y1 = d["bbox"]
        side = max(x1 - x0, y1 - y0) * margin
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        xa, ya = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        crop = bgr[ya:int(cy + side / 2), xa:int(cx + side / 2)]
        if crop.size == 0:
            crop = np.zeros((size, size, 3), np.uint8)
        out.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA))
    return out


class PartKindClassifier:
    """The part-kind classifier exported to ExecuTorch. Used once per taught board, never during inspection."""

    def __init__(self, pte_path, names: list[str], batch: int = 1):
        from executorch.runtime import Runtime
        self.names = list(names)
        self._m = Runtime.get().load_program(str(pte_path)).load_method("forward")

    def __call__(self, crops) -> list[tuple[str, float]]:
        import numpy as np
        import torch
        out = []
        for c in crops:
            x = np.ascontiguousarray(c[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
            p = np.asarray(self._m.execute([torch.from_numpy(x)])[0]).reshape(-1)
            if p.min() < 0 or abs(float(p.sum()) - 1.0) > 1e-3:       # logits, not probabilities
                e = np.exp(p - p.max())
                p = e / e.sum()
            i = int(p.argmax())
            out.append((self.names[i], float(p[i])))
        return out


SEARCH_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)     # tried when the picture's mm scale is not known


def teach_parts(detector, classifier, goods, mm_per_px: float | None = None, max_goods: int = 4) -> PartsMap:
    """The parts map of a good board, from one good picture or several.

    goods[0] is the reference picture. With more good pictures the map is CHECKED against them: each is lined
    up with the reference and searched for parts, and
      - a part the reference shows but another good picture does not is left off the map (the detector does
        not see it reliably, so it would later be reported missing on good boards);
      - a place where another good picture shows a confident part and the reference has none becomes an
        ignore zone (it would later be reported as unexpected on good boards).
    When the mm scale is unknown, a few resampling factors are tried and the one that leaves the most
    reliable parts is kept; that factor is stored and reused at inspection."""
    import numpy as np
    goods = [goods] if isinstance(goods, np.ndarray) else list(goods)
    ref, others = goods[0], goods[1:max_goods]
    h, w = ref.shape[:2]
    warps = []
    if others:
        from .golden import MIN_ALIGN_CC, align_warp
        for g in others:
            warp, cc = align_warp(g, ref)
            if warp is not None and cc >= MIN_ALIGN_CC:
                warps.append((g, warp))
    best = None
    for s in ([scale_for(mm_per_px)] if mm_per_px else SEARCH_SCALES):
        dets, _ = find_parts(detector, ref, scale=s)
        m = build_map(dets, (w, h), (1.0 / mm_per_px) if mm_per_px else None)
        m.scale, m.taught_from = s, 1 + len(warps)
        keep, zones = set(range(len(m.parts))), []
        for g, warp in warps:
            seen, _ = find_parts(detector, g, scale=s)
            for f in check_board(m, seen, warp, (g.shape[1], g.shape[0]), raw=True):
                if f[0] == "missing_part":
                    keep.discard(f[1])
                else:
                    zones.append(f[1])
        m.dropped = len(m.parts) - len(keep)
        m.parts, m.ignore = [p for i, p in enumerate(m.parts) if i in keep], zones
        for g, warp in warps:          # how alike each part looks from one good board to the next
            there = moved(m, np.linalg.inv(np.vstack([np.asarray(warp, float), [0.0, 0.0, 1.0]]))
                          @ np.diag([g.shape[1] / w, g.shape[0] / h, 1.0]), (g.shape[1], g.shape[0]))
            for p, v in zip(m.parts, likenesses(there, np.asarray(warp, np.float32), ref, g)):
                if v is not None:
                    p.sim_min = v if p.sim_min is None else min(p.sim_min, v)
        # most reliable parts wins; on a tie, the factor closest to leaving the picture as it is
        if best is None or (len(m.parts), -abs(np.log(s))) > (len(best.parts), -abs(np.log(best.scale))):
            best = m
    if classifier is not None and best.parts:
        named = classifier(part_crops(ref, [{"bbox": p.box} for p in best.parts]))
        for p, (k, kc) in zip(best.parts, named):
            p.kind, p.kind_conf = k, float(kc)
    return best


def moved(pmap: PartsMap, matrix, size: tuple[int, int]) -> PartsMap:
    """The same map carried into another picture (the board under inspection) through a 3x3 matrix taking
    the good picture's points to that picture's points."""
    boxes = warp_boxes([{"bbox": p.box, "conf": p.conf} for p in pmap.parts], matrix)
    parts = [Part(b["bbox"], p.kind, p.kind_conf, p.conf, p.sim_min) for p, b in zip(pmap.parts, boxes)]
    zones = [b["bbox"] for b in warp_boxes([{"bbox": z} for z in pmap.ignore], matrix)]
    return PartsMap([int(size[0]), int(size[1])], pmap.px_per_mm, parts, pmap.scale, zones)


def likenesses(there: PartsMap, warp, ref_bgr, test_bgr) -> list[float | None]:
    """likeness() for every part of a map already carried into the inspected picture."""
    import cv2
    h, w = test_bgr.shape[:2]
    ref = ref_bgr if ref_bgr.shape[:2] == (h, w) else cv2.resize(ref_bgr, (w, h))
    ref = cv2.warpAffine(ref, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REPLICATE)
    # compared in colour: a part swapped for one of another colour but similar brightness must not look alike
    return [likeness(ref, test_bgr, p.box) for p in there.parts]


def check_board(pmap: PartsMap, dets: list[dict], warp, test_size: tuple[int, int], raw: bool = False,
                ref=None, test=None):
    """Missing, changed and extra parts on the inspected picture, in ITS pixels. warp: golden.align_warp's 2x3
    matrix (inspected -> good, with the good picture resized to the inspected one's size).
    ref, test: the good board's picture and the inspected one; when given, each part's place is also compared
    with the good board (see compare()).
    raw (used while teaching): [("missing_part", index of the map part) | ("extra_part", box in the GOOD
    picture)] instead of findings."""
    import numpy as np
    w, h = test_size
    to_test = np.linalg.inv(np.vstack([np.asarray(warp, float), [0.0, 0.0, 1.0]]))
    resize = np.diag([w / pmap.size[0], h / pmap.size[1], 1.0])        # good picture -> its resized self
    there = moved(pmap, to_test @ resize, (w, h))
    likes = likenesses(there, np.asarray(warp, np.float32), ref, test) if ref is not None and test is not None else None
    found = compare(there, dets, likes=likes)
    if not raw:
        return found
    back = np.linalg.inv(to_test @ resize)
    out = []
    for f in found:
        if f.what == "missing_part":
            out.append(("missing_part", next(i for i, p in enumerate(there.parts) if p.box == f.box)))
        else:
            out.append(("extra_part", warp_boxes([{"bbox": f.box}], back)[0]["bbox"]))
    return out
