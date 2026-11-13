"""Fuse the learned detector with the golden-board difference channel.

Each golden difference that overlaps a detection (IoU >= 0.1, or either box's centre inside the
other) is merged with it: evidence = max(detector confidence, difference score); the detector's
class name is kept (it was trained on labelled defects), the topology class is kept alongside.
Differences the detector did not see become candidates of their own (source "golden"), named by
topology. Detections with no difference stay as they are: on a board whose golden comparison is
unavailable or refused, the station is exactly the detector-only station.
"""
from __future__ import annotations


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _inside(p, box) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def _overlap(a, b) -> bool:
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return _iou(a, b) >= 0.1 or _inside(ca, b) or _inside(cb, a)


def fuse(dets: list[dict], diffs) -> list[dict]:
    """dets: [{bbox, conf, cls}]; diffs: golden_diff.DiffCandidate list. Returns fused candidates
    [{bbox, conf, cls, source, det_conf, diff_score, topo_cls, diff_size_px}]."""
    out = [{**d, "bbox": tuple(d["bbox"]), "source": "detector", "det_conf": d["conf"], "diff_score": None,
            "topo_cls": None, "diff_size_px": None} for d in dets]
    for c in diffs:
        m = [o for o in out if o["source"] != "golden" and _overlap(o["bbox"], c.bbox)]
        if m:
            for o in m:
                if o["diff_score"] is None or c.score > o["diff_score"]:
                    o.update(diff_score=c.score, topo_cls=c.cls, diff_size_px=c.size_px)
                o["conf"] = max(o["conf"], c.score)
                o["source"] = "detector+golden"
                # the finding covers BOTH pieces of evidence (a detector box can sit beside the change)
                b = o["bbox"]
                o["bbox"] = (min(b[0], c.bbox[0]), min(b[1], c.bbox[1]), max(b[2], c.bbox[2]), max(b[3], c.bbox[3]))
        else:
            out.append({"bbox": tuple(c.bbox), "conf": c.score, "cls": c.cls, "source": "golden",
                        "det_conf": None, "diff_score": c.score, "topo_cls": c.cls, "diff_size_px": c.size_px})
    return out
