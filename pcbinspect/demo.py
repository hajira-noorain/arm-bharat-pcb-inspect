"""Demo mode: the held-out test boards a visitor can pick, how findings are matched to a board's known answer,
safe decoding of an uploaded photo, and when the shared demo log resets."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

MAX_UPLOAD_BYTES = 10_000_000
MAX_UPLOAD_SIDE = 4000


@dataclass
class Sample:
    file: str
    design: str
    kind: str                     # "bare" | "assembled"
    good: bool
    boxes: list[list[int]] = field(default_factory=list)
    has_golden: bool = False


def load_samples(root: Path, per_design: int = 4) -> list[Sample]:
    """Test boards from <root>/demo/manifest.json whose picture exists: per design, defective boards first,
    then one good board if there is one, at most `per_design` in all. Order is stable."""
    root = Path(root)
    man = root / "demo" / "manifest.json"
    if not man.is_file():
        return []
    boards = root / "demo" / "boards"
    by_design: dict[str, list[dict]] = {}
    for e in json.loads(man.read_text(encoding="utf-8")):
        if (boards / e["file"]).is_file():
            by_design.setdefault(e["design"], []).append(e)
    out = []
    for design, items in by_design.items():
        bad = [e for e in items if not e.get("good")]
        good = [e for e in items if e.get("good")]
        pick = bad[:per_design - 1] + good[:1] if good else bad[:per_design]
        for e in pick:
            stem = Path(e["file"]).stem
            out.append(Sample(e["file"], design, "assembled" if design.startswith("visa_") else "bare",
                              bool(e.get("good")), [list(map(int, b["bbox"] if isinstance(b, dict) else b))
                                                    for b in e.get("boxes") or []],
                              (boards / f"{stem}.golden.jpg").is_file()))
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _hits(f, k) -> bool:
    """Same rule as the evaluation (training/eval_golden.py): overlap of at least 0.1, or the finding's centre
    inside the marked box."""
    cx, cy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
    return _iou(f, k) >= 0.1 or (k[0] <= cx <= k[2] and k[1] <= cy <= k[3])


def match_known(found: list, known: list) -> dict:
    """How many of the dataset's marked defects were found, and how many findings match no marked defect."""
    n_found = sum(any(_hits(f, k) for f in found) for k in known)
    extra = sum(not any(_hits(f, k) for k in known) for f in found)
    return {"marked": len(known), "found": n_found, "extra": extra}


class UploadError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def decode_upload(data: bytes, max_bytes: int = MAX_UPLOAD_BYTES, max_side: int = MAX_UPLOAD_SIDE) -> np.ndarray:
    """An uploaded photo as 8-bit BGR, or UploadError. Only JPEG and PNG, judged by their bytes, not the name."""
    if len(data) > max_bytes:
        raise UploadError(413, f"the photo is larger than {max_bytes // 1_000_000} MB")
    if not (data[:3] == b"\xff\xd8\xff" or data[:4] == b"\x89PNG"):
        raise UploadError(400, "upload a JPEG or PNG photo")
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise UploadError(400, "this file could not be read as a photo")
    if max(im.shape[:2]) > max_side:
        raise UploadError(400, f"the photo is larger than {max_side} pixels on a side")
    if im.dtype == np.uint16:
        im = (im >> 8).astype(np.uint8)
    elif im.dtype != np.uint8:
        raise UploadError(400, "unsupported image depth")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(im)


def seconds_until(hour: int, now: datetime) -> float:
    """Seconds from `now` to the next hour:00 (a full day if it is exactly that time now)."""
    nxt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if nxt <= now:
        nxt += timedelta(days=1)
    return (nxt - now).total_seconds()
