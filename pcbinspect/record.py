"""The structured defect record: the single contract between vision, severity, gate, log and Llama.

The PS5 mentor guidance is "give Llama structured findings, not raw pixels" — this module is that structure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

# Bare-board classes (DeepPCB + PKU/HRIPCB, unified). Order == YOLO class index. Never reorder.
BARE_CLASSES: tuple[str, ...] = (
    "missing_hole",
    "mouse_bite",
    "open",
    "short",
    "spur",
    "spurious_copper",
    "pin_hole",
    "missing_pad",   # damaged / partly etched-away pad (Mendeley PCB-Defect real boards)
)
# Anything the anomaly model flags that the detector cannot name.
ANOMALY_CLASS = "unknown_anomaly"

Source = Literal["detector", "anomaly"]
Severity = Literal["minor", "major", "critical"]
Verdict = Literal["PASS", "FAIL", "REVIEW"]


@dataclass
class Defect:
    id: str                      # "D1", "D2", ... stable within one inspection
    cls: str                     # one of BARE_CLASSES or ANOMALY_CLASS (or an operator relabel)
    source: Source
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in full-image pixels
    confidence: float            # detector confidence, or normalised anomaly score in [0, 1]
    size_mm: float | None = None     # longest bbox side in mm (None if scale unknown)
    width_mm: float | None = None    # shortest bbox side in mm — proxy for depth of a nick / width of a bridge
    area_mm2: float | None = None
    location: str = ""               # coarse zone, e.g. "top-left"
    severity: Severity | None = None
    ipc: dict[str, str] = field(default_factory=dict)   # {"class1": "accept", "class2": "reject", ...}
    severity_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


@dataclass
class Inspection:
    board_id: str
    design: str
    image_path: str
    defects: list[Defect]
    board_score: float           # max defect evidence on the board, used by the conformal gate
    verdict: Verdict
    alpha: float                 # miss-rate target the gate was operating at
    mm_per_px: float | None
    latency_ms: dict[str, float] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)

    def to_llm_record(self) -> dict:
        """The compact, numbers-only record Llama sees. No pixels, no paths."""
        return {
            "board_id": self.board_id,
            "design": self.design,
            "verdict": self.verdict,
            "defects": [
                {
                    "id": d.id,
                    "type": d.cls,
                    "severity": d.severity,
                    "confidence": round(d.confidence, 2),
                    "size_mm": None if d.size_mm is None else round(d.size_mm, 2),
                    "location": d.location,
                    "ipc": d.ipc,
                }
                for d in self.defects
            ],
        }


def zone_of(bbox: tuple[int, int, int, int], img_w: int, img_h: int) -> str:
    """3x3 coarse location name for operator-facing text."""
    cx = (bbox[0] + bbox[2]) / 2 / max(img_w, 1)
    cy = (bbox[1] + bbox[3]) / 2 / max(img_h, 1)
    col = "left" if cx < 1 / 3 else ("centre" if cx < 2 / 3 else "right")
    row = "top" if cy < 1 / 3 else ("middle" if cy < 2 / 3 else "bottom")
    return "centre" if (row, col) == ("middle", "centre") else f"{row}-{col}"
