"""Severity and disposition: what a defect means for THIS customer's IPC class.

For every defect the engine produces:
  ipc          per IPC class: 'accept' | 'reject' | 'uncertain' (the limit lies within the
               measurement's +/-1 pixel uncertainty: a person must decide)
  disposition  for the design's customer class: ACCEPT | REWORK | SCRAP | REVIEW
  severity     a sort key derived from the lowest class that rejects it:
               critical = rejected even at Class 1, major = from Class 2, minor = Class 3 only
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .record import Defect

IPC_CLASSES = ("class1", "class2", "class3")
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "severity.yaml"
DISPOSITION_RANK = {"ACCEPT": 0, "REVIEW": 1, "REWORK": 2, "SCRAP": 3}


@dataclass
class DesignRules:
    min_trace_width_mm: float
    min_spacing_mm: float
    customer_class: int = 2
    rules_from_design: bool = True   # False = still on the generic default rules


class SeverityEngine:
    def __init__(self, config: dict | None = None, config_path: Path = DEFAULT_CONFIG):
        self.cfg = config if config is not None else yaml.safe_load(Path(config_path).read_text())
        dr = self.cfg["default_design_rules"]
        self.default_rules = DesignRules(dr["min_trace_width_mm"], dr["min_spacing_mm"], 2, False)

    # ------------------------------------------------------------------ public
    def grade(self, d: Defect, rules: DesignRules | None = None, measurement=None) -> Defect:
        rules = rules or self.default_rules
        spec = self.cfg["classes"].get(d.cls)
        if spec is None:  # operator relabel to a type we have no rule for: be conservative
            spec = {"rule": "reject_all", "fallback_severity": "critical",
                    "repair": {c: "review" for c in IPC_CLASSES}}
        rule = spec["rule"]

        if measurement is not None and rule in ("width_reduction", "spacing_reduction", "fixed"):
            self._grade_measured(d, measurement)
        elif rule == "reject_all":
            d.ipc = {c: "reject" for c in IPC_CLASSES}
            d.severity_reason = f"{d.cls.replace('_', ' ')} is a functional defect: rejected at every IPC class"
        elif rule in ("width_reduction", "spacing_reduction", "fixed"):
            d.ipc = self._ipc_from_severity(spec["fallback_severity"])
            why = "no golden-board measurement" if rule != "fixed" else "graded by defect type"
            d.severity_reason = f"default for {d.cls.replace('_', ' ')} ({why})"
        elif rule == "area":
            sev = self._anomaly_severity(d)
            d.ipc = self._ipc_from_severity(sev)
        else:
            raise ValueError(f"unknown severity rule {rule!r} for class {d.cls}")

        d.severity = self._severity_from_ipc(d.ipc)
        d.disposition = self._disposition(d, spec, rules.customer_class)
        return d

    @staticmethod
    def board_disposition(defects: list[Defect]) -> str:
        if not defects:
            return "ACCEPT"
        return max((d.disposition or "REVIEW" for d in defects), key=DISPOSITION_RANK.get)

    # ------------------------------------------------------------------ helpers
    def _grade_measured(self, d: Defect, m) -> None:
        limits = self.cfg["width_reduction_limit_pct" if m.kind == "width" else "spacing_reduction_limit_pct"]
        d.ipc = {}
        for c in IPC_CLASSES:
            lim = limits[c]
            lim = 100.0 if lim is None else float(lim)      # functional only: cut through / bridged
            d.ipc[c] = m.decide(lim)
        d.width_mm = round(m.defect_mm, 3)
        d.uncertainty_mm = round(m.px_mm, 3)
        d.severity_reason = m.text()

    def _anomaly_severity(self, d: Defect) -> str:
        if d.area_mm2 is not None:
            t = self.cfg["anomaly_area_mm2"]
            d.severity_reason = f"unnamed anomaly covering {d.area_mm2:.2f} mm2"
            return "minor" if d.area_mm2 < t["minor_below"] else (
                "major" if d.area_mm2 < t["major_below"] else "critical")
        t = self.cfg["anomaly_score"]
        d.severity_reason = f"unnamed anomaly, score {d.confidence:.2f} (no mm scale)"
        return "minor" if d.confidence < t["minor_below"] else (
            "major" if d.confidence < t["major_below"] else "critical")

    @staticmethod
    def _disposition(d: Defect, spec: dict, customer_class: int) -> str:
        key = f"class{customer_class}"
        call = d.ipc.get(key, "reject")
        if call == "accept":
            return "ACCEPT"
        if call == "uncertain":
            return "REVIEW"
        repair = spec.get("repair", {}).get(key, "review")
        return {"rework": "REWORK", "scrap": "SCRAP"}.get(repair, "REVIEW")

    @staticmethod
    def _severity_from_ipc(ipc: dict[str, str]) -> str:
        if ipc["class1"] != "accept":
            return "critical"
        if ipc["class2"] != "accept":
            return "major"
        return "minor"

    @staticmethod
    def _ipc_from_severity(sev: str) -> dict[str, str]:
        reject_from = {"critical": 1, "major": 2, "minor": 3}[sev]
        return {f"class{i}": ("reject" if i >= reject_from else "accept") for i in (1, 2, 3)}
