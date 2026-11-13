"""Severity engine: class criticality x size in mm vs IPC-style limits x location.

Severity levels are derived from the IPC-class outcome, so they mean something concrete:
  critical -> rejected even at Class 1 (general electronics): the board will not work
  major    -> acceptable at Class 1, rejected at Class 2 (dedicated service, the usual default)
  minor    -> acceptable at Class 2, rejected only at Class 3 (high reliability) — or accepted everywhere
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .record import Defect

IPC_CLASSES = ("class1", "class2", "class3")
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "severity.yaml"


@dataclass
class DesignRules:
    min_trace_width_mm: float
    min_spacing_mm: float


class SeverityEngine:
    def __init__(self, config: dict | None = None, config_path: Path = DEFAULT_CONFIG):
        self.cfg = config if config is not None else yaml.safe_load(Path(config_path).read_text())
        dr = self.cfg["default_design_rules"]
        self.default_rules = DesignRules(dr["min_trace_width_mm"], dr["min_spacing_mm"])

    def grade(self, d: Defect, rules: DesignRules | None = None, measurement=None) -> Defect:
        """Fill d.severity, d.ipc and d.severity_reason in place and return d.

        measurement: a golden.Measurement (golden-board comparison). When present it replaces the
        bounding-box proxy, because it measures the copper actually lost or added.
        """
        rules = rules or self.default_rules
        spec = self.cfg["classes"].get(d.cls)
        if spec is None:  # operator relabel to a class we have no rule for: be conservative
            spec = {"rule": "reject_all", "fallback_severity": "critical"}
        rule = spec["rule"]

        if measurement is not None and rule in ("width_reduction", "spacing_reduction", "fixed"):
            limits = self.cfg["width_reduction_limit_pct" if measurement.kind == "width"
                              else "spacing_reduction_limit_pct"]
            d.ipc = {c: ("reject" if measurement.pct > limits[c] else "accept") for c in IPC_CLASSES}
            d.severity = self._severity_from_ipc(d.ipc)
            d.width_mm = round(measurement.defect_mm, 3)
            d.severity_reason = measurement.text()
            return d

        if rule == "reject_all":
            d.ipc = {c: "reject" for c in IPC_CLASSES}
            d.severity = "critical"
            d.severity_reason = f"{d.cls} is a functional defect: rejected at every IPC class"
            return d

        if rule == "fixed":
            return self._fallback(d, spec, "graded by defect type")

        if rule in ("width_reduction", "spacing_reduction"):
            if d.width_mm is None:
                return self._fallback(d, spec, "no mm scale available")
            nominal = rules.min_trace_width_mm if rule == "width_reduction" else rules.min_spacing_mm
            limits = self.cfg[f"{rule}_limit_pct"]
            pct = 100.0 * d.width_mm / nominal
            d.ipc = {c: ("reject" if pct > limits[c] else "accept") for c in IPC_CLASSES}
            what = "conductor width" if rule == "width_reduction" else "conductor spacing"
            d.severity = self._severity_from_ipc(d.ipc)
            d.severity_reason = (
                f"{d.width_mm:.2f} mm reduces {what} ({nominal:.2f} mm nominal) by {pct:.0f}%"
            )
            return d

        if rule == "area":
            if d.area_mm2 is not None:
                t = self.cfg["anomaly_area_mm2"]
                sev = "minor" if d.area_mm2 < t["minor_below"] else (
                    "major" if d.area_mm2 < t["major_below"] else "critical")
                d.severity_reason = f"unnamed anomaly covering {d.area_mm2:.2f} mm2"
            else:
                t = self.cfg["anomaly_score"]
                sev = "minor" if d.confidence < t["minor_below"] else (
                    "major" if d.confidence < t["major_below"] else "critical")
                d.severity_reason = f"unnamed anomaly, score {d.confidence:.2f} (no mm scale)"
            d.severity = sev
            d.ipc = self._ipc_from_severity(sev)
            return d

        raise ValueError(f"unknown severity rule {rule!r} for class {d.cls}")

    def _fallback(self, d: Defect, spec: dict, why: str) -> Defect:
        d.severity = spec["fallback_severity"]
        d.ipc = self._ipc_from_severity(d.severity)
        d.severity_reason = f"default severity for {d.cls} ({why})"
        return d

    @staticmethod
    def _severity_from_ipc(ipc: dict[str, str]) -> str:
        if ipc["class1"] == "reject":
            return "critical"
        if ipc["class2"] == "reject":
            return "major"
        return "minor"

    @staticmethod
    def _ipc_from_severity(sev: str) -> dict[str, str]:
        reject_from = {"critical": 1, "major": 2, "minor": 3}[sev]
        return {f"class{i}": ("reject" if i >= reject_from else "accept") for i in (1, 2, 3)}
