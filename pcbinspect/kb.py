"""Loads the defect knowledge base and serves exact-match retrieval by defect type."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_KB = Path(__file__).resolve().parent.parent / "kb" / "defects.yaml"


class KnowledgeBase:
    def __init__(self, path: Path = DEFAULT_KB):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.process_steps: dict[str, str] = raw["process_steps"]
        self.defects: dict[str, dict] = raw["defects"]
        self.by_id: dict[str, dict] = {}
        for cls, entry in self.defects.items():
            for kind in ("causes", "actions"):
                for item in entry[kind]:
                    if item["id"] in self.by_id:
                        raise ValueError(f"duplicate KB id {item['id']}")
                    self.by_id[item["id"]] = {**item, "cls": cls, "kind": kind[:-1]}

    def entry(self, cls: str) -> dict:
        return self.defects.get(cls) or self.defects["unknown_anomaly"]

    def allowed_ids(self, cls: str) -> tuple[set[str], set[str]]:
        e = self.entry(cls)
        return {c["id"] for c in e["causes"]}, {a["id"] for a in e["actions"]}

    def text(self, kb_id: str) -> str:
        return self.by_id[kb_id]["text"]

    def display_name(self, cls: str) -> str:
        return self.entry(cls)["name"]

    def all_class_names(self) -> dict[str, list[str]]:
        """Words that identify each defect type in free text (used by the grounding checker)."""
        out = {}
        for cls, e in self.defects.items():
            words = {cls.replace("_", " "), e["name"].split("(")[0].strip().lower()}
            out[cls] = sorted(w.lower() for w in words if w)
        return out


@lru_cache(maxsize=1)
def default_kb() -> KnowledgeBase:
    return KnowledgeBase()
