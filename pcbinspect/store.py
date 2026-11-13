"""Offline inspection log: SQLite (WAL) + content-addressed image store + CSV export.

Every inspection is committed BEFORE the UI/alerts update, so a power cut never loses a verdict
that an operator has already seen.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from .record import Inspection

SCHEMA = """
CREATE TABLE IF NOT EXISTS designs (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('bare', 'assembled')),
    min_trace_width_mm REAL,
    min_spacing_mm REAL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    board_id TEXT NOT NULL,
    design TEXT NOT NULL,
    image_sha TEXT,
    verdict TEXT NOT NULL CHECK (verdict IN ('PASS', 'FAIL', 'REVIEW')),
    board_score REAL NOT NULL,
    alpha REAL NOT NULL,
    mm_per_px REAL,
    latency_json TEXT,
    model_versions_json TEXT,
    final_verdict TEXT,           -- operator's decision for REVIEW boards (or an override)
    disposition TEXT,             -- ACCEPT | REWORK | SCRAP | REVIEW for the customer class
    customer_class INTEGER
);
CREATE INDEX IF NOT EXISTS ix_insp_ts ON inspections(ts);
CREATE INDEX IF NOT EXISTS ix_insp_board ON inspections(board_id);
CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL REFERENCES inspections(id),
    defect_key TEXT NOT NULL,
    cls TEXT NOT NULL,
    source TEXT NOT NULL,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    confidence REAL,
    size_mm REAL, width_mm REAL, area_mm2 REAL,
    location TEXT,
    severity TEXT,
    ipc_json TEXT,
    severity_reason TEXT,
    operator_label TEXT,          -- 'confirmed' | 'false_alarm' | 'relabel:<cls>' | NULL
    disposition TEXT,
    uncertainty_mm REAL
);
CREATE INDEX IF NOT EXISTS ix_def_insp ON defects(inspection_id);
CREATE INDEX IF NOT EXISTS ix_def_cls ON defects(cls);
CREATE TABLE IF NOT EXISTS advice (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL REFERENCES inspections(id),
    ts REAL NOT NULL,
    backend TEXT NOT NULL,        -- 'llama-executorch' | 'template'
    model TEXT,
    advice_json TEXT NOT NULL,
    grounded INTEGER NOT NULL,
    rejected_reason TEXT,         -- why the Llama output was rejected, if it was
    latency_ms REAL,
    prefill_tps REAL,
    decode_tps REAL,
    operator_rating INTEGER       -- +1 / -1 / NULL
);
CREATE TABLE IF NOT EXISTS feedback_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    inspection_id INTEGER,
    defect_id INTEGER,
    action TEXT NOT NULL,
    payload_json TEXT,
    undone INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    version TEXT NOT NULL,
    metrics_json TEXT,
    active INTEGER NOT NULL DEFAULT 0,
    note TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    design TEXT,
    rule TEXT NOT NULL,
    message TEXT NOT NULL,
    data_json TEXT,
    acknowledged INTEGER NOT NULL DEFAULT 0
);
"""


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(exist_ok=True)
        self.db_path = self.root / "inspections.db"
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a station database was created (idempotent)."""
        for table, col, typ in (("inspections", "disposition", "TEXT"),
                                ("inspections", "customer_class", "INTEGER"),
                                ("defects", "disposition", "TEXT"),
                                ("defects", "uncertainty_mm", "REAL")):
            cols = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    # ---------- images ----------
    def put_image(self, data: bytes, ext: str = ".jpg") -> str:
        sha = hashlib.sha256(data).hexdigest()
        p = self.image_path(sha, ext)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(p)
        return sha

    def image_path(self, sha: str, ext: str = ".jpg") -> Path:
        return self.root / "images" / sha[:2] / f"{sha}{ext}"

    # ---------- designs ----------
    def upsert_design(self, name: str, kind: str, min_trace_width_mm: float | None = None,
                      min_spacing_mm: float | None = None) -> None:
        self.conn.execute(
            "INSERT INTO designs(name, kind, min_trace_width_mm, min_spacing_mm, created) "
            "VALUES (?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET kind=excluded.kind, "
            "min_trace_width_mm=excluded.min_trace_width_mm, min_spacing_mm=excluded.min_spacing_mm",
            (name, kind, min_trace_width_mm, min_spacing_mm, time.time()))

    def design(self, name: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM designs WHERE name=?", (name,)).fetchone()
        return dict(r) if r else None

    # ---------- inspections ----------
    def log_inspection(self, insp: Inspection, image_sha: str | None) -> int:
        with self.conn:  # one transaction: the inspection and all its defects, or nothing
            self.conn.execute("BEGIN")
            cur = self.conn.execute(
                "INSERT INTO inspections(ts, board_id, design, image_sha, verdict, board_score, alpha, "
                "mm_per_px, latency_json, model_versions_json, disposition, customer_class) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), insp.board_id, insp.design, image_sha, insp.verdict, insp.board_score,
                 insp.alpha, insp.mm_per_px, json.dumps(insp.latency_ms),
                 json.dumps(insp.model_versions), insp.disposition, insp.customer_class))
            iid = cur.lastrowid
            for d in insp.defects:
                self.conn.execute(
                    "INSERT INTO defects(inspection_id, defect_key, cls, source, x1, y1, x2, y2, "
                    "confidence, size_mm, width_mm, area_mm2, location, severity, ipc_json, "
                    "severity_reason, disposition, uncertainty_mm) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (iid, d.id, d.cls, d.source, *d.bbox, d.confidence, d.size_mm, d.width_mm,
                     d.area_mm2, d.location, d.severity, json.dumps(d.ipc), d.severity_reason,
                     d.disposition, d.uncertainty_mm))
        return iid

    def inspection(self, iid: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM inspections WHERE id=?", (iid,)).fetchone()
        if not r:
            return None
        out = dict(r)
        out["defects"] = [dict(x) for x in self.conn.execute(
            "SELECT * FROM defects WHERE inspection_id=? ORDER BY id", (iid,))]
        out["advice"] = [dict(x) for x in self.conn.execute(
            "SELECT * FROM advice WHERE inspection_id=? ORDER BY id", (iid,))]
        return out

    def recent(self, limit: int = 50, design: str | None = None) -> list[dict]:
        q = "SELECT * FROM inspections"
        args: tuple = ()
        if design:
            q += " WHERE design=?"
            args = (design,)
        q += " ORDER BY id DESC LIMIT ?"
        return [dict(r) for r in self.conn.execute(q, args + (limit,))]

    def board_history(self, board_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM inspections WHERE board_id=? ORDER BY id", (board_id,))]

    def set_final_verdict(self, iid: int, verdict: str) -> None:
        if verdict not in ("PASS", "FAIL"):
            raise ValueError(verdict)
        self.conn.execute("UPDATE inspections SET final_verdict=? WHERE id=?", (verdict, iid))

    # ---------- advice ----------
    def log_advice(self, iid: int, backend: str, model: str | None, advice: dict, grounded: bool,
                   rejected_reason: str | None = None, latency_ms: float | None = None,
                   prefill_tps: float | None = None, decode_tps: float | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO advice(inspection_id, ts, backend, model, advice_json, grounded, "
            "rejected_reason, latency_ms, prefill_tps, decode_tps) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (iid, time.time(), backend, model, json.dumps(advice), int(grounded), rejected_reason,
             latency_ms, prefill_tps, decode_tps))
        return cur.lastrowid

    def rate_advice(self, advice_id: int, rating: int) -> None:
        if rating not in (-1, 1):
            raise ValueError(rating)
        self.conn.execute("UPDATE advice SET operator_rating=? WHERE id=?", (rating, advice_id))

    # ---------- feedback ----------
    def log_feedback(self, action: str, inspection_id: int | None = None,
                     defect_id: int | None = None, payload: dict | None = None) -> int:
        with self.conn:
            self.conn.execute("BEGIN")
            cur = self.conn.execute(
                "INSERT INTO feedback_events(ts, inspection_id, defect_id, action, payload_json) "
                "VALUES (?,?,?,?,?)",
                (time.time(), inspection_id, defect_id, action, json.dumps(payload or {})))
            if defect_id is not None and action in ("confirmed", "false_alarm") or (
                    defect_id is not None and action.startswith("relabel:")):
                self.conn.execute("UPDATE defects SET operator_label=? WHERE id=?",
                                  (action, defect_id))
        return cur.lastrowid

    def undo_feedback(self, event_id: int) -> dict:
        with self.conn:
            self.conn.execute("BEGIN")
            r = self.conn.execute("SELECT * FROM feedback_events WHERE id=?", (event_id,)).fetchone()
            if r is None or r["undone"]:
                raise ValueError(f"feedback event {event_id} missing or already undone")
            self.conn.execute("UPDATE feedback_events SET undone=1 WHERE id=?", (event_id,))
            if r["defect_id"] is not None:
                self.conn.execute("UPDATE defects SET operator_label=NULL WHERE id=?",
                                  (r["defect_id"],))
        return dict(r)

    # ---------- model versions ----------
    def record_model_version(self, kind: str, version: str, metrics: dict, active: bool,
                             note: str = "") -> None:
        with self.conn:
            self.conn.execute("BEGIN")
            if active:
                self.conn.execute("UPDATE model_versions SET active=0 WHERE kind=?", (kind,))
            self.conn.execute(
                "INSERT INTO model_versions(ts, kind, version, metrics_json, active, note) "
                "VALUES (?,?,?,?,?,?)",
                (time.time(), kind, version, json.dumps(metrics), int(active), note))

    def model_versions(self, kind: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM model_versions WHERE kind=? ORDER BY id", (kind,))]

    # ---------- alerts ----------
    def log_alert(self, design: str | None, rule: str, message: str, data: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO alerts(ts, design, rule, message, data_json) VALUES (?,?,?,?,?)",
            (time.time(), design, rule, message, json.dumps(data)))
        return cur.lastrowid

    def open_alerts(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM alerts WHERE acknowledged=0 ORDER BY id DESC")]

    # ---------- export ----------
    def export_csv(self, path: Path, since_ts: float = 0.0) -> int:
        rows = self.conn.execute(
            "SELECT i.id, datetime(i.ts,'unixepoch','localtime') AS time, i.board_id, i.design, "
            "i.verdict, i.final_verdict, i.board_score, i.alpha, d.defect_key, d.cls, d.severity, "
            "d.size_mm, d.location, d.confidence, d.operator_label "
            "FROM inspections i LEFT JOIN defects d ON d.inspection_id=i.id "
            "WHERE i.ts>=? ORDER BY i.id, d.id", (since_ts,)).fetchall()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(rows[0].keys() if rows else [
                "id", "time", "board_id", "design", "verdict", "final_verdict", "board_score",
                "alpha", "defect_key", "cls", "severity", "size_mm", "location", "confidence",
                "operator_label"])
            for r in rows:
                w.writerow(list(r))
        return len(rows)

    def close(self) -> None:
        self.conn.close()
