"""Operator questions over the inspection log — whitelisted, parameterised, read-only.

Llama is never allowed to write SQL. It only (1) picks one of the queries below and fills its
parameters, and (2) summarises the returned rows. Both steps are validated; the numbers in the
answer must appear in the rows. Keyword routing + a plain table is the fallback.
"""
from __future__ import annotations

import json
import re
import time

from .grounding import NUM_RE, extract_json

QUERIES = {
    "defect_counts": {
        "desc": "how many of each defect type were found in the last N hours",
        "params": {"hours": 8},
        "sql": ("SELECT d.cls AS defect, COUNT(*) AS n FROM defects d JOIN inspections i "
                "ON i.id=d.inspection_id WHERE i.ts>=? GROUP BY d.cls ORDER BY n DESC"),
    },
    "yield": {
        "desc": "how many boards were inspected, passed, failed or sent to review in the last N hours",
        "params": {"hours": 8},
        "sql": ("SELECT COUNT(*) AS boards, SUM(verdict='PASS') AS passed, SUM(verdict='FAIL') AS failed, "
                "SUM(verdict='REVIEW') AS review FROM inspections WHERE ts>=?"),
    },
    "board_history": {
        "desc": "the inspection history of one board by its ID",
        "params": {"board_id": ""},
        "sql": ("SELECT i.id AS inspection, datetime(i.ts,'unixepoch','localtime') AS time, i.verdict, "
                "COUNT(d.id) AS defects FROM inspections i LEFT JOIN defects d ON d.inspection_id=i.id "
                "WHERE i.board_id=? GROUP BY i.id ORDER BY i.id"),
    },
    "severity_counts": {
        "desc": "how many critical, major and minor defects were found in the last N hours",
        "params": {"hours": 8},
        "sql": ("SELECT d.severity, COUNT(*) AS n FROM defects d JOIN inspections i ON i.id=d.inspection_id "
                "WHERE i.ts>=? GROUP BY d.severity ORDER BY n DESC"),
    },
    "open_alerts": {
        "desc": "process drift alerts that are still open",
        "params": {},
        "sql": "SELECT datetime(ts,'unixepoch','localtime') AS time, message FROM alerts WHERE acknowledged=0 ORDER BY id DESC LIMIT 10",
    },
}

LLAMA_HDR = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"


def _chat(system: str, user: str, prefix: str) -> str:
    return (LLAMA_HDR + system + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + prefix)


def keyword_route(q: str) -> tuple[str, dict]:
    ql = q.lower()
    hours = 8
    m = re.search(r"(\d+)\s*(hours?|hrs?|h)\b", ql)
    if m:
        hours = int(m.group(1))
    elif "today" in ql or "shift" in ql:
        hours = 8 if "shift" in ql else 24
    elif "week" in ql:
        hours = 24 * 7
    b = re.search(r"\b(?:board|serial|id)\s*[:#]?\s*([A-Za-z0-9_-]{3,})", q)
    if b and any(w in ql for w in ("history", "board", "serial")):
        return "board_history", {"board_id": b.group(1)}
    if "alert" in ql or "drift" in ql:
        return "open_alerts", {}
    if any(w in ql for w in ("critical", "major", "minor", "severity")):
        return "severity_counts", {"hours": hours}
    if any(w in ql for w in ("yield", "pass", "fail", "how many boards", "inspected", "review")):
        return "yield", {"hours": hours}
    return "defect_counts", {"hours": hours}


def run_query(conn, qid: str, params: dict) -> list[dict]:
    q = QUERIES[qid]
    if qid == "board_history":
        args = (str(params.get("board_id", ""))[:64],)
    elif "hours" in q["params"]:
        hours = float(params.get("hours", q["params"]["hours"]))
        hours = min(max(hours, 0.1), 24 * 365)
        args = (time.time() - hours * 3600,)
    else:
        args = ()
    return [dict(r) for r in conn.execute(q["sql"], args)]


def table_text(rows: list[dict]) -> str:
    if not rows:
        return "No matching records."
    return "; ".join(", ".join(f"{k} {v}" for k, v in r.items()) for r in rows)


def answer(question: str, conn, backend=None, max_new_tokens: int = 96) -> dict:
    out = {"question": question, "route": "keyword", "summary_by": "template"}
    qid, params = keyword_route(question)
    if backend is not None:
        menu = "\n".join(f"{k}: {v['desc']} (params: {json.dumps(v['params'])})" for k, v in QUERIES.items())
        p = _chat("You route questions from a factory operator to a fixed list of database queries. "
                  "Reply with JSON only.",
                  f"Queries:\n{menu}\n\nQuestion: {question}\n"
                  'Reply as {"query": "<one id from the list>", "params": {...}}', '{"query": "')
        try:
            g = backend.generate(p, 48)
            obj = extract_json('{"query": "' + g.text)
            if obj and obj.get("query") in QUERIES and isinstance(obj.get("params", {}), dict):
                qid, params = obj["query"], {**QUERIES[obj["query"]]["params"], **obj.get("params", {})}
                out["route"] = "llama"
        except Exception as exc:  # routing failure falls back to keywords
            out["route_error"] = repr(exc)
    rows = run_query(conn, qid, params)
    out.update({"query": qid, "params": params, "rows": rows, "answer": table_text(rows)})
    if backend is not None and rows:
        p = _chat("You summarise database rows for a factory operator in one or two plain sentences. "
                  "Use only numbers that appear in the rows.",
                  f"Question: {question}\nRows: {json.dumps(rows, default=str)}", "")
        try:
            g = backend.generate(p, max_new_tokens)
            text = g.text.strip().split("<|eot_id|>")[0].strip()
            allowed = {round(float(x), 2) for x in NUM_RE.findall(json.dumps(rows, default=str))}
            bad = [m for m in NUM_RE.findall(text) if round(float(m), 2) not in allowed]
            if text and not bad:
                out["answer"], out["summary_by"] = text, "llama"
            else:
                out["summary_rejected"] = f"numbers not in rows: {bad}" if bad else "empty"
        except Exception as exc:
            out["summary_error"] = repr(exc)
    return out
