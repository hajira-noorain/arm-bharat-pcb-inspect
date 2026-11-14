"""Record what the station does on every test board, for the static online demo (deploy/build_static_demo.py).

Runs against a station started in demo mode, inspects each test board one at a time, waits for Llama's advice
for that board before going on (so every board gets Llama's own answer, never the busy fallback), and saves
the station's replies exactly as it gave them:

    python deploy/record_demo.py --station http://127.0.0.1:8000 --out <dir>

<dir>/data/ then holds state.json, samples.json, stats.json, log.json, spc_<design>.json, inspections/<file>.json,
images/<id>.jpg and thumbs/<file>.jpg. Nothing is edited after the station wrote it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Llama's other two jobs, recorded against the same log: shift questions (one per kind of query the station
# answers) and customer requirements compiled into rules (from the held-out clause set).
QUESTIONS = ["Which defect is the most common this shift?", "What is the pass rate today?",
             "How many boards went to a person for checking today?",
             "Split the defects from today into critical, major and minor", "What happened with board B00005?",
             "Are there any open alerts?"]
CLAUSES = ["Mouse bites up to 0.15 mm are acceptable.", "Conductor width reduction from nicks or bites not to exceed 30%.",
           "Spurs longer than 100 µm must be removed.", "Shorted nets are to be held for MRB disposition.",
           "Pinholes exceeding 75 microns shall be referred to engineering."]


def get(base: str, path: str) -> bytes:
    with urllib.request.urlopen(base + path, timeout=600) as r:
        return r.read()


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--advice-timeout", type=float, default=600.0, help="seconds to wait for one board's advice")
    ap.add_argument("--extras-only", action="store_true", help="only the shift questions and customer rules")
    a = ap.parse_args()
    out = Path(a.out) / "data"
    for sub in ("inspections", "images", "thumbs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    base = a.station.rstrip("/")
    if a.extras_only:
        return record_extras(base, out)
    samples = json.loads(get(base, "/api/samples"))
    (out / "samples.json").write_text(json.dumps(samples, indent=1))
    for i, s in enumerate(samples):
        q = urllib.parse.quote(s["file"])
        (out / "thumbs" / f"{s['file']}.jpg").write_bytes(get(base, f"/api/samples/{q}.jpg"))
        t0 = time.time()
        r = post(base, "/api/inspect", {"design": s["design"], "sample": s["file"], "alpha": 0.05})
        extra = {"known": r.get("known"), "sample": r.get("sample")}
        while r.get("advice_pending") and time.time() - t0 < a.advice_timeout:
            time.sleep(2)
            r = {**json.loads(get(base, f"/api/inspection/{r['id']}")), **extra}
        (out / "inspections" / f"{s['file']}.json").write_text(json.dumps(r, indent=1))
        (out / "images" / f"{r['id']}.jpg").write_bytes(get(base, f"/api/image/{r['id']}.jpg"))
        adv = (r.get("advice") or [{}])[-1]
        print(f"{i + 1}/{len(samples)} {s['file']}: {r['verdict']}, {len(r['defects'])} findings, advice by "
              f"{adv.get('backend', 'none')} in {(adv.get('latency_ms') or 0) / 1000:.1f} s", flush=True)
    (out / "state.json").write_text(get(base, "/api/state").decode())
    (out / "stats.json").write_text(get(base, "/api/stats").decode())
    (out / "log.json").write_text(get(base, "/api/log?limit=200").decode())
    for d in sorted({s["design"] for s in samples}):
        (out / f"spc_{d}.json").write_text(get(base, f"/api/spc/{urllib.parse.quote(d)}").decode())
    print("RECORDED", len(samples), "boards into", out)
    return record_extras(base, out)


def record_extras(base: str, out: Path) -> int:
    asked = []
    for q in QUESTIONS:
        r = post(base, "/api/ask", {"question": q})
        asked.append(r)
        print(f"ask: {q} -> {r.get('answer', '')[:90]!r} ({r.get('summary_by')})", flush=True)
    (out / "ask.json").write_text(json.dumps(asked, indent=1))
    read = []
    for c in CLAUSES:
        r = post(base, "/api/rules/compile", {"design": "deeppcb", "clause": c})
        read.append(r)
        print(f"rule: {c} -> {[x.get('sentence') for x in r.get('readers', [])]} ({r.get('status')})", flush=True)
    (out / "rules.json").write_text(json.dumps(read, indent=1))
    print("RECORDED", len(asked), "questions and", len(read), "requirements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
