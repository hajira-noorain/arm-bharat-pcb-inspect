"""One board through the station's evidence-mode advisor with a fine-tuned .pte: causes, probabilities, why lines.

    python llm/try_evidence_advice.py --pte <model.pte> --tok <tokenizer.model> [--why-key]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.advisor import Advisor  # noqa: E402
from pcbinspect.kb import default_kb  # noqa: E402
from pcbinspect.llm_logits import LogitLlama  # noqa: E402

RECORD = {"board_id": "B00042", "design": "demo", "verdict": "FAIL", "defects": [
    {"id": "D1", "type": "open", "severity": "critical", "size_mm": 0.4, "location": "centre", "confidence": 0.93,
     "ipc": {}, "evidence": {"repeat_same_spot": {"boards": 6, "of": 10}, "repeat_in_line": {"boards": 0, "of": 10},
                             "near_edge": False, "shape": "long_thin", "count_on_board": 1, "trend": "steady",
                             "central": True, "roll_pitch": False, "co_occurring": ["short"]}},
    {"id": "D2", "type": "short", "severity": "critical", "size_mm": 0.2, "location": "top-left", "confidence": 0.88,
     "ipc": {}, "evidence": {"repeat_same_spot": {"boards": 0, "of": 10}, "repeat_in_line": {"boards": 1, "of": 10},
                             "near_edge": False, "shape": "irregular", "count_on_board": 5, "trend": "rising",
                             "central": True, "roll_pitch": False, "co_occurring": ["open"]}}]}

ap = argparse.ArgumentParser()
ap.add_argument("--pte", required=True)
ap.add_argument("--tok", required=True)
ap.add_argument("--why-key", action="store_true")
args = ap.parse_args()
be = LogitLlama(args.pte, args.tok)
t0 = time.perf_counter()
a = Advisor(be, default_kb(), cause_mode="evidence", why=True, why_key=args.why_key).advise(RECORD)
print(round(time.perf_counter() - t0, 1), "s", a.backend, a.summary_by)
for s in a.steps:
    top = sorted(s["p"].items(), key=lambda x: -x[1])[:2]
    print(s["defect"], s["cause_id"], [(c, round(p, 2)) for c, p in top], "abstain" if s["abstain"] else "",
          "|", s.get("why_by"), "|", s.get("why"), "|", s.get("why_rejected", ""))
