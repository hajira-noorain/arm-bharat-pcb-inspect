"""Run the real advice prompt through the exported Llama .pte and the grounding checker.

python llm/try_advice.py --pte /work/models/llm/llama3_2_1b_spinquant.pte --tok /work/models/llm/tokenizer.model
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.advisor import Advisor  # noqa: E402
from pcbinspect.llm import ExecuTorchLlama  # noqa: E402

RECORD = {"board_id": "B-0042", "design": "deeppcb", "verdict": "FAIL", "defects": [
    {"id": "D1", "type": "short", "severity": "critical", "confidence": 0.91, "size_mm": 0.31,
     "location": "top-left", "ipc": {"class1": "reject", "class2": "reject", "class3": "reject"}},
    {"id": "D2", "type": "mouse_bite", "severity": "minor", "confidence": 0.64, "size_mm": 0.12,
     "location": "centre", "ipc": {"class1": "accept", "class2": "accept", "class3": "reject"}}]}

ap = argparse.ArgumentParser()
ap.add_argument("--pte", required=True)
ap.add_argument("--tok", required=True)
a = ap.parse_args()
llm = ExecuTorchLlama(a.pte, a.tok)
print(f"load {llm.load_ms:.0f} ms")
adv = Advisor(llm).advise(RECORD, {"short": 4})
print("RAW:", adv.raw_output)
print("backend:", adv.backend, "grounded:", adv.grounded, "rejected:", adv.rejected_reasons)
print(json.dumps(adv.to_json(), indent=1))
g = adv.gen
print(f"wall {g.latency_ms:.0f} ms, prefill {g.prefill_tps} tok/s, decode {g.decode_tps} tok/s, n={g.n_generated}")
print(llm.last_stats)
