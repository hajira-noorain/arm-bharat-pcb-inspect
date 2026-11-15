"""The evidence prompt for Llama's cause choice. One wording, used by the station (pcbinspect/advisor.py) and by
the training data (llm/data/gen_cause.py), so the fine-tuned model sees exactly what it was trained on.

Each defect is described with the clues the station computed (pcbinspect/evidence.py); each candidate cause is
then asked about alone, as a yes/no question, and the station reads Llama's yes/no probabilities.
`variant` (training data only) changes the wording and order of the clues, never the facts.
"""
from __future__ import annotations

HDR = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
SYSTEM = ("You are the quality assistant on a PCB inspection station. You judge the likely cause of each defect "
          "from the clues the station measured.")


def _clues(ev: dict, v: int) -> list[str]:
    rep = ev.get("repeat_same_spot", "unknown")
    if isinstance(rep, dict):
        same = (f"same spot on {rep['boards']} of the last {rep['of']} boards" if v % 2 == 0 else
                f"seen at this same place on {rep['boards']} of the previous {rep['of']} boards")
    else:
        same = "same spot unknown" if v % 2 == 0 else "same spot unknown (no earlier boards)"
    edge = ev.get("near_edge", "unknown")
    edge = ("near the edge" if edge is True else "not near the edge" if edge is False else "edge unknown")
    shape = {"long_thin": "long and thin", "round": "round", "irregular": "irregular shape"}.get(
        ev.get("shape"), "shape unknown")
    n = ev.get("count_on_board")
    count = f"{n} of this type on the board" if isinstance(n, int) else "count unknown"
    trend = {"rising": "rising on recent boards", "steady": "steady on recent boards"}.get(ev.get("trend"),
                                                                                         "trend unknown")
    out = [same, edge, shape, count, trend]
    if ev.get("co_occurring"):
        out.append("also on the board: " + ", ".join(t.replace("_", " ") for t in ev["co_occurring"]))
    if v % 4 == 3:                  # a different order for some training examples
        out = out[2:] + out[:2]
    return out


def clue_text(ev: dict, variant: int = 0) -> str:
    return "Clues: " + "; ".join(_clues(ev or {}, variant)) + "."


def defect_line(d: dict) -> str:
    what = d["type"].replace("_", " ") + (f" ({d['part']})" if d.get("part") and d["part"] != "part" else "")
    size = f", {d['size_mm']} mm" if d.get("size_mm") is not None else ""
    return f"{d['id']}: {what}, {d.get('severity')}{size}, {d.get('location')}"


def build_evidence_prompt(record: dict, kb, variant: int = 0, max_defects: int = 6) -> str:
    lines = []
    for d in record["defects"][:max_defects]:
        lines.append(defect_line(d) + ". " + clue_text(d.get("evidence") or {}, variant))
    user = f"Board {record['board_id']} verdict {record['verdict']}. Defects:\n" + "\n".join(lines)
    return (HDR + SYSTEM + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def cause_question(defect_id: str, cause_id: str, kb) -> str:
    c = kb.by_id[cause_id]
    return f'\nQuestion: is "{c["short"]}" ({c.get("step", "process")}) the likely cause of {defect_id}? Answer:'
