"""Fill-in sentences: Llama writes the sentence with {field} placeholders and the station writes the numbers.

Why: a small model reading database rows can copy the wrong column or name the wrong row ("28 boards were sent for
checking" when 25 were). Here every number comes from a named field the station filled in, rankings are computed
by code (top_defect, top_count), and a checker refuses a sentence that (1) types a number itself, (2) uses a field
that does not exist, (3) leaves out what the question asked for, (4) says "most common" without the top fields,
or (5) says nothing. A refused sentence is replaced by the standard sentence, so the operator always gets the
right numbers. Used by pcbinspect/ask.py and by the training data (llm/data/gen_sentences.py).
"""
from __future__ import annotations

import re

# {field} or {field|noun}: with a noun the station writes the number and the noun, singular or plural
PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)(?:\|([a-z][a-z ]*))?\}")
DIGIT = re.compile(r"\d")
SUPERLATIVE = re.compile(r"\b(most|top|highest|largest|biggest|commonest|main)\b", re.I)
SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
SYSTEM = ("You answer a factory operator's question about the inspection log in one or two plain sentences. "
          "Write every number or name from the facts as {name}, using the names listed; never type a number.")


def _int(x) -> int:
    return int(x or 0)


def _words(s) -> str:
    return str(s).replace("_", " ")


# words an operator uses for each defect type (plural forms included by the regex below)
TYPE_WORDS = {"open": "opens?", "short": "shorts?", "mouse_bite": "mouse ?bites?|nicks?", "spur": "spurs?",
              "spurious_copper": "spurious copper|copper islands?", "pin_hole": "pin ?holes?",
              "missing_hole": "missing holes?", "missing_pad": "missing pads?|damaged pads?",
              "hole_breakout": "hole breakouts?|breakouts?", "scratch": "scratch(?:es)?",
              "foreign_object": "foreign objects?", "missing_part": "missing parts?", "extra_part": "extra parts?",
              "changed_part": "changed parts?|wrong parts?"}


def named_types(question: str) -> list[str]:
    q = question.lower()
    return [t for t, pat in TYPE_WORDS.items() if re.search(rf"\b(?:{pat})\b", q)]


def facts_for(qid: str, rows: list[dict], params: dict, question: str = "") -> dict:
    hours = params.get("hours")
    f: dict = {} if hours is None else {"hours": int(hours)}
    if qid == "yield":
        r = rows[0] if rows else {}
        f.update({k: _int(r.get(k)) for k in ("boards", "passed", "failed", "review")})
    elif qid in ("defect_counts", "severity_counts"):
        key = "defect" if qid == "defect_counts" else "severity"
        rs = sorted(rows, key=lambda r: (-_int(r["n"]), str(r[key])))
        f["total_defects"] = sum(_int(r["n"]) for r in rs)
        if qid == "defect_counts":
            f["n_types"] = len(rs)
            for i, name in ((0, "top"), (1, "second")):
                if len(rs) > i:
                    f[f"{name}_defect"], f[f"{name}_count"] = _words(rs[i]["defect"]), _int(rs[i]["n"])
            for r in rs:
                f[f"count_{r['defect']}"] = _int(r["n"])
            for t in named_types(question):      # a type the question names but nobody found: say 0, not nothing
                f.setdefault(f"count_{t}", 0)
        else:
            by = {str(r["severity"]): _int(r["n"]) for r in rs}
            f.update({s: by.get(s, 0) for s in ("critical", "major", "minor")})
    elif qid == "board_history":
        f["board"] = params.get("board_id", "")
        f["times_inspected"] = len(rows)
        if rows:
            last = rows[-1]
            f.update({"last_verdict": "CHECK" if last["verdict"] == "REVIEW" else last["verdict"],
                      "last_defects": _int(last["defects"]), "last_time": str(last["time"])})
    elif qid == "open_alerts":
        f["n_alerts"] = len(rows)
        if rows:
            f["latest_alert"], f["latest_time"] = str(rows[0]["message"]), str(rows[0]["time"])
    return f


def required_fields(question: str, qid: str) -> set[str]:
    """What the question asks for, which the answer must therefore contain."""
    q = question.lower()
    if qid == "yield":
        need = set()
        if any(w in q for w in ("person", "check", "review", "manual", "operator", "second look")):
            need.add("review")
        if any(w in q for w in ("fail", "reject")):
            need.add("failed")
        if "pass" in q:
            need.add("passed")
        if any(w in q for w in ("rate", "share", "yield", "percent", "%", "versus")):
            return (need or {"passed"}) | {"boards"}
        return need or {"boards"}
    if qid == "defect_counts":
        need = {f"count_{t}" for t in named_types(question)}
        if any(w in q for w in ("most", "common", "top", "main", "kept", "which")):
            need.add("top_defect")
        return need
    if qid == "severity_counts":
        named = {s for s in ("critical", "major", "minor") if s in q}
        if any(w in q for w in ("split", "severity", "severe", "serious", "small", "how bad")):
            return {"critical", "major", "minor"}
        return named
    if qid == "board_history":
        return {"times_inspected"}
    if qid == "open_alerts":
        return {"n_alerts"}
    return set()


def check_template(tpl: str, question: str, qid: str, facts: dict) -> list[str]:
    t = (tpl or "").strip()
    used = [m.group(1) for m in PLACEHOLDER.finditer(t)]
    out = []
    if not t or not used:
        return ["the answer gives no fact" if t else "empty answer"]
    if DIGIT.search(PLACEHOLDER.sub("", t)):
        out.append("a number was typed instead of written as a field")
    unknown = sorted(set(used) - set(facts))
    if unknown:
        out.append(f"fields that do not exist: {unknown}")
    missing = sorted(required_fields(question, qid) - set(used))
    if missing:
        out.append(f"does not say what was asked: {missing}")
    if qid == "defect_counts" and SUPERLATIVE.search(t) and "top_defect" not in used:
        out.append("says 'most' without the top defect")
    if len(SENTENCE_END.findall(t + " ")) > 2:
        out.append("more than two sentences")
    return out


def plural(noun: str, n) -> str:
    if n == 1:
        return noun
    return noun + ("es" if re.search(r"(s|x|ch|sh)$", noun) else "s")


def render(tpl: str, facts: dict) -> str:
    def one(m):
        v = facts[m.group(1)]
        return f"{v} {plural(m.group(2), v)}" if m.group(2) else str(v)
    out = PLACEHOLDER.sub(one, tpl.strip())
    return out[:1].upper() + out[1:]


def nothing_to_say(qid: str, f: dict) -> bool:
    """Nothing inspected or found: the standard sentence says so; Llama is not asked."""
    return ((qid == "yield" and not f.get("boards")) or
            (qid in ("defect_counts", "severity_counts") and not f.get("total_defects")) or
            (qid == "board_history" and not f.get("times_inspected")) or
            (qid == "open_alerts" and not f.get("n_alerts")))


def standard_sentence(qid: str, f: dict) -> str:
    h = f.get("hours")
    span = f" in the last {h} hours" if h is not None else ""
    if qid == "yield":
        if not f.get("boards"):
            return f"No boards inspected{span}."
        return (f"{f['boards']} boards inspected{span}: {f['passed']} passed, {f['failed']} failed, "
                f"{f['review']} sent to a person.")
    if qid == "defect_counts":
        if not f.get("total_defects"):
            return f"No defects{span}."
        s = f"{f['total_defects']} defects{span}; most common: {f['top_defect']} ({f['top_count']})"
        if "second_defect" in f:
            s += f", then {f['second_defect']} ({f['second_count']})"
        return s + "."
    if qid == "severity_counts":
        if not f.get("total_defects"):
            return f"No defects{span}."
        return (f"{f['total_defects']} defects{span}: {f['critical']} critical, {f['major']} major, "
                f"{f['minor']} minor.")
    if qid == "board_history":
        if not f.get("times_inspected"):
            return f"No inspection of board {f.get('board', '')} in the log."
        return (f"Board {f['board']} was inspected {f['times_inspected']} time(s); last verdict "
                f"{f['last_verdict']} with {f['last_defects']} defect(s), at {f['last_time']}.")
    if qid == "open_alerts":
        if not f.get("n_alerts"):
            return "No open alerts."
        return f"{f['n_alerts']} open alert(s); latest at {f['latest_time']}: {f['latest_alert']}"
    return ""


def template_prompt(question: str, facts: dict) -> str:
    listed = "; ".join(f"{k} = {v}" for k, v in facts.items())
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM +
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
            f"Question: {question}\nFacts: {listed}" +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
