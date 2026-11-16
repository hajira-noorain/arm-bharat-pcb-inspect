"""Evidence for Llama's cause choice, computed by the station from its own log (never guessed).

A process engineer tells causes apart from a few clues: a defect at the same spot on board after board points to
the phototool; one near the panel edge to handling; a long thin break to a hair; many of one kind across the board
to the etch chemistry; a type that is rising to something that just changed. The station can compute all of these
from the boxes it already logs, so it does, per defect. Anything it cannot know is "unknown", not zero.
"""
from __future__ import annotations

import math

UNKNOWN = "unknown"
SPOT_MM = 1.5           # same spot: centres within this distance (fixture repeatability plus box jitter)
SPOT_FRACTION = 0.03    # without a scale: a share of the frame's short side
EDGE_MM = 2.0           # near the edge: the box comes within this distance of the frame border
EDGE_FRACTION = 0.025
LONG_THIN = 3.0         # longest / shortest box side at or above this: long and thin
ROUND = 1.4             # below this: round
TREND_HALF = 10         # trend compares the last 10 boards with the 10 before
TREND_MIN = 3           # ... and needs the type on at least this many recent boards
CENTRAL = (0.3, 0.7)    # central: the box centre in the middle 40% of the frame both ways (etchant puddling)
ROLL_MM = (190.0, 264.0)  # laminator roll circumferences (TB-0173): a repeat this far apart, in line
ALIGN_MM = 5.0          # "in line" for the roll repeat: within this distance across the direction of travel


def _centre(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def shape_of(b) -> str:
    w, h = max(1, b[2] - b[0]), max(1, b[3] - b[1])
    r = max(w, h) / min(w, h)
    return "long_thin" if r >= LONG_THIN else "round" if r < ROUND else "irregular"


def near_edge(b, img_w, img_h, mm_per_px):
    if not img_w or not img_h:
        return UNKNOWN
    lim = EDGE_MM / mm_per_px if mm_per_px else EDGE_FRACTION * min(img_w, img_h)
    return min(b[0], b[1], img_w - b[2], img_h - b[3]) <= lim


def central(b, img_w, img_h):
    if not img_w or not img_h:
        return UNKNOWN
    cx, cy = _centre(b)
    return CENTRAL[0] <= cx / img_w <= CENTRAL[1] and CENTRAL[0] <= cy / img_h <= CENTRAL[1]


def roll_pitch(b, others, mm_per_px):
    """Another defect of the same type on this board, in line and one laminator-roll circumference away."""
    if not mm_per_px:
        return UNKNOWN
    cx, cy = _centre(b)
    for ob in others:
        ox, oy = _centre(ob)
        for along, across in ((abs(ox - cx), abs(oy - cy)), (abs(oy - cy), abs(ox - cx))):
            if across * mm_per_px <= ALIGN_MM and ROLL_MM[0] <= along * mm_per_px <= ROLL_MM[1]:
                return True
    return False


def board_evidence(store, iid: int, img_w: int | None, img_h: int | None, mm_per_px: float | None,
                   window: int = 20) -> dict[str, dict]:
    """{defect_key: evidence} for inspection `iid`, from the boards of the same design logged before it."""
    conn = store.conn
    me = conn.execute("SELECT design FROM inspections WHERE id=?", (iid,)).fetchone()
    if me is None:
        return {}
    rows = conn.execute("SELECT defect_key, cls, x1, y1, x2, y2 FROM defects WHERE inspection_id=? ORDER BY id",
                        (iid,)).fetchall()
    prev = [r[0] for r in conn.execute(
        "SELECT id FROM inspections WHERE design=? AND id<? ORDER BY id DESC LIMIT ?", (me["design"], iid, window))]
    hist: dict[int, list] = {p: [] for p in prev}
    if prev:
        q = "SELECT inspection_id, cls, x1, y1, x2, y2 FROM defects WHERE inspection_id IN (%s)" % ",".join("?" * len(prev))
        for r in conn.execute(q, prev):
            hist[r["inspection_id"]].append((r["cls"], (r["x1"], r["y1"], r["x2"], r["y2"])))
    tol = (SPOT_MM / mm_per_px if mm_per_px
           else SPOT_FRACTION * min(img_w, img_h) if img_w and img_h else 30.0)
    types = [r["cls"] for r in rows]
    boxes = [(r["defect_key"], (r["x1"], r["y1"], r["x2"], r["y2"])) for r in rows]
    kinds = {r["defect_key"]: r["cls"] for r in rows}
    out = {}
    for r in rows:
        b = (r["x1"], r["y1"], r["x2"], r["y2"])
        cx, cy = _centre(b)
        if prev:
            n_same = sum(any(c == r["cls"] and math.dist(_centre(bb), (cx, cy)) <= tol for c, bb in hist[p])
                         for p in prev)
            repeat = {"boards": n_same, "of": len(prev)}

            def in_line(bb):     # same type in line with this one (same x or same y) but not at the same spot
                ox, oy = _centre(bb)
                return math.dist((ox, oy), (cx, cy)) > tol and (abs(ox - cx) <= tol or abs(oy - cy) <= tol)
            n_line = sum(any(c == r["cls"] and in_line(bb) for c, bb in hist[p]) for p in prev)
            line = {"boards": n_line, "of": len(prev)}
        else:
            repeat = line = UNKNOWN
        if len(prev) >= 2 * TREND_HALF:
            seen = [any(c == r["cls"] for c, _ in hist[p]) for p in prev]
            recent, older = sum(seen[:TREND_HALF]), sum(seen[TREND_HALF:2 * TREND_HALF])
            trend = "rising" if recent >= TREND_MIN and recent >= 2 * max(older, 1) else "steady"
        else:
            trend = UNKNOWN
        out[r["defect_key"]] = {
            "repeat_same_spot": repeat,
            "repeat_in_line": line,
            "central": central(b, img_w, img_h),
            "roll_pitch": roll_pitch(b, [ob for k, ob in boxes if k != r["defect_key"] and kinds[k] == r["cls"]],
                                     mm_per_px),
            "near_edge": near_edge(b, img_w, img_h, mm_per_px),
            "shape": shape_of(b),
            "count_on_board": types.count(r["cls"]),
            "trend": trend,
            "co_occurring": sorted(set(types) - {r["cls"]}),
        }
    return out


def attach_evidence(record: dict, store, iid: int | None, img_w: int | None, img_h: int | None,
                    mm_per_px: float | None) -> dict:
    """A copy of the Llama record with each defect's evidence added. A board that is not in the log (a demo upload)
    has no history, so its record goes to Llama unchanged and the prompt says the clues are unknown."""
    if iid is None:
        return record
    ev = board_evidence(store, iid, img_w, img_h, mm_per_px)
    return {**record, "defects": [{**d, **({"evidence": ev[d["id"]]} if d["id"] in ev else {})}
                                  for d in record.get("defects", [])]}
