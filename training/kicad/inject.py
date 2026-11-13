"""Design-level defect injection into KiCad boards (.kicad_pcb, format versions 5-10).

Method: the defect is made in the *design file* (copper geometry), exactly as the DefectDet dataset did, so a
renderer produces both a defective board and a pixel-aligned defect-free twin (the untouched original text), and
the defect's bounding box is known analytically in board millimetres.  No raster editing is involved.

All coordinates are board millimetres, y pointing down (KiCad convention).  Only the top copper (F.Cu) is
edited; inner layers and B.Cu are untouched.  Everything is seeded: ``inject_many(text, counts, seed)`` returns the
same result for the same inputs.

KiCad tracks are capsules: a ``(segment ...)`` has ROUND end caps of radius width/2.  Every injector below accounts
for the caps, so the bounding boxes describe the visible defect, not the raw segment endpoints.

Injectors (``cls`` is the pcbinspect BARE_CLASSES name):

open             one interior piece of a track is removed.  The two stubs end ``size`` mm apart (cap tip to cap tip).
                 Accepted only if a connectivity graph of the net (segments, vias, pads, T-junctions) shows the cut
                 really separates two parts that each reach a pad or via, and the net has no copper zone.
short            a ``(segment)`` with net 0 and width ``bridge_w`` is added between a point on one track and a nearby
                 track or pad of a DIFFERENT net, where the edge-to-edge gap is about ``size``.  The bridge must not
                 touch a third net.  The bbox is the gap region that the bridge fills.
spur             a thin stub (same net as its host track) starts on the track centreline and protrudes ``size`` mm past
                 the track edge.  Rejected unless it keeps clearance from every other copper object.
spurious_copper  an isolated irregular ``gr_poly`` blob of diameter ~``size`` on F.Cu in empty space (clearance from
                 every track/pad/via and from every zone fill).
mouse_bite       a scoop is bitten out of one edge of a track: the section between two cuts is replaced by a narrower
                 segment, flush with the opposite edge, so that the round caps of the full-width neighbours leave a
                 half-round notch of depth ``depth`` and opening ``size`` along the track.  (Replacing a section with a
                 narrower width over a short length, as the brief allows.)
pin_hole         a small enclosed void inside a track: the middle section is replaced by two parallel rails, leaving a
                 closed hole of width ``size`` between them (a keepout zone or a zone void would not render, because
                 kicad-cli does not refill zones).  Needs a track at least 0.3 mm wide.
missing_pad      the copper of one top-side SMD pad is removed by deleting the ``(pad ...)`` node from the footprint
                 instance (mask opening and paste go with it; the trace now ends at bare laminate).
missing_hole     a plated through-hole pad is turned into a plain pad without a drill: ``thru_hole`` -> ``smd`` with the
                 ``(drill ...)`` removed and the copper layers listed explicitly, so ring copper remains but no hole.

Not done: nothing is skipped; ``pin_hole`` uses the rails method above.  Defects whose location search fails
return ``None`` and ``inject_many`` simply reports fewer defects than requested.
"""
from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from . import sexpr
from .sexpr import Node, fmt

DEFECT_CLASSES = ("open", "short", "spur", "spurious_copper", "mouse_bite", "pin_hole", "missing_pad", "missing_hole")

# default size ranges in mm (the characteristic size recorded as size_mm)
SIZE_RANGE = {
    "open": (0.15, 0.5), "short": (0.12, 0.5), "spur": (0.15, 0.5), "spurious_copper": (0.2, 0.7),
    "mouse_bite": (0.2, 0.5), "pin_hole": (0.1, 0.25),
}


def check_class_names() -> None:
    """Assert our names are real pcbinspect classes (imported lazily so the module works stand-alone)."""
    from pcbinspect.record import BARE_CLASSES
    missing = [c for c in DEFECT_CLASSES if c not in BARE_CLASSES]
    if missing:
        raise AssertionError(f"not in pcbinspect.record.BARE_CLASSES: {missing}")


# ======================================================================================================
# geometry
# ======================================================================================================
def _pt_seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-18 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy), t


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _seg_intersect(a, b, c, d) -> bool:
    d1 = _cross(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])
    d2 = _cross(b[0] - a[0], b[1] - a[1], d[0] - a[0], d[1] - a[1])
    d3 = _cross(d[0] - c[0], d[1] - c[1], a[0] - c[0], a[1] - c[1])
    d4 = _cross(d[0] - c[0], d[1] - c[1], b[0] - c[0], b[1] - c[1])
    return d1 * d2 < 0 and d3 * d4 < 0


def seg_seg_dist(a, b, c, d) -> float:
    if _seg_intersect(a, b, c, d):
        return 0.0
    return min(_pt_seg(a[0], a[1], c[0], c[1], d[0], d[1])[0], _pt_seg(b[0], b[1], c[0], c[1], d[0], d[1])[0],
               _pt_seg(c[0], c[1], a[0], a[1], b[0], b[1])[0], _pt_seg(d[0], d[1], a[0], a[1], b[0], b[1])[0])


def pt_box_dist(px, py, x1, y1, x2, y2) -> float:
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def seg_box_dist(a, b, x1, y1, x2, y2) -> float:
    if pt_box_dist(a[0], a[1], x1, y1, x2, y2) == 0 or pt_box_dist(b[0], b[1], x1, y1, x2, y2) == 0:
        return 0.0
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for i in range(4):
        if _seg_intersect(a, b, corners[i], corners[(i + 1) % 4]):
            return 0.0
    return min(min(_pt_seg(cx, cy, a[0], a[1], b[0], b[1])[0] for cx, cy in corners),
               pt_box_dist(a[0], a[1], x1, y1, x2, y2), pt_box_dist(b[0], b[1], x1, y1, x2, y2))


@dataclass
class Shape:
    """Copper obstacle: kind 'cap' (capsule: ax,ay,bx,by,r) or 'box' (x1,y1,x2,y2)."""
    kind: str
    g: tuple
    net: str | None
    ref: object = None
    what: str = ""

    @property
    def bbox(self):
        if self.kind == "cap":
            ax, ay, bx, by, r = self.g
            return (min(ax, bx) - r, min(ay, by) - r, max(ax, bx) + r, max(ay, by) + r)
        return self.g


def cap(a, b, r, net=None, ref=None, what="") -> Shape:
    return Shape("cap", (a[0], a[1], b[0], b[1], r), net, ref, what)


def shape_dist(s: Shape, t: Shape) -> float:
    """Edge-to-edge distance (0 when touching/overlapping)."""
    if s.kind == "cap" and t.kind == "cap":
        d = seg_seg_dist(s.g[:2], s.g[2:4], t.g[:2], t.g[2:4]) - s.g[4] - t.g[4]
    elif s.kind == "cap":
        d = seg_box_dist(s.g[:2], s.g[2:4], *t.g) - s.g[4]
    elif t.kind == "cap":
        d = seg_box_dist(t.g[:2], t.g[2:4], *s.g) - t.g[4]
    else:
        dx = max(s.g[0] - t.g[2], t.g[0] - s.g[2], 0.0)
        dy = max(s.g[1] - t.g[3], t.g[1] - s.g[3], 0.0)
        d = math.hypot(dx, dy)
    return max(d, 0.0)


class Grid:
    """Uniform spatial hash over shape bounding boxes."""

    def __init__(self, cell: float = 2.0):
        self.cell = cell
        self.items: list[Shape] = []
        self.h: dict[tuple[int, int], list[int]] = {}

    def add(self, s: Shape) -> None:
        i = len(self.items)
        self.items.append(s)
        x1, y1, x2, y2 = s.bbox
        c = self.cell
        for gx in range(int(math.floor(x1 / c)), int(math.floor(x2 / c)) + 1):
            for gy in range(int(math.floor(y1 / c)), int(math.floor(y2 / c)) + 1):
                self.h.setdefault((gx, gy), []).append(i)

    def near(self, bbox, radius: float) -> list[Shape]:
        x1, y1, x2, y2 = bbox
        c = self.cell
        seen: set[int] = set()
        out = []
        for gx in range(int(math.floor((x1 - radius) / c)), int(math.floor((x2 + radius) / c)) + 1):
            for gy in range(int(math.floor((y1 - radius) / c)), int(math.floor((y2 + radius) / c)) + 1):
                for i in self.h.get((gx, gy), ()):
                    if i not in seen:
                        seen.add(i)
                        out.append(self.items[i])
        return out


class Fills:
    """F.Cu zone fills (``filled_polygon``) as edge arrays: exact 'is there copper here' and distance queries."""

    def __init__(self, polys: list[np.ndarray]):
        if polys:
            a = []
            for p in polys:
                q = np.roll(p, -1, axis=0)
                a.append(np.hstack([p, q]))
            self.e = np.vstack(a)
        else:
            self.e = np.zeros((0, 4))

    def inside(self, x: float, y: float) -> bool:
        e = self.e
        if not len(e):
            return False
        x1, y1, x2, y2 = e[:, 0], e[:, 1], e[:, 2], e[:, 3]
        m = (y1 > y) != (y2 > y)
        if not m.any():
            return False
        xi = x1[m] + (y - y1[m]) * (x2[m] - x1[m]) / (y2[m] - y1[m])
        return bool((xi > x).sum() % 2 == 1)

    def dist(self, x: float, y: float, radius: float = 1.5) -> float:
        """Distance to the nearest fill edge within ``radius`` (else ``radius``); 0 if inside copper."""
        e = self.e
        if not len(e):
            return radius
        if self.inside(x, y):
            return 0.0
        m = ((np.minimum(e[:, 0], e[:, 2]) < x + radius) & (np.maximum(e[:, 0], e[:, 2]) > x - radius) &
             (np.minimum(e[:, 1], e[:, 3]) < y + radius) & (np.maximum(e[:, 1], e[:, 3]) > y - radius))
        if not m.any():
            return radius
        s = e[m]
        dx, dy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
        L2 = np.maximum(dx * dx + dy * dy, 1e-18)
        t = np.clip(((x - s[:, 0]) * dx + (y - s[:, 1]) * dy) / L2, 0, 1)
        d = np.hypot(x - (s[:, 0] + t * dx), y - (s[:, 1] + t * dy))
        return float(min(d.min(), radius))


# ======================================================================================================
# board model
# ======================================================================================================
@dataclass
class Seg:
    node: Node
    a: tuple
    b: tuple
    w: float
    layer: str
    net: str | None

    @property
    def length(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])


@dataclass
class Pad:
    node: Node
    fp: Node
    x: float
    y: float
    hx: float          # half extent of the axis-aligned bounding box
    hy: float
    shape: str
    ptype: str         # smd / thru_hole / np_thru_hole / connect
    layers: list
    net: str | None
    drill: float | None
    on_top: bool       # has copper on F.Cu
    flipped: bool

    @property
    def box(self):
        return (self.x - self.hx, self.y - self.hy, self.x + self.hx, self.y + self.hy)


@dataclass
class Via:
    node: Node
    x: float
    y: float
    size: float
    drill: float
    net: str | None


@dataclass
class Defect:
    cls: str
    bbox_mm: tuple          # x1, y1, x2, y2 in board mm
    size_mm: float          # characteristic size (gap / protrusion / notch opening / drill / pad size)
    layer: str = "F.Cu"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"cls": self.cls, "bbox_mm": [round(v, 5) for v in self.bbox_mm], "size_mm": round(self.size_mm, 5),
                "layer": self.layer, **self.meta}


def _is_none_net(n: str | None) -> bool:
    return n is None or n in ("", "0")


def _rot_extent(w: float, h: float, ang: float) -> tuple[float, float]:
    a = math.radians(ang)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return (c * w + s * h) / 2, (s * w + c * h) / 2


def _poly_pts(node: Node) -> list[tuple[float, float]]:
    pts = node.find("pts")
    out = []
    if pts is not None:
        for xy in pts.findall("xy"):
            v = [float(a) for a in xy.args[1:3]]
            out.append((v[0], v[1]))
    return out


class Board:
    """Parsed board with indexes of F.Cu copper for geometric queries.  Mutations go to ``self.doc``."""

    def __init__(self, text: str):
        self.doc = sexpr.parse(text)
        self.root = self.doc.root
        v = self.root.val("version", "0")
        self.version = int(v) if str(v).isdigit() else 0
        self.fp_key = "footprint" if self.root.findall("footprint") else "module"
        self.segs: list[Seg] = []
        self.vias: list[Via] = []
        self.pads: list[Pad] = []
        self.zone_nets: set[str] = set()
        self._load()
        self.fills = Fills(self._fill_polys())
        self.grid = Grid(2.0)
        self._build_grid()
        self.edge = board_edge_bbox(self.root)
        self.net_style_int = any((s.net or "").isdigit() for s in self.segs)

    # ---- loading ----
    def _load(self) -> None:
        for n in self.root.findall("segment"):
            st, en, w = n.floats("start"), n.floats("end"), n.floats("width")
            if len(st) < 2 or len(en) < 2 or not w:
                continue
            self.segs.append(Seg(n, (st[0], st[1]), (en[0], en[1]), w[0], n.val("layer", ""), n.val("net")))
        for n in self.root.findall("via"):
            at, sz, dr = n.floats("at"), n.floats("size"), n.floats("drill")
            if len(at) >= 2 and sz:
                self.vias.append(Via(n, at[0], at[1], sz[0], dr[0] if dr else 0.0, n.val("net")))
        for fp in self.root.findall(self.fp_key):
            at = fp.floats("at")
            if len(at) < 2:
                continue
            fx, fy = at[0], at[1]
            rot = at[2] if len(at) > 2 else 0.0
            flipped = fp.val("layer", "F.Cu") == "B.Cu"
            ca, sa = math.cos(math.radians(rot)), math.sin(math.radians(rot))
            for p in fp.findall("pad"):
                a = p.args
                if len(a) < 4:
                    continue
                pat, size = p.floats("at"), p.floats("size")
                if len(pat) < 2 or len(size) < 2:
                    continue
                # pad coordinates are stored in the footprint frame (already mirrored for flipped footprints);
                # KiCad rotation: x' = x cos + y sin, y' = -x sin + y cos.  Verified on real boards (track ends
                # coincide with pad centres).
                px, py = pat[0], pat[1]
                X, Y = fx + px * ca + py * sa, fy - px * sa + py * ca
                pang = pat[2] if len(pat) > 2 else rot
                hx, hy = _rot_extent(size[0], size[1], pang)
                layers = p.vals("layers")
                dr = None
                dn = p.find("drill")
                if dn is not None:
                    nums = []
                    for t in dn.args[1:]:
                        if isinstance(t, str):
                            try:
                                nums.append(float(t))
                            except ValueError:
                                pass
                    dr = max(nums) if nums else None
                net = p.find("net")
                netv = sexpr.unquote(net.args[1]) if net is not None and len(net.args) > 1 else None
                ptype = sexpr.unquote(a[2]) if isinstance(a[2], str) else ""
                shape = sexpr.unquote(a[3]) if isinstance(a[3], str) else ""
                on_top = any(l in ("F.Cu", "*.Cu", "F&B.Cu") for l in layers)
                self.pads.append(Pad(p, fp, X, Y, hx, hy, shape, ptype, layers, netv, dr, on_top, flipped))
        for z in self.root.findall("zone"):
            if z.find("keepout") is not None:
                continue
            zn = z.val("net")
            if zn and zn != "0":
                self.zone_nets.add(zn)

    def _fill_polys(self) -> list[np.ndarray]:
        out = []
        for z in self.root.findall("zone"):
            if z.find("keepout") is not None:
                continue
            for fpn in z.findall("filled_polygon"):
                if fpn.val("layer", "") != "F.Cu":
                    continue
                pts = _poly_pts(fpn)
                if len(pts) >= 3:
                    out.append(np.array(pts, dtype=float))
        return out

    def _build_grid(self) -> None:
        for s in self.segs:
            if s.layer == "F.Cu":
                self.grid.add(cap(s.a, s.b, s.w / 2, s.net, s, "seg"))
        for v in self.vias:
            self.grid.add(cap((v.x, v.y), (v.x, v.y), v.size / 2, v.net, v, "via"))
        for p in self.pads:
            if p.on_top or p.ptype in ("thru_hole", "np_thru_hole"):
                if p.shape == "circle":
                    self.grid.add(Shape("cap", (p.x, p.y, p.x, p.y, p.hx), p.net, p, "pad"))
                elif p.shape == "oval":
                    d = abs(p.hx - p.hy)
                    ax, ay = (d, 0.0) if p.hx >= p.hy else (0.0, d)
                    self.grid.add(Shape("cap", (p.x - ax, p.y - ay, p.x + ax, p.y + ay, min(p.hx, p.hy)), p.net, p, "pad"))
                else:
                    self.grid.add(Shape("box", p.box, p.net, p, "pad"))
        for fp in self.root.findall(self.fp_key):          # copper graphics inside footprints (logos etc.)
            for k in fp.nodes():
                if k.name.startswith("fp_") and k.val("layer", "") == "F.Cu":
                    pts = []
                    for sub in k.nodes():
                        if sub.name in ("start", "end", "center", "mid", "at"):
                            f = sub.floats(sub.name)
                            if len(f) >= 2:
                                pts.append(f[:2])
                        elif sub.name == "pts":
                            pts += _poly_pts(k)
                    at = fp.floats("at")
                    if pts and len(at) >= 2:
                        # coordinates of fp_* are footprint-local; use a generous box around the footprint origin
                        # for rotated/flipped parts we cannot cheaply transform, so cover the local extent.
                        r = max(max(abs(x), abs(y)) for x, y in pts) + 0.3
                        self.grid.add(Shape("box", (at[0] - r, at[1] - r, at[0] + r, at[1] + r), None, k, "fpcu"))
        for k in self.root.nodes():
            if k.name.startswith("gr_") and k.val("layer", "") == "F.Cu":
                pts = []
                for sub in k.nodes():
                    if sub.name in ("start", "end", "center", "mid"):
                        f = sub.floats(sub.name)
                        if len(f) >= 2:
                            pts.append(f[:2])
                    elif sub.name == "pts":
                        pts += _poly_pts(k)
                if pts:
                    xs, ys = zip(*pts)
                    self.grid.add(Shape("box", (min(xs) - 0.1, min(ys) - 0.1, max(xs) + 0.1, max(ys) + 0.1), None, k, "grcu"))

    # ---- helpers for the injectors ----
    def clearance(self, s: Shape, ignore: Iterable = (), radius: float = 1.0) -> float:
        """Distance from ``s`` to the nearest F.Cu copper object (tracks, vias, pads, graphics, fills)."""
        ign = {id(i) for i in ignore}
        best = radius
        for t in self.grid.near(s.bbox, radius):
            if id(t.ref) in ign:
                continue
            d = shape_dist(s, t)
            if d < best:
                best = d
        if s.kind == "cap" and len(self.fills.e):
            ax, ay, bx, by, r = s.g
            n = max(2, int(math.hypot(bx - ax, by - ay) / 0.1) + 1)
            for i in range(n + 1):
                t = i / n
                best = min(best, self.fills.dist(ax + t * (bx - ax), ay + t * (by - ay), radius + r) - r)
        return max(best, 0.0)

    def inside_edge(self, bbox, inset: float = 0.6) -> bool:
        if self.edge is None:
            return True
        e = self.edge
        return bbox[0] > e[0] + inset and bbox[1] > e[1] + inset and bbox[2] < e[2] - inset and bbox[3] < e[3] - inset

    def next_uid(self, rng: random.Random) -> str:
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))

    def _tail(self, rng: random.Random) -> str:
        if self.version >= 20230620:
            return f' (uuid "{self.next_uid(rng)}")'
        if self.version >= 20211014:
            return f" (tstamp {self.next_uid(rng)})"
        return ""

    def _net_txt(self, net: str | None) -> str:
        if net is None or net == "0":
            return " (net 0)" if (self.net_style_int or self.version < 20260101) else ""
        return f" (net {net})"

    def new_segment(self, a, b, w, layer: str, net: str | None, rng) -> Node:
        lay = f'"{layer}"' if self.version >= 20211014 else layer
        return sexpr.parse_node(f"(segment (start {fmt(a[0])} {fmt(a[1])}) (end {fmt(b[0])} {fmt(b[1])}) "
                                f"(width {fmt(w)}) (layer {lay}){self._net_txt(net)}{self._tail(rng)})")

    def clone_segment(self, host: Seg, a, b, w, rng) -> Node:
        n = host.node.copy()
        n.set_val("start", fmt(a[0]), fmt(a[1]))
        n.set_val("end", fmt(b[0]), fmt(b[1]))
        n.set_val("width", fmt(w))
        for key in ("uuid", "tstamp"):
            if n.find(key) is not None:
                n.set_val(key, f'"{self.next_uid(rng)}"' if key == "uuid" else self.next_uid(rng))
        return n

    def dumps(self) -> str:
        return self.doc.dumps()


def board_edge_bbox(root: Node) -> tuple | None:
    """Bounding box of everything on Edge.Cuts (stroke width included), board mm.  None if there is no outline."""
    pts: list[tuple[float, float]] = []
    pad = 0.0

    def add_from(k: Node, ox=0.0, oy=0.0) -> None:
        nonlocal pad
        nm = k.name
        w = k.floats("width") or [0.0]
        st = k.find("stroke")
        if st is not None and st.floats("width"):
            w = st.floats("width")
        pad = max(pad, w[0] / 2)
        if nm in ("gr_line", "gr_rect", "gr_circle", "gr_arc", "gr_curve", "gr_poly"):
            for key in ("start", "end", "mid", "center"):
                f = k.floats(key)
                if len(f) >= 2:
                    pts.append((f[0], f[1]))
            if nm == "gr_poly" or nm == "gr_curve":
                pts.extend(_poly_pts(k))
            if nm == "gr_circle":
                c, e = k.floats("center"), k.floats("end")
                if len(c) >= 2 and len(e) >= 2:
                    r = math.hypot(e[0] - c[0], e[1] - c[1])
                    pts.extend([(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)])
            if nm == "gr_arc":
                _arc_points(k, pts)
    for k in root.nodes():
        if k.name.startswith("gr_") and k.val("layer", "") == "Edge.Cuts":
            add_from(k)
    if not pts:
        return None
    xs, ys = zip(*pts)
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _arc_points(k: Node, pts: list) -> None:
    """Extreme points of an arc.  v6+ has start/mid/end; v5 has centre/start-point/angle."""
    st, mid, en = k.floats("start"), k.floats("mid"), k.floats("end")
    if len(mid) >= 2 and len(st) >= 2 and len(en) >= 2:
        pts.extend([tuple(st[:2]), tuple(mid[:2]), tuple(en[:2])])
        (x1, y1), (x2, y2), (x3, y3) = st[:2], mid[:2], en[:2]
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d) < 1e-9:
            return
        ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
        uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
        r = math.hypot(x1 - ux, y1 - uy)
        a1, a2, a3 = (math.atan2(y - uy, x - ux) for (x, y) in ((x1, y1), (x2, y2), (x3, y3)))

        def between(a, lo, hi, ccw):
            return ((a - lo) % (2 * math.pi)) <= ((hi - lo) % (2 * math.pi)) if ccw else ((lo - a) % (2 * math.pi)) <= ((lo - hi) % (2 * math.pi))
        ccw = ((a2 - a1) % (2 * math.pi)) < ((a3 - a1) % (2 * math.pi))
        for ang in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
            if between(ang, a1, a3, ccw):
                pts.append((ux + r * math.cos(ang), uy + r * math.sin(ang)))
    elif len(st) >= 2 and len(en) >= 2:        # v5: start = centre, end = a point on the arc, angle in degrees
        r = math.hypot(en[0] - st[0], en[1] - st[1])
        pts.extend([(st[0] - r, st[1] - r), (st[0] + r, st[1] + r)])     # conservative: full circle


# ======================================================================================================
# connectivity (for 'open')
# ======================================================================================================
class _UF:
    def __init__(self):
        self.p: dict = {}

    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        self.p[self.f(a)] = self.f(b)


def _net_components(b: Board, net: str, replace: tuple | None = None):
    """Union-find of the copper items of ``net`` (all layers).  ``replace=(host_seg, [(a,b),(a,b)])`` swaps one segment.

    Returns (uf, key_fn, terminal_keys).  Items connect where endpoints coincide (1 um), where an endpoint lies on
    another segment's centreline (T-junction), at vias (both layers) and where an endpoint lies inside a pad.
    """
    uf = _UF()
    key = lambda p, l: (round(p[0], 3), round(p[1], 3), l)
    segs = []
    for s in b.segs:
        if s.net != net or s.layer not in ("F.Cu", "B.Cu"):
            continue
        if replace is not None and s is replace[0]:
            for (pa, pb) in replace[1]:
                segs.append((pa, pb, s.w, s.layer))
        else:
            segs.append((s.a, s.b, s.w, s.layer))
    terms: set = set()
    ends = []
    for (pa, pb, w, l) in segs:
        uf.u(key(pa, l), key(pb, l))
        ends.append((pa, l))
        ends.append((pb, l))
    # T-junctions: endpoint on the interior of another same-layer segment
    for (pa, pb, w, l) in segs:
        for (q, ql) in ends:
            if ql != l:
                continue
            if _pt_seg(q[0], q[1], pa[0], pa[1], pb[0], pb[1])[0] < 0.005:
                uf.u(key(q, l), key(pa, l))
    for v in b.vias:
        if v.net != net:
            continue
        for l in ("F.Cu", "B.Cu"):
            uf.u(("via", id(v)), key((v.x, v.y), l))
        for (q, ql) in ends:
            if math.hypot(q[0] - v.x, q[1] - v.y) <= v.size / 2 + 0.005:
                uf.u(("via", id(v)), key(q, ql))
        terms.add(("via", id(v)))
    for p in b.pads:
        if p.net != net:
            continue
        pk = ("pad", id(p))
        terms.add(pk)
        layers = ("F.Cu", "B.Cu") if p.ptype in ("thru_hole",) or "*.Cu" in p.layers or "F&B.Cu" in p.layers else (("B.Cu",) if p.flipped else ("F.Cu",))
        for (q, ql) in ends:
            if ql in layers and pt_box_dist(q[0], q[1], *p.box) <= 0.005:
                uf.u(pk, key(q, ql))
        uf.f(pk)
    return uf, key, terms


def breaks_net(b: Board, host: Seg, pieces: list[tuple]) -> bool:
    """True if replacing ``host`` by ``pieces`` ([(a,b),(a,b)]) leaves the two cut ends in different components that
    each contain a pad or via (a genuine open between two terminals)."""
    net = host.net
    if _is_none_net(net) or net in b.zone_nets:
        return False
    uf, key, terms = _net_components(b, net, (host, pieces))
    l = host.layer
    ka, kb = key(pieces[0][1], l), key(pieces[1][0], l)
    ra, rb = uf.f(ka), uf.f(kb)
    if ra == rb:
        return False
    ta = any(uf.f(t) == ra for t in terms)
    tb = any(uf.f(t) == rb for t in terms)
    return ta and tb


# ======================================================================================================
# injectors
# ======================================================================================================
def _sep_ok(bbox, existing: list[tuple], sep: float) -> bool:
    x1, y1, x2, y2 = bbox
    for (a, b, c, d) in existing:
        if not (x2 + sep < a or c + sep < x1 or y2 + sep < b or d + sep < y1):
            return False
    return True


def _bbox_pts(pts: Iterable[tuple]) -> tuple:
    xs, ys = zip(*pts)
    return (min(xs), min(ys), max(xs), max(ys))


def _unit(a, b):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    return ((b[0] - a[0]) / L, (b[1] - a[1]) / L), L


class _Ctx:
    def __init__(self, b: Board, rng: random.Random, sep: float):
        self.b, self.rng, self.sep = b, rng, sep
        self.bboxes: list[tuple] = []
        self.consumed: set[int] = set()

    def ok(self, bbox) -> bool:
        return _sep_ok(bbox, self.bboxes, self.sep) and self.b.inside_edge(bbox, 0.3)


def _top_segs(c: _Ctx, min_len: float, min_w: float = 0.0, need_net: bool = False) -> list[Seg]:
    out = [s for s in c.b.segs if s.layer == "F.Cu" and s.length >= min_len and s.w >= min_w and id(s) not in c.consumed
           and (not need_net or not _is_none_net(s.net))]
    c.rng.shuffle(out)
    return out


def inject_open(c: _Ctx, size: float, tries: int = 300) -> Defect | None:
    b, rng = c.b, c.rng
    for s in _top_segs(c, size + 0.6, need_net=True)[:tries]:
        u, L = _unit(s.a, s.b)
        span = size + s.w                         # centre-to-centre distance of the new cap centres
        if L < span + 0.5:
            continue
        t = rng.uniform(span / 2 + 0.25, L - span / 2 - 0.25)
        p1 = (s.a[0] + u[0] * (t - span / 2), s.a[1] + u[1] * (t - span / 2))
        p2 = (s.a[0] + u[0] * (t + span / 2), s.a[1] + u[1] * (t + span / 2))
        n = (-u[1], u[0])
        tip1 = (p1[0] + u[0] * s.w / 2, p1[1] + u[1] * s.w / 2)
        tip2 = (p2[0] - u[0] * s.w / 2, p2[1] - u[1] * s.w / 2)
        # removed copper = track between the cap centres minus the two cap half-discs: its bbox is p1..p2 x width
        bb = _bbox_pts([(q[0] + n[0] * sg * s.w / 2, q[1] + n[1] * sg * s.w / 2) for q in (p1, p2) for sg in (1, -1)])
        if not c.ok(bb):
            continue
        pieces = [(s.a, p1), (p2, s.b)]
        if not breaks_net(b, s, pieces):
            continue
        # the gap must not be bridged by foreign copper (e.g. another track crossing right there)
        gap = cap(tip1, tip2, s.w / 2 * 0.98, None)
        if b.clearance(gap, ignore=[s], radius=0.05) < 0.02:
            continue
        n1 = b.clone_segment(s, s.a, p1, s.w, rng)
        n2 = b.clone_segment(s, p2, s.b, s.w, rng)
        b.root.replace(s.node, [n1, n2])
        c.consumed.add(id(s))
        return Defect("open", bb, size, "F.Cu", {"net": s.net, "track_w_mm": s.w})
    return None


def _short_candidates(c: _Ctx, size: float, limit_segs: int = 400):
    b, rng = c.b, c.rng
    cands = []
    segs = [s for s in b.segs if s.layer == "F.Cu" and not _is_none_net(s.net) and id(s) not in c.consumed and s.length > 0.3]
    rng.shuffle(segs)
    for s in segs[:limit_segs]:
        u, L = _unit(s.a, s.b)
        for _ in range(3):
            t = rng.uniform(0.15, 0.85) * L
            p = (s.a[0] + u[0] * t, s.a[1] + u[1] * t)
            for o in b.grid.near((p[0], p[1], p[0], p[1]), size * 2 + 1.0):
                if o.net is None or _is_none_net(o.net) or o.net == s.net or o.ref is s or id(o.ref) in c.consumed:
                    continue
                if o.what == "seg" or (o.what == "pad" and o.kind == "cap"):
                    d, tt = _pt_seg(p[0], p[1], *o.g[:4])
                    qx = o.g[0] + tt * (o.g[2] - o.g[0])
                    qy = o.g[1] + tt * (o.g[3] - o.g[1])
                    ro = o.g[4]
                elif o.what == "pad":
                    x1, y1, x2, y2 = o.g
                    qx, qy = min(max(p[0], x1), x2), min(max(p[1], y1), y2)
                    d, ro = math.hypot(p[0] - qx, p[1] - qy), 0.0
                    if d < 1e-9:
                        continue
                elif o.what == "via":
                    d = math.hypot(p[0] - o.g[0], p[1] - o.g[1])
                    qx, qy, ro = o.g[0], o.g[1], o.g[4]
                else:
                    continue
                gap = d - s.w / 2 - ro
                if 0.05 <= gap <= size * 2.0 + 0.2:
                    cands.append((abs(gap - size), gap, s, o, p, (qx, qy), ro))
    cands.sort(key=lambda t: t[0])
    return cands


def inject_short(c: _Ctx, size: float, bridge_w: float | None = None) -> Defect | None:
    b, rng = c.b, c.rng
    bw = bridge_w if bridge_w is not None else round(rng.uniform(0.12, 0.3), 4)
    cands = _short_candidates(c, size)
    for (_, gap, s, o, p, q, ro) in cands[:80]:
        u, dist = _unit(p, q)
        p_edge = (p[0] + u[0] * s.w / 2, p[1] + u[1] * s.w / 2)
        q_edge = (q[0] - u[0] * ro, q[1] - u[1] * ro)
        if o.what == "pad" and o.kind == "box":
            q = (q[0] + u[0] * 0.05, q[1] + u[1] * 0.05)          # end slightly inside the pad
        else:                                                      # keep the round end caps inside the tracks' width
            ep, eq = max(0.0, bw / 2 - s.w / 2), max(0.0, bw / 2 - ro)
            p = (p[0] + u[0] * ep, p[1] + u[1] * ep)
            q = (q[0] - u[0] * eq, q[1] - u[1] * eq)
        bridge = cap(p, q, bw / 2, None)
        third = [t for t in b.grid.near(bridge.bbox, 0.1) if t.ref is not s and t.ref is not o.ref
                 and t.net not in (s.net, o.net) and shape_dist(bridge, t) < 0.1]
        if third or b.fills.dist((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, 0.2) < 0.1:
            continue
        # the visible bridge = bridge copper outside both objects; sample along it with true distances
        ptsx = []
        for i in range(41):
            tt = i / 40
            sp = (p[0] + (q[0] - p[0]) * tt, p[1] + (q[1] - p[1]) * tt)
            pt = Shape("cap", (sp[0], sp[1], sp[0], sp[1], 0.0), None)
            if shape_dist(pt, cap(s.a, s.b, s.w / 2)) > 0.0 and shape_dist(pt, o) > 0.0:
                ptsx.append(sp)
        if not ptsx:
            ptsx = [((p_edge[0] + q_edge[0]) / 2, (p_edge[1] + q_edge[1]) / 2)]
        nrm = (-u[1], u[0])
        x1, y1, x2, y2 = _bbox_pts([(sp[0] + nrm[0] * sg * bw / 2, sp[1] + nrm[1] * sg * bw / 2) for sp in ptsx for sg in (1, -1)])
        bb = (x1 - 0.03, y1 - 0.03, x2 + 0.03, y2 + 0.03)
        if not c.ok(bb):
            continue
        node = b.new_segment(p, q, bw, "F.Cu", None, rng)
        b.root.append(node)
        c.consumed.add(id(s))
        if o.what != "via":
            c.consumed.add(id(o.ref))
        return Defect("short", bb, gap, "F.Cu", {"bridge_w_mm": bw, "nets": [s.net, o.net]})
    return None


def inject_spur(c: _Ctx, size: float, tries: int = 300) -> Defect | None:
    b, rng = c.b, c.rng
    for s in _top_segs(c, 0.8, 0.1, need_net=True)[:tries]:
        u, L = _unit(s.a, s.b)
        t = rng.uniform(0.25, 0.75) * L
        c0 = (s.a[0] + u[0] * t, s.a[1] + u[1] * t)
        sw = round(min(rng.uniform(0.08, 0.2), s.w * 0.8), 4)
        for side in rng.sample([1, -1], 2):
            n = (-u[1] * side, u[0] * side)
            ang = rng.uniform(-0.5, 0.5)                          # slight lean from the normal
            d = (n[0] * math.cos(ang) - n[1] * math.sin(ang), n[0] * math.sin(ang) + n[1] * math.cos(ang))
            reach = s.w / 2 + size - sw / 2
            tip = (c0[0] + d[0] * reach, c0[1] + d[1] * reach)
            stub = cap(c0, tip, sw / 2, None)
            if b.clearance(stub, ignore=[s], radius=0.2) < 0.12:
                continue
            ptsx = []
            for i in range(31):
                tt = i / 30
                sp = (c0[0] + (tip[0] - c0[0]) * tt, c0[1] + (tip[1] - c0[1]) * tt)
                if _pt_seg(sp[0], sp[1], s.a[0], s.a[1], s.b[0], s.b[1])[0] > s.w / 2:
                    ptsx.append(sp)
            if not ptsx:
                continue
            x1, y1, x2, y2 = _bbox_pts(ptsx)
            bb = (x1 - sw / 2, y1 - sw / 2, x2 + sw / 2, y2 + sw / 2)
            if not c.ok(bb):
                continue
            node = b.new_segment(c0, tip, sw, "F.Cu", s.net, rng)
            b.root.append(node)
            c.consumed.add(id(s))
            return Defect("spur", bb, size, "F.Cu", {"net": s.net, "stub_w_mm": sw})
    return None


def _blob(rng: random.Random, cx: float, cy: float, diam: float) -> list[tuple]:
    n = rng.randint(6, 9)
    a0 = rng.uniform(0, 2 * math.pi)
    ex = rng.uniform(0.75, 1.0)
    pts = []
    for i in range(n):
        a = a0 + 2 * math.pi * i / n
        r = diam / 2 * rng.uniform(0.7, 1.1)
        pts.append((cx + r * math.cos(a), cy + r * ex * math.sin(a)))
    return pts


def _poly_text(b: Board, pts: list[tuple], rng) -> str:
    xy = " ".join(f"(xy {fmt(x)} {fmt(y)})" for x, y in pts)
    tail = b._tail(rng)
    v = b.version
    if v >= 20221018:
        return f'(gr_poly (pts {xy}) (stroke (width 0) (type solid)) (fill solid) (layer "F.Cu"){tail})'
    if v >= 20211014:
        return f'(gr_poly (pts {xy}) (layer "F.Cu") (width 0) (fill solid){tail})'
    return f"(gr_poly (pts {xy}) (layer F.Cu) (width 0))"


def inject_spurious_copper(c: _Ctx, size: float, tries: int = 400) -> Defect | None:
    b, rng = c.b, c.rng
    anchors = [s for s in b.grid.items if s.what in ("seg", "pad")]
    if not anchors:
        return None
    for _ in range(tries):
        a = rng.choice(anchors)
        x1, y1, x2, y2 = a.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        reach = max(x2 - x1, y2 - y1) / 2 + size / 2 + rng.uniform(0.25, 1.2)
        th = rng.uniform(0, 2 * math.pi)
        px, py = cx + reach * math.cos(th), cy + reach * math.sin(th)
        pts = _blob(rng, px, py, size)
        bb = _bbox_pts(pts)
        if not c.ok(bb):
            continue
        probe = Shape("box", (bb[0], bb[1], bb[2], bb[3]), None)
        if b.clearance(probe, radius=0.3) < 0.2:
            continue
        if b.fills.dist(px, py, 1.0) < size / 2 + 0.2:
            continue
        b.root.append(_poly_text(b, pts, rng))
        return Defect("spurious_copper", bb, size, "F.Cu", {"vertices": len(pts)})
    return None


def inject_mouse_bite(c: _Ctx, size: float, tries: int = 300) -> Defect | None:
    b, rng = c.b, c.rng
    for s in _top_segs(c, size + 0.8, 0.2)[:tries]:
        u, L = _unit(s.a, s.b)
        w = s.w
        if w > 1.0:
            continue
        depth = round(rng.uniform(0.3, 0.5) * w, 4)
        wn = w - depth
        # the neighbours' round caps (radius w/2) must not close the notch: opening at the edge >= 2.6 * cap chord
        size = max(size, 2.6 * math.sqrt(max(w * depth - depth * depth, 0.0)))
        if L < size + 0.6 + w:
            continue
        t0 = rng.uniform(0.3 + w / 2, L - size - 0.3 - w / 2)
        p0 = (s.a[0] + u[0] * t0, s.a[1] + u[1] * t0)               # centre of the first full-width cap
        p1 = (p0[0] + u[0] * size, p0[1] + u[1] * size)
        side = rng.choice([1, -1])
        n = (-u[1] * side, u[0] * side)                              # direction of the bitten edge
        off = -depth / 2                                              # narrow piece flush with the opposite edge
        q0 = (p0[0] + n[0] * off, p0[1] + n[1] * off)
        q1 = (p1[0] + n[0] * off, p1[1] + n[1] * off)
        e0 = (p0[0] + n[0] * w / 2, p0[1] + n[1] * w / 2)
        e1 = (p1[0] + n[0] * w / 2, p1[1] + n[1] * w / 2)
        i0 = (p0[0] + n[0] * (w / 2 - depth), p0[1] + n[1] * (w / 2 - depth))
        i1 = (p1[0] + n[0] * (w / 2 - depth), p1[1] + n[1] * (w / 2 - depth))
        bb = _bbox_pts([e0, e1, i0, i1])
        if not c.ok(bb):
            continue
        if b.clearance(Shape("box", bb, None), ignore=[s], radius=0.05) < 0.02:     # other copper would fill the notch
            continue
        parts = [b.clone_segment(s, s.a, p0, w, rng), b.clone_segment(s, q0, q1, wn, rng), b.clone_segment(s, p1, s.b, w, rng)]
        b.root.replace(s.node, parts)
        c.consumed.add(id(s))
        return Defect("mouse_bite", bb, size, "F.Cu", {"depth_mm": depth, "track_w_mm": w, "net": s.net})
    return None


def inject_pin_hole(c: _Ctx, size: float, tries: int = 300) -> Defect | None:
    b, rng = c.b, c.rng
    for s in _top_segs(c, 1.0, 0.3)[:tries]:
        u, L = _unit(s.a, s.b)
        w = s.w
        hw = min(size, w * 0.5)
        rw = (w - hw) / 2
        if rw < 0.06:
            continue
        r = w / 2
        lb = hw                                                       # hole length between the full-width caps
        if L < lb + 2 * r + 0.6:
            continue
        t0 = rng.uniform(0.3, L - lb - 2 * r - 0.3)
        p0 = (s.a[0] + u[0] * t0, s.a[1] + u[1] * t0)
        p1 = (p0[0] + u[0] * (lb + 2 * r), p0[1] + u[1] * (lb + 2 * r))
        n = (-u[1], u[0])
        k = math.sqrt(max(r * r - (hw / 2) ** 2, 0.0))
        hole_a = (p0[0] + u[0] * k, p0[1] + u[1] * k)
        hole_b = (p1[0] - u[0] * k, p1[1] - u[1] * k)
        bb = _bbox_pts([(hole_a[0] + n[0] * hw / 2, hole_a[1] + n[1] * hw / 2), (hole_a[0] - n[0] * hw / 2, hole_a[1] - n[1] * hw / 2),
                        (hole_b[0] + n[0] * hw / 2, hole_b[1] + n[1] * hw / 2), (hole_b[0] - n[0] * hw / 2, hole_b[1] - n[1] * hw / 2)])
        if not c.ok(bb):
            continue
        if b.clearance(Shape("box", bb, None), ignore=[s], radius=0.05) < 0.02:     # other copper would fill the hole
            continue
        sh = hw / 2 + rw / 2
        parts = [b.clone_segment(s, s.a, p0, w, rng)]
        for sgn in (1, -1):
            o = (n[0] * sh * sgn, n[1] * sh * sgn)
            parts.append(b.clone_segment(s, (p0[0] + o[0], p0[1] + o[1]), (p1[0] + o[0], p1[1] + o[1]), rw, rng))
        parts.append(b.clone_segment(s, p1, s.b, w, rng))
        b.root.replace(s.node, parts)
        c.consumed.add(id(s))
        return Defect("pin_hole", bb, hw, "F.Cu", {"track_w_mm": w, "net": s.net})
    return None


def inject_missing_pad(c: _Ctx, size: float = 0.0, tries: int = 300) -> Defect | None:
    b, rng = c.b, c.rng
    pads = [p for p in b.pads if p.ptype == "smd" and p.on_top and not p.flipped and p.shape != "custom"
            and min(p.hx, p.hy) * 2 >= 0.3 and id(p) not in c.consumed]
    rng.shuffle(pads)
    for p in pads[:tries]:
        bb = p.box
        if not c.ok(bb):
            continue
        p.fp.remove(p.node)
        c.consumed.add(id(p))
        return Defect("missing_pad", bb, max(p.hx, p.hy) * 2, "F.Cu", {"net": p.net, "pad_shape": p.shape})
    return None


def inject_missing_hole(c: _Ctx, size: float = 0.0, tries: int = 300) -> Defect | None:
    b, rng = c.b, c.rng
    pads = [p for p in b.pads if p.ptype == "thru_hole" and p.drill and p.drill >= 0.4 and id(p) not in c.consumed
            and p.shape != "custom"]
    rng.shuffle(pads)
    for p in pads[:tries]:
        r = p.drill / 2 + 0.05
        bb = (p.x - r, p.y - r, p.x + r, p.y + r)
        if not c.ok(bb):
            continue
        pn = p.node
        a = pn.kids
        idx = [i for i, k in enumerate(a) if isinstance(k, str) and not k.isspace()]
        a[idx[2]] = "smd"
        dn = pn.find("drill")
        if dn is not None:
            pn.remove(dn)
        for extra in ("remove_unused_layers", "keep_end_layers"):
            e = pn.find(extra)
            if e is not None:
                pn.remove(e)
        ln = pn.find("layers")
        if ln is not None:
            vals = []
            for v in ln.args[1:]:
                v = sexpr.unquote(v)
                vals += {"*.Cu": ["F.Cu", "B.Cu"], "*.Mask": ["F.Mask", "B.Mask"], "F&B.Cu": ["F.Cu", "B.Cu"]}.get(v, [v])
            ln.kids = [ln.kids[0]] + [x for v in vals for x in (" ", f'"{v}"' if b.version >= 20211014 else v)]
        c.consumed.add(id(p))
        return Defect("missing_hole", bb, p.drill, "F.Cu", {"net": p.net, "drill_mm": p.drill})
    return None


_INJECTORS: dict[str, Callable] = {
    "open": inject_open, "short": inject_short, "spur": inject_spur, "spurious_copper": inject_spurious_copper,
    "mouse_bite": inject_mouse_bite, "pin_hole": inject_pin_hole, "missing_pad": inject_missing_pad,
    "missing_hole": inject_missing_hole,
}


def inject_many(text: str, counts: dict[str, int], seed: int = 0, min_sep_mm: float = 2.5,
                sizes: dict[str, float] | None = None) -> tuple[str, list[Defect]]:
    """Inject ``counts[cls]`` defects of each class into the board ``text``.

    Returns (modified board text, defects).  Defects are separated by at least ``min_sep_mm`` between bounding boxes.
    Seeded: identical inputs give identical output.  Fewer defects than requested are returned when the board has
    no suitable location.
    """
    b = Board(text)
    rng = random.Random(seed)
    ctx = _Ctx(b, rng, min_sep_mm)
    order: list[str] = []
    for cls, n in counts.items():
        if cls not in _INJECTORS:
            raise KeyError(f"unknown defect class {cls!r}")
        order += [cls] * int(n)
    rng.shuffle(order)
    out: list[Defect] = []
    for cls in order:
        lo, hi = SIZE_RANGE.get(cls, (0.0, 0.0))
        size = (sizes or {}).get(cls) or (rng.uniform(lo, hi) if hi else 0.0)
        d = _INJECTORS[cls](ctx, size)
        if d is not None:
            ctx.bboxes.append(d.bbox_mm)
            out.append(d)
    return b.dumps(), out


def defects_to_json(defects: list[Defect]) -> list[dict]:
    return [d.to_dict() for d in defects]
