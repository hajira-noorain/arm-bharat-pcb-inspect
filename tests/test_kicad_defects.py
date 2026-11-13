"""Design-level defect pipeline: s-expression round trip, injectors, bounding boxes, mm->pixel mapping.

Uses a tiny hand-written .kicad_pcb (two nets, tracks, SMD + through-hole pads, one zone with a fill, an outline).
No KiCad is needed: the renderer's numpy rasteriser draws the same copper the injector edited.
"""
import math
import random

import numpy as np
import pytest

from training.kicad import inject as I
from training.kicad import render as R
from training.kicad import sexpr
from training.kicad import sources as S


def fixture(version=20221018):
    q = (lambda s: f'"{s}"') if version >= 20211014 else (lambda s: s)
    tail = (lambda u: f" (tstamp {u})") if version < 20230620 else (lambda u: f' (uuid "{u}")')
    if version < 20211014:
        tail = lambda u: ""
    seg = lambda x1, y1, x2, y2, w, n, u: (f"  (segment (start {x1} {y1}) (end {x2} {y2}) (width {w}) (layer {q('F.Cu')}) "
                                          f"(net {n}){tail(u)})")
    fp_key = "footprint" if version >= 20211014 else "module"
    edge = (f'  (gr_rect (start 0 0) (end 40 30) (stroke (width 0.1) (type default)) (fill none) (layer "Edge.Cuts"))'
            if version >= 20221018 else
            "\n".join(f"  (gr_line (start {a}) (end {b}) (layer Edge.Cuts) (width 0.1))" for a, b in
                      (("0 0", "40 0"), ("40 0", "40 30"), ("40 30", "0 30"), ("0 30", "0 0"))))
    return f"""(kicad_pcb (version {version}) (generator pcbnew)

  (general
    (thickness 1.6)
  )
  (layers
    (0 {q('F.Cu')} signal)
    (31 {q('B.Cu')} signal)
    (37 {q('F.SilkS')} user)
    (39 {q('F.Mask')} user)
    (44 {q('Edge.Cuts')} user)
  )
  (net 0 "")
  (net 1 "A")
  (net 2 "B")
  ({fp_key} "R:R1" (layer {q('F.Cu')}) (at 20 10)
    (pad "1" smd rect (at -10 0) (size 0.6 0.6) (layers {q('F.Cu')} {q('F.Mask')}) (net 1 "A"))
    (pad "2" smd rect (at 10 0) (size 0.6 0.6) (layers {q('F.Cu')} {q('F.Mask')}) (net 1 "A"))
  )
  ({fp_key} "R:R2" (layer {q('F.Cu')}) (at 20 11)
    (pad "1" smd rect (at -10 0) (size 0.6 0.6) (layers {q('F.Cu')} {q('F.Mask')}) (net 2 "B"))
    (pad "2" smd roundrect (at 10 0) (size 0.6 0.6) (layers {q('F.Cu')} {q('F.Mask')}) (net 2 "B"))
  )
  ({fp_key} "J:J1" (layer {q('F.Cu')}) (at 20 22)
    (pad "1" thru_hole circle (at 0 0) (size 1.7 1.7) (drill 1) (layers *.Cu *.Mask) (net 2 "B"))
  )
{seg(10, 10, 30, 10, 0.4, 1, "11111111-1111-1111-1111-111111111111")}
{seg(10, 11, 30, 11, 0.25, 2, "22222222-2222-2222-2222-222222222222")}
{seg(10, 11, 10, 20, 0.25, 2, "33333333-3333-3333-3333-333333333333")}
{edge}
  (zone (net 2) (net_name "B") (layer {q('F.Cu')}) (hatch edge 0.5)
    (connect_pads (clearance 0.2))
    (min_thickness 0.2)
    (polygon (pts (xy 5 24) (xy 35 24) (xy 35 28) (xy 5 28)))
    (filled_polygon (layer {q('F.Cu')}) (pts (xy 5 24) (xy 35 24) (xy 35 28) (xy 5 28)))
  )
)
"""


FIX = fixture()
TRACK1 = ((10.0, 10.0), (30.0, 10.0), 0.4)     # net 1
TRACK2 = ((10.0, 11.0), (30.0, 11.0), 0.25)    # net 2


def mapping(board, ppm=50.0):
    return R.Mapping.from_edge(board, ppm, 1.0)


def one(cls, seed=1, sizes=None, text=FIX, sep=2.5):
    out, ds = I.inject_many(text, {cls: 1}, seed=seed, min_sep_mm=sep, sizes=sizes)
    return out, ds


def f(*a):
    return float(a[0])


# ---------------------------------------------------------------------------------------------- s-expr
@pytest.mark.parametrize("version", [20171130, 20211014, 20221018, 20241229])
def test_roundtrip_identity_fixture(version):
    t = fixture(version)
    assert sexpr.parse(t).dumps() == t


def test_roundtrip_awkward_text():
    t = '\r\n(kicad_pcb  (version 20221018)\t(x "a ( b ) \\" c")\r\n  (empty)()  (n -1.5e-3 +2 .5)  \n) \n\n'
    d = sexpr.parse(t)
    assert d.dumps() == t
    assert sexpr.unquote(d.root.find("x").args[1]) == 'a ( b ) " c'


def test_unbalanced_rejected():
    for bad in ["(a (b)", "(a))", "", "(a)(b)"]:
        with pytest.raises(ValueError):
            sexpr.parse(bad)


def test_edit_changes_only_the_edited_part():
    d = sexpr.parse(FIX)
    seg = d.root.findall("segment")[0]
    seg.set_val("width", 0.5)
    new = d.dumps()
    assert new.replace("(width 0.5)", "(width 0.4)", 1) == FIX


def test_real_board_roundtrip_if_present():
    import glob, os
    sp = os.environ.get("KICAD_SAMPLE_DIR")
    for p in (glob.glob(os.path.join(sp, "*.kicad_pcb")) if sp else []):
        t = open(p, encoding="utf8", newline="").read()
        assert sexpr.parse(t).dumps() == t


# ---------------------------------------------------------------------------------------------- model
def test_board_model_reads_fixture():
    b = I.Board(FIX)
    assert b.version == 20221018 and len(b.segs) == 3 and len(b.pads) == 5 and len(b.vias) == 0
    assert b.zone_nets == {"2"} and b.edge == pytest.approx((-0.05, -0.05, 40.05, 30.05))
    smd = [p for p in b.pads if p.ptype == "smd"]
    assert (smd[0].x, smd[0].y) == pytest.approx((10.0, 10.0))
    assert b.pads[-1].drill == 1.0 and b.pads[-1].ptype == "thru_hole"


def test_class_names_are_pcbinspect_classes():
    from pcbinspect.record import BARE_CLASSES
    I.check_class_names()
    assert set(I.DEFECT_CLASSES) <= set(BARE_CLASSES)


# ---------------------------------------------------------------------------------------------- injectors
def segs_of(text):
    return I.Board(text).segs


def test_open_removes_an_interior_piece_and_breaks_the_net():
    out, ds = one("open", sizes={"open": 0.3})
    assert len(ds) == 1 and ds[0].cls == "open"
    sexpr.parse(out)
    n1 = [s for s in segs_of(out) if s.net == "1"]
    assert len(n1) == 2
    (a, b) = sorted(n1, key=lambda s: s.a[0])
    assert a.a == pytest.approx((10, 10)) and b.b == pytest.approx((30, 10))
    # cap tip to cap tip the stubs are `size` apart
    tip_gap = (b.a[0] - b.w / 2) - (a.b[0] + a.w / 2)
    assert tip_gap == pytest.approx(0.3, abs=1e-4)
    d = ds[0]
    assert d.size_mm == pytest.approx(0.3)
    x1, y1, x2, y2 = d.bbox_mm
    assert (x2 - x1) == pytest.approx(0.3 + 0.4, abs=1e-4) and (y2 - y1) == pytest.approx(0.4, abs=1e-4)
    assert 10.5 < x1 and x2 < 29.5
    assert I.Board(FIX).segs[0].length == pytest.approx(20.0)
    # the other net is untouched
    assert len([s for s in segs_of(out) if s.net == "2"]) == 2


def test_open_never_picks_a_net_with_a_zone():
    for seed in range(15):
        out, ds = one("open", seed=seed)
        for d in ds:
            assert d.meta["net"] == "1"


def test_short_bridges_different_nets_with_the_expected_width():
    out, ds = I.inject_many(FIX, {"short": 1}, seed=3, sizes={"short": 0.5})
    assert len(ds) == 1
    d = ds[0]
    new = [s for s in segs_of(out) if s.net in (None, "0")]
    assert len(new) == 1
    br = new[0]
    assert br.w == pytest.approx(d.meta["bridge_w_mm"], abs=1e-5) and 0.12 <= br.w <= 0.3
    assert set(d.meta["nets"]) == {"1", "2"}
    # gap between 0.5*0.7 and 2*size; recorded gap is the true edge-to-edge distance
    ba = I.Board(out)
    assert 0.05 <= d.size_mm <= 2 * 0.5 + 0.2                  # edge-to-edge gap near the target
    # both ends of the bridge lie on (or within) copper of the two different nets
    shapes = [s for s in I.Board(FIX).grid.items]
    ends_nets = set()
    for e in (br.a, br.b):
        pt = I.Shape("cap", (e[0], e[1], e[0], e[1], 0.0), None)
        for s in shapes:
            if I.shape_dist(pt, s) <= 1e-6:
                ends_nets.add(s.net)
    assert {"1", "2"} <= ends_nets
    x1, y1, x2, y2 = d.bbox_mm
    assert x1 <= min(br.a[0], br.b[0]) + 0.5 and x2 >= max(br.a[0], br.b[0]) - 0.5


def test_spur_is_a_stub_on_its_own_net_that_keeps_clearance():
    out, ds = one("spur", seed=5, sizes={"spur": 0.3})
    assert len(ds) == 1
    d = ds[0]
    base = {(s.a, s.b, s.w, s.net) for s in segs_of(FIX)}
    new = [s for s in segs_of(out) if (s.a, s.b, s.w, s.net) not in base]
    assert len(new) == 1
    sp = new[0]
    host_net = d.meta["net"]
    assert sp.net == host_net and sp.w < 0.41
    host = next(s for s in segs_of(FIX) if s.net == host_net and I._pt_seg(sp.a[0], sp.a[1], *s.a, *s.b)[0] < 1e-6)
    far = max(I._pt_seg(e[0], e[1], *host.a, *host.b)[0] for e in (sp.a, sp.b)) + sp.w / 2
    # protrusion measured along the stub is `size`; perpendicular to the track it is size*cos(lean), lean <= 0.5 rad
    assert 0.3 * math.cos(0.5) - 1e-3 <= far - host.w / 2 <= 0.3 + 1e-4
    other = [s for s in I.Board(FIX).grid.items if s.what in ("seg", "pad") and s.g[:4] != (*host.a, *host.b)]
    stub = I.cap(sp.a, sp.b, sp.w / 2)
    assert min(I.shape_dist(stub, o) for o in other) >= 0.12 - 1e-9


def test_spurious_copper_is_an_isolated_polygon():
    out, ds = one("spurious_copper", seed=2, sizes={"spurious_copper": 0.5})
    assert len(ds) == 1
    sexpr.parse(out)
    polys = [g for g in sexpr.parse(out).root.findall("gr_poly")]
    assert len(polys) == 1 and polys[0].val("layer") == "F.Cu"
    x1, y1, x2, y2 = ds[0].bbox_mm
    assert 0.2 < max(x2 - x1, y2 - y1) < 0.7
    probe = I.Shape("box", (x1, y1, x2, y2), None)
    assert min(I.shape_dist(probe, s) for s in I.Board(FIX).grid.items) >= 0.2 - 1e-9
    zx = I.Board(FIX).fills
    assert not zx.inside((x1 + x2) / 2, (y1 + y2) / 2)


def test_mouse_bite_narrows_a_section_flush_with_one_edge():
    out, ds = one("mouse_bite", seed=1, sizes={"mouse_bite": 0.4})
    assert len(ds) == 1
    d = ds[0]
    host = [s for s in segs_of(out) if s.net == d.meta["net"]]
    narrow = [s for s in host if s.w < d.meta["track_w_mm"] - 1e-6]
    assert len(narrow) == 1
    assert narrow[0].w == pytest.approx(d.meta["track_w_mm"] - d.meta["depth_mm"], abs=1e-4)
    # flush with the opposite edge: centre shifted by depth/2 perpendicular to the track
    full = next(s for s in host if s.w == pytest.approx(d.meta["track_w_mm"], abs=1e-6))
    assert abs(narrow[0].a[1] - full.a[1]) == pytest.approx(d.meta["depth_mm"] / 2, abs=1e-4)
    assert d.size_mm >= 0.4 - 1e-9


def test_pin_hole_leaves_an_enclosed_void():
    out, ds = one("pin_hole", seed=1, sizes={"pin_hole": 0.15})
    assert len(ds) == 1
    d = ds[0]
    b = I.Board(out)
    m = mapping(b, 100)
    img = R.raster_copper(b, m)
    x1, y1, x2, y2 = d.bbox_mm
    cx, cy = m.px((x1 + x2) / 2, (y1 + y2) / 2)
    assert img[int(cy), int(cx)] < 0.05                      # the void
    # copper on both sides across the track (rails) and on both ends along it
    w = d.meta["track_w_mm"]
    yc = (y1 + y2) / 2
    assert img[int(m.px(0, yc + (y2 - y1) / 2 + 0.03)[1]), int(cx)] > 0.9
    assert img[int(m.px(0, yc - (y2 - y1) / 2 - 0.03)[1]), int(cx)] > 0.9
    assert img[int(cy), int(m.px(x1 - w, 0)[0])] > 0.9 and img[int(cy), int(m.px(x2 + w, 0)[0])] > 0.9
    # the net is not opened: the rails carry current -> two parallel segments over the hole
    assert len([s for s in b.segs if s.net == d.meta["net"]]) >= 4


def test_missing_pad_removes_one_smd_pad():
    out, ds = one("missing_pad", seed=4)
    assert len(ds) == 1
    b0, b1 = I.Board(FIX), I.Board(out)
    assert len(b1.pads) == len(b0.pads) - 1
    d = ds[0]
    gone = [p for p in b0.pads if p.box == pytest.approx(d.bbox_mm)]
    assert len(gone) == 1 and gone[0].ptype == "smd"
    assert not any(p.box == pytest.approx(d.bbox_mm) for p in b1.pads)


def test_missing_hole_drops_the_drill():
    out, ds = one("missing_hole", seed=1)
    assert len(ds) == 1
    b1 = I.Board(out)
    p = [p for p in b1.pads if p.net == "2" and abs(p.x - 20) < 1e-6 and abs(p.y - 22) < 1e-6][0]
    assert p.ptype == "smd" and p.drill is None
    assert p.node.find("drill") is None
    layers = p.node.vals("layers")
    assert "F.Cu" in layers and "B.Cu" in layers and "*.Cu" not in layers
    d = ds[0]
    assert d.bbox_mm[0] == pytest.approx(20 - 0.55) and d.size_mm == pytest.approx(1.0)
    m = mapping(b1, 100)
    h0, h1 = R.raster_holes(I.Board(FIX), m), R.raster_holes(b1, m)
    assert h0.sum() > 0 and h1.sum() == 0


@pytest.mark.parametrize("version", [20171130, 20211014, 20221018, 20241229])
@pytest.mark.parametrize("cls", list(I.DEFECT_CLASSES))
def test_every_injector_writes_a_valid_file_in_every_format(version, cls):
    t = fixture(version)
    for seed in range(1, 12):
        out, ds = I.inject_many(t, {cls: 1}, seed=seed)
        if ds:
            break
    assert ds and ds[0].cls == cls, f"{cls} not injected in format {version}"
    doc = sexpr.parse(out)
    assert doc.dumps() == out
    I.Board(out)                                              # re-parses into the model
    if cls == "spurious_copper":
        g = doc.root.findall("gr_poly")[0]
        assert g.find("pts") is not None
        assert ("(layer F.Cu)" in str(g)) == (version < 20211014)
    if cls in ("open", "spur", "short", "mouse_bite", "pin_hole"):
        last = str(doc.root.findall("segment")[-1])
        if version >= 20230620:
            assert "(uuid" in last
        elif version >= 20211014:
            assert "(tstamp" in last
        else:
            assert "(tstamp" not in last and "(layer F.Cu)" in last


def test_defects_are_separated_and_seeded():
    t = FIX
    out1, d1 = I.inject_many(t, {c: 2 for c in I.DEFECT_CLASSES}, seed=11, min_sep_mm=2.0)
    out2, d2 = I.inject_many(t, {c: 2 for c in I.DEFECT_CLASSES}, seed=11, min_sep_mm=2.0)
    out3, d3 = I.inject_many(t, {c: 2 for c in I.DEFECT_CLASSES}, seed=12, min_sep_mm=2.0)
    assert out1 == out2 and [d.to_dict() for d in d1] == [d.to_dict() for d in d2]
    assert out1 != out3
    for i, a in enumerate(d1):
        for b in d1[i + 1:]:
            ax1, ay1, ax2, ay2 = a.bbox_mm
            bx1, by1, bx2, by2 = b.bbox_mm
            assert ax2 + 2.0 < bx1 or bx2 + 2.0 < ax1 or ay2 + 2.0 < by1 or by2 + 2.0 < ay1


def test_unknown_class_rejected():
    with pytest.raises(KeyError):
        I.inject_many(FIX, {"scratch": 1})


# ---------------------------------------------------------------------------------------------- bboxes + mapping
@pytest.mark.parametrize("cls", list(I.DEFECT_CLASSES))
def test_bbox_contains_exactly_the_copper_change(cls):
    ppm = 100.0
    found = 0
    for seed in range(1, 12):
        out, ds = one(cls, seed=seed)
        if not ds:
            continue
        found += 1
        d = ds[0]
        b0, b1 = I.Board(FIX), I.Board(out)
        m = mapping(b0, ppm)
        a = R.raster_copper(b0, m) + R.raster_holes(b0, m) * 0.0
        c = R.raster_copper(b1, m)
        diff = np.abs(a - c) + np.abs(R.raster_holes(b0, m) - R.raster_holes(b1, m))
        box = R.changed_bbox_mm(diff, np.zeros_like(diff), m)
        assert box is not None, "defect changed nothing"
        tol = 2.0 / ppm
        x1, y1, x2, y2 = d.bbox_mm
        assert box[0] >= x1 - tol and box[1] >= y1 - tol and box[2] <= x2 + tol and box[3] <= y2 + tol, (cls, box, d.bbox_mm)
        if cls in ("open", "spurious_copper", "missing_hole", "pin_hole"):
            # tight: the label is not much larger than the change
            assert (x2 - x1) - (box[2] - box[0]) < 0.12 and (y2 - y1) - (box[3] - box[1]) < 0.12, (cls, box, d.bbox_mm)
    assert found, f"{cls} never injected"


def test_mapping_places_open_defect_at_the_missing_copper_pixels():
    out, ds = one("open", seed=2, sizes={"open": 0.4})
    d = ds[0]
    b0, b1 = I.Board(FIX), I.Board(out)
    m = mapping(b0, 80)
    a, c = R.raster_copper(b0, m), R.raster_copper(b1, m)
    x1, y1, x2, y2 = m.bbox_px(d.bbox_mm)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    assert a[cy, cx] > 0.99 and c[cy, cx] < 0.01             # copper in the golden, gone in the defect, at the box centre
    ys, xs = np.nonzero(np.abs(a - c) > 0.25)
    assert xs.min() >= x1 - 1.5 and xs.max() + 1 <= x2 + 1.5 and ys.min() >= y1 - 1.5 and ys.max() + 1 <= y2 + 1.5
    # exactness: pixel <-> mm round trip
    assert m.mm(*m.px(12.34, 5.67)) == pytest.approx((12.34, 5.67))
    # a pixel's mm centre lies where the geometry says: the track centre row is y = 10
    row = int(round(m.px(0, 10.0)[1] - 0.5))
    assert a[row, int(m.px(20, 0)[0])] > 0.99


def test_mapping_is_not_symmetric_in_x_and_y():
    b = I.Board(FIX)
    m = mapping(b, 50)
    # track 1 is horizontal at y=10: copper rows span ~0.4 mm = 20 px tall and ~20 mm = 1000 px wide
    img = R.raster_copper(b, m)
    rows = np.nonzero(img[:int(m.px(0, 10.5)[1]), int(m.px(15, 0)[0])] > 0.5)[0]
    cols = np.nonzero(img[int(m.px(0, 10)[1]), :] > 0.5)[0]
    assert 18 <= (rows.max() - rows.min() + 1) <= 22          # 0.4 mm * 50 px/mm
    assert cols.max() - cols.min() > 900


def test_calibrate_recovers_a_page_offset():
    b = I.Board(FIX)
    ppm = 40.0
    gx, gy = b.edge[0], b.edge[1]
    own = R.raster_copper(b, R.Mapping.from_edge(b, ppm, 0.0))
    true_x0, true_y0 = gx - 2.37, gy - 1.18                  # the page starts 2.37 / 1.18 mm up-left of the outline
    shifted = R.raster_copper(b, R.Mapping(ppm, ppm, true_x0, true_y0, own.shape[1], own.shape[0]))
    x0, y0, score = R.calibrate(shifted, own, ppm, gx, gy)
    assert abs(x0 - true_x0) < 0.5 / ppm and abs(y0 - true_y0) < 0.5 / ppm and score > 0.9


def test_yolo_line_uses_pcbinspect_indices():
    from pcbinspect.record import BARE_CLASSES
    line = R.yolo_line(BARE_CLASSES.index("short"), (100, 200, 120, 230), 640)
    cls, cx, cy, w, h = line.split()
    assert int(cls) == BARE_CLASSES.index("short")
    assert float(cx) == pytest.approx(110 / 640, abs=1e-6) and float(cy) == pytest.approx(215 / 640, abs=1e-6)
    assert float(w) == pytest.approx(20 / 640, abs=1e-6) and float(h) == pytest.approx(30 / 640, abs=1e-6)
    tiny = R.yolo_line(0, (10, 10, 12, 12), 640).split()
    assert float(tiny[3]) * 640 == pytest.approx(6.0)


def test_appearance_pair_differs_only_at_the_defect():
    out, ds = one("open", seed=2, sizes={"open": 0.4})
    b0, b1 = I.Board(FIX), I.Board(out)
    m = mapping(b0, 40)
    app = R.sample_appearance(random.Random(5))

    def tile(b):
        cu = R.raster_copper(b, m)
        zero = np.zeros_like(cu)
        board = np.ones_like(cu)
        L = {"cu": cu, "open": zero, "silk": zero, "board": board, "holes": R.raster_holes(b, m)}
        return R.compose_tile(L, app, (0, 0), (cu.shape[1], cu.shape[0]), 40.0, 0)
    a, c = tile(b0), tile(b1)
    diff = np.abs(a.astype(int) - c.astype(int)).sum(-1)
    ys, xs = np.nonzero(diff > 6)
    x1, y1, x2, y2 = m.bbox_px(ds[0].bbox_mm)
    assert len(xs) > 5
    assert xs.min() >= x1 - 12 and xs.max() <= x2 + 12 and ys.min() >= y1 - 12 and ys.max() <= y2 + 12


# ---------------------------------------------------------------------------------------------- sources
def test_sources_are_permissive_and_complete():
    assert len(S.SOURCES) >= 20 and len({s["repo"] for s in S.SOURCES}) >= 20
    for s in S.SOURCES:
        assert s["licence"] in S.PERMISSIVE
        assert "-SA" not in s["licence"] and "OHL-S" not in s["licence"] and "OHL-W" not in s["licence"]
        assert s["path"].endswith(".kicad_pcb") and len(s["commit"]) == 40 and s["why"]
        assert S.url(s).startswith("https://raw.githubusercontent.com/")
    assert not any("kicad/kicad" in s["repo"].lower() or "kicad-demos" in s["repo"].lower() for s in S.SOURCES)


def test_exact_calibration_from_outline_recovers_page_offset():
    """The SVG page origin depends on every board item (not just the outline), so it is measured from the
    outline primitives, which appear in both files in mm. Shift a known outline and recover the offset."""
    from training.kicad.render import calibrate_from_edges
    board = """(kicad_pcb (version 20221018) (generator pcbnew)
  (gr_line (start 10 20) (end 60 20) (layer "Edge.Cuts") (width 0.1))
  (gr_line (start 60 20) (end 60 50) (layer "Edge.Cuts") (width 0.1))
  (gr_line (start 60 50) (end 10 50) (layer "Edge.Cuts") (width 0.1))
  (gr_line (start 10 50) (end 10 20) (layer "Edge.Cuts") (width 0.1))
)"""
    dx, dy = -9.2063, -18.75   # page offset as KiCad would place it
    pts = [(10, 20), (60, 20), (60, 50), (10, 50)]
    paths = "".join(f'<path d="M{a[0] + dx:.4f} {a[1] + dy:.4f}\nL{b[0] + dx:.4f} {b[1] + dy:.4f}\n" />\n'
                    for a, b in zip(pts, pts[1:] + pts[:1]))
    svg = f'<svg width="52mm" height="32mm" viewBox="0 0 52 32"><g>{paths}</g></svg>'
    x0, y0, score = calibrate_from_edges(svg, board)
    assert score == 1.0 and abs(x0 + dx) < 1e-6 and abs(y0 + dy) < 1e-6
    # an outline that does not match (different size) must not be trusted
    bad = svg.replace("M", "M").replace(f"{60 + dx:.4f}", f"{61 + dx:.4f}")
    assert calibrate_from_edges(bad, board) is None


def test_svg_layer_files_follow_the_boards_own_layer_names(tmp_path, monkeypatch):
    """KiCad 7+ boards name the silkscreen "F.Silkscreen"; kicad-cli names the SVG after that user name.
    Older boards use the canonical name. A board with no front silkscreen still renders."""
    import subprocess
    from types import SimpleNamespace
    from training.kicad import render as R
    new_style = '(kicad_pcb (layers (0 "F.Cu" signal) (37 "F.SilkS" user "F.Silkscreen") (39 "F.Mask" user) ' \
                '(44 "Edge.Cuts" user)) )'
    old_style = '(kicad_pcb (layers (0 F.Cu signal) (37 F.SilkS user) (39 F.Mask user) (44 Edge.Cuts user)) )'
    assert "F_Silkscreen" in R.layer_file_stems(new_style)["F.SilkS"]
    assert R.layer_file_stems(old_style)["F.SilkS"] >= {"F_SilkS"}
    monkeypatch.setattr(R.shutil, "which", lambda _: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stderr=""))
    for text, names, silk in ((new_style, ["F_Cu", "F_Silkscreen", "F_Mask", "Edge_Cuts"], "b-F_Silkscreen.svg"),
                              (old_style, ["F_Cu", "F_SilkS", "F_Mask", "Edge_Cuts"], "b-F_SilkS.svg"),
                              (old_style, ["F_Cu", "F_Mask", "Edge_Cuts"], None)):
        d = tmp_path / str(len(list(tmp_path.iterdir())))
        d.mkdir()
        (d / "b.kicad_pcb").write_text(text)
        out = d / "svg"
        out.mkdir()
        for n in names:
            (out / f"b-{n}.svg").write_text("<svg/>")
        files = R.export_svg_layers(d / "b.kicad_pcb", out)
        assert files["F.Cu"].name == "b-F_Cu.svg" and files["Edge.Cuts"].name == "b-Edge_Cuts.svg"
        assert (files["F.SilkS"].name if "F.SilkS" in files else None) == silk
    (out / "b-F_Cu.svg").unlink()
    with pytest.raises(RuntimeError, match="F_Mask"):
        R.export_svg_layers(d / "b.kicad_pcb", out)
