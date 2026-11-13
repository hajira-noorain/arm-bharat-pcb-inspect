"""Golden-board difference channel: find ANY copper change against a known-good board.

A learned detector only finds defects that look like the ones it was trained on. The detection-limit
"eye test" showed what that means: large, clean-looking defects (a wide bridge between two pads, a
big round hole) are missed because they look like legitimate features. Comparing with a golden
(known-good) image of the same design has no such blind spot: bigger changes are always easier.

  1. align golden onto the test image (ECC affine, refused below MIN_ALIGN_CC)
  2. copper masks with ONE polarity for both images (chosen on the golden, then the test polarity
     that agrees best with it), so a board near 50% copper cannot flip between the two
  3. per pixel COPPER FRACTION (0 bare .. 1 copper, from each image's own grey levels);
     extra = fraction rises by > T_FRAC, missing = falls by > T_FRAC (on binary scans this is exactly
     mask XOR; on blurred captures it also sees a narrow grey gap being bridged); an opening of
     (2*tol+1) px removes thin registration slivers; each remaining difference is re-registered
     LOCALLY (phase correlation in a window around it) and kept only if it survives: residual
     warp / lens distortion produces differences that vanish when re-registered, defects do not
  4. thin, elongated differences lying along a golden edge for their whole length are registration
     slivers (warp, scale) and are dropped; real defects are compact or cross a conductor
  5. connectivity (netlist) check: two golden conductors joined in the test = short, one golden
     conductor split = open; electrical evidence, kept however thin (never removed as a sliver)
  6. each remaining component is classified by TOPOLOGY against the golden conductors:
       extra   touching >= 2 conductors -> short; 1 -> spur; 0 -> spurious_copper;
               filling a golden hole (enclosed substrate) -> missing_hole
       missing splitting its conductor -> open; enclosed in copper -> pin_hole; else mouse_bite
  7. score = area / (area + A0), A0 = 4 (2*tol+1)^2: monotone in the size of the change; the
     conformal gate turns scores into decisions, as for the detector

Scope: binarised scans and bare copper on laminate. Solder-masked boards need a different copper
segmentation (not validated).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .golden import MIN_ALIGN_CC, align


@dataclass
class DiffCandidate:
    cls: str
    bbox: tuple[int, int, int, int]
    score: float
    area_px: int
    kind: str           # extra | missing
    size_px: float      # characteristic size (bridge / gap width, diameter, bite length)


def _otsu(ch: np.ndarray) -> np.ndarray:
    c8 = cv2.GaussianBlur(ch, (3, 3), 0)
    _, m = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m > 0


@dataclass
class Seg:
    """How copper is segmented: channel images, Otsu thresholds and polarity for test and golden."""
    ct: np.ndarray
    cg: np.ndarray
    thr_t: float
    thr_g: float
    inv_t: bool
    inv_g: bool
    levels_t: tuple = (0.0, 1.0)   # (bare, copper) grey level of the test image
    levels_g: tuple = (0.0, 1.0)
    ft: np.ndarray | None = None   # copper fraction per pixel, test: 0 = bare laminate, 1 = full copper
    fg: np.ndarray | None = None

    def golden_mask(self, ch: np.ndarray) -> np.ndarray:
        m = ch > self.thr_g
        return ~m if self.inv_g else m

    def golden_frac(self, ch: np.ndarray) -> np.ndarray:
        return _frac(ch, self.levels_g)


T_FRAC = 0.3  # a pixel's copper fraction must change by 30% of full copper contrast to count


def _frac(ch: np.ndarray, levels) -> np.ndarray:
    bare, cu = levels
    return np.clip((ch - bare) / (cu - bare if abs(cu - bare) > 1e-3 else 1e-3), 0.0, 1.0).astype(np.float32)


def _otsu_thr(ch: np.ndarray) -> float:
    c8 = cv2.GaussianBlur(ch, (3, 3), 0)
    thr, _ = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr)


def consistent_masks(test: np.ndarray, golden: np.ndarray):
    """Copper masks for test and (aligned) golden on the same LAB channel with matching polarity.
    Copper is taken as the golden's minority phase. Returns (test_mask, golden_mask, Seg)."""
    lt, lg = cv2.cvtColor(test, cv2.COLOR_BGR2LAB), cv2.cvtColor(golden, cv2.COLOR_BGR2LAB)
    best = None
    for c in range(3):
        ch = cv2.GaussianBlur(lg[:, :, c], (3, 3), 0)
        thr = _otsu_thr(lg[:, :, c])
        mg = ch > thr
        a, b = ch[mg], ch[~mg]
        if len(a) < 10 or len(b) < 10:
            continue
        # ABSOLUTE contrast between the two classes: a colour channel holding only (JPEG/sensor) colour
        # noise can look well separated relative to its own small spread, but its contrast is tiny
        sep = abs(float(a.mean()) - float(b.mean()))
        if best is None or sep > best[0]:
            best = (sep, c, thr, mg)
    if best is None:
        z = np.zeros(test.shape[:2], bool)
        return z, z, None
    _, c, thr_g, mg = best
    inv_g = bool(mg.mean() > 0.5)
    if inv_g:
        mg = ~mg
    ct = cv2.GaussianBlur(lt[:, :, c], (3, 3), 0)
    cg = cv2.GaussianBlur(lg[:, :, c], (3, 3), 0)
    thr_t = _otsu_thr(lt[:, :, c])
    mt = ct > thr_t
    inv_t = bool((mt == mg).mean() < (~mt == mg).mean())
    if inv_t:
        mt = ~mt
    ct, cg = ct.astype(np.float32), cg.astype(np.float32)
    # per-image grey levels of bare laminate and copper (robust medians of each class)
    lv_t = (float(np.median(ct[~mt])), float(np.median(ct[mt]))) if mt.any() and (~mt).any() else (0.0, 255.0)
    lv_g = (float(np.median(cg[~mg])), float(np.median(cg[mg]))) if mg.any() and (~mg).any() else (0.0, 255.0)
    seg = Seg(ct, cg, thr_t, thr_g, inv_t, inv_g, lv_t, lv_g, _frac(ct, lv_t), _frac(cg, lv_g))
    return mt, mg, seg


def _local_residual(comp, kind, t, seg: Seg, box, ker, min_area):
    """Re-register a window around one difference (phase correlation of the golden onto the test)
    and recompute it. Differences caused by local misregistration (warp, lens distortion) vanish;
    real defects stay. Returns the residual area (0 = it was registration)."""
    h, w = t.shape
    x0, y0, x1, y1 = box
    pad = 24
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    a, b = seg.ct[Y0:Y1, X0:X1], seg.cg[Y0:Y1, X0:X1]
    if min(a.shape) < 16:
        return int(comp.sum())
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(b, a, win)
    if resp < 0.05 or max(abs(dx), abs(dy)) > 6 or max(abs(dx), abs(dy)) < 0.25:
        return int(comp.sum())
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    gw = cv2.warpAffine(b, M, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    g_frac = seg.golden_frac(gw)
    t_frac = seg.ft[Y0:Y1, X0:X1]
    r = ((t_frac - g_frac) > T_FRAC) if kind == "extra" else ((g_frac - t_frac) > T_FRAC)
    r = cv2.morphologyEx(r.astype(np.uint8), cv2.MORPH_OPEN, ker) > 0
    return int((r & comp[Y0:Y1, X0:X1]).sum()) if (r & comp[Y0:Y1, X0:X1]).sum() >= min_area else 0


def _min_side(mask_u8):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0, 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    return float(min(w, h)), float(max(w, h))


SLIVER_ELONGATION = 4.0
SLIVER_EDGE_FRACTION = 0.9


def _is_sliver(comp, d_edge, tol_px) -> bool:
    """Registration artefact: thin (<= 2*tol+2 px), elongated, and lying along a golden edge for
    (nearly) its whole extent. Real defects are compact or cross a conductor instead."""
    c8 = comp.astype(np.uint8)
    thick = 2.0 * float(cv2.distanceTransform(np.pad(c8, 1), cv2.DIST_L2, 3).max())
    if thick > 2 * tol_px + 2:
        return False
    short_side, long_side = _min_side(c8)
    if long_side < SLIVER_ELONGATION * max(short_side, 1.0):
        return False
    return float((d_edge[comp] <= tol_px + 1.5).mean()) >= SLIVER_EDGE_FRACTION


MIN_PIECE_PX = 30  # a conductor piece smaller than this is noise, not a separate net


def _overlaps(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _connectivity(t, g, glab, n_g, existing, A0) -> list[DiffCandidate]:
    """Netlist comparison: two golden conductors joined into one piece of test copper = SHORT; one
    golden conductor fallen apart into two pieces = OPEN. This is electrical evidence, so it is kept
    however thin the bridge or cut (the shape-based sliver filter must never remove it). Each
    finding is located on the raw (unopened) difference pixels; findings already reported by the
    shape analysis are not repeated."""
    out: list[DiffCandidate] = []
    n_t, tlab = cv2.connectedComponents(t.astype(np.uint8), connectivity=8)
    both = t & g
    pairs = np.bincount(tlab[both].astype(np.int64) * n_g + glab[both], minlength=n_t * n_g).reshape(n_t, n_g)
    pairs[0, :] = 0
    pairs[:, 0] = 0
    sig = pairs >= MIN_PIECE_PX
    h, w = t.shape

    def locate(raw, labels_img, a_id, b_id, kind, cls):
        n, lab, st, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            x0, y0, x1, y1 = max(0, x - 2), max(0, y - 2), min(w, x + bw + 2), min(h, y + bh + 2)
            ring = cv2.dilate((lab[y0:y1, x0:x1] == i).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            near = set(np.unique(labels_img[y0:y1, x0:x1][ring]).tolist())
            if a_id in near and b_id in near:
                box = (int(x), int(y), int(x + bw), int(y + bh))
                score = float(0.9 + 0.09 * min(1.0, area / A0))
                prior = [c for c in existing + out if _overlaps(box, c.bbox) and c.kind == kind]
                if prior:
                    # already seen by the shape analysis (often with a small-area score): the netlist
                    # evidence decides both its name and its strength
                    for c in prior:
                        c.cls, c.score = cls, max(c.score, score)
                    return
                comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
                short_side, _ = _min_side(comp)
                out.append(DiffCandidate(cls, box, score, int(area), kind, float(short_side)))
                return

    # shorts: one test piece overlapping >= 2 golden conductors
    for tid in np.nonzero(sig.sum(1) >= 2)[0]:
        gids = np.nonzero(sig[tid])[0]
        raw = (tlab == tid) & ~g
        for j in range(len(gids) - 1):
            locate(raw, glab, int(gids[j]), int(gids[j + 1]), "extra", "short")
    # opens: one golden conductor overlapping >= 2 test pieces
    for gid in np.nonzero(sig.sum(0) >= 2)[0]:
        tids = np.nonzero(sig[:, gid])[0]
        raw = (glab == gid) & ~t
        for j in range(len(tids) - 1):
            locate(raw, tlab, int(tids[j]), int(tids[j + 1]), "missing", "open")
    return out


def diff_candidates(test_bgr: np.ndarray, golden_bgr: np.ndarray, tol_px: int = 1, aligned: bool = False,
                    min_align_cc: float = MIN_ALIGN_CC) -> tuple[list[DiffCandidate], dict]:
    if aligned:
        g_img, cc = golden_bgr, 1.0
    else:
        g_img, cc = align(test_bgr, golden_bgr)
    info = {"align_cc": cc, "ok": cc >= min_align_cc}
    if not info["ok"]:
        info["reason"] = f"golden alignment too poor (cc {cc:.2f})"
        return [], info
    t, g, seg = consistent_masks(test_bgr, g_img)
    if seg is None:
        info.update(ok=False, reason="no copper contrast")
        return [], info
    k = 2 * tol_px + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # differences in COPPER FRACTION, not in yes/no masks: a narrow gap that blur and print spread
    # render grey (counted as copper by any global threshold) still changes clearly when bridged
    extra = cv2.morphologyEx(((seg.ft - seg.fg) > T_FRAC).astype(np.uint8), cv2.MORPH_OPEN, ker)
    missing = cv2.morphologyEx(((seg.fg - seg.ft) > T_FRAC).astype(np.uint8), cv2.MORPH_OPEN, ker)
    A0 = 4.0 * k * k
    min_area = k * k
    h, w = t.shape
    # distance (px) to the nearest golden copper edge: registration slivers hug it along their length
    edge = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    d_edge = cv2.distanceTransform((1 - edge).astype(np.uint8), cv2.DIST_L2, 3)
    n_g, glab, gst, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), connectivity=8)
    n_s, slab, sst, _ = cv2.connectedComponentsWithStats((~g).astype(np.uint8), connectivity=4)
    border_sub = set(np.unique(np.r_[slab[0], slab[-1], slab[:, 0], slab[:, -1]]).tolist())
    out: list[DiffCandidate] = []
    for kind, m in (("extra", extra), ("missing", missing)):
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if area < min_area:
                continue
            comp = lab == i
            area = _local_residual(comp, kind, t, seg, (x, y, x + bw, y + bh), ker, min_area)
            if area < min_area:
                continue
            x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(w, x + bw + 3), min(h, y + bh + 3)
            if _is_sliver(comp[y0:y1, x0:x1], d_edge[y0:y1, x0:x1], tol_px):
                continue
            ring = cv2.dilate(comp[y0:y1, x0:x1].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            short_side, long_side = _min_side(comp[y0:y1, x0:x1].astype(np.uint8))
            if kind == "extra":
                touched = set(np.unique(glab[y0:y1, x0:x1][ring & g[y0:y1, x0:x1]]).tolist()) - {0}
                subs = set(np.unique(slab[y0:y1, x0:x1][comp[y0:y1, x0:x1]]).tolist()) - {0}
                in_hole = bool(subs) and not (subs & border_sub) and \
                    all(sst[s_, cv2.CC_STAT_AREA] <= 4 * area for s_ in subs)
                if in_hole:
                    cls, size = "missing_hole", float(np.sqrt(4 * area / np.pi))
                elif len(touched) >= 2:
                    cls, size = "short", short_side
                elif len(touched) == 1:
                    cls, size = "spur", short_side
                else:
                    cls, size = "spurious_copper", float(np.sqrt(4 * area / np.pi))
            else:
                L = np.bincount(glab[y0:y1, x0:x1][comp[y0:y1, x0:x1]].ravel(), minlength=n_g)
                L[0] = 0
                cond = int(L.argmax())
                cx0, cy0 = gst[cond, cv2.CC_STAT_LEFT], gst[cond, cv2.CC_STAT_TOP]
                cx1, cy1 = cx0 + gst[cond, cv2.CC_STAT_WIDTH], cy0 + gst[cond, cv2.CC_STAT_HEIGHT]
                cm = glab[cy0:cy1, cx0:cx1] == cond
                pieces, _, pst, _ = cv2.connectedComponentsWithStats(
                    (t[cy0:cy1, cx0:cx1] & cm).astype(np.uint8), connectivity=8)
                big = int((pst[1:, cv2.CC_STAT_AREA] >= min_area).sum()) if pieces > 1 else 0
                touches_sub = bool((ring & ~g[y0:y1, x0:x1]).any())
                if big >= 2:
                    cls, size = "open", short_side
                elif not touches_sub:
                    cls, size = "pin_hole", float(np.sqrt(4 * area / np.pi))
                else:
                    cls, size = "mouse_bite", long_side
            out.append(DiffCandidate(cls, (int(x), int(y), int(x + bw), int(y + bh)),
                                     float(area / (area + A0)), int(area), kind, float(size)))
    out += _connectivity(t, g, glab, n_g, out, A0)
    info["n"] = len(out)
    return out, info
