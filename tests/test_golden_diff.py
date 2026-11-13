import cv2
import numpy as np
import pytest

from pcbinspect.fusion import fuse
from pcbinspect.golden_diff import diff_candidates


def board(invert=False):
    """Synthetic bare board, 640 px: parallel traces (narrow 6 px gaps), pads with holes, a pour."""
    im = np.full((640, 640), 235, np.uint8)
    for i in range(6):
        y = 80 + i * 26
        cv2.line(im, (60, y), (580, y), 20, 20)          # 20 px traces, 6 px gaps
    for x in range(100, 560, 90):
        cv2.circle(im, (x, 420), 28, 20, -1)
        cv2.circle(im, (x, 420), 10, 235, -1)            # drilled hole
    cv2.rectangle(im, (60, 500), (580, 600), 20, -1)     # copper pour
    if invert:
        im = 255 - im
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


def recapture(img, rng, noise=3.0, jpeg=None):
    M = cv2.getRotationMatrix2D((320, 320), rng.uniform(-0.3, 0.3), 1.0)
    M[:, 2] += rng.uniform(-2, 2, 2)
    out = cv2.warpAffine(img, M, (640, 640), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out = np.clip(out + rng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    if jpeg:
        out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_COLOR)
    return out


def found(cands, x, y, cls=None, min_score=0.5):
    return [c for c in cands if c.bbox[0] - 3 <= x <= c.bbox[2] + 3 and c.bbox[1] - 3 <= y <= c.bbox[3] + 3
            and c.score >= min_score and (cls is None or c.cls == cls)]


@pytest.mark.parametrize("jpeg", [None, 60])
def test_recaptured_good_board_gives_no_confident_findings(jpeg):
    rng = np.random.default_rng(0)
    g = board()
    for _ in range(3):
        cands, info = diff_candidates(recapture(g, rng, noise=6, jpeg=jpeg), g)
        assert info["ok"] and not [c for c in cands if c.score >= 0.5]


def test_colour_noise_channel_is_not_chosen():
    # strong chroma noise (JPEG colour blocks) on an otherwise grey board must not become 'copper'
    rng = np.random.default_rng(1)
    g = board()
    t = recapture(g, rng).astype(np.float32)
    t[..., 0] += rng.normal(0, 12, t.shape[:2])  # colour noise in one channel only
    t = np.clip(t, 0, 255).astype(np.uint8)
    t = cv2.imdecode(cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], cv2.IMREAD_COLOR)
    cands, _ = diff_candidates(t, g)
    assert not [c for c in cands if c.score >= 0.5]


def test_hairline_short_across_narrow_gap_is_found_by_connectivity():
    rng = np.random.default_rng(2)
    g = board()
    t = g.copy()
    cv2.line(t, (300, 90), (300, 96), (20, 20, 20), 3)   # 3 px bridge across the 6 px gap at y=90..96
    cands, _ = diff_candidates(recapture(t, rng, noise=2), g)
    hit = found(cands, 300, 93, "short", min_score=0.9)
    assert hit, [(c.cls, c.bbox, round(c.score, 2)) for c in cands]


def test_thin_open_is_found_by_connectivity():
    rng = np.random.default_rng(3)
    g = board()
    t = g.copy()
    cv2.line(t, (400, 165), (400, 195), (235, 235, 235), 3)  # 3 px cut through the 20 px trace at y=184
    cands, _ = diff_candidates(recapture(t, rng, noise=2), g)
    assert found(cands, 400, 184, "open", min_score=0.9), [(c.cls, c.bbox) for c in cands]


def test_pin_hole_and_spurious_copper_by_topology():
    rng = np.random.default_rng(4)
    g = board()
    t = g.copy()
    cv2.circle(t, (330, 550), 8, (235, 235, 235), -1)   # hole inside the pour
    cv2.circle(t, (600, 300), 8, (20, 20, 20), -1)      # isolated dot on bare laminate
    cands, _ = diff_candidates(recapture(t, rng, noise=2), g)
    assert found(cands, 330, 550, "pin_hole")
    assert found(cands, 600, 300, "spurious_copper")


def test_inverted_polarity_board():
    rng = np.random.default_rng(5)
    g = board(invert=True)
    t = g.copy()
    cv2.circle(t, (600, 300), 8, (235, 235, 235), -1)   # on an inverted board copper is LIGHT
    cands, _ = diff_candidates(recapture(t, rng, noise=2), g)
    assert found(cands, 600, 300, "spurious_copper")
    assert len([c for c in cands if c.score >= 0.5]) == 1


def test_wrong_board_is_refused():
    g = board()
    other = np.full_like(g, 235)
    cv2.circle(other, (320, 320), 200, (20, 20, 20), -1)
    cands, info = diff_candidates(other, g)
    assert not info["ok"] and cands == []


def test_fusion_merges_and_keeps_both_sources():
    rng = np.random.default_rng(6)
    g = board()
    t = g.copy()
    cv2.circle(t, (600, 300), 8, (20, 20, 20), -1)
    cands, _ = diff_candidates(recapture(t, rng, noise=2), g)
    dets = [{"bbox": (590, 290, 612, 312), "conf": 0.4, "cls": "spurious_copper"},
            {"bbox": (100, 300, 120, 320), "conf": 0.3, "cls": "spur"}]
    fz = fuse(dets, cands)
    merged = [f for f in fz if f["source"] == "detector+golden"]
    assert merged and merged[0]["conf"] >= max(0.4, merged[0]["diff_score"]) and merged[0]["cls"] == "spurious_copper"
    assert any(f["source"] == "detector" and f["conf"] == 0.3 for f in fz)


def test_alignment_recovers_a_board_put_down_off_position_and_turned():
    """A board 8% of the frame off and 4 degrees turned (millimetres on a real fixture) must still align;
    single-scale ECC from the identity lost it beyond ~2% and ~1 degree."""
    from pcbinspect.golden import align
    rng = np.random.default_rng(0)
    g = np.full((900, 1200, 3), 30, np.uint8)
    for _ in range(60):                                   # irregular copper-like blobs and traces
        p1 = (int(rng.integers(40, 1160)), int(rng.integers(40, 860)))
        p2 = (p1[0] + int(rng.integers(-200, 200)), p1[1] + int(rng.integers(-200, 200)))
        cv2.line(g, p1, p2, (200, 190, 180), int(rng.integers(4, 14)))
    h, w = g.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 4.0, 1.0)
    M[:, 2] += (0.08 * w, -0.05 * w)
    t = cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out, cc = align(t, g)
    assert cc > 0.95
    inner = (slice(200, 700), slice(300, 900))
    assert np.abs(out[inner].astype(int) - t[inner].astype(int)).mean() < 6
