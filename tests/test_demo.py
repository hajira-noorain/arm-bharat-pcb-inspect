"""Demo mode helpers: the test-board catalogue, matching findings to a board's known answer, safe upload decoding."""
import json
from datetime import datetime

import cv2
import numpy as np
import pytest

from pcbinspect.demo import UploadError, decode_upload, load_samples, match_known, seconds_until


def _jpg(path, size=20):
    cv2.imwrite(str(path), np.full((size, size, 3), 128, np.uint8))


def _png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_load_samples_picks_defective_then_good_per_design(tmp_path):
    boards = tmp_path / "demo" / "boards"
    boards.mkdir(parents=True)
    entries = [{"file": f"deeppcb_{i:03d}.jpg", "design": "deeppcb", "good": False, "boxes": [[1, 2, 3, 4]]}
               for i in range(3)]
    entries += [{"file": f"deeppcb_g{i}.jpg", "design": "deeppcb", "good": True, "boxes": []} for i in range(2)]
    entries += [{"file": "visa_pcb1_000.jpg", "design": "visa_pcb1", "good": True}]
    entries[0]["boxes"] = [{"cls": "open", "bbox": [207, 195, 244, 223]}]      # the bundle's manifest format
    for e in entries:
        _jpg(boards / e["file"])
    (tmp_path / "demo" / "manifest.json").write_text(json.dumps(entries))
    got = load_samples(tmp_path, per_design=2)
    assert [(s.file, s.good) for s in got] == [("deeppcb_000.jpg", False), ("deeppcb_g0.jpg", True),
                                               ("visa_pcb1_000.jpg", True)]
    assert got[0].kind == "bare" and got[2].kind == "assembled" and got[2].boxes == []
    assert got[0].boxes == [[207, 195, 244, 223]] and got[1].boxes == []


def test_load_samples_skips_missing_files_and_missing_manifest(tmp_path):
    assert load_samples(tmp_path) == []
    (tmp_path / "demo" / "boards").mkdir(parents=True)
    _jpg(tmp_path / "demo" / "boards" / "here.jpg")
    (tmp_path / "demo" / "manifest.json").write_text(json.dumps([
        {"file": "gone.jpg", "design": "d", "good": False, "boxes": []},
        {"file": "here.jpg", "design": "d", "good": False, "boxes": []}]))
    assert [s.file for s in load_samples(tmp_path)] == ["here.jpg"]


def test_match_known_counts_found_and_extra():
    known = [[0, 0, 10, 10], [50, 50, 60, 60]]
    assert match_known([(2, 2, 12, 12), (100, 100, 110, 110)], known) == {"marked": 2, "found": 1, "extra": 1}
    # centre inside a large marked box counts even when the overlap is tiny
    assert match_known([(4, 4, 6, 6)], [[0, 0, 40, 40]]) == {"marked": 1, "found": 1, "extra": 0}
    assert match_known([], []) == {"marked": 0, "found": 0, "extra": 0}


def test_decode_upload_accepts_gray_rgba_and_16bit_png():
    gray = np.full((30, 40), 90, np.uint8)
    rgba = np.zeros((30, 40, 4), np.uint8)
    deep = np.full((30, 40, 3), 40000, np.uint16)
    ok, gray_jpg = cv2.imencode(".jpg", gray)
    for data in (gray_jpg.tobytes(), _png_bytes(rgba), _png_bytes(deep)):
        im = decode_upload(data)
        assert im.shape == (30, 40, 3) and im.dtype == np.uint8
    assert decode_upload(_png_bytes(deep))[0, 0, 0] == 40000 >> 8


def test_decode_upload_rejects_bad_input():
    with pytest.raises(UploadError) as e:
        decode_upload(b"hello")
    assert e.value.status == 400
    with pytest.raises(UploadError) as e:
        decode_upload(b"\xff\xd8\xff" + b"0" * 10_000_000)
    assert e.value.status == 413
    with pytest.raises(UploadError) as e:
        decode_upload(_png_bytes(np.zeros((10, 4001, 3), np.uint8)))
    assert e.value.status == 400
    gif = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    with pytest.raises(UploadError) as e:
        decode_upload(gif)
    assert e.value.status == 400


def test_seconds_until_next_reset():
    assert seconds_until(3, datetime(2026, 11, 14, 2, 30)) == 1800
    assert seconds_until(3, datetime(2026, 11, 14, 3, 0, 0)) == 86400
