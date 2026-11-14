"""Package a recording (deploy/record_demo.py) as a static site: the station's own dashboard in recorded mode.

    python deploy/build_static_demo.py --recording <record dir> --out <site dir>

The site needs no server code: it serves index.html, static/ and data/. It is published as a free static
Hugging Face Space (deploy/space/README.static.md becomes its README.md).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", nargs="*", default=None,
                    help="recorded shift questions to show as examples (default: all). Answers are never edited.")
    a = ap.parse_args()
    rec, out = Path(a.recording) / "data", Path(a.out)
    if not (rec / "samples.json").is_file():
        raise SystemExit(f"no recording at {rec}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "pcbinspect" / "static", out / "static")
    shutil.copytree(rec, out / "data")
    index = {}
    for f in sorted((out / "data" / "inspections").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        index[str(r["id"])] = f.stem          # file name of the test board, e.g. deeppcb_000.jpg
    (out / "data" / "index.json").write_text(json.dumps(index))
    ask = out / "data" / "ask.json"
    if a.questions is not None and ask.is_file():
        kept = [r for r in json.loads(ask.read_text(encoding="utf-8")) if r["question"] in a.questions]
        missing = set(a.questions) - {r["question"] for r in kept}
        if missing:
            raise SystemExit(f"not in the recording: {sorted(missing)}")
        ask.write_text(json.dumps(kept, indent=1))
    page = (ROOT / "pcbinspect" / "static" / "index.html").read_text(encoding="utf-8")
    page = page.replace('href="/static/', 'href="static/').replace('src="/static/', 'src="static/')
    page = page.replace('<script src="static/labels.js">', '<script>window.PCB_RECORDED = true;</script>\n'
                        '<script src="static/labels.js">', 1)
    page = page.replace('href="/api/export.csv"', 'href="data/log.json"')
    page = page.replace('<a href="/api/eyetest/card.pdf">Download the card</a>', 'Download the card on the station')
    tail = '</div>\n\n<div class="toast"'
    credits = ('<p class="meta" style="margin-top:24px">Test boards from DeepPCB (MIT), PKU-Market-PCB (released for '
               'research use), Mendeley PCB-Defect and VisA (CC BY 4.0). These boards were held out: the station was '
               'never trained on them. Built with Llama.</p>\n' + tail)
    assert page.count(tail) == 1
    page = page.replace(tail, credits, 1)
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "static" / "index.html").unlink()
    shutil.copy(ROOT / "deploy" / "space" / "README.static.md", out / "README.md")
    n = len(index)
    print(f"static demo: {n} recorded boards, {sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e6:.1f} MB in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
