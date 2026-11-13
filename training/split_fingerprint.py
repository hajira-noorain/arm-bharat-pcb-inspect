"""Fingerprint the held-out cal/test splits so two machines can prove they built the same split.

python training/split_fingerprint.py <bare_yolo dir>
"""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for split in ("cal", "test"):
    items = json.loads((root / f"boards_{split}.json").read_text())
    canon = sorted((b["source"], Path(b["path"]).name, b["good"],
                    tuple(sorted((x["cls"], tuple(round(v) for v in x["bbox"])) for x in b["boxes"])))
                   for b in items)
    h = hashlib.sha256(json.dumps(canon).encode()).hexdigest()[:16]
    print(split, len(items), h)
tr = sorted(p.name for p in (root / "images" / "train").glob("*.jpg") if not p.stem.endswith("_cam"))
print("train_tiles", len(tr), hashlib.sha256("\n".join(tr).encode()).hexdigest()[:16])
