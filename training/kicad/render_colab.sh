#!/usr/bin/env bash
# Colab job (Ubuntu 22.04, root): synthetic bare-PCB defect data from KiCad designs.
#   bash training/kicad/render_colab.sh                 # all boards, defaults
#   bash training/kicad/render_colab.sh --boards 5      # extra args go to render.py
# Needs this repo unpacked anywhere (training/kicad/*.py + pcbinspect/record.py); internet for apt/pip/GitHub raw.
# Output: /content/kicad_out/{images,labels,golden}/{train,val,test}/*  + manifest.json   (and /content/kicad_out.tgz)
set -euxo pipefail
CODE=$(cd "$(dirname "$0")/../.." && pwd)
export DEBIAN_FRONTEND=noninteractive
export PYTHONPATH="$CODE:${PYTHONPATH:-}"

# ---- KiCad 9 headless (official PPA); recommends skipped: no symbol/footprint/3D libraries needed for plots ----
if ! command -v kicad-cli >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq software-properties-common xvfb libcairo2
  add-apt-repository -y ppa:kicad/kicad-9.0-releases
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends kicad
fi
kicad-cli --version
# options we rely on (documented for 9.0 at https://docs.kicad.org/9.0/en/cli/cli.html)
kicad-cli pcb export svg --help | grep -E -- "--page-size-mode|--mode-multi|--black-and-white|--exclude-drawing-sheet|--drill-shape-opt"
kicad-cli pcb render --help | head -40 || true      # 3D ray-traced render exists in 9.0 (optional; the 2D path is the default)

pip install -q cairosvg scipy pillow numpy

OUT=/content/kicad_out
# 1) one real board end to end; abort early if nothing came out (e.g. SVG naming/option mismatch)
python "$CODE/training/kicad/render.py" --out /content/kicad_try --work /content/kicad_work_try --boards 1 --defect-sets 1 "$@"
python - <<'PY'
import json, sys
m = json.load(open('/content/kicad_try/manifest.json'))
b = m['boards'][0]
n = sum(1 for t in b.get('tiles', []) if 'name' in t)
print('trial board:', b['repo'], b['status'], 'tiles', n, 'mapping', b.get('mapping'), b.get('reason', ''))
sys.exit(0 if n else 1)
PY

# 2) everything
python "$CODE/training/kicad/render.py" --out "$OUT" --work /content/kicad_work "$@"
tar czf /content/kicad_out.tgz -C /content kicad_out
echo "done: $OUT  (/content/kicad_out.tgz)"
