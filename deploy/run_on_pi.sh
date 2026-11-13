#!/usr/bin/env bash
# Start the inspection station. Open http://<pi-address>:8000 on the Pi screen or a phone.
#   bash deploy/run_on_pi.sh              camera auto-detect (Pi Camera, else USB webcam)
#   bash deploy/run_on_pi.sh replay       cycle through the demo boards (no camera needed)
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
CAMERA="${1:-auto}"
exec python -m pcbinspect.app --config config/app.yaml --set "camera=${CAMERA}"
