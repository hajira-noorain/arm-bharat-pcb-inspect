#!/usr/bin/env bash
# One-time setup on a Raspberry Pi 5 running Raspberry Pi OS (64-bit, Trixie or Bookworm).
# Run from the unpacked bundle directory:   bash deploy/pi_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== system packages (camera, GPIO, OpenCV runtime libs)"
sudo apt-get update
sudo apt-get install -y python3-venv python3-picamera2 python3-gpiozero python3-lgpio libgl1 libglib2.0-0

echo "== Python environment (system site packages so picamera2/gpiozero from apt are visible)"
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r deploy/requirements-pi.txt

echo "== checks"
python - <<'PY'
import platform, executorch
from executorch.extension.llm.runner import TextLLMRunner  # noqa: F401  (Llama runtime present)
from executorch.runtime import Runtime  # noqa: F401
print("arch", platform.machine(), "executorch", getattr(executorch, "__version__", "?"))
PY
vcgencmd measure_temp || true
vcgencmd get_throttled || true
echo "Setup done. Start the station with:  bash deploy/run_on_pi.sh"
