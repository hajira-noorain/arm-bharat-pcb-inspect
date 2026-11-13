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
echo "== Arm Runtime for YOLO26 (github.com/arm/Optimized-Runtimes, Apache-2.0): Arm's optimised Pi 5 detector runtime"
# Prebuilt library from the project's releases + conversion of OUR fine-tuned checkpoint. Optional: if anything
# fails the station keeps using ExecuTorch for vision (vision_backend falls back automatically).
if [ -f models/vision/pcb_y26n.pt ]; then
  ( set +e
    [ -d third_party/Optimized-Runtimes ] || git clone --depth 1 https://github.com/arm/Optimized-Runtimes.git third_party/Optimized-Runtimes
    cd third_party/Optimized-Runtimes/yolo26-raspberry-pi-5
    make model CHECKPOINT="$(cd ../../.. && pwd)/models/vision/pcb_y26n.pt" RESOLUTION=640       && pip install -e src/runtime/api/python       && echo "Arm runtime ready: $(ls dist/runtime/*/libarm-yolo26n.so* 2>/dev/null | head -1)"       || echo "Arm runtime not set up (station will use ExecuTorch for vision)" )
fi
vcgencmd measure_temp || true
vcgencmd get_throttled || true
echo "Setup done. Start the station with:  bash deploy/run_on_pi.sh"
