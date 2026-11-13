#!/usr/bin/env bash
# Every on-device measurement for the report, in one command, from the bundle folder on the Pi 5:
#   bash deploy/measure_on_pi.sh            # full run: functional check, benchmarks, 30-minute soak (~45 min)
#   bash deploy/measure_on_pi.sh quick      # a few repetitions and a 3-minute soak, to check the setup first
# Stops at the first failure. Writes everything to results/ and packs it as results_pi_<time>.tgz to copy back.
# Keep the active cooler on and close other programs; do not use the Pi while it runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODE="${1:-full}"
STAMP=$(date +%Y%m%d_%H%M)
mkdir -p results
LOG=results/measure_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1

echo "== system"
uname -a
grep -m1 -i "model" /proc/cpuinfo || true
cat /proc/device-tree/model 2>/dev/null; echo
free -m
vcgencmd measure_temp 2>/dev/null || true
vcgencmd get_throttled 2>/dev/null || true
$PY -c "import executorch, platform; print('python', platform.python_version())"
$PY -m pip freeze 2>/dev/null | grep -iE "^(executorch|torch|numpy|opencv)" || true

echo "== 1/3 functional check (Llama on ExecuTorch is mandatory: a failure stops here)"
$PY deploy/arm_check.py --root .

if [ "$MODE" = "quick" ]; then
  echo "== 2/3 benchmarks (quick)"
  $PY bench/bench_device.py --quick --out results/device_bench_quick.json
  echo "== 3/3 soak (3 minutes)"
  $PY bench/soak.py --minutes 3
else
  echo "== 2/3 benchmarks: vision at 1/2/4 threads, ExecuTorch vs Arm runtime, end to end, Llama"
  $PY bench/bench_device.py --out results/device_bench.json
  echo "== 3/3 soak: 30 minutes of continuous inspection with advice"
  $PY bench/soak.py --minutes 30
fi

vcgencmd measure_temp 2>/dev/null || true
vcgencmd get_throttled 2>/dev/null || true
tar czf "results_pi_${STAMP}.tgz" results
echo "DONE: copy results_pi_${STAMP}.tgz back to the laptop"
