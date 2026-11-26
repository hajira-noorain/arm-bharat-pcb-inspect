#!/usr/bin/env bash
# Rebuild ExecuTorch's Llama kernel library (libcustom_ops_aot_lib.so) for ARMv8.0 so Llama runs on a Raspberry Pi 4.
#
# Why: the executorch 1.6.0 aarch64 wheel ships this library built for ARMv8.2 (atomic instructions such as
# ldadd/casal). The Pi 4's Cortex-A72 is ARMv8.0 and dies with "illegal instruction" the moment Llama loads.
# The other ExecuTorch libraries in the wheel are fine on the Pi 4; only this one is replaced.
#
# What it does (the same steps as the by-hand rebuild on the Pi 4):
#   1. clone ExecuTorch v1.6.0 (the tag matching the wheel) into ~/src/executorch and fetch only the submodules needed
#   2. patch extension/llm/custom_ops/CMakeLists.txt: -march=armv8.2-a+dotprod -> -march=armv8-a+dotprod
#      (the torchao dot-product kernels need the dotprod flag to compile; they are only used when the CPU reports it)
#   3. configure with -march=armv8-a everywhere else and build only the custom_ops_aot_lib target (about 6 min on a Pi 4)
#   4. re-point its library names/rpath at the wheel's own libs with patchelf and drop it into the wheel
#   5. check that Llama loads
#
# Run on the Pi from the unpacked bundle, after deploy/pi_setup.sh:   bash deploy/rebuild_llm_ops_pi4.sh
# Safe to re-run: it does nothing if the installed library is already the rebuilt one. FORCE=1 rebuilds anyway.
# The original library is kept in ~/backup/libcustom_ops_aot_lib.so.orig (outside the package folder on purpose:
# a second libcustom_ops_aot_lib.* inside it makes executorch refuse to load).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

# the same virtual environment deploy/pi_setup.sh creates
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "No $REPO/.venv - run bash deploy/pi_setup.sh first"; exit 1; }
[ "$(uname -m)" = "aarch64" ] || { echo "This is for a 64-bit Raspberry Pi (aarch64), not $(uname -m)"; exit 1; }

ET_TAG=v1.6.0                    # must match executorch==1.6.0 in deploy/requirements-pi.txt
SRC="$HOME/src/executorch"       # the folder must be called "executorch" (the build refers to its parent as ../..)
BUILD="$SRC/build-a72"
SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
DEST="$SP/executorch/extension/llm/custom_ops"
LIB="$DEST/libcustom_ops_aot_lib.so"
STAMP="$HOME/.cache/arm_bharat_llm_ops_armv8_0.sha256"
[ -f "$LIB" ] || { echo "executorch 1.6.0 is not installed in .venv (no $LIB)"; exit 1; }

# 0. already done? (the stamp holds the checksum of the library we installed last time)
if [ -z "${FORCE:-}" ] && [ -f "$STAMP" ] && [ "$(sha256sum "$LIB" | cut -d' ' -f1)" = "$(cat "$STAMP")" ]; then
  echo "Installed library is already the ARMv8.0 rebuild (FORCE=1 to rebuild)."
  SKIP_BUILD=1
fi

if [ -z "${SKIP_BUILD:-}" ]; then
  echo "== build tools"
  sudo apt-get install -y cmake ninja-build g++ git patchelf binutils

  echo "== ExecuTorch $ET_TAG source (only the submodules this target needs)"
  mkdir -p "$HOME/src"
  if [ ! -d "$SRC/.git" ]; then
    git clone -q --depth 1 --branch "$ET_TAG" https://github.com/pytorch/executorch.git "$SRC"
  fi
  cd "$SRC"
  git submodule update --init --depth 1 \
    third-party/flatbuffers third-party/flatcc third-party/json third-party/gflags third-party/ao shim \
    backends/xnnpack/third-party/cpuinfo backends/xnnpack/third-party/pthreadpool backends/xnnpack/third-party/FXdiv \
    kernels/optimized/third-party/eigen

  echo "== patch: custom ops need dotprod to compile, but not ARMv8.2"
  sed -i 's/-march=armv8.2-a+dotprod>/-march=armv8-a+dotprod>/' extension/llm/custom_ops/CMakeLists.txt
  grep -n "march" extension/llm/custom_ops/CMakeLists.txt

  echo "== configure (ARMv8.0 everywhere else)"
  TP="$("$PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
  rm -rf "$BUILD"
  cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$PY" -DCMAKE_PREFIX_PATH="$TP" \
    -DCMAKE_C_FLAGS="-march=armv8-a" -DCMAKE_CXX_FLAGS="-march=armv8-a" \
    -DEXECUTORCH_BUILD_SHARED=ON -DEXECUTORCH_BUILD_KERNELS_LLM=ON -DEXECUTORCH_BUILD_KERNELS_LLM_AOT=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_EXTENSION_THREADPOOL=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON -DEXECUTORCH_BUILD_XNNPACK=OFF \
    -DEXECUTORCH_BUILD_PYBIND=OFF -DEXECUTORCH_BUILD_TESTS=OFF

  echo "== build custom_ops_aot_lib on all 4 cores (the Pi runs warm; a fan helps)"
  cmake --build "$BUILD" --target custom_ops_aot_lib -j4
  NEW="$BUILD/extension/llm/custom_ops/libcustom_ops_aot_lib.so"
  [ -f "$NEW" ] || { echo "build produced no library"; exit 1; }

  echo "== install into the wheel"
  mkdir -p "$HOME/backup" "$HOME/.cache"
  [ -f "$HOME/backup/libcustom_ops_aot_lib.so.orig" ] || cp "$LIB" "$HOME/backup/libcustom_ops_aot_lib.so.orig"
  cp "$NEW" /tmp/new_custom_ops.so
  # the build links against versioned names (libexecutorch.so.1); the wheel ships unversioned ones, and its
  # libs/torch are found relative to the library itself
  patchelf --replace-needed libexecutorch.so.1 libexecutorch.so \
           --replace-needed libexecutorch_threadpool.so.1 libexecutorch_threadpool.so \
           --set-rpath '$ORIGIN/../../pybindings:$ORIGIN/../../../../torch/lib:$ORIGIN/../../../lib' \
           /tmp/new_custom_ops.so
  cp /tmp/new_custom_ops.so "$LIB"
  rm -f /tmp/new_custom_ops.so
  sha256sum "$LIB" | cut -d' ' -f1 > "$STAMP"
  echo "unresolved libraries (should print nothing):"
  ldd "$LIB" | grep "not found" || true
fi

echo "== check: Llama loads and answers"
cd "$REPO"
"$PY" - <<'PY'
import glob, os
import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401  (this is the import that used to die with SIGILL)
from pcbinspect.llm import ExecuTorchLlama
models = sorted(glob.glob("models/llm/llama3_2_1b*.pte"))
if not models or not os.path.exists("models/llm/tokenizer.model"):
    print("custom ops import OK (no Llama model files here to run a prompt)")
else:
    llm = ExecuTorchLlama(models[0], "models/llm/tokenizer.model")
    p = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nIn one sentence, what is a short circuit on a PCB?"
         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    g = llm.generate(p, 32)
    print(os.path.basename(models[0]), "decode tok/s", g.decode_tps, "->", repr(g.text))
PY
echo "Done. Restart the station to pick it up (it loads the library at start)."
