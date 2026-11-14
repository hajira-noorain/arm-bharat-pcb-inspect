#!/usr/bin/env bash
# Retrained bare-board detectors (YOLO26s, YOLO26m) for the Pi:
#   1. ExecuTorch FP32 + INT8 export (XNNPACK)
#   2. the exported FP32 file re-scored on the held-out test boards, and the PASS/FAIL/CHECK gates calibrated
#      on its cal-board scores (what ships is what is calibrated)
#   3. INT8 accuracy on the test boards
#   4. the golden-board (fused) gate for DeepPCB, same trust setting as the deployed detector (0.7)
set -x
export MPLBACKEND=Agg PCB_UNSUPPORTED_SCALE=0.7
P=/content/etv/bin/python
CODE=/content/code21
rm -rf $CODE && mkdir -p $CODE && unzip -q -o /content/code21.zip -d $CODE
until ! pgrep -f dump_parts_visa.py > /dev/null; do sleep 30; done
# the ExecuTorch environment was made for Llama: add what the vision export and evaluation need, without
# letting the installer move torch (executorch 1.6 is built against the installed version)
TV=$($P -c "import torch; print(torch.__version__)")
uv pip install -q --python $P opencv-python-headless ultralytics "torch==$TV" 2>&1 | tail -n 3
$P -c "import cv2, ultralytics, torch, executorch; print('ENV_OK', cv2.__version__, ultralytics.__version__, torch.__version__)"
for M in s m; do
  W=/content/ex5$M; mkdir -p $W/data $W/results $W/models/vision
  ln -sfn /content/data/bare_yolo_v3 $W/data/bare_yolo
  cd $CODE
  $P training/export_vision.py --work $W --yolo /content/out5/runs/y26${M}_v5/weights/best.pt --only detector --name pcb_y26$M 2>&1 \
     | grep -E "^detector|failed|rror|Traceback" | cut -c1-400
  ls -la $W/models/vision
  $P training/eval_bare.py --work $W --backend executorch --weights $W/models/vision/pcb_y26${M}_xnnpack.pte \
     --tag v5${M}_et --data /content/data/bare_yolo_v3 2>&1 | grep -E "mAP50|gate |rror|Traceback" | cut -c1-300
  if [ -f $W/models/vision/pcb_y26${M}_xnnpack_int8.pte ]; then
    $P training/eval_bare.py --work $W --backend executorch --weights $W/models/vision/pcb_y26${M}_xnnpack_int8.pte \
       --tag v5${M}_et_int8 --data /content/data/bare_yolo_v3 --no-gate --splits test 2>&1 | grep -E "mAP50|rror|Traceback" | cut -c1-300
  fi
  $P training/eval_golden.py --work $W --tag v5${M}_et --design deeppcb 2>&1 | grep -E "fused|golden_diff|^detector|rror|Traceback" | cut -c1-300
  ls -la $W/models/gate $W/results
  echo EXPORT_${M}_DONE
done
cd /content && rm -rf collect4 final4.zip && mkdir -p collect4/models/vision collect4/gates_s collect4/gates_m collect4/results
for M in s m; do
  W=/content/ex5$M
  cp $W/models/vision/pcb_y26${M}_xnnpack*.pte $W/models/vision/pcb_y26$M.pt collect4/models/vision/ 2>/dev/null
  cp $W/models/gate/*.json collect4/gates_$M/
  cp $W/results/bare_eval_*.json $W/results/golden_eval_*.json $W/results/export_vision_*.json collect4/results/ 2>/dev/null
done
if grep -rlE --include='*.json' 'hf_[A-Za-z0-9]{20,}' collect4; then echo TOKEN_FOUND; exit 1; fi
zip -qr final4.zip collect4 && ls -l final4.zip
echo EXPORT_V5_ALL_DONE
