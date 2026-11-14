#!/usr/bin/env bash
# The runtime was reset. Rebuild, then redo the two lost steps:
#   A. YOLO26s / YOLO26m: ExecuTorch export, re-score on the held-out test boards, PASS/FAIL/CHECK gates, golden gate
#   B. parts check with the chosen rule: the station's own code on held-out FPIC and on VisA
# Every input is fetched from its original host or re-uploaded with a checksum; nothing is reused unchecked.
set -x
export MPLBACKEND=Agg
U=/content/up; W=/content/w; D=/content/data
mkdir -p $W $D /content/ext /content/res
# ---------------------------------------------------------------- inputs
cat $U/y26s.part* > $W/pcb_y26s_v5.pt && cat $U/y26m.part* > $W/pcb_y26m_v5.pt
cp $U/parts_y26n.pt $U/parts_kind.pt $W/
(cd $W && sha256sum -c $U/sums.txt) || { echo "CHECKSUM_FAILED"; exit 1; }
rm -rf /content/code && mkdir -p /content/code && unzip -q -o $U/code22.zip -d /content/code
# ---------------------------------------------------------------- environments
pip -q install ultralytics opencv-python-headless 2>&1 | tail -n 1
pip -q install uv 2>&1 | tail -n 1
uv venv -q /content/etv --python 3.12
uv pip install -q --python /content/etv/bin/python executorch==1.6.0 ultralytics opencv-python-headless torchvision pyyaml numpy 2>&1 | tail -n 3
/content/etv/bin/python -c "import cv2, torchvision, ultralytics, torch, executorch; print('ENV_OK', torch.__version__, torchvision.__version__)"
echo SETUP_DONE
# ---------------------------------------------------------------- data
cd $D
git clone --depth 1 -q https://github.com/tangsanli5201/DeepPCB.git &
git clone --depth 1 -q https://github.com/Ironbrotherstyle/PCB-DATASET.git HRIPCB &
(curl -sSL --retry 5 -o PCB_Defect_mendeley.zip "https://data.mendeley.com/public-files/datasets/vdj74sngvn/files/9a62af5d-f3ea-4b3d-a99b-e48115d51c2f/file_downloaded" \
   && mkdir -p mendeley && unzip -q -o PCB_Defect_mendeley.zip -d mendeley) &
(mkdir -p dspcbsd && curl -sSL --retry 5 -o dspcbsd/DsPCBSD+.zip https://ndownloader.figshare.com/files/44069552 \
   && cd dspcbsd && unzip -q -o "DsPCBSD+.zip" "Data_YOLO/images/*/*" "Data_YOLO/labels/*/*" -d x) &
python $U/fetch_ext.py defectdet &
bash $U/fetch_fpic.sh > /content/run_fpic.log 2>&1 &
(mkdir -p /content/VisA/split_csv && cd /content/VisA \
   && curl -sSL --retry 5 -o split_csv/1cls.csv https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv \
   && curl -sSL --retry 5 -o VisA.tar https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar \
   && tar -xf VisA.tar --wildcards '*pcb1/*' '*pcb2/*' '*pcb3/*' '*pcb4/*' && rm -f VisA.tar) &
wait
tail -n 2 /content/ext/fetch.log; grep -E "FPIC_DONE|UNZIP_FAILED" /content/run_fpic.log; ls /content/VisA
cd /content/code
python training/prepare_bare.py --data $D --out $D/bare_yolo_v3 --defectdet /content/ext/defectdet 2>&1 | tail -n 3
python training/split_fingerprint.py $D/bare_yolo_v3 | tee /content/res/split_fingerprint_rebuild.txt
if ! grep -q "cal 744 356da8f161db0786" /content/res/split_fingerprint_rebuild.txt || ! grep -q "test 1006 3f6578122e0e601a" /content/res/split_fingerprint_rebuild.txt; then
  echo "SPLIT_MISMATCH"; exit 1
fi
echo DATA_DONE
# ---------------------------------------------------------------- A: detectors for the Pi (CPU)
(
export PCB_UNSUPPORTED_SCALE=0.7
P=/content/etv/bin/python
for M in s m; do
  X=/content/ex5$M; mkdir -p $X/data $X/results $X/models/vision
  ln -sfn $D/bare_yolo_v3 $X/data/bare_yolo
  cd /content/code
  $P training/export_vision.py --work $X --yolo $W/pcb_y26${M}_v5.pt --only detector --name pcb_y26$M 2>&1 | grep -E "^detector|failed|rror|Traceback" | cut -c1-300
  $P training/eval_bare.py --work $X --backend executorch --weights $X/models/vision/pcb_y26${M}_xnnpack.pte \
     --tag v5${M}_et --data $D/bare_yolo_v3 2>&1 | grep -E "mAP50|gate |rror|Traceback" | cut -c1-300
  [ -f $X/models/vision/pcb_y26${M}_xnnpack_int8.pte ] && $P training/eval_bare.py --work $X --backend executorch \
     --weights $X/models/vision/pcb_y26${M}_xnnpack_int8.pte --tag v5${M}_et_int8 --data $D/bare_yolo_v3 --no-gate --splits test 2>&1 \
     | grep -E "mAP50|rror|Traceback" | cut -c1-300
  $P training/eval_golden.py --work $X --tag v5${M}_et --design deeppcb 2>&1 | grep -E "fused|golden_diff|^detector|rror|Traceback" | cut -c1-300
  mkdir -p /content/res/gates_$M /content/res/models
  cp $X/models/gate/*.json /content/res/gates_$M/ 2>/dev/null
  cp $X/models/vision/pcb_y26${M}_xnnpack*.pte /content/res/models/ 2>/dev/null
  cp $X/results/bare_eval_*.json $X/results/golden_eval_*.json $X/results/export_vision_*.json /content/res/ 2>/dev/null
  echo EXPORT_${M}_DONE
done
) > /content/run_export.log 2>&1 &
# ---------------------------------------------------------------- B: parts check with the chosen rule (GPU + CPU)
(
cd /content/code
python training/eval_parts.py --fpic /content/ext/fpic --manifest $U/parts_manifest.json --detector $W/parts_y26n.pt \
   --classifier $W/parts_kind.pt --out /content/res/parts_check_eval.json 2>&1 | grep -E "RESULT|rror|Traceback" | cut -c1-3000
for d in pcb1 pcb2 pcb3 pcb4; do
  OMP_NUM_THREADS=2 python -u training/eval_parts_visa.py --visa /content/VisA --detector $W/parts_y26n.pt \
     --classifier $W/parts_kind.pt --designs $d --out /content/res/parts_check_visa_$d.json > /content/run_visa_$d.log 2>&1 &
done
wait
grep -h -E "RESULT|rror|Traceback" /content/run_visa_pcb*.log | cut -c1-300
echo PARTS_DONE
) > /content/run_parts.log 2>&1 &
wait
cd /content && if grep -rlE --include='*.json' 'hf_[A-Za-z0-9]{20,}' res; then echo TOKEN_FOUND; exit 1; fi
rm -f res.zip && zip -qr res.zip res && ls -l res.zip
echo REBUILD_ALL_DONE
