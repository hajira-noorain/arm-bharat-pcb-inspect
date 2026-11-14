#!/usr/bin/env bash
# Raw parts-check measurements on VisA, one process per design (the alignment is single-threaded CPU work).
pkill -f dump_parts_visa.py
sleep 2
CODE=/content/code20; OUT=/content/outp
cd $CODE
for d in pcb1 pcb2 pcb3 pcb4; do
  OMP_NUM_THREADS=3 OPENCV_FOR_THREADS_NUM=3 nohup python -u training/dump_parts_visa.py --visa /content/VisA \
     --detector $OUT/runs/parts_det/weights/best.pt --designs $d --out $OUT/results/parts_visa_raw_$d.json \
     > /content/run_dump_$d.log 2>&1 &
done
wait
cd $OUT/results && rm -f /content/raw.zip && zip -q /content/raw.zip parts_visa_raw_pcb*.json && ls -l /content/raw.zip
echo DUMP_ALL_DONE
