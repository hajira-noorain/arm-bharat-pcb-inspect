#!/usr/bin/env bash
# Parts check with the rule chosen on good training photos: the station's own code end to end, once.
set -x
until grep -q EXPORT_V5_ALL_DONE /content/run_export_v5.log; do sleep 30; done
CODE=/content/code22; OUT=/content/outp; D=/content/data/parts
rm -rf $CODE && mkdir -p $CODE && unzip -q -o /content/code22.zip -d $CODE
cd $OUT/results && cp -n parts_check_eval.json parts_check_eval_fifth.json && cp -n parts_check_visa.json parts_check_visa_fifth.json
cd $CODE
DET=$OUT/runs/parts_det/weights/best.pt; CLS=$OUT/runs/parts_cls/weights/best.pt
python training/eval_parts.py --fpic /content/ext/fpic --manifest $D/manifest.json --detector $DET --classifier $CLS \
   --out $OUT/results/parts_check_eval.json 2>&1 | grep -E "RESULT|rror|Traceback" | cut -c1-3000
echo PARTS8_EVAL_DONE
for d in pcb1 pcb2 pcb3 pcb4; do
  OMP_NUM_THREADS=3 python -u training/eval_parts_visa.py --visa /content/VisA --detector $DET --classifier $CLS --designs $d \
     --out $OUT/results/parts_check_visa_$d.json > /content/run_parts8_$d.log 2>&1 &
done
wait
grep -h -E "RESULT|rror|Traceback" /content/run_parts8_pcb*.log | cut -c1-300
cd $OUT/results && rm -f /content/p8.zip && zip -q /content/p8.zip parts_check_eval.json parts_check_visa_pcb*.json && ls -l /content/p8.zip
echo PARTS8_ALL_DONE
