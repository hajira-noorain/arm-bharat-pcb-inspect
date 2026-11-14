#!/usr/bin/env bash
# FPIC (FICS PCB Image Collection, University of Florida) straight onto Colab, from the file links printed on
# its own page. Images are 7.2 GB. Nothing here goes into the repo.
set -x
B=https://physicaldb.ece.ufl.edu/wp-content/uploads/2025/04/PCB_DATA_UPLOAD
D=/content/ext/fpic
mkdir -p $D && cd $D
for f in metadata smd_annotation ocr_annotation vtp_annotation color_checker pcb_image; do
  [ -s $f.zip ] || curl -sSL --retry 5 -C - -A "Mozilla/5.0" -o $f.zip $B/$f.zip
  ls -l $f.zip
  mkdir -p $f && unzip -q -o $f.zip -d $f || echo "UNZIP_FAILED $f"
done
du -sh $D/*/ | cat
find $D -maxdepth 3 -iname '*readme*' -o -maxdepth 3 -iname '*licen*' -o -maxdepth 3 -iname '*.txt' | head -20
echo FPIC_DONE
