"""Train the bare-board defect detector (runs on the laptop GPU, BEFORE deployment).

Usage: python training/train_bare.py --model yolo26n.pt --name y26n
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\sulai\pcb_work\data\bare_yolo\data.yaml")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--name", default="y26n")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--project", default=r"C:\Users\sulai\pcb_work\runs")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=640, batch=args.batch, patience=30,
        project=args.project, name=args.name, exist_ok=True, seed=0, deterministic=False,
        workers=8, cos_lr=True,
        # PCB defects have no canonical orientation: flip both ways
        fliplr=0.5, flipud=0.5,
        # colour jitter covers binarised scans, green solder mask and bare copper
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
        mosaic=1.0, close_mosaic=15, scale=0.4,
        plots=True,
    )
    print("best:", Path(args.project) / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
