import multiprocessing
from ultralytics import YOLO
import torch
import os



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    model = YOLO(r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\runs\detect\train5\weights\best.pt", task="detect")

    model.train(
        data="D:\\UTS\\Semester 3\\Deep Learning\\Assignment 3\\PresentPerfect\\Kavi's Workspace\\data.yaml",
        epochs=100,
        lr0=0.02,
        # ─── maximise GPU utilisation ───────────────────────────────────────
        imgsz=640,             # standard size; bump up if you need higher res
        batch=32,              # n=64 often fits on 12 GB with this tiny weights file
                              # if you OOM, drop to 32 or 48

        # ─── data loading I/O ──────────────────────────────────────────────
        workers=16,            # match your logical CPU cores :contentReference[oaicite:3]{index=3}
        cache='disk',           # preload dataset into RAM (fast) :contentReference[oaicite:4]{index=4}

        # ─── precision & scheduling ───────────────────────────────────────
        device='cuda:0',
        amp=True,              # mixed‑precision for speed & memory :contentReference[oaicite:5]{index=5}

        # ─── checkpoint & plotting ────────────────────────────────────────
        save=True,
        save_period=10,        # save a .pt every 10 epochs :contentReference[oaicite:6]{index=6}
        plots=True,            # dump loss/metric curves at end :contentReference[oaicite:7]{index=7}

        # (other hyperparameters: lr0, momentum, etc., tunable per dataset)
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()