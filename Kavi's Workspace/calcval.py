from ultralytics import YOLO
import torch.multiprocessing as mp
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_validation():
    # Load your trained model
    model = YOLO(r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\models\affectnet_yolov11\affectnet_yolov11.pt")  

    # Run validation
    metrics = model.val(
        data=r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\Kavi's Workspace\data.yaml",  # same data config used during training
        imgsz=96,
        batch=32,
        workers=16  ,
        split='test',
        cache='disk', 
    )

    print(f"\nValidation Results:")
    print(f"  mAP@0.5: {metrics.map50:.3f}")
    print(f"  mAP@0.5:0.95: {metrics.map:.3f}")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")

if __name__ == "__main__":
    mp.freeze_support()
    run_validation()
