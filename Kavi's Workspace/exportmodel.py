from ultralytics import YOLO

# Load your trained .pt
model = YOLO(r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\models\imagesbase\bestdetection.pt")

# Export to a TensorRT engine (FP16)
model.export(format="tensorrt", device=0, half=True)