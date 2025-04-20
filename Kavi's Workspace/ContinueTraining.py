from ultralytics import YOLO
import multiprocessing
import os 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # point to the checkpoint you want to resume from
    model = YOLO(r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\runs\detect\train6\weights\last.pt")
    model.info()
    print(model.model)
    #model.train(resume=True, epochs=300)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()