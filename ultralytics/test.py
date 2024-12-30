#from ultralytics import YOLO
import sys, os
# abs path
cur_dir = os.getcwd()
yolo_path = os.path.join(cur_dir, "ultralytics")
print(yolo_path)
sys.path.insert(0, yolo_path)

from ultralytics import YOLO  # Now you can import as usual

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model 


path = 'coco8.yaml'
model.train(data=path, epochs=1)  # train the model
