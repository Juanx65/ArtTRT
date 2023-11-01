from ultralytics import YOLO

# Load a model
model = YOLO('weights/yolov8m-pose.pt')  # load an official model

# Export the model
#model.export(format='engine',imgsz=640,half=False, dynamic=False, batch=1)
model.export(format='engine',half=True)