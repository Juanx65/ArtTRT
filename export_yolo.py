from ultralytics import YOLO

# Load a model
model = YOLO('weights/best_salmon.pt')  # load an official model

# Export the model
model.export(format='engine',imgsz=640,half=True, dynamic=False, batch=4)