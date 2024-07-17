from ultralytics import YOLO

# Load model
model = YOLO('yolov8n-cls.pt')

model.train(data="/Users/tirthpatel/Desktop/Computer Vision/Image Classification/Weather Classification/data", 
            epochs=20, imgsz=64, project="/Users/tirthpatel/Desktop/Computer Vision/Image Classification/Weather Classification/yolo logs", device="mps")