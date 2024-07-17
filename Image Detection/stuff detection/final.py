from ultralytics import YOLO
import cv2

# Load model
model = YOLO('/Users/tirthpatel/Desktop/Computer Vision/Image Detection/stuff detection/yolo logs/train2/weights/best.pt')
img = cv2.imread('Image Detection/stuff detection/data/images/train/00d9a357b10f81d8.jpg')

# Run inference
results = model(img)

color_dict = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (0, 255, 255)}
class_dict = model.names

# display using cv2
for box in results[0].boxes.data:
    x1, y1, x2, y2, conf, cls = box.tolist()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    conf = round(conf, 2)
    color = color_dict[int(cls)]
    class_label = class_dict[int(cls)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(img, str(conf), (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


cv2.imshow('image', img)
cv2.waitKey(0)