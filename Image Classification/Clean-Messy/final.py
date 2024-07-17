from ultralytics import YOLO
import cv2

model = YOLO("/Users/tirthpatel/Desktop/Computer Vision/Image Classification/Clean-Messy/yolo logs/train2/weights/best.pt")
img = cv2.imread("/Users/tirthpatel/Desktop/Computer Vision/Image Classification/Clean-Messy/data/train/clean/0.png")

results = model(img)

names_dict = results[0].names

label = names_dict[results[0].probs.top1]
confidence = round(float(results[0].probs.top1conf),2)

# just adding the label and confidence to the image
cv2.putText(img, label, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.putText(img, str(confidence), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('image', img)

cv2.waitKey(0)