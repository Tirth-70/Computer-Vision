from ultralytics import YOLO
import cv2

# Load model
model1 = YOLO("/Users/tirthpatel/Desktop/Computer Vision/Image Classification/Clean-Messy/yolo logs/train2/weights/best.pt")
model2 = YOLO('/Users/tirthpatel/Desktop/Computer Vision/Image Detection/stuff detection/yolo logs/train2/weights/best.pt')

cap = cv2.VideoCapture("ROOM_PROJECT/data/6612061-uhd_4096_2160_25fps.mp4")

class_dict = model2.names
color_dict = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (0, 255, 255)}
names_dict = model1.names
ret = True
while ret:
    ret, frame = cap.read()

    if ret:
        # Run inference
        res1 = model2(frame)
        res2 = model1(frame)


        # display using cv2
        for box in res1[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = round(conf, 2)
            color = color_dict[int(cls)]
            class_label = model2.names[int(cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, str(conf), (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)



        label = names_dict[res2[0].probs.top1]
        confidence = round(float(res2[0].probs.top1conf),2)

        # just adding the label and confidence to the image
        cv2.putText(frame, label, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, str(confidence), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()