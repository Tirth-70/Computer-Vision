import cv2
import mediapipe as mp
import numpy as np

def get_limits(color):

    c = np.uint8([[color]])
    hsv_color = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    lower_limit = hsv_color[0][0][0] - 10, 100, 100
    upper_limit = hsv_color[0][0][0] + 10, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)

    return lower_limit, upper_limit

def face_detection(image, face_detection):

    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            y = y - 30
            image = cv2.rectangle(image, (x, y), (x+w, y+h+30), (0, 0, 0), 2)

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            face = image[y:y+h+30, x:x+w]
            face = cv2.blur(face, (99, 99))
            image[y:y+h+30, x:x+w] = face
    return image