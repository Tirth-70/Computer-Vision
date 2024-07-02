# Blur the face and detect yellow color

import cv2
import mediapipe as mp
from PIL import Image

from utils import face_detection, get_limits

cam = cv2.VideoCapture(0)

mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection = 0)

# color = [0, 165, 255] # BGR Orange
color = [0, 255, 255] # BGR Yellow

while True:

    ret, frame = cam.read()

    frame = face_detection(frame, mp_face_detection)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_limit, upper_limit = get_limits(color)

    mask = cv2.inRange(hsv, lower_limit, upper_limit, cv2.THRESH_BINARY)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Yellow - Yellow color detection', (0, 100), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Black - face blur', (0, 130), font, 1, (200, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()