# Color detection using OpenCV

import cv2
from utils import get_limits
from PIL import Image

cap = cv2.VideoCapture(0)

color = [0, 255, 255] # BGR


while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_limit, upper_limit = get_limits(color)

    mask = cv2.inRange(hsv, lower_limit, upper_limit, cv2.THRESH_BINARY)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('original', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()