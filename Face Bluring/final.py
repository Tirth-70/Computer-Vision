# Face Detection and Bluring

import cv2
import mediapipe as mp

def face_detection(image, face_detection):

    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            y = y - 30
            image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            face = image[y:y+h, x:x+w]
            face = cv2.blur(face, (99, 99))
            image[y:y+h, x:x+w] = face
    return image

cam = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection = 0)
while True:
    ret, frame = cam.read()
    cv2.imshow('Face Detection', face_detection(frame, mp_face_detection))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()